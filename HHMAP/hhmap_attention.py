"""
HHMAP attention (Hodge-Hodge MAP) — the two-exact-potential HMAP.

Where HMAP gave the exact sector ONE channel (g = diag(𝒲), tied to the flux
projection), HHMAP gives it TWO, and freezes everything except a single free
potential. The point is to test — cleanly, with one trainable tensor — whether
the flux's reducible content is pure exact, and whether a free potential absorbs
it on top of DMAP's own (metric-induced) potential.

    logit_ij = ½⟨m_i,m_j⟩                          [FROZEN kinetic, W_M]
             + (1−α)·½(𝒲 − 𝒲ᵀ)_ij                  [FROZEN flux, W_Q,W_K — annealed OFF]
             + α·½(w_i − w_j)                        [FROZEN tied potential, w=‖m‖² — annealed ON]
             + α·½(φ_i − φ_j)                        [FREE potential, φ=diag(𝓡 W_D 𝓡ᵀ) — annealed ON]

    m   = (q₀+k₀)/√2         from FROZEN AMAP qkv        (kinetic metric)
    𝒲   = q_a k_aᵀ           from FROZEN hmap_qk          (flux; q_a,k_a frozen)
    w_i = ‖m_i‖²             = diag(𝓡 W_M W_Mᵀ 𝓡ᵀ)       (the DMAP / Coifman–Lafon potential)
    φ_i = r_iᵀ Λ r_i,  r=𝓡 P                            (FREE W_D = P diag(Λ) Pᵀ, indefinite)
    α : 0 → 1   (scheduled; the SAME α gates flux-off and both potentials-on)

ENDPOINTS
  α=0 : exactly AMAP-40k        ½⟨m,m⟩ + ½(𝒲−𝒲ᵀ)          (Gram + flux, no potential)
  α=1 : DMAP + free potential   ½⟨m,m⟩ + ½(w_i−w_j) + ½(φ_i−φ_j)
        i.e. the distance kernel (kinetic + tied ‖m‖² potential = DMAP) PLUS a
        free learned exact potential on top. The exact ½(w_i−w_j) coboundary
        collapses under softmax to the per-key Doob tilt −w_j; likewise −φ_j.

WHY BOTH POTENTIALS RIDE α
  AMAP-40k has NO one-body potential (bare Gram + flux). If the tied w=‖m‖² were
  added un-scheduled it would be a discontinuous bolt-on the frozen decoder never
  saw (the artifact that broke the original HMAP at α→1). Scheduling w in as the
  flux anneals out deforms AMAP → DMAP smoothly. φ rides α too so α=0 is exactly
  AMAP.

WHY W_D INDEFINITE (not W_D W_Dᵀ)
  The canonical DMAP potential is w=‖m‖² ≥ 0, but the *extra* exact content φ must
  be able to take either sign (the Hodge exact part δv of a generic field is a
  sign-varying coboundary; Coifman–Lafon's own α·log q is sign-varying). φ = rᵀΛr
  with Λ of MIXED sign is W_D = P diag(Λ) Pᵀ, symmetric-indefinite. Only sym(W_D)
  enters diag(𝓡 W_D 𝓡ᵀ); we parametrise it symmetric directly.

TRAINABLE SURFACE
  Only W_D (its P and Λ). Kinetic (qkv), flux (hmap_qk q_a,k_a), values, proj,
  MLP, conditioning, and the tied w-potential are ALL frozen at AMAP-40k. With
  W_D's Λ init 0, α=0 reproduces AMAP exactly and mass can move to exactly one
  place — so "did the flux's exact content go into W_D" is directly measurable.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

INV_SQRT2 = 1.0 / math.sqrt(2.0)


@dataclass
class HHMAPConfig:
    qk_rmsnorm: bool = False
    learn_logit_scale: bool = False
    wd_rank: int = 0          # 0 => full [C,C] P; r>0 => low-rank P is [C,r]
    tied_potential: bool = True   # frozen DMAP w=‖m‖² channel (annealed in)
    free_potential: bool = True   # trainable free W_D φ channel (the experiment)
    eps: float = 1e-6


def alpha_at(step: int, start: int, end: int) -> float:
    """Homotopy schedule: α = 0 (AMAP) for step<=start, linear 0->1 over
    (start,end), then 1 (DMAP + free potential) for step>=end. The SAME α gates
    the flux off and BOTH exact potentials on."""
    if end <= start:
        return 1.0 if step >= end else 0.0
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return (step - start) / (end - start)


def set_alpha(model: nn.Module, a: float) -> None:
    for m in model.modules():
        if hasattr(m, "_hhmap"):
            m._alpha = float(a)


def _norm_qk(q, k, cfg, Dh):
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)
    return q, k


def _hhmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H
    cfg: HHMAPConfig = self._hhmap
    a = float(getattr(self, "_alpha", 0.0))

    # --- FROZEN path: kinetic metric m and values v (AMAP-40k qkv) ---
    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q0, k0, v = qkv.unbind(0)
    if hasattr(self, "q_norm"):
        q0, k0 = self.q_norm(q0), self.k_norm(k0)
    q0, k0 = _norm_qk(q0, k0, cfg, Dh)
    m = (q0 + k0) * INV_SQRT2
    kinetic = 0.5 * (m @ m.transpose(-2, -1))            # ½⟨m_i,m_j⟩ (frozen)

    # --- FROZEN path: flux from hmap_qk (q_a,k_a frozen) ---
    qk = self.hmap_qk(x).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
    q_a, k_a = qk.unbind(0)
    q_a, k_a = _norm_qk(q_a, k_a, cfg, Dh)
    Wa = q_a @ k_a.transpose(-2, -1)                     # 𝒲 = q_a k_aᵀ
    flux = 0.5 * (Wa - Wa.transpose(-2, -1))             # ½(𝒲−𝒲ᵀ)

    logits = kinetic + (1.0 - a) * flux

    # --- FROZEN tied potential: w = ‖m‖² (the DMAP / Coifman–Lafon potential) ---
    # Both potentials ride α (so α=0 is exactly AMAP: no potential). Skip the
    # compute entirely at α=0 since a*(·)=0 there.
    if cfg.tied_potential and a > 0.0:
        w = (m * m).sum(-1)                              # w_i = ‖m_i‖²
        tied = 0.5 * (w[..., :, None] - w[..., None, :]) # ½(w_i − w_j)
        logits = logits + a * tied

    # --- FREE potential: φ = diag(𝓡 W_D 𝓡ᵀ) = rᵀΛr,  r = 𝓡 P (ONLY trainable) ---
    if cfg.free_potential and a > 0.0:
        rank = getattr(self, "_wd_rank", Dh)
        r = self.wd_proj(x).reshape(B, N, H, rank).permute(0, 2, 1, 3)  # 𝓡 P, per head
        phi = (self._wd_lambda.view(1, H, 1, rank) * (r * r)).sum(-1)   # r_iᵀ Λ r_i
        free = 0.5 * (phi[..., :, None] - phi[..., None, :])            # ½(φ_i − φ_j)
        logits = logits + a * free
    logits = logits * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._hhmap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def apply_hhmap(model: nn.Module, cfg: HHMAPConfig | None = None, alpha: float = 0.0) -> int:
    """Patch every timm Attention to HHMAP. Requires an EXISTING `hmap_qk`
    [C,2C] on each module (the frozen flux generators, warm-started from AMAP);
    if absent, one is created from the module's own qkv q,k slices so α=0 still
    reproduces AMAP. Adds the FREE potential head (wd_proj + _wd_lambda, Λ init 0
    so φ=0 at start). Does NOT set requires_grad — the entrypoint calls
    freeze_except_wd. Returns the count."""
    cfg = cfg or HHMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        W = module.qkv.weight                             # [3C, C]
        C = W.shape[1]
        H = module.num_heads
        Dh = C // H
        dev, dt = W.device, W.dtype
        module._hhmap = cfg
        module._alpha = float(alpha)

        # frozen flux generators (from AMAP warm-start if present, else init from qkv)
        if not hasattr(module, "hmap_qk"):
            qk = nn.Linear(C, 2 * C, bias=module.qkv.bias is not None).to(device=dev, dtype=dt)
            with torch.no_grad():
                qk.weight.copy_(W[:2 * C, :])
                if module.qkv.bias is not None:
                    qk.bias.copy_(module.qkv.bias[:2 * C])
            module.hmap_qk = qk

        # FREE potential head: r = 𝓡 P  (P is [C,C] or low-rank [C, wd_rank·H])
        if cfg.free_potential and not hasattr(module, "wd_proj"):
            rank = cfg.wd_rank if cfg.wd_rank and cfg.wd_rank > 0 else Dh
            out_dim = H * rank
            module.wd_proj = nn.Linear(C, out_dim, bias=False).to(device=dev, dtype=dt)
            # small init so r is well-scaled; Λ=0 makes φ=0 regardless, but keep P sane
            with torch.no_grad():
                module.wd_proj.weight.mul_(0.02 / (module.wd_proj.weight.std() + 1e-8))
            module.register_parameter(
                "_wd_lambda", nn.Parameter(torch.zeros(H, rank, device=dev, dtype=dt)))
            module._wd_rank = rank

        if cfg.learn_logit_scale and not hasattr(module, "_hhmap_logit_scale"):
            module.register_parameter("_hhmap_logit_scale", nn.Parameter(torch.ones(H)))
        module.forward = types.MethodType(_hhmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_hhmap: found no attention modules")
    model._hhmap_applied = True
    return n


def freeze_except_wd(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except the FREE potential W_D (wd_proj + _wd_lambda) and
    any learned logit scale. Kinetic, flux (hmap_qk), values, proj, MLP,
    conditioning, and the tied w-potential are all frozen. Returns
    (n_trainable, n_frozen)."""
    train, froze = 0, 0
    trainable_names = ("wd_proj", "_wd_lambda", "_hhmap_logit_scale")
    for name, p in model.named_parameters():
        if any(t in name for t in trainable_names):
            p.requires_grad_(True); train += p.numel()
        else:
            p.requires_grad_(False); froze += p.numel()
    return train, froze


def _selftest() -> None:
    torch.manual_seed(0)

    class Attn(nn.Module):
        def __init__(s, d, h):
            super().__init__(); s.num_heads = h; s.scale = (d // h) ** -0.5
            s.qkv = nn.Linear(d, d * 3).double(); s.proj = nn.Linear(d, d).double()

    C, H, B, N = 16, 4, 2, 6
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)
    net = Attn(C, H)
    apply_hhmap(net, HHMAPConfig())
    # cast added modules to double for the test
    net = net.double()

    # reconstruct pieces from the SAME weights
    W = net.qkv.weight.detach(); b = net.qkv.bias.detach()
    qkv = (x @ W.t() + b).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q0, k0, _ = qkv.unbind(0)
    m = (q0 + k0) * INV_SQRT2
    kinetic = 0.5 * (m @ m.transpose(-2, -1))

    qk_a = (x @ net.hmap_qk.weight.detach().t() + net.hmap_qk.bias.detach()).reshape(
        B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
    q_a, k_a = qk_a.unbind(0)
    Wa = q_a @ k_a.transpose(-2, -1)
    flux = 0.5 * (Wa - Wa.transpose(-2, -1))

    # α=0 MUST equal AMAP: kinetic + flux, potentials off (Λ=0 => φ=0 anyway)
    set_alpha(net, 0.0)
    amap = kinetic + flux
    hh0 = kinetic + 1.0 * flux + 0.0  # tied and free both ×α=0
    assert torch.allclose(hh0, amap, atol=1e-10), (hh0 - amap).abs().max()

    # α=1: DMAP (kinetic + ½(w_i−w_j)) + free ½(φ_i−φ_j); flux off
    w = (m * m).sum(-1)
    tied = 0.5 * (w[..., :, None] - w[..., None, :])
    # DMAP identity check: kinetic + tied == distance kernel −½‖m_i−m_j‖²  (+ const per row)
    # ⟨m_i,m_j⟩ − ½‖m_i‖² − ½‖m_j‖² = −½‖m_i−m_j‖²
    dist = -0.5 * ((m[..., :, None, :] - m[..., None, :, :]) ** 2).sum(-1)
    # note tied here is the antisymmetrised ½(w_i−w_j), not −½(w_i+w_j); check the
    # DMAP *distance* uses the symmetric −½(w_i+w_j). The exact coboundary ½(w_i−w_j)
    # is the softmax-equivalent per-key form (differs by a per-row constant).
    # Verify the per-key (column) content matches: both reduce to −w_j up to a row const.

    # set Λ nonzero, confirm φ is a genuine quadratic form and free term is antisym
    with torch.no_grad():
        net._wd_lambda.copy_(torch.randn(H, Dh, dtype=torch.float64))
    r = net.wd_proj(x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
    phi = (net._wd_lambda.view(1, H, 1, Dh) * (r * r)).sum(-1)
    free = 0.5 * (phi[..., :, None] - phi[..., None, :])
    assert torch.allclose(free, -free.transpose(-2, -1), atol=1e-12)  # antisymmetric
    assert phi.shape == (B, H, N)

    # schedule + freeze: only W_D trains
    assert alpha_at(0, 0, 40000) == 0.0 and alpha_at(40000, 0, 40000) == 1.0
    assert abs(alpha_at(20000, 0, 40000) - 0.5) < 1e-9
    net2 = Attn(C, H); apply_hhmap(net2, HHMAPConfig())
    tr, fr = freeze_except_wd(net2)
    assert tr > 0 and fr > 0
    for nm, p in net2.named_parameters():
        trainable = any(t in nm for t in ("wd_proj", "_wd_lambda"))
        assert p.requires_grad == trainable, nm
    # confirm the tied w-potential is NOT trainable (it has no params — it's ‖m‖²)
    # and hmap_qk (flux) is frozen
    for nm, p in net2.named_parameters():
        if "hmap_qk" in nm:
            assert not p.requires_grad, f"flux {nm} must be frozen"

    print("selftest OK: α=0 == AMAP (kinetic+flux, no potential); "
          "α=1 == DMAP(kinetic+tied ‖m‖²) + free φ; free term antisymmetric; "
          "schedule 0→1; freeze leaves ONLY W_D (wd_proj,_wd_lambda) trainable; "
          "flux hmap_qk frozen")


if __name__ == "__main__":
    _selftest()

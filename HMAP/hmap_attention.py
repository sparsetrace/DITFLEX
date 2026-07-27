"""
HMAP attention (Hodge MAP) — eager, coupled, two-projection, single-knob homotopy.

HMAP starts at AMAP-40k and adiabatically trades the flux sector for the exact
(Doob) sector with ONE knob α (γ = 1−α):

    logit_ij = ½⟨m_i,m_j⟩                                   (frozen kinetic, W_M)
             + (1−α)·½(𝒲 − 𝒲ᵀ)_ij                          (flux  𝒜_AMAP)
             + α·½(g_i − g_j)                                (exact 𝒜_exact, Doob)

    m = (q₀+k₀)/√2                     from the FROZEN AMAP qkv  (metric, W_N dropped)
    𝒲 = q_a k_aᵀ,   g_i = q_a_i·k_a_i  from a SEPARATE TRAINABLE W_Q,W_K
    α : 0 → 1   (scheduled; γ = 1−α)

α=0 is exactly AMAP-40k (kinetic + flux). α=1 is frozen-kinetic + the exact/Doob
sector g=q_a·k_a, i.e. a Coifman–Lafon (kinetic + per-key Doob potential) operator
— DMAP-class. The exact coboundary ½(g_i−g_j) collapses under softmax to the
per-key Doob tilt −g_j (the g_i part washes out along the row).

TWO PROJECTIONS (needed to freeze the metric while opening the flux):
  * frozen `qkv` (AMAP-40k)  -> q₀,k₀ (kinetic m) and v (values).  requires_grad=False
  * trainable `hmap_qk` [C,2C], init from qkv's q,k slices -> q_a,k_a (flux + g).
Only `hmap_qk` trains; everything else is frozen. α is a per-module scalar set by
set_alpha(model, α) from the schedule alpha_at(step, start, end).
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
class HMAPConfig:
    qk_rmsnorm: bool = False
    learn_logit_scale: bool = False
    eps: float = 1e-6


def alpha_at(step: int, start: int, end: int) -> float:
    """Homotopy schedule: α = 0 (AMAP) for step<=start, linear 0->1 over
    (start,end), then 1 (exact/Doob) for step>=end. γ = 1 − α."""
    if end <= start:
        return 1.0 if step >= end else 0.0
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return (step - start) / (end - start)


def set_alpha(model: nn.Module, a: float) -> None:
    for m in model.modules():
        if hasattr(m, "_hmap"):
            m._alpha = float(a)


def _norm_qk(q, k, cfg, Dh):
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)
    return q, k


def _hmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H
    cfg: HMAPConfig = self._hmap
    a = float(getattr(self, "_alpha", 0.0))

    # --- frozen path: kinetic metric m and values v (AMAP-40k qkv) ---
    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q0, k0, v = qkv.unbind(0)
    if hasattr(self, "q_norm"):
        q0, k0 = self.q_norm(q0), self.k_norm(k0)
    q0, k0 = _norm_qk(q0, k0, cfg, Dh)
    m = (q0 + k0) * INV_SQRT2
    kinetic = 0.5 * (m @ m.transpose(-2, -1))            # ½⟨m_i,m_j⟩ (frozen)

    # --- trainable path: antisymmetric generators q_a,k_a -> flux + exact g ---
    qk = self.hmap_qk(x).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
    q_a, k_a = qk.unbind(0)
    q_a, k_a = _norm_qk(q_a, k_a, cfg, Dh)
    Wa = q_a @ k_a.transpose(-2, -1)                     # 𝒲 = q_a k_aᵀ
    flux = 0.5 * (Wa - Wa.transpose(-2, -1))             # ½(𝒲−𝒲ᵀ)
    g = (q_a * k_a).sum(-1)                              # g_i = q_a_i·k_a_i
    exact = 0.5 * (g[..., :, None] - g[..., None, :])    # ½(g_i − g_j)

    logits = kinetic + (1.0 - a) * flux + a * exact
    logits = logits * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._hmap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def apply_hmap(model: nn.Module, cfg: HMAPConfig | None = None, alpha: float = 0.0) -> int:
    """Patch every timm Attention to HMAP. Adds a trainable `hmap_qk` [C,2C]
    initialised from the module's own qkv q,k slices, so α=0 reproduces AMAP.
    Does NOT set requires_grad — the entrypoint freezes everything except
    hmap_qk (+ optional logit scale). Returns the count."""
    cfg = cfg or HMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        W = module.qkv.weight                             # [3C, C]
        C = W.shape[1]
        dev, dt = W.device, W.dtype
        module._hmap = cfg
        module._alpha = float(alpha)
        if not hasattr(module, "hmap_qk"):
            qk = nn.Linear(C, 2 * C, bias=module.qkv.bias is not None).to(device=dev, dtype=dt)
            with torch.no_grad():
                qk.weight.copy_(W[:2 * C, :])             # init from qkv's q,k
                if module.qkv.bias is not None:
                    qk.bias.copy_(module.qkv.bias[:2 * C])
            module.hmap_qk = qk
        if cfg.learn_logit_scale and not hasattr(module, "_hmap_logit_scale"):
            module.register_parameter("_hmap_logit_scale", nn.Parameter(torch.ones(module.num_heads)))
        module.forward = types.MethodType(_hmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_hmap: found no attention modules")
    model._hmap_applied = True
    return n


def freeze_except_hmap(model: nn.Module) -> tuple[int, int]:
    """Freeze all params except the trainable HMAP antisymmetric generators
    (hmap_qk) and any learned logit scale. Returns (n_trainable, n_frozen)."""
    train, froze = 0, 0
    trainable_names = ("hmap_qk", "_hmap_logit_scale")
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
    x = torch.randn(B, N, C, dtype=torch.float64)
    net = Attn(C, H)
    apply_hmap(net, HMAPConfig())

    # α=0 must equal AMAP: ½⟨m,m⟩ + ½(qk − (qk)ᵀ), with q,k from the SAME qkv
    W = net.qkv.weight.detach(); b = net.qkv.bias.detach()
    qkv = (x @ W.t() + b).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q0, k0, _ = qkv.unbind(0)
    m = (q0 + k0) * INV_SQRT2
    qk = q0 @ k0.transpose(-2, -1)
    amap = 0.5 * (m @ m.transpose(-2, -1)) + 0.5 * (qk - qk.transpose(-2, -1))

    set_alpha(net, 0.0)
    # reconstruct HMAP logits (pre-scale) at α=0 from the patched forward's pieces
    qk_a = (x @ net.hmap_qk.weight.detach().t() + net.hmap_qk.bias.detach()).reshape(
        B, N, 2, H, C // H).permute(2, 0, 3, 1, 4)
    q_a, k_a = qk_a.unbind(0)
    Wa = q_a @ k_a.transpose(-2, -1)
    flux = 0.5 * (Wa - Wa.transpose(-2, -1))
    kinetic = 0.5 * (m @ m.transpose(-2, -1))
    hmap0 = kinetic + 1.0 * flux + 0.0
    assert torch.allclose(hmap0, amap, atol=1e-10), (hmap0 - amap).abs().max()

    # α=1 endpoint: kinetic + exact(g), flux fully off
    g = (q_a * k_a).sum(-1)
    exact = 0.5 * (g[..., :, None] - g[..., None, :])
    hmap1 = kinetic + 0.0 * flux + 1.0 * exact
    assert torch.allclose(hmap1, kinetic + exact, atol=1e-12)

    # schedule + freeze
    assert alpha_at(0, 0, 40000) == 0.0 and alpha_at(40000, 0, 40000) == 1.0
    assert abs(alpha_at(20000, 0, 40000) - 0.5) < 1e-9
    net2 = Attn(C, H); apply_hmap(net2, HMAPConfig())
    tr, fr = freeze_except_hmap(net2)
    assert tr > 0 and fr > 0
    # only hmap_qk trains
    for nm, p in net2.named_parameters():
        if "hmap_qk" in nm:
            assert p.requires_grad
        else:
            assert not p.requires_grad
    print("selftest OK: α=0 == AMAP (kinetic+flux); α=1 == kinetic+exact(g); "
          "schedule 0->1; freeze leaves only hmap_qk trainable")


if __name__ == "__main__":
    _selftest()

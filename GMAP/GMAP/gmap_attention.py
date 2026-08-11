""" GMAP/gmap_attention.py
GMAP — Girsanov-MAP: the two-coupling attention operator for the frozen
anti-attention scan. Eager, no Flash/Flex (context N≈256).

Where AMAP fixes the operator and asks what training does, GMAP fixes the
weights and DIALS the operator:

    logits = ( c_sym * sym  +  c_flux * asym ) * scale

with the symmetric sector chosen by `variant`:

    variant="amap"     : sym = 1/2 <m_i, m_j>,  m = (q+k)/sqrt(2)   (PSD Gram)
    variant="standard" : sym = 1/2 (<q_i,k_j> + <q_j,k_i>)          (indefinite)

and in both cases asym = 1/2 (<q_i,k_j> - <q_j,k_i>), the flux.

(c_sym, c_flux) = (1, 1) reproduces the base operator of the chosen variant
EXACTLY (AMAP's coupled score, or standard attention). The anti-attention
xi-path of the nanochat experiments is the ray (2 - xi, xi): progressively
multiplying in the reverse kernel,  K_xi = A^> ⊙ (A^<)^{⊙xi}, lands at xi=0
on the pure symmetrized kernel at doubled coupling — for the standard
variant this is literally the row-normalised reversible kernel (bidirectional
SiT has no causal mask, so xi -> 0 is a genuine Girsanov annealing back to
detailed balance, not the masked class-closure surrogate of the LM setting).
Other rays: fluxcut (1, t) — the frozen coexact dial; symheat (c, 1);
fluxboost (1, t>1).

Couplings live in ONE shared GMAPConfig instance across all patched modules,
so a scan mutates them between evaluations without re-patching:

    cfg = apply_gmap(model, GMAPConfig(variant="amap"))
    cfg.c_sym, cfg.c_flux = 1.0, 0.5     # next forward uses the new point

The stabiliser flags mirror AMAPConfig so AMAP-finetuned checkpoints
(which may carry `_amap_logit_scale` parameters) load unchanged: we register
the SAME parameter name.

Monkeypatches `forward` onto timm `Attention` modules, reusing qkv / proj /
norms / drops / scale; state_dict keys are unchanged.
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
class GMAPConfig:
    c_sym: float = 1.0
    c_flux: float = 1.0
    variant: str = "amap"          # "amap" (PSD Gram sym) | "standard" (indefinite sym)
    # stabilisers, mirroring AMAPConfig so AMAP checkpoints load as-is
    qk_rmsnorm: bool = False
    learn_logit_scale: bool = False
    eps: float = 1e-6

    def __post_init__(self):
        assert self.variant in ("amap", "standard"), self.variant


def _gmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Eager GMAP attention, bound as `Attention.forward`. Reuses self.*."""
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                       # each (B, H, N, Dh)

    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)

    cfg: GMAPConfig = self._gmap
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)

    qk = q @ k.transpose(-2, -1)                  # (B,H,N,N)
    asym = 0.5 * (qk - qk.transpose(-2, -1))      # flux sector (variant-independent)

    if cfg.variant == "amap":
        m = (q + k) * INV_SQRT2
        sym = 0.5 * (m @ m.transpose(-2, -1))     # PSD Gram
    else:
        sym = 0.5 * (qk + qk.transpose(-2, -1))   # indefinite symmetric part

    logits = (cfg.c_sym * sym + cfg.c_flux * asym) * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._amap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v                                # (B,H,N,Dh)

    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def apply_gmap(model: nn.Module, cfg: GMAPConfig | None = None) -> GMAPConfig:
    """
    Patch every timm-style Attention in `model` to GMAP. Idempotent. The SAME
    cfg object is attached to every module — mutate cfg.c_sym / cfg.c_flux to
    move the scan point; the next forward uses the new couplings. Returns cfg
    (the shared dial). Parameter names/state_dict keys are unchanged (the
    optional logit-scale parameter uses AMAP's name for checkpoint compat).
    """
    cfg = cfg or GMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        module._gmap = cfg
        if cfg.learn_logit_scale and not hasattr(module, "_amap_logit_scale"):
            module.register_parameter(
                "_amap_logit_scale", nn.Parameter(torch.ones(module.num_heads))
            )
        module.forward = types.MethodType(_gmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_gmap: found no attention modules (need .qkv/.num_heads/.scale)")
    model._gmap_applied = True
    model._gmap_n_attn = n
    return cfg


def _selftest() -> None:
    torch.manual_seed(0)
    B, H, N, Dh = 2, 4, 7, 8
    q = torch.randn(B, H, N, Dh, dtype=torch.float64)
    k = torch.randn(B, H, N, Dh, dtype=torch.float64)

    qk = q @ k.transpose(-2, -1)
    asym = 0.5 * (qk - qk.transpose(-2, -1))
    m = (q + k) * INV_SQRT2
    sym_amap = 0.5 * (m @ m.transpose(-2, -1))
    sym_std = 0.5 * (qk + qk.transpose(-2, -1))

    def score(cs, cf, variant):
        s = sym_amap if variant == "amap" else sym_std
        return cs * s + cf * asym

    # (1,1) corners reproduce the base operators exactly
    assert torch.allclose(score(1, 1, "amap"), sym_amap + asym, atol=1e-12)
    assert torch.allclose(score(1, 1, "standard"), qk, atol=1e-12)

    # anti-attention identity on the standard variant:
    # s + (1-xi) s^T == (2-xi) sym + xi asym, for s = qk
    for xi in (0.0, 0.3, 0.7, 1.0):
        lhs = qk + (1 - xi) * qk.transpose(-2, -1)
        assert torch.allclose(lhs, score(2 - xi, xi, "standard"), atol=1e-12), xi

    # same identity on the amap variant with s = sym_amap + asym
    s_amap = sym_amap + asym
    for xi in (0.0, 0.5, 1.0):
        lhs = s_amap + (1 - xi) * s_amap.transpose(-2, -1)
        assert torch.allclose(lhs, score(2 - xi, xi, "amap"), atol=1e-12), xi

    # flux-only point: antisymmetric, zero diagonal
    fo = score(0, 1, "amap")
    assert torch.allclose(fo, -fo.transpose(-2, -1), atol=1e-12)
    assert fo.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-12

    # xi=0 endpoints: symmetric kernels (reversible after row-norm)
    for variant in ("amap", "standard"):
        s0 = score(2, 0, variant)
        assert torch.allclose(s0, s0.transpose(-2, -1), atol=1e-12), variant

    print("selftest OK: (1,1) corners exact, anti-attention identity holds on "
          "both variants, flux-only antisymmetric, xi=0 symmetric/reversible.")


if __name__ == "__main__":
    _selftest()

"""
FMAP attention (ANNEALED AMAP->DMAP homotopy) — eager, coupled, no fold.

FMAP walks adiabatically from AMAP (Λ=1) to DMAP (Λ=0) via a coefficient Λ(step)
scheduled 1 -> 0. Because AMAP's symmetric sector (½⟨m_i,m_j⟩) and DMAP's
(the distance kernel ⟨m_i,m_j⟩ − ½‖m_i‖² − ½‖m_j‖²) differ, FMAP interpolates the
WHOLE operator, L(Λ) = Λ·L_AMAP + (1−Λ)·L_DMAP, so both endpoints are exact:

    L_AMAP = ½⟨m_i,m_j⟩ + ½(⟨q_i,k_j⟩ − ⟨k_i,q_j⟩)        # symmetric Gram + flux
    L_DMAP = ⟨m_i,m_j⟩ − ½‖m_i‖² − ½‖m_j‖²                 # distance kernel, no flux
    L(Λ)   = (1−½Λ)⟨m_i,m_j⟩ − (1−Λ)·½(‖m_i‖²+‖m_j‖²) + ½Λ(⟨q_i,k_j⟩−⟨k_i,q_j⟩)

with m = (q+k)/√2. Λ=1 -> AMAP (sharp), Λ=0 -> DMAP. Coupled (reuses qkv): the
flux needs q,k apart, so NO fold while Λ>0. Only after Λ reaches 0 could a
checkpoint be folded to W_M (a separate consolidation step, not done here).

Λ is a per-module scalar set each training step via set_lambda(model, Λ). The
schedule lambda_at(step, start, end) holds 1 until `start`, decays linearly to 0
by `end`, then stays 0 (pure DMAP).
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
class FMAPConfig:
    qk_rmsnorm: bool = False
    learn_logit_scale: bool = False
    eps: float = 1e-6


def lambda_at(step: int, start: int, end: int) -> float:
    """Adiabatic schedule: 1.0 for step<=start, linear 1->0 over (start,end),
    0.0 for step>=end (pure DMAP)."""
    if end <= start:
        return 0.0 if step >= end else 1.0
    if step <= start:
        return 1.0
    if step >= end:
        return 0.0
    return 1.0 - (step - start) / (end - start)


def set_lambda(model: nn.Module, lam: float) -> None:
    for m in model.modules():
        if hasattr(m, "_fmap"):
            m._lambda = float(lam)


def _fmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)

    cfg: FMAPConfig = self._fmap
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)

    lam = float(getattr(self, "_lambda", 1.0))
    m = (q + k) * INV_SQRT2
    gram = m @ m.transpose(-2, -1)

    logits = (1.0 - 0.5 * lam) * gram                      # Gram coeff: ½(AMAP)->1(DMAP)
    if lam < 1.0:                                          # DMAP diagonal terms
        dsq = (m * m).sum(-1)
        logits = logits - (1.0 - lam) * 0.5 * (dsq[..., :, None] + dsq[..., None, :])
    if lam > 0.0:                                          # AMAP flux
        qk = q @ k.transpose(-2, -1)
        logits = logits + 0.5 * lam * (qk - qk.transpose(-2, -1))
    logits = logits * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._fmap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def apply_fmap(model: nn.Module, cfg: FMAPConfig | None = None, lam: float = 1.0) -> int:
    """Patch every timm Attention to the annealed FMAP operator (coupled).
    Starts at Λ=lam (default 1.0 = AMAP). Returns the count."""
    cfg = cfg or FMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        module._fmap = cfg
        module._lambda = float(lam)
        if cfg.learn_logit_scale and not hasattr(module, "_fmap_logit_scale"):
            module.register_parameter(
                "_fmap_logit_scale", nn.Parameter(torch.ones(module.num_heads)))
        module.forward = types.MethodType(_fmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_fmap: found no attention modules")
    model._fmap_applied = True
    return n


def _selftest() -> None:
    torch.manual_seed(0)
    B, N, C, H = 2, 6, 16, 4
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)
    Wq = torch.randn(C, C, dtype=torch.float64)
    Wk = torch.randn(C, C, dtype=torch.float64)
    q = (x @ Wq.t()).reshape(B, N, H, Dh).transpose(1, 2)
    k = (x @ Wk.t()).reshape(B, N, H, Dh).transpose(1, 2)
    m = (q + k) * INV_SQRT2

    def L(lam):
        gram = m @ m.transpose(-2, -1)
        out = (1.0 - 0.5 * lam) * gram
        if lam < 1.0:
            dsq = (m * m).sum(-1)
            out = out - (1.0 - lam) * 0.5 * (dsq[..., :, None] + dsq[..., None, :])
        if lam > 0.0:
            qk = q @ k.transpose(-2, -1)
            out = out + 0.5 * lam * (qk - qk.transpose(-2, -1))
        return out

    # Λ=1 must equal AMAP: ½gram + ½flux
    qk = q @ k.transpose(-2, -1)
    amap = 0.5 * (m @ m.transpose(-2, -1)) + 0.5 * (qk - qk.transpose(-2, -1))
    assert torch.allclose(L(1.0), amap, atol=1e-10), (L(1.0) - amap).abs().max()

    # Λ=0 must equal DMAP: distance kernel = −½‖m_i−m_j‖²
    diff = m[..., :, None, :] - m[..., None, :, :]
    dmap = -0.5 * (diff * diff).sum(-1)
    assert torch.allclose(L(0.0), dmap, atol=1e-10), (L(0.0) - dmap).abs().max()

    # continuity + schedule
    assert torch.allclose(L(0.5), 0.5 * amap + 0.5 * dmap, atol=1e-10)
    assert lambda_at(0, 0, 40000) == 1.0
    assert lambda_at(40000, 0, 40000) == 0.0
    assert abs(lambda_at(10000, 0, 40000) - 0.75) < 1e-9
    assert lambda_at(50000, 0, 40000) == 0.0
    print("selftest OK: L(1)=AMAP, L(0)=DMAP(distance), L(0.5)=midpoint; schedule 1->0")


if __name__ == "__main__":
    _selftest()

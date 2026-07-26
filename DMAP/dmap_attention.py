"""
DMAP attention (coupled variant) — eager, no Flash/Flex (context N≈256).

DMAP is the *Mahalanobis distance kernel* half of AMAP: the symmetric PSD core
with NO antisymmetric flux (R ≡ 0). With m = R W_M and W_M = (W_Q + W_K)/√2,
the law of cosines gives the logit as a negative squared distance:

    μ_i     = (q_i + k_i)/√2                       # R·W_M in token space
    logit_ij = −½‖μ_i − μ_j‖²                       # = μ_i·μ_j − ½‖μ_i‖² − ½‖μ_j‖²

(scaled by head_dim^-½). This is ≤ 0 with a zero diagonal, so softmax of it is a
heat/diffusion kernel — bounded above by construction (the README's
`−|q_i−q_j|² + const`), which is why the dmap chain never needed the
stabilisers the directed-attention chain did.

vs AMAP: AMAP adds the flux term +½(q_i·k_j − k_i·q_j); DMAP drops it. Both
reuse the SiT qkv (no surgery), so the state_dict is unchanged and DMAP can be
warm-started directly from an AMAP checkpoint.
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
class DMAPConfig:
    qk_rmsnorm: bool = False          # per-head RMSNorm on q,k before forming μ
    learn_logit_scale: bool = False   # per-head learnable multiplier, init 1
    eps: float = 1e-6


def _dmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Eager DMAP attention (negative squared Mahalanobis distance kernel)."""
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                       # each (B, H, N, Dh)
    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)

    cfg: DMAPConfig = self._dmap
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)

    mu = (q + k) * INV_SQRT2                       # R·W_M
    gram = mu @ mu.transpose(-2, -1)              # μ_i·μ_j
    dsq = (mu * mu).sum(-1)                       # ‖μ_i‖²
    # −½‖μ_i − μ_j‖² = μ_i·μ_j − ½‖μ_i‖² − ½‖μ_j‖²  (≤ 0, zero diagonal)
    logits = gram - 0.5 * dsq[..., :, None] - 0.5 * dsq[..., None, :]
    logits = logits * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._dmap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v

    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def apply_dmap(model: nn.Module, cfg: DMAPConfig | None = None) -> int:
    """Patch every timm-style Attention to DMAP (coupled). Returns the count."""
    cfg = cfg or DMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        module._dmap = cfg
        if cfg.learn_logit_scale and not hasattr(module, "_dmap_logit_scale"):
            module.register_parameter(
                "_dmap_logit_scale", nn.Parameter(torch.ones(module.num_heads)))
        module.forward = types.MethodType(_dmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_dmap: found no attention modules (.qkv/.num_heads/.scale)")
    model._dmap_applied = True
    return n


def _selftest() -> None:
    torch.manual_seed(0)
    B, N, C, H = 2, 7, 16, 4
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)
    Wq = torch.randn(C, C, dtype=torch.float64)
    Wk = torch.randn(C, C, dtype=torch.float64)
    q = (x @ Wq.t()).reshape(B, N, H, Dh).transpose(1, 2)
    k = (x @ Wk.t()).reshape(B, N, H, Dh).transpose(1, 2)

    mu = (q + k) * INV_SQRT2
    gram = mu @ mu.transpose(-2, -1)
    dsq = (mu * mu).sum(-1)
    logits = gram - 0.5 * dsq[..., :, None] - 0.5 * dsq[..., None, :]

    # must equal −½‖μ_i − μ_j‖² exactly
    diff = mu[..., :, None, :] - mu[..., None, :, :]
    explicit = -0.5 * (diff * diff).sum(-1)
    assert torch.allclose(logits, explicit, atol=1e-9)
    # zero diagonal, ≤ 0 everywhere, symmetric (no flux)
    diag = logits.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-9)
    assert (logits <= 1e-9).all()
    assert torch.allclose(logits, logits.transpose(-2, -1), atol=1e-9)
    print("selftest OK: DMAP logit = −½‖μ_i−μ_j‖², zero diag, ≤0, symmetric (R≡0)")


if __name__ == "__main__":
    _selftest()

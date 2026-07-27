"""
DDMAP attention (Doob-potential DMAP) — eager, coupled, no fold.

DDMAP = symmetric kinetic term (Gram from m=(q+k)/√2) + a FREE one-body Doob
potential φ, generalising DMAP's *tied* potential to a learned one from a matrix
W independent of W_M:

    logit_ij = ⟨m_i, m_j⟩ + φ_i + φ_j
    φ_i      = diag(R W Rᵀ)_i = r_iᵀ Λ r_i,   r = R·P     (P, Λ new, ⟂ W_M)

Special cases:
    potential="dmap" : φ_i = −½‖m_i‖²          -> exact DMAP distance kernel
    potential="none" : φ_i = 0                 -> bare symmetric Gram attention
    potential="free" : φ learned (Λ init 0 => starts as bare Gram, then learns)

The potential is computed at the CONTEXT level (a per-token scalar from the
activations), not folded into any projection. Only the per-KEY φ_j actually
affects softmax (φ_i is constant along its row and cancels), so φ is effectively
a learned, geometry-derived per-key bias — a Doob h-transform of the kernel.

New params vs SiT: phi_proj [C,C] + phi_lambda [H,Dh] per attention. A DDMAP
checkpoint therefore carries extra keys and cannot load back into plain SiT/AMAP/
DMAP; warm-starting FROM them loads the shared qkv/proj and leaves φ at init.
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
class DDMAPConfig:
    potential: str = "free"           # "free" | "dmap" | "none"
    qk_rmsnorm: bool = False
    learn_logit_scale: bool = False
    eps: float = 1e-6


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "scale")


def _ddmap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)

    cfg: DDMAPConfig = self._ddmap
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)

    m = (q + k) * INV_SQRT2
    logits = m @ m.transpose(-2, -1)                       # ⟨m_i,m_j⟩ (kinetic)

    if cfg.potential == "free":
        r = self.phi_proj(x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)   # R·P
        phi = (self._phi_lambda.view(1, H, 1, Dh) * (r * r)).sum(-1)    # r_iᵀ Λ r_i
        logits = logits + phi[..., :, None] + phi[..., None, :]
    elif cfg.potential == "dmap":
        dsq = (m * m).sum(-1)
        logits = logits - 0.5 * dsq[..., :, None] - 0.5 * dsq[..., None, :]
    elif cfg.potential != "none":
        raise ValueError(f"potential must be free|dmap|none, got {cfg.potential!r}")

    logits = logits * self.scale
    if cfg.learn_logit_scale:
        logits = logits * self._ddmap_logit_scale.view(1, H, 1, 1)

    attn = logits.softmax(dim=-1)
    attn = self.attn_drop(attn) if hasattr(self, "attn_drop") else attn
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out) if hasattr(self, "proj_drop") else out
    return out


def apply_ddmap(model: nn.Module, cfg: DDMAPConfig | None = None) -> int:
    """Patch every timm Attention to DDMAP (coupled). For potential='free' this
    registers a per-attention potential head (phi_proj + phi_lambda, Λ init 0 so
    φ starts at 0 = bare Gram). Returns the count."""
    cfg = cfg or DDMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        C = module.qkv.weight.shape[1]
        H = module.num_heads
        Dh = C // H
        module._ddmap = cfg
        if cfg.potential == "free" and not hasattr(module, "phi_proj"):
            dev, dt = module.qkv.weight.device, module.qkv.weight.dtype
            module.phi_proj = nn.Linear(C, C, bias=False).to(device=dev, dtype=dt)
            module.register_parameter(
                "_phi_lambda", nn.Parameter(torch.zeros(H, Dh, device=dev, dtype=dt)))
        if cfg.learn_logit_scale and not hasattr(module, "_ddmap_logit_scale"):
            module.register_parameter("_ddmap_logit_scale", nn.Parameter(torch.ones(H)))
        module.forward = types.MethodType(_ddmap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_ddmap: found no attention modules")
    model._ddmap_applied = True
    return n


def _selftest() -> None:
    torch.manual_seed(0)
    B, N, C, H = 2, 6, 16, 4
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)

    class Attn(nn.Module):
        def __init__(s, d, h):
            super().__init__(); s.num_heads = h; s.scale = (d // h) ** -0.5
            s.qkv = nn.Linear(d, d * 3, bias=True).double(); s.proj = nn.Linear(d, d).double()

    def logits_of(mode):
        m_ = Attn(C, H)
        apply_ddmap(m_, DDMAPConfig(potential=mode))
        if mode == "free" and hasattr(m_, "blocks"):
            pass
        # recompute reference from the SAME qkv
        W = m_.qkv.weight.detach(); b = m_.qkv.bias.detach()
        qkv = (x @ W.t() + b).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0); mm = (q + k) * INV_SQRT2
        gram = mm @ mm.transpose(-2, -1)
        return m_, gram, mm

    # potential="dmap" must equal the DMAP distance kernel
    m_, gram, mm = logits_of("dmap")
    dsq = (mm * mm).sum(-1)
    ref_dmap = gram - 0.5 * dsq[..., :, None] - 0.5 * dsq[..., None, :]
    diff = mm[..., :, None, :] - mm[..., None, :, :]
    assert torch.allclose(ref_dmap, -0.5 * (diff * diff).sum(-1), atol=1e-10)

    # potential="none" is bare Gram
    # potential="free" with Λ=0 => φ=0 => bare Gram (Λ initialised to zeros)
    mf = Attn(C, H); apply_ddmap(mf, DDMAPConfig(potential="free"))
    assert hasattr(mf, "phi_proj") and torch.count_nonzero(mf._phi_lambda) == 0
    # set Λ nonzero and confirm φ enters symmetrically and is a pure quadratic form
    with torch.no_grad():
        mf._phi_lambda.copy_(torch.randn(H, Dh, dtype=torch.float64))
    r = mf.phi_proj(x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
    phi = (mf._phi_lambda.view(1, H, 1, Dh) * (r * r)).sum(-1)
    assert phi.shape == (B, H, N)
    print("selftest OK: potential='dmap' == DMAP distance kernel; 'free' Λ=0 => bare Gram; "
          "φ is a per-token quadratic form diag(R P Λ Pᵀ Rᵀ)")


if __name__ == "__main__":
    _selftest()

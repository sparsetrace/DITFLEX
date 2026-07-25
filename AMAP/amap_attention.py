"""
AMAP attention (coupled variant) — eager, no Flash/Flex (context N≈256).

Standard attention logit matrix (bilinear form W = W_Q W_K^T):
    L = R W R^T ,   L_ij = <q_i, k_j>

AMAP replaces W with
    W_AMAP = 1/2 W_M W_M^T + 1/2 (W - W^T),   W_M = (W_Q + W_K)/sqrt(2)
           = W + 1/2 W_N W_N^T                (W_N = (W_Q - W_K)/sqrt(2))

i.e. the symmetric sector is forced PSD (1/2 W_M W_M^T, a Gram/kernel) while the
antisymmetric "flux" sector is left identical to standard attention. Only the
negative-definite symmetric piece -1/2 W_N W_N^T is dropped (functionally).

COUPLED variant: we do NOT materialise W_M. In token space m = (q + k)/sqrt(2),
so everything is a function of the existing q, k — no new parameters:

    m_i     = (q_i + k_i)/sqrt(2)
    sym_ij  = 1/2 <m_i, m_j>                       # PSD Gram
    asym_ij = 1/2 (<q_i,k_j> - <k_i,q_j>)          # = 1/2 (QK^T - (QK^T)^T)
    logit   = (sym + asym) * scale                 # scale = head_dim**-0.5

Because q,k are shared, W_M and the flux field are trained *jointly-constrained*
by one set of weights. The DECOUPLED arm (independent W_M) is a later experiment
(see decouple_amap / AMAP/surgery.py) and only differs in where `m` comes from.

We monkeypatch `forward` onto the existing timm `Attention` modules, reusing
their qkv / proj / q_norm / k_norm / drops / scale. The module structure and
therefore the checkpoint state_dict are UNCHANGED — transactional resume and the
7M SiT weights load as-is.
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
class AMAPConfig:
    # Optional stabilisers, OFF by default so the first run is the faithful
    # operator. The symmetric Gram has a large positive diagonal (1/2||m_i||^2),
    # so AMAP logits sit at a higher, more self-biased scale than the <q,k> the
    # SiT was trained at — the exact axis this project's 240K–344K blowup lived
    # on. Flip these on if the finetune won't settle.
    qk_rmsnorm: bool = False       # per-head RMSNorm on q and k before AMAP
    learn_logit_scale: bool = False  # per-head learnable multiplier, init 1
    eps: float = 1e-6


def _amap_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Eager AMAP attention, bound as `Attention.forward`. Reuses self.*."""
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                      # each (B, H, N, Dh)

    # timm's per-head norms (Identity unless the checkpoint carries qk-norm;
    # absent entirely on older timm)
    if hasattr(self, "q_norm"):
        q, k = self.q_norm(q), self.k_norm(k)

    cfg: AMAPConfig = self._amap
    if cfg.qk_rmsnorm:
        q = F.rms_norm(q, (Dh,), eps=cfg.eps)
        k = F.rms_norm(k, (Dh,), eps=cfg.eps)

    m = (q + k) * INV_SQRT2                       # W_M in token space
    sym = 0.5 * (m @ m.transpose(-2, -1))         # PSD Gram, (B,H,N,N)

    qk = q @ k.transpose(-2, -1)
    asym = 0.5 * (qk - qk.transpose(-2, -1))      # directed flux field

    logits = (sym + asym) * self.scale
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


def apply_amap(model: nn.Module, cfg: AMAPConfig | None = None) -> int:
    """
    Patch every timm-style Attention in `model` to AMAP (coupled). Idempotent.
    Returns the number of attention modules patched. Does not change parameters
    or state_dict keys.
    """
    cfg = cfg or AMAPConfig()
    n = 0
    for module in model.modules():
        if not _is_attention(module):
            continue
        module._amap = cfg
        if cfg.learn_logit_scale and not hasattr(module, "_amap_logit_scale"):
            # registered as a real parameter so it trains and checkpoints
            module.register_parameter(
                "_amap_logit_scale", nn.Parameter(torch.ones(module.num_heads))
            )
        module.forward = types.MethodType(_amap_forward, module)
        n += 1
    if n == 0:
        raise ValueError("apply_amap: found no attention modules (need .qkv/.num_heads/.scale)")
    model._amap_applied = True
    return n


# --------------------------------------------------------------------------- #
# Correctness: the eager formula must equal the explicit bilinear form, and the
# W + 1/2 W_N W_N^T identity must hold. Run `python amap_attention.py`.
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    B, N, C, H = 2, 7, 16, 4
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)

    # random projections, per head, as [C, C] fused blocks
    Wq = torch.randn(C, C, dtype=torch.float64)
    Wk = torch.randn(C, C, dtype=torch.float64)

    # per-head q,k like timm (split C into H*Dh)
    q = (x @ Wq.t()).reshape(B, N, H, Dh).transpose(1, 2)   # (B,H,N,Dh)
    k = (x @ Wk.t()).reshape(B, N, H, Dh).transpose(1, 2)

    # (a) eager AMAP formula
    m = (q + k) * INV_SQRT2
    sym = 0.5 * (m @ m.transpose(-2, -1))
    qk = q @ k.transpose(-2, -1)
    asym = 0.5 * (qk - qk.transpose(-2, -1))
    L_eager = sym + asym                                     # (B,H,N,N), unscaled

    # (b) explicit per-head bilinear W_AMAP = 1/2 Wm Wm^T + 1/2 (W - W^T)
    for h in range(H):
        Wqh = Wq.reshape(H, Dh, C)[h]                        # (Dh, C)
        Wkh = Wk.reshape(H, Dh, C)[h]
        W = Wqh.t() @ Wkh                                    # (C,C) bilinear form
        Wm = (Wqh + Wkh) * INV_SQRT2                         # (Dh, C)
        Wn = (Wqh - Wkh) * INV_SQRT2
        W_amap = 0.5 * (Wm.t() @ Wm) + 0.5 * (W - W.t())
        L_explicit = torch.einsum("bnc,cd,bmd->bnm", x, W_amap, x)   # (B,N,N)
        assert torch.allclose(L_eager[:, h], L_explicit, atol=1e-9), h
        # identity: W_amap == W + 1/2 Wn^T Wn
        assert torch.allclose(W_amap, W + 0.5 * (Wn.t() @ Wn), atol=1e-9), h
        # PSD symmetric sector
        assert torch.linalg.eigvalsh(0.5 * (Wm.t() @ Wm)).min() > -1e-9, h

    # antisym is exactly skew in i<->j
    assert torch.allclose(asym, -asym.transpose(-2, -1), atol=1e-12)

    # scale-inflation diagnostic: AMAP diagonal bias vs standard off-diagonal
    diag = sym.diagonal(dim1=-2, dim2=-1).mean().item()
    offd = qk.mean().item()
    print(f"selftest OK: eager==explicit, W+½WnWnᵀ identity holds, sym PSD.")
    print(f"  self-attn diagonal bias ½‖m‖²≈{diag:.2f}  vs mean ⟨q,k⟩≈{offd:.2f} "
          f"(→ watch logit scale; qk_rmsnorm/learn_logit_scale available)")


if __name__ == "__main__":
    _selftest()

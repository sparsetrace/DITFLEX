"""
DMAP attention (FOLDED variant) — eager, no Flash/Flex (context N≈256).

DMAP is the Mahalanobis distance kernel; it uses q,k only through
μ = (q+k)/√2 = R·W_M, so W_N = (W_Q−W_K)/√2 is dead weight. This folded
variant discards W_N *structurally*: the fused qkv [3d,d] is replaced by a
fused wmv [2d,d] = [W_M ; W_V], and the forward projects μ directly.

    μ = R·W_M
    logit_ij = −½‖μ_i − μ_j‖²   (× d_h^-½)          # ≤ 0, zero diagonal

vs the coupled DMAP: identical operator and identical logits, but ~⅓ fewer
attention-projection params (and optimizer state), because W_N is gone rather
than computed-and-discarded each forward. The fold W_M = (W_Q+W_K)/√2 is EXACT
for DMAP (the loss never depended on W_N), so folding at step 0 loses no
reachable solution. It is a one-way door: a folded checkpoint can no longer
become AMAP or standard attention (those need q,k apart).

Folded checkpoints carry `attn.wmv.{weight,bias}` instead of `attn.qkv.*`.
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
    qk_rmsnorm: bool = False          # per-head RMSNorm on μ
    learn_logit_scale: bool = False   # per-head learnable multiplier, init 1
    eps: float = 1e-6


def _is_attention(module: nn.Module) -> bool:
    return hasattr(module, "num_heads") and hasattr(module, "scale") and (
        hasattr(module, "qkv") or hasattr(module, "wmv"))


def _folded_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Eager folded-DMAP attention: project μ,v from wmv; distance kernel."""
    B, N, C = x.shape
    H = self.num_heads
    Dh = C // H

    wmv = self.wmv(x).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
    mu, v = wmv.unbind(0)                          # each (B, H, N, Dh)

    cfg: DMAPConfig = self._dmap
    if cfg.qk_rmsnorm:
        mu = F.rms_norm(mu, (Dh,), eps=cfg.eps)

    gram = mu @ mu.transpose(-2, -1)
    dsq = (mu * mu).sum(-1)
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


def install_folded_dmap(model: nn.Module, cfg: DMAPConfig | None = None,
                        fold_weights: bool = True) -> int:
    """Replace each timm Attention's fused qkv [3d,d] with a fused wmv [2d,d]
    and bind the folded distance-kernel forward.

    fold_weights=True  : initialise W_M = (W_Q+W_K)/sqrt2 from the module's
                         current qkv (model already holds source weights, e.g.
                         base SiT for a from-scratch DMAP).
    fold_weights=False : create a correctly-shaped wmv to be filled by
                         load_state_dict (loading a folded / fold_state_dict()'d
                         source).

    Returns the number of attention modules folded.
    """
    cfg = cfg or DMAPConfig()
    n = 0
    for m in model.modules():
        if not _is_attention(m) or hasattr(m, "wmv"):
            continue
        d = m.qkv.weight.shape[1]
        has_bias = m.qkv.bias is not None
        wmv = nn.Linear(d, 2 * d, bias=has_bias).to(
            device=m.qkv.weight.device, dtype=m.qkv.weight.dtype)
        if fold_weights:
            with torch.no_grad():
                W = m.qkv.weight
                Wq, Wk, Wv = W[:d], W[d:2 * d], W[2 * d:]
                wmv.weight.copy_(torch.cat([(Wq + Wk) * INV_SQRT2, Wv], 0))
                if has_bias:
                    b = m.qkv.bias
                    bq, bk, bv = b[:d], b[d:2 * d], b[2 * d:]
                    wmv.bias.copy_(torch.cat([(bq + bk) * INV_SQRT2, bv], 0))
        m.wmv = wmv
        delattr(m, "qkv")                          # drop W_N direction structurally
        m._dmap = cfg
        if cfg.learn_logit_scale and not hasattr(m, "_dmap_logit_scale"):
            m.register_parameter("_dmap_logit_scale", nn.Parameter(torch.ones(m.num_heads)))
        m.forward = types.MethodType(_folded_forward, m)
        n += 1
    if n == 0:
        raise ValueError("install_folded_dmap: no attention modules found")
    model._dmap_folded = True
    return n


def fold_state_dict(sd: dict) -> dict:
    """Fold a full-qkv state_dict into the folded wmv format:
        *.attn.qkv.{weight,bias} -> *.attn.wmv.{weight,bias}   (W_M=(W_Q+W_K)/sqrt2)
    q_norm/k_norm entries are dropped (Identity on the SiT checkpoint); every
    other tensor is passed through. Use to fold an AMAP checkpoint for warm-start."""
    import re
    out = {}
    rx = re.compile(r"^(?P<attn>.*\.attn)\.qkv\.(?P<kind>weight|bias)$")
    for k, v in sd.items():
        m = rx.match(k)
        if m:
            attn, kind = m.group("attn"), m.group("kind")
            if kind == "weight":
                d = v.shape[1]
                Wq, Wk, Wv = v[:d], v[d:2 * d], v[2 * d:]
                out[f"{attn}.wmv.weight"] = torch.cat([(Wq + Wk) * INV_SQRT2, Wv], 0)
            else:
                d = v.shape[0] // 3
                bq, bk, bv = v[:d], v[d:2 * d], v[2 * d:]
                out[f"{attn}.wmv.bias"] = torch.cat([(bq + bk) * INV_SQRT2, bv], 0)
            continue
        if ".attn.q_norm" in k or ".attn.k_norm" in k:
            continue
        out[k] = v
    return out


def _selftest() -> None:
    torch.manual_seed(0)
    B, N, C, H = 2, 7, 16, 4
    Dh = C // H
    x = torch.randn(B, N, C, dtype=torch.float64)

    class Attn(nn.Module):
        def __init__(self, dim, h):
            super().__init__()
            self.num_heads = h
            self.scale = (dim // h) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=True).double()
            self.proj = nn.Linear(dim, dim).double()

    m = Attn(C, H)
    W = m.qkv.weight.detach(); b = m.qkv.bias.detach()
    qkv = (x @ W.t() + b).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    mu_ref = (q + k) * INV_SQRT2
    gram = mu_ref @ mu_ref.transpose(-2, -1)
    dsq = (mu_ref * mu_ref).sum(-1)
    logit_ref = (gram - 0.5 * dsq[..., :, None] - 0.5 * dsq[..., None, :]) * m.scale

    install_folded_dmap(m, DMAPConfig(), fold_weights=True)
    assert not hasattr(m, "qkv") and hasattr(m, "wmv")
    wmv = (x @ m.wmv.weight.detach().t() + m.wmv.bias.detach()).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
    mu_fold, _ = wmv.unbind(0)
    assert torch.allclose(mu_fold, mu_ref, atol=1e-10), (mu_fold - mu_ref).abs().max()

    sd = {"blocks.0.attn.qkv.weight": W, "blocks.0.attn.qkv.bias": b,
          "blocks.0.attn.proj.weight": torch.randn(C, C, dtype=torch.float64)}
    folded = fold_state_dict(sd)
    assert "blocks.0.attn.wmv.weight" in folded and "blocks.0.attn.qkv.weight" not in folded
    assert torch.allclose(folded["blocks.0.attn.wmv.weight"], m.wmv.weight.detach(), atol=1e-12)
    assert m.wmv.weight.shape[0] == 2 * C and W.shape[0] == 3 * C
    print("selftest OK: fold exact (mu_fold==mu_ref), fold_state_dict matches, "
          f"attn proj {3*C}->{2*C} rows (1/3 smaller)")


if __name__ == "__main__":
    _selftest()

"""FlexAttention self-attention for diffusers' Attention module.

This is the ONLY attention implementation in the repo. There is no SDPA path.

Three things live here:

  1. identity_score_mod -- the baseline score function. It is a *real*
     score_mod (not None) so that the baseline traverses exactly the same
     FlexAttention machinery as any experimental score_mod: swapping the
     experiment in changes only the function, never the dispatch path.
     Compiled, the identity inlines to nothing.

  2. FlexSelfAttnProcessor -- a diffusers attention processor whose score
     function is a swappable component.

  3. reference_self_attention -- straight-line softmax attention written
     directly from the math (explicit matmuls, explicit softmax) using the
     module's own weights, intended to run in fp64. It exists so
     scripts/verify_identity.py can check the Flex path against something
     that depends on no fused kernel at all. Test utility only, never a
     training path.

QK-NORM (added at the 344K migration; DEVIATION from published DiT/SiT):
if the module carries ``norm_q`` / ``norm_k`` (RMSNorm over head_dim),
the processor applies them per-head AFTER the [B, N, H*D] -> [B, H, N, D]
reshape and BEFORE flex_attention -- entirely upstream of the score_mod,
so the swappable-score dispatch path is untouched. The fp64 reference
applies the identical normalization from the same weights, so the
identity gate certifies both configurations. Rationale recorded in the
migration commit and README: at ~270-312K steps the un-normalized chain
developed unbounded QK logits (block 1 reaching 8.6e6), driven by adaLN
weight growth, producing the gradient-spike instability; RMSNorm on Q and
K bounds per-head logits by construction. The DMAP chain does NOT use
qk-norm (see model.py / diffusion_model.py guards): it shows no logit
pathology, untied norms would break its R == 0 symmetry, and tied norms
would change the squared-distance kernel's geometry.

Scope: DiT-L/2 self-attention on [B, N, C] tokens. Cross-attention,
attention masks, 4D inputs, group/spatial norm, in-processor residuals,
and output rescaling are all rejected loudly rather than handled -- in a
repo whose premise is that every deviation from the baseline is known, a
config surprise should fail at the gate, not show up later as an
uninterpretable training curve. qk-norm moved from the rejected list to
the handled list in the 344K migration, with gate coverage for both the
with-norm and without-norm configurations.

Performance note: eager flex_attention is the slow-but-correct fallback
and is what the gates use. The fast path is whole-model torch.compile in
train.py, which fuses the score_mod into the generated kernel. Do not
compile here -- tests and verify_identity.py need the uncompiled module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    from torch.nn.attention.flex_attention import flex_attention
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import torch.nn.attention.flex_attention.flex_attention. "
        "ditflex requires a PyTorch build with FlexAttention support (>= 2.5)."
    ) from e

# score_mod(score, batch, head, q_idx, kv_idx) -> score
ScoreMod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

QK_NORM_EPS = 1e-6


def identity_score_mod(score, b, h, q_idx, kv_idx):
    """No modification: standard full attention routed through FlexAttention."""
    return score


class FlexSelfAttnProcessor:
    """Self-attention through FlexAttention, for diffusers' Attention module.

    Args:
        score_mod: FlexAttention score modification. Defaults to
            ``identity_score_mod`` (the DiT/SiT baseline). The softmax scale
            is always taken from ``attn.scale`` -- never from Flex's default
            -- so the computation is exactly the module's configured
            attention regardless of how the module was built.
    """

    def __init__(self, score_mod: ScoreMod | None = None):
        self.score_mod: ScoreMod = score_mod if score_mod is not None else identity_score_mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("FlexSelfAttnProcessor is self-attention only.")
        if attention_mask is not None:
            raise ValueError("Fixed-shape training uses no attention mask.")
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got ndim={hidden_states.ndim}.")
        if getattr(attn, "group_norm", None) is not None:
            raise ValueError("group_norm is not handled by this processor.")
        if getattr(attn, "spatial_norm", None) is not None:
            raise ValueError("spatial_norm is not handled by this processor.")
        if getattr(attn, "residual_connection", False):
            raise ValueError("residual_connection is handled by the block, not the processor.")
        if getattr(attn, "rescale_output_factor", 1.0) != 1.0:
            raise ValueError("rescale_output_factor != 1 is not handled.")

        norm_q = getattr(attn, "norm_q", None)
        norm_k = getattr(attn, "norm_k", None)
        if (norm_q is None) != (norm_k is None):
            raise ValueError(
                "qk-norm must be installed on BOTH norm_q and norm_k or neither; "
                "a half-installed configuration is a migration bug."
            )

        batch, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        # [B, N, H*D] -> [B, H, N, D]
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        # QK-norm: per-head RMSNorm over head_dim, applied before the kernel
        # and entirely outside the score_mod dispatch path.
        if norm_q is not None:
            query = norm_q(query)
            key = norm_k(key)

        out = flex_attention(
            query,
            key,
            value,
            score_mod=self.score_mod,
            scale=attn.scale,  # explicit: never rely on Flex's 1/sqrt(D) default
        )

        # [B, H, N, D] -> [B, N, H*D]
        out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)

        out = attn.to_out[0](out)  # linear
        out = attn.to_out[1](out)  # dropout (identity in eval)
        return out


class IdentityFlexSelfAttnProcessor(FlexSelfAttnProcessor):
    """Baseline processor: FlexAttention with the identity score_mod."""

    def __init__(self):
        super().__init__(score_mod=identity_score_mod)


def _reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = QK_NORM_EPS,
) -> torch.Tensor:
    """RMSNorm written straight from the math, for the fp64 reference path.

    Matches torch.nn.RMSNorm semantics: x / sqrt(mean(x^2) + eps) * weight,
    normalized over the last dimension. Written explicitly (no fused op) so
    the reference depends on nothing but arithmetic.
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms * weight


def reference_self_attention(
    attn,
    hidden_states: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Softmax self-attention written straight from the math.

    Uses the weights of ``attn`` but none of its forward code and no fused
    attention kernel of any kind: explicit projections, optional explicit
    per-head RMSNorm on Q/K (when the module carries norm_q/norm_k), an
    explicit ``q @ k^T * scale`` score matrix, an explicit softmax, explicit
    output projection.

    Args:
        attn: a diffusers Attention module (self-attention config).
        hidden_states: [B, N, C].
        dtype: if given, inputs and weights are cast to this dtype for the
            computation (use torch.float64 for a high-precision reference
            against a bf16/fp32 module). If None, computes in the module's
            native dtype with autograd intact -- used by the gradient gate.

    Note: skips attn.to_out[1] (dropout), so compare against modules in
    eval mode only.
    """

    def cast(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None or dtype is None:
            return t
        return t.to(dtype)

    x = hidden_states if dtype is None else hidden_states.to(dtype)

    def linear(t, layer):
        w = cast(layer.weight)
        b = cast(layer.bias)
        out = t @ w.transpose(0, 1)
        return out if b is None else out + b

    query = linear(x, attn.to_q)
    key = linear(x, attn.to_k)
    value = linear(x, attn.to_v)

    batch, seq_len, _ = x.shape
    heads = attn.heads
    head_dim = query.shape[-1] // heads

    query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

    norm_q = getattr(attn, "norm_q", None)
    norm_k = getattr(attn, "norm_k", None)
    if norm_q is not None:
        eps_q = float(getattr(norm_q, "eps", QK_NORM_EPS) or QK_NORM_EPS)
        eps_k = float(getattr(norm_k, "eps", QK_NORM_EPS) or QK_NORM_EPS)
        query = _reference_rms_norm(query, cast(norm_q.weight), eps_q)
        key = _reference_rms_norm(key, cast(norm_k.weight), eps_k)

    scores = (query @ key.transpose(-2, -1)) * attn.scale
    probs = scores.softmax(dim=-1)
    out = probs @ value

    out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
    out = linear(out, attn.to_out[0])
    return out

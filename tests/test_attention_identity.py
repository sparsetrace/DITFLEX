"""Pytest form of Gate 1 (scripts/verify_identity.py).

Same checks, same reference: FlexAttention vs explicit fp64 math built from
the same weights. No SDPA anywhere. Skips (does not fail) on machines
without CUDA so the CPU test workflow stays green.

Parametrized over qk_norm since the 344K migration: the with-norm
configuration attaches RMSNorm(head_dim) to norm_q/norm_k with
NON-TRIVIAL weights (1 + noise), so a silently-skipped norm cannot pass
by accident, and the fp64 reference applies the identical normalization.
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Flex GPU kernels are the thing under test"
)

DIM, HEADS, HEAD_DIM, SEQ_LEN, BATCH = 1024, 16, 64, 256, 4
REL_TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


@pytest.fixture(autouse=True)
def strict_fp32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def attach_qk_norms(attn, dtype: torch.dtype, seed: int = 7) -> None:
    """Install RMSNorm(head_dim) with non-trivial weights (1 + 0.1*noise).

    Ones-weights would make a dropped multiply invisible; perturbed weights
    make the norm's presence measurable in every comparison.
    """
    from ditflex.attention import QK_NORM_EPS

    g = torch.Generator().manual_seed(seed)
    for name in ("norm_q", "norm_k"):
        norm = torch.nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        with torch.no_grad():
            norm.weight.add_(0.1 * torch.randn(HEAD_DIM, generator=g))
        setattr(attn, name, norm.to(device="cuda", dtype=dtype))


def build_attention(dtype: torch.dtype, qk_norm: bool, requires_grad: bool = False):
    from diffusers.models.attention_processor import Attention

    attn = Attention(
        query_dim=DIM, heads=HEADS, dim_head=HEAD_DIM, dropout=0.0, bias=True, out_bias=True
    )
    attn = attn.to(device="cuda", dtype=dtype).eval()
    if qk_norm:
        attach_qk_norms(attn, dtype)
    for p in attn.parameters():
        p.requires_grad_(requires_grad)
    return attn


def agree(got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float = 1e-8) -> bool:
    """|a-b| <= atol + rtol*|ref|. The atol term matters for mathematically
    zero quantities (e.g. d/d(to_k.bias): softmax is shift-invariant, so the
    key bias has exactly zero gradient) where a pure relative comparison is
    rounding noise divided by rounding noise."""
    got, ref = got.double(), ref.double()
    return ((got - ref).abs().max() <= atol + rtol * ref.abs().max()).item()


@requires_cuda
@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qknorm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_matches_math_reference(dtype, qk_norm):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(dtype, qk_norm)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=dtype)

    assert abs(attn.scale - HEAD_DIM**-0.5) < 1e-9

    with torch.no_grad():
        ref = reference_self_attention(attn, x, dtype=torch.float64)

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        out = attn(x)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert agree(out, ref, REL_TOL[dtype])


@requires_cuda
def test_qk_norm_changes_the_output():
    """A dropped norm is invisible to the identity comparison when weights
    are ones; with perturbed weights, with-norm and without-norm outputs
    must measurably differ -- proving the norm is live in the Flex path."""
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm=False)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")
    with torch.no_grad():
        plain = attn(x)

    attach_qk_norms(attn, torch.float32)
    with torch.no_grad():
        normed = attn(x)

    assert (normed - plain).abs().max().item() > 1e-3


@requires_cuda
def test_half_installed_norms_are_rejected():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32, qk_norm=True)
    attn.norm_k = None  # simulate a broken migration
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")
    with pytest.raises(ValueError, match="BOTH"):
        attn(x)


@requires_cuda
def test_score_mod_is_wired():
    """Identity comparison cannot catch a silently-dropped score_mod
    (identity == no-mod). A zero score_mod forces uniform attention, which
    must change the output."""
    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm=False)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        identity_out = attn(x)

    attn.set_processor(FlexSelfAttnProcessor(score_mod=lambda s, b, h, q, kv: s * 0.0))
    with torch.no_grad():
        uniform_out = attn(x)

    assert (uniform_out - identity_out).abs().max().item() > 1e-3


@requires_cuda
@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qknorm"])
def test_flex_backward_matches_reference(qk_norm):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm, requires_grad=True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }
    if qk_norm:
        assert any("norm_q" in n for n in ref_grads), "reference gave no grad to norm_q"

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            assert agree(param.grad, ref_grads[name], 1e-4), f"grad mismatch: {name}"


@requires_cuda
def test_processor_rejects_out_of_contract_inputs():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32, qk_norm=False)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    with pytest.raises(ValueError):
        attn(x, encoder_hidden_states=torch.randn_like(x))
    with pytest.raises(ValueError):
        attn(x, attention_mask=torch.ones(BATCH, 1, SEQ_LEN, device="cuda"))

"""Pytest form of Gate 1 (scripts/verify_identity.py).

Same checks, same reference: FlexAttention vs explicit fp64 math built from
the same weights. No SDPA anywhere. Skips (does not fail) on machines
without CUDA so the CPU test workflow stays green.
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


def build_attention(dtype: torch.dtype, requires_grad: bool = False):
    from diffusers.models.attention_processor import Attention

    attn = Attention(
        query_dim=DIM, heads=HEADS, dim_head=HEAD_DIM, dropout=0.0, bias=True, out_bias=True
    )
    attn = attn.to(device="cuda", dtype=dtype).eval()
    for p in attn.parameters():
        p.requires_grad_(requires_grad)
    return attn


def max_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    return ((got - ref).abs().max() / (ref.abs().max() + 1e-12)).item()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_matches_math_reference(dtype):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(dtype)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=dtype)

    assert abs(attn.scale - HEAD_DIM**-0.5) < 1e-9

    with torch.no_grad():
        ref = reference_self_attention(attn, x, dtype=torch.float64)

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        out = attn(x)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert max_rel(out, ref) < REL_TOL[dtype]


@requires_cuda
def test_score_mod_is_wired():
    """Identity comparison cannot catch a silently-dropped score_mod
    (identity == no-mod). A zero score_mod forces uniform attention, which
    must change the output."""
    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        identity_out = attn(x)

    attn.set_processor(FlexSelfAttnProcessor(score_mod=lambda s, b, h, q, kv: s * 0.0))
    with torch.no_grad():
        uniform_out = attn(x)

    assert (uniform_out - identity_out).abs().max().item() > 1e-3


@requires_cuda
def test_flex_backward_matches_reference():
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(torch.float32, requires_grad=True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            assert max_rel(param.grad, ref_grads[name]) < 1e-4, f"grad mismatch: {name}"


@requires_cuda
def test_processor_rejects_out_of_contract_inputs():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    with pytest.raises(ValueError):
        attn(x, encoder_hidden_states=torch.randn_like(x))
    with pytest.raises(ValueError):
        attn(x, attention_mask=torch.ones(BATCH, 1, SEQ_LEN, device="cuda"))

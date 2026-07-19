#!/usr/bin/env python
"""Gate 1: the FlexAttention path must compute softmax attention.

Reference: NOT another fused kernel. The comparison target is
ditflex.attention.reference_self_attention -- explicit matmuls and an
explicit softmax in fp64, built from the same weights. If the Flex path
matches that, it matches the math.

Checks, in order:
  1. scale        -- attn.scale equals what the processor passes to Flex
  2. forward fp32 -- flex vs fp64 reference, rel tol 1e-4
  3. forward bf16 -- flex vs fp64 reference, rel tol 2e-2
                     (bf16 has ~8 mantissa bits; tighter is not meaningful)
  4. score_mod wiring -- a constant-zero score_mod must produce uniform
     attention and therefore a measurably different output. An identity
     comparison alone CANNOT catch a bug where score_mod is silently
     dropped, because identity == no-mod. This check can.
  5. gradients fp32 -- flex backward vs autograd through the reference
  6. (--compile) the compiled Flex path vs the same reference

If this fails, nothing downstream is interpretable: a training curve that
differs from the DiT/SiT baseline could be the score_mod or could be the
plumbing, and you will not be able to tell which.

Run:
    python scripts/verify_identity.py
    python scripts/verify_identity.py --compile

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
from diffusers.models.attention_processor import Attention

from ditflex.attention import (
    FlexSelfAttnProcessor,
    IdentityFlexSelfAttnProcessor,
    reference_self_attention,
)

# DiT-L/2 self-attention geometry: width 1024, 16 heads, head_dim 64,
# 32x32 latents at patch 2 -> 256 tokens.
DIM = 1024
HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 256
BATCH = 4

REL_TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


def build_attention(device: torch.device, dtype: torch.dtype) -> Attention:
    attn = Attention(
        query_dim=DIM,
        heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
    )
    attn = attn.to(device=device, dtype=dtype).eval()
    for p in attn.parameters():
        p.requires_grad_(False)
    return attn


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> bool:
    got64, ref64 = got.double(), ref.double()
    max_abs = (got64 - ref64).abs().max().item()
    denom = ref64.abs().max().item() + 1e-12
    max_rel = max_abs / denom
    ok = max_rel < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<30} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  tol={tol:.1e}")
    return ok


def check_scale(attn: Attention) -> bool:
    """The processor passes scale=attn.scale explicitly, so the only thing to
    verify is that the module's scale is the expected 1/sqrt(head_dim) for
    this config (scale_qk=True). A surprise here means the module was built
    differently than the DiT-L/2 recipe assumes."""
    expected = HEAD_DIM ** -0.5
    ok = abs(attn.scale - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'scale':<30} attn.scale={attn.scale:.8f}  expected={expected:.8f}")
    return ok


def check_score_mod_wiring(attn: Attention, x: torch.Tensor, identity_out: torch.Tensor) -> bool:
    """score_mod that zeroes every score -> uniform attention. If the output
    does not change, score_mod is not wired through and the swappable
    component is not swappable."""

    def zero_score(score, b, h, q_idx, kv_idx):
        return score * 0.0

    attn.set_processor(FlexSelfAttnProcessor(score_mod=zero_score))
    with torch.no_grad():
        uniform_out = attn(x)

    diff = (uniform_out.double() - identity_out.double()).abs().max().item()
    ok = diff > 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'score_mod wiring':<30} |uniform - identity|_max={diff:.3e} (must be > 1e-3)")
    if not ok:
        print("         -> score_mod appears to be silently ignored by the Flex call.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also verify the compiled Flex path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required: the Flex kernels under test are the GPU ones.")
        return 1

    # fp32 means fp32: no TF32 in the path under test, or the fp32 tolerance
    # is meaningless.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print(f"shape [B={BATCH}, N={SEQ_LEN}, C={DIM}]  heads={HEADS} head_dim={HEAD_DIM}\n")

    all_ok = True

    for dtype in (torch.float32, torch.bfloat16):
        print(f"dtype = {dtype}")
        attn = build_attention(device, dtype)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=dtype)

        all_ok &= check_scale(attn)

        with torch.no_grad():
            ref = reference_self_attention(attn, x, dtype=torch.float64)

        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            flex_out = attn(x)

        if flex_out.shape != ref.shape:
            print(f"  [FAIL] shape mismatch {tuple(flex_out.shape)} vs {tuple(ref.shape)}")
            all_ok = False
        if not torch.isfinite(flex_out).all():
            print("  [FAIL] non-finite values in flex output")
            all_ok = False

        all_ok &= compare("flex vs fp64 reference", flex_out, ref, REL_TOL[dtype])
        all_ok &= check_score_mod_wiring(attn, x, flex_out)

        if args.compile:
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            compiled = torch.compile(attn)
            with torch.no_grad():
                out_c = compiled(x)
            all_ok &= compare("compiled flex vs reference", out_c, ref, REL_TOL[dtype])

        print()

    # Forward agreement does not guarantee backward agreement. Compare the
    # Flex backward against autograd through the explicit-math reference,
    # in fp32, on the same parameters.
    print("gradient check (fp32)")
    attn = build_attention(device, torch.float32)
    for p in attn.parameters():
        p.requires_grad_(True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.float32)

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None}

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name not in ref_grads:
            continue
        all_ok &= compare(f"grad {name}", param.grad, ref_grads[name], 1e-4)

    print()
    if all_ok:
        print("ALL CHECKS PASSED -- the Flex path computes the math, and score_mod is live.")
        return 0
    print("FAILURES ABOVE -- do not proceed to training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

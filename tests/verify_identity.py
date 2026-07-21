#!/usr/bin/env python
"""Gate 1: the FlexAttention path must compute softmax attention.

Reference: NOT another fused kernel. The comparison target is
ditflex.attention.reference_self_attention -- explicit matmuls and an
explicit softmax in fp64, built from the same weights. If the Flex path
matches that, it matches the math.

Since the 344K migration the gate runs EVERY check in BOTH configurations:
without qk-norm (the pre-migration baseline) and with qk-norm attached
using NON-TRIVIAL weights (1 + 0.1*noise -- ones-weights would make a
dropped norm invisible). The fp64 reference applies the identical
normalization from the same weights.

Checks, per configuration:
  1. scale        -- attn.scale equals what the processor passes to Flex
  2. forward fp32 -- flex vs fp64 reference, rel tol 1e-4
  3. forward bf16 -- flex vs fp64 reference, rel tol 2e-2
                     (bf16 has ~8 mantissa bits; tighter is not meaningful)
  4. score_mod wiring -- a constant-zero score_mod must produce uniform
     attention and therefore a measurably different output
  5. norm liveness (qk-norm config only) -- with-norm output must differ
     from without-norm output on the same weights/input
  6. gradients fp32 -- flex backward vs autograd through the reference,
     including the norm weights when present
  7. (--compile) the compiled Flex path vs the same reference

If this fails, nothing downstream is interpretable.

Run:
    python tests/verify_identity.py
    python tests/verify_identity.py --compile

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
from diffusers.models.attention_processor import Attention

from ditflex.attention import (
    QK_NORM_EPS,
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


def attach_qk_norms(attn: Attention, device, dtype, seed: int = 7) -> None:
    g = torch.Generator().manual_seed(seed)
    for name in ("norm_q", "norm_k"):
        norm = torch.nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        with torch.no_grad():
            norm.weight.add_(0.1 * torch.randn(HEAD_DIM, generator=g))
        setattr(attn, name, norm.to(device=device, dtype=dtype))


def build_attention(device, dtype, qk_norm: bool) -> Attention:
    attn = Attention(
        query_dim=DIM,
        heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
    )
    attn = attn.to(device=device, dtype=dtype).eval()
    if qk_norm:
        attach_qk_norms(attn, device, dtype)
    for p in attn.parameters():
        p.requires_grad_(False)
    return attn


def compare(
    name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float = 1e-8
) -> bool:
    got64, ref64 = got.double(), ref.double()
    max_abs = (got64 - ref64).abs().max().item()
    denom = ref64.abs().max().item()
    ok = max_abs <= atol + rtol * denom
    max_rel = max_abs / (denom + 1e-12)
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:<34} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"rtol={rtol:.1e} atol={atol:.1e}"
    )
    return ok


def check_scale(attn: Attention) -> bool:
    expected = HEAD_DIM ** -0.5
    ok = abs(attn.scale - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'scale':<34} attn.scale={attn.scale:.8f}  expected={expected:.8f}")
    return ok


def check_score_mod_wiring(attn: Attention, x: torch.Tensor, identity_out: torch.Tensor) -> bool:
    def zero_score(score, b, h, q_idx, kv_idx):
        return score * 0.0

    attn.set_processor(FlexSelfAttnProcessor(score_mod=zero_score))
    with torch.no_grad():
        uniform_out = attn(x)

    diff = (uniform_out.double() - identity_out.double()).abs().max().item()
    ok = diff > 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'score_mod wiring':<34} |uniform - identity|_max={diff:.3e} (> 1e-3)")
    return ok


def run_configuration(device, qk_norm: bool, compile_check: bool) -> bool:
    tag = "qk-norm" if qk_norm else "plain"
    all_ok = True

    plain_outputs: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float32, torch.bfloat16):
        print(f"[{tag}] dtype = {dtype}")
        torch.manual_seed(0)
        attn = build_attention(device, dtype, qk_norm)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=dtype)

        all_ok &= check_scale(attn)

        with torch.no_grad():
            ref = reference_self_attention(attn, x, dtype=torch.float64)

        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            flex_out = attn(x)

        if flex_out.shape != ref.shape or not torch.isfinite(flex_out).all():
            print("  [FAIL] shape or finiteness")
            all_ok = False

        all_ok &= compare("flex vs fp64 reference", flex_out, ref, REL_TOL[dtype])
        all_ok &= check_score_mod_wiring(attn, x, flex_out)

        # Norm liveness: same weights and input, norms removed, output must move.
        if qk_norm and dtype is torch.float32:
            attn.norm_q, attn.norm_k = None, None
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            with torch.no_grad():
                unnormed = attn(x)
            diff = (unnormed.double() - flex_out.double()).abs().max().item()
            ok = diff > 1e-3
            print(f"  [{'PASS' if ok else 'FAIL'}] {'qk-norm liveness':<34} "
                  f"|normed - plain|_max={diff:.3e} (> 1e-3)")
            all_ok &= ok
            attach_qk_norms(attn, device, dtype)

        if compile_check:
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            compiled = torch.compile(attn)
            with torch.no_grad():
                out_c = compiled(x)
            all_ok &= compare("compiled flex vs reference", out_c, ref, REL_TOL[dtype])

        plain_outputs[dtype] = flex_out
        print()

    # Gradient check (fp32), including norm weights when present.
    print(f"[{tag}] gradient check (fp32)")
    attn = build_attention(device, torch.float32, qk_norm)
    for p in attn.parameters():
        p.requires_grad_(True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.float32)

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }
    if qk_norm and not any("norm_q" in n for n in ref_grads):
        print("  [FAIL] reference produced no gradient for norm_q")
        all_ok = False

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            all_ok &= compare(f"grad {name}", param.grad, ref_grads[name], 1e-4)
    print()
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also verify the compiled Flex path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required: the Flex kernels under test are the GPU ones.")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print(f"shape [B={BATCH}, N={SEQ_LEN}, C={DIM}]  heads={HEADS} head_dim={HEAD_DIM}\n")

    all_ok = True
    for qk_norm in (False, True):
        all_ok &= run_configuration(device, qk_norm, args.compile)

    if all_ok:
        print("ALL CHECKS PASSED -- Flex computes the math in BOTH configurations, "
              "score_mod and qk-norm are live.")
        return 0
    print("FAILURES ABOVE -- do not proceed to training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

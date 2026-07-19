#!/usr/bin/env python
"""
Gate 1: IdentityFlexSelfAttnProcessor must match the default Diffusers
processor to within numerical tolerance.

If this fails, nothing downstream is interpretable -- a training curve that
differs from the DiT baseline could be the score_mod, or it could be the
plumbing, and you will not be able to tell which.

Run:
    python scripts/verify_identity.py
    python scripts/verify_identity.py --compile     # also check the compiled path

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from ditflex.attention import IdentityFlexSelfAttnProcessor

# DiT-L/2 self-attention geometry: width 1024, 16 heads, head_dim 64,
# 32x32 latents at patch 2 -> 256 tokens.
DIM = 1024
HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 256
BATCH = 4

# bf16 has ~3 decimal digits of mantissa; agreement to 1e-4 relative is about
# as tight as is meaningful. fp32 should be far tighter.
TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 1e-2}


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


@torch.no_grad()
def run_with(attn: Attention, processor, x: torch.Tensor) -> torch.Tensor:
    attn.set_processor(processor)
    return attn(x)


def compare(name: str, a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    a32, b32 = a.float(), b.float()
    max_abs = (a32 - b32).abs().max().item()
    denom = b32.abs().max().item() + 1e-12
    max_rel = max_abs / denom
    ok = max_rel < tol
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:<28} max_abs={max_abs:.3e}  "
        f"max_rel={max_rel:.3e}  tol={tol:.1e}"
    )
    return ok


def check_scale(attn: Attention) -> bool:
    """FlexAttention defaults to 1/sqrt(head_dim). Diffusers carries its own
    attn.scale, which is normally identical -- but not if the module was built
    with scale_qk=False. A silent mismatch here would look like a training bug
    much later, so check it explicitly."""
    flex_default = HEAD_DIM ** -0.5
    ok = abs(attn.scale - flex_default) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] scale                        attn.scale={attn.scale:.8f}  "
        f"flex_default={flex_default:.8f}"
    )
    if not ok:
        print(
            "         -> pass scale=attn.scale explicitly to flex_attention() "
            "in IdentityFlexSelfAttnProcessor."
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true",
                        help="also verify the torch.compile'd Flex path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required (FlexAttention has no meaningful CPU path).")
        return 1

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

        ref = run_with(attn, AttnProcessor2_0(), x)
        flex = run_with(attn, IdentityFlexSelfAttnProcessor(), x)

        all_ok &= compare("sdpa vs flex (eager)", flex, ref, TOL[dtype])

        # Shape / finiteness sanity -- catches a reshape that "works" but
        # transposes heads and sequence.
        if flex.shape != ref.shape:
            print(f"  [FAIL] shape mismatch {flex.shape} vs {ref.shape}")
            all_ok = False
        if not torch.isfinite(flex).all():
            print("  [FAIL] non-finite values in flex output")
            all_ok = False

        if args.compile:
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            compiled = torch.compile(attn)
            with torch.no_grad():
                out_c = compiled(x)
            all_ok &= compare("sdpa vs flex (compiled)", out_c, ref, TOL[dtype])

        print()

    # Gradients: the forward matching does not guarantee the backward does.
    print("gradient check (fp32)")
    attn = build_attention(device, torch.float32)
    for p in attn.parameters():
        p.requires_grad_(True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.float32)

    grads = {}
    for label, proc in (("sdpa", AttnProcessor2_0()),
                        ("flex", IdentityFlexSelfAttnProcessor())):
        attn.zero_grad(set_to_none=True)
        attn.set_processor(proc)
        attn(x).square().mean().backward()
        grads[label] = attn.to_q.weight.grad.detach().clone()

    all_ok &= compare("d/d(to_q.weight)", grads["flex"], grads["sdpa"], 1e-4)

    print()
    if all_ok:
        print("ALL CHECKS PASSED -- the Flex path is numerically the default path.")
        return 0
    print("FAILURES ABOVE -- do not proceed to training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

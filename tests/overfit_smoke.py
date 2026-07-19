#!/usr/bin/env python
"""tests/overfit_smoke.py
Gate 3: overfit a tiny fixed batch. Loss must collapse toward zero.

This is the end-to-end test of the training path -- model construction,
FlexAttention processor swap, timestep conditioning, class conditioning,
objective, and optimizer -- on a problem small enough that failure to learn
can only mean something is wired wrong. A model that cannot memorise 128
samples will not learn 1.28M.

Attention is ALWAYS FlexAttention (identity score_mod). There is no SDPA
path in this repo; a smoke that certified the stock diffusers processor
would certify a code path we never train on.

It is NOT a test of generative quality. Loss going to ~0 is the pass
condition; the resulting model is worthless.

Known smoke-only shortcuts (fine for memorisation, NOT for objective.py):
  - flow t is logit-normal here; SiT-L/2 parity requires uniform t
  - timestep=(t*1000).long() discretises continuous time
  - pred[:, :4] ignores DiT's learned-sigma channels; the published DiT
    recipe trains hybrid MSE+VLB on them -- decide and document in Phase 1

Run:
    python tests/overfit_smoke.py                          # random latents, DDPM
    python tests/overfit_smoke.py --objective flow
    python tests/overfit_smoke.py --latents ./dlatents/latents_0000.safetensors
    python tests/overfit_smoke.py --small                  # 6-layer model, fast

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel

from ditflex.attention import IdentityFlexSelfAttnProcessor

N_SAMPLES = 128
BATCH = 32
LATENT_SHAPE = (4, 32, 32)
N_CLASSES = 1000

# A fixed batch this small should drop by well over an order of magnitude.
# The absolute floor differs between objectives, so gate on the ratio.
PASS_RATIO = 0.10


def build_model(small: bool, device, dtype) -> DiTTransformer2DModel:
    # DiT-L/2: width 1024 = 16 heads x 64, depth 24, patch 2.
    # out_channels 8 = 4 eps + 4 sigma (DiT learns sigma; we use the first 4).
    model = DiTTransformer2DModel(
        num_attention_heads=16,
        attention_head_dim=64,
        in_channels=4,
        out_channels=8,
        num_layers=6 if small else 24,
        sample_size=32,
        patch_size=2,
        num_embeds_ada_norm=N_CLASSES + 1,   # +1 for the CFG null class
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )
    # The only attention path in this repo.
    model.set_attn_processor(IdentityFlexSelfAttnProcessor())
    return model.to(device=device, dtype=dtype)


def make_data(args, device):
    if args.latents:
        from safetensors import safe_open
        with safe_open(args.latents, framework="pt", device="cpu") as f:
            lat = f.get_tensor("latents")[:N_SAMPLES]
            lab = f.get_tensor("labels")[:N_SAMPLES]
        x0 = lat.view(-1, *LATENT_SHAPE).float().to(device)
        y = lab.long().to(device)
        print(f"data: {args.latents}  std={x0.std().item():.4f}")
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        x0 = torch.randn(N_SAMPLES, *LATENT_SHAPE, generator=g).to(device)
        y = torch.randint(0, N_CLASSES, (N_SAMPLES,), generator=g).to(device)
        print("data: random gaussian latents")
    return x0, y


def loss_ddpm(model, x0, y, alphas_cumprod):
    """eps-prediction against the linear-beta DDPM schedule."""
    t = torch.randint(0, len(alphas_cumprod), (x0.shape[0],), device=x0.device)
    ab = alphas_cumprod[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    pred = model(hidden_states=xt, timestep=t, class_labels=y).sample[:, :4]
    return F.mse_loss(pred, eps)


def loss_flow(model, x0, y, _):
    """Rectified flow / linear interpolant: x_t = (1-t) x0 + t eps, target v = eps - x0.
    Logit-normal t is a smoke-only choice -- see module docstring."""
    t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device))
    tb = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = (1 - tb) * x0 + tb * eps
    pred = model(hidden_states=xt,
                 timestep=(t * 1000).long(),
                 class_labels=y).sample[:, :4]
    return F.mse_loss(pred, eps - x0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["ddpm", "flow"], default="ddpm")
    parser.add_argument("--small", action="store_true", help="6 layers instead of 24")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latents", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"objective={args.objective}  attention=Flex(identity)  "
          f"layers={6 if args.small else 24}  steps={args.steps}")

    model = build_model(args.small, device, torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M\n")

    x0, y = make_data(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # DDPM schedule: linear betas, 1000 steps, as in the DiT reference.
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)

    loss_fn = loss_ddpm if args.objective == "ddpm" else loss_flow

    model.train()
    first, recent = None, []
    t_start = time.time()

    for step in range(args.steps):
        idx = torch.randint(0, N_SAMPLES, (BATCH,), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model, x0[idx], y[idx], abar)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        val = loss.item()
        if not torch.isfinite(loss):
            print(f"  step {step}: loss is {val} -- diverged")
            return 1

        if first is None:
            first = val
        if step >= args.steps - 20:
            recent.append(val)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  loss {val:.5f}")

    final = sum(recent) / len(recent)
    ratio = final / first
    dt = time.time() - t_start

    print(f"\nfirst={first:.5f}  final(avg last 20)={final:.5f}  ratio={ratio:.4f}")
    print(f"{args.steps} steps in {dt:.1f}s  ({args.steps / dt:.1f} steps/s)")

    if ratio < PASS_RATIO:
        print(f"\nPASS -- loss fell to {ratio:.1%} of initial (< {PASS_RATIO:.0%}).")
        return 0

    print(f"\nFAIL -- loss only fell to {ratio:.1%} of initial.")
    print("Check: class_labels wired through? timestep in the right range? "
          "LR sane? output slice [:, :4] correct?")
    return 1


if __name__ == "__main__":
    sys.exit(main())

# ditflex -- repo snapshot

Generated 2026-07-19 15:57 UTC by context/context.py. 11 files.

## Tree

```
DITFLEX/
├── .github/
    ├── workflows/
        ├── context.yml
├── README.md
├── pyproject.toml
├── run/
    ├── modal_train.py
├── scripts/
    ├── overfit_smoke.py
    ├── verify_identity.py
    ├── verify_latents.py
├── src/
    ├── ditflex/
        ├── __init__.py
├── tests/
    ├── modal_ci.py
    ├── test_attention_identity.py
    ├── verify_identity.py
```

## Files

### `.github/workflows/context.yml`

```yaml
name: context

# Regenerate context/context.md (a one-file snapshot of the repo for
# sharing) and commit it back.
#
# Triggers: manual dispatch, or a push to main that modifies the
# generator itself -- and ONLY the generator. Editing source files does
# not refresh the snapshot automatically; dispatch when you want a fresh
# one. (The workflow's own commit cannot retrigger it: it touches only
# context/context.md, carries [skip ci], and GITHUB_TOKEN pushes do not
# start workflows.)
#
# Runs entirely on the Actions runner -- no Modal, no GPU, no secrets
# beyond the automatic GITHUB_TOKEN. Reading files and writing markdown
# is not a job for an ephemeral server.

on:
  workflow_dispatch:

  push:
    branches: [main]
    paths:
      - context/context.py

permissions:
  contents: write   # allows the push with the automatic GITHUB_TOKEN

jobs:
  snapshot:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Generate context.md
        run: python context/context.py

      - name: Commit if changed
        run: |
          git config user.name "ditflex context bot"
          git config user.email "actions@github.com"
          git add context/context.md
          if git diff --cached --quiet; then
            echo "context.md unchanged -- nothing to commit"
          else
            git commit -m "context: refresh repo snapshot [skip ci]"
            git push
          fi
```

### `README.md`

````markdown
# dit-flex

DiT-L/2 on ImageNet-256 latents, with self-attention routed through PyTorch
FlexAttention so the attention score function is a swappable component.

Baselines: **DiT** (Peebles & Xie, 2023) for the DDPM objective, **SiT**
(Ma et al., 2024) for flow matching — SiT is the same architecture with the
objective swapped, so both are directly comparable at the DiT-L/2 config.

Training runs are **time-boxed and chained**: each job trains for a fixed wall
clock, pushes a checkpoint to the HF Hub, and exits. The next job resumes from
it. Long training is many short runs, not one long one.

---

## Status / TODO

Ordered. Do not skip ahead — each step makes the next one interpretable.

### Phase 0 — correctness gates
- [x] Encode ImageNet-256 → SD-VAE latents, upload to HF
- [x] Verify latent reconstruction (decode → image looks right)
- [ ] `scripts/verify_identity.py` — identity FlexAttention vs default Diffusers
      processor, assert max abs diff < 1e-4 in bf16. **Blocks everything.**
- [ ] `scripts/verify_latents.py` — load from Hub, assert shape `[N, 4096]`,
      `std ≈ 1.0` (scaling factor already applied — see Data Notes), labels in
      `[0, 999]`, N matches manifest
- [ ] `scripts/overfit_smoke.py` — 128 samples, loss → ~0 in a few hundred steps

### Phase 1 — plumbing
- [ ] `config.py` dataclasses, JSON round-trip test
- [ ] `latents.py` — GPU-resident store, rank-offset deterministic sampling
- [ ] `objective.py` — DDPM eps and flow matching behind one interface
- [ ] `checkpoint.py` — save/load/resume, HF Hub push/pull
- [ ] `train.py` — DDP, `torch.compile`, time-boxed loop
- [ ] `modal_app.py` — the only Modal-aware file

### Phase 2 — CI and launch
- [ ] `.github/workflows/test.yml` — CPU unit tests on push
- [ ] `.github/workflows/smoke.yml` — manual, 2 GPU / 10 min, full path
      (download → train → checkpoint → upload) end to end
- [ ] `.github/workflows/run.yml` — manual, 8 GPU / 5 h, detached

### Phase 3 — first real run
- [ ] 2×B300 smoke: 1000 steps, confirm loss decreasing and checkpoint round-trips
- [ ] 8×B300 × 5 h, DDPM objective, batch 256 — matches published DiT-L/2 recipe
- [ ] Chain runs to 400K steps
- [ ] `eval.py` + cached ImageNet reference stats → FID vs published DiT-L/2

### Phase 4 — the actual experiment
- [ ] Flow matching objective, same config → compare against SiT-L/2
- [ ] The score_mod modification
- [ ] Horizontal-flip latents (see Data Notes) if chasing published numbers

---

## Repo structure

```
dit-flex/
├── README.md
├── pyproject.toml
├── .github/workflows/
│   ├── test.yml                 # CPU unit tests, on push
│   ├── smoke.yml                # manual: 2 GPU, ~10 min, full path
│   └── run.yml                  # manual: 8 GPU, ~5 h, --detach
├── modal_app.py                 # the ONLY Modal-aware file
├── src/ditflex/
│   ├── config.py                # dataclasses -> JSON -> checkpoint
│   ├── distributed.py           # rank/world/init, rank0-only helpers
│   ├── attention.py             # IdentityFlexSelfAttnProcessor + score_mods
│   ├── model.py                 # build DiT-L/2, swap processors
│   ├── latents.py               # GPU-resident store; NO DataLoader
│   ├── objective.py             # DDPM eps | flow matching
│   ├── ema.py
│   ├── train.py                 # loop, compile, DDP, time-box
│   ├── checkpoint.py            # save/load/resume + HF Hub
│   ├── sample.py                # DDIM / ODE sampling + CFG
│   └── eval.py                  # FID vs cached reference stats
├── scripts/
│   ├── prepare_latents.py       # cleaned from imagenet-processing.ipynb
│   ├── verify_identity.py
│   ├── verify_latents.py
│   └── overfit_smoke.py
└── tests/
    ├── test_attention_identity.py
    ├── test_latents_shapes.py
    └── test_config_roundtrip.py
```

`torch.compile` lives in `train.py`, not `model.py` — tests and
`verify_identity.py` need the uncompiled model.

---

## Secrets

The important thing is **where** each lives. GitHub only launches; Modal does
the work and needs the data credentials.

### GitHub repository secrets
Used by the workflow launcher only.

| Secret | Purpose |
|---|---|
| `MODAL_TOKEN_ID` | authenticate `modal run` from CI |
| `MODAL_TOKEN_SECRET` | same |

`GITHUB_TOKEN` is injected automatically and does not need to be created.
The repo being private does not require an extra token: `actions/checkout`
uses the automatic token, and `modal run` uploads the checked-out source
into the container, so Modal never clones from GitHub.

### Modal secrets
Created with `modal secret create <name> KEY=value`. Referenced by name in
`modal_app.py`.

| Modal secret | Keys | Purpose |
|---|---|---|
| `huggingface` | `HF_TOKEN` | pull latents dataset, push checkpoints — **needs write scope** |
| `wandb` *(optional)* | `WANDB_API_KEY` | run logging |

One token with write access is simplest. If you want least privilege, split
into a read token for `sparsetrace/dlatentzz` and a write token scoped to the
checkpoint repo, as `HF_TOKEN_READ` / `HF_TOKEN_WRITE`.

### Local development
`.env` (gitignored), or just export:

```bash
export HF_TOKEN=hf_...
modal token new          # writes ~/.modal.toml, no env var needed
```

**Never** commit tokens. The source notebook had `HF_TOKEN` read from env —
keep that pattern in `scripts/prepare_latents.py`.

---

## Data notes

Latents: `sparsetrace/dlatentzz` — 32 safetensors files, ~10.5 GB total,
1.28 M ImageNet train images.

Four properties of the encoding that the training code must respect:

1. **Flat storage.** Shape is `[N, 4096]`, not `[N, 4, 32, 32]`.
   `latents.py` must `.view(-1, 4, 32, 32)`.
2. **Scaling factor already applied.** `z = z * 0.18215` happened at encode
   time. **Do not apply it again.** Assert `std ≈ 1.0` on load — if you see
   `≈ 5.5`, something is double-scaling.
3. **Deterministic latents.** Encoded with `posterior.mode()`, not
   `.sample()`. DiT samples the posterior each epoch; we froze the mean.
   Deviation from the published recipe — acceptable, but state it in any writeup.
4. **No horizontal flips.** DiT trains with random h-flip before the VAE.
   Latents cannot be flipped directly (the conv VAE is not exactly
   equivariant), so matching the reference recipe requires encoding a second
   flipped pass (+10.5 GB, trivial at 96 GB/GPU).

`dtype` is bf16 on disk, cast to fp32 or bf16 at load depending on the
training precision.

---

## Design decisions

**No DataLoader, no DistributedSampler.** The full 10.5 GB tensor lives on
each GPU. Each rank draws indices from a generator seeded by
`(global_step, rank)` — stateless, so resume is exact and survives a change
of world size.

**DDP, not FSDP.** DiT-L is 458 M params; weights + EMA + AdamW states are
~7.3 GB in fp32. Sharding buys nothing at this scale and costs complexity.

**Fixed shapes everywhere.** Fixed batch, fixed 256-token sequence,
`drop_last=True` → one `torch.compile`, no recompiles.

**Compile inner, then DDP-wrap.** Test the other order once; this interaction
has been version-sensitive.

**Save uncompiled, unwrapped state dicts.** Strip `_orig_mod.` and `module.`
prefixes before writing, or checkpoints will not load into a bare model for
sampling.

---

## Checkpointing

Runs are time-boxed. The loop checks a deadline every 500 steps (rank 0
decides, broadcast to all — avoids clock drift), stops cleanly, saves, uploads.

Budget: ~7.3 GB per checkpoint, ~12 min up and ~12 min down at 100 MB/s.
A 5 h job is ~4.5 h of training. `torch.compile` costs another 2–5 min on a
cold container.

Hub layout (`sparsetrace/dit-flex-L2`, model repo):

```
state.json              # step, wall-clock, config, git sha, run_history
model.safetensors       # fp32 weights
ema.safetensors         # fp32 EMA (0.9999)
optim.safetensors       # AdamW m, v
archive/step_0200000/   # periodic, EMA + state only, kept forever
```

Top level is always "latest" and is overwritten each run; HF repos are git,
so prior revisions remain recoverable. `run_history` in `state.json` records
each run's step range and duration — worth having when a loss discontinuity
turns out to align with a run boundary.

---

## Quickstart

```bash
# gates
python scripts/verify_identity.py
python scripts/verify_latents.py
python scripts/overfit_smoke.py

# short run on Modal
modal run modal_app.py::train --gpus 2 --train-seconds 600 --objective ddpm

# real run, detached (survives the CI job exiting)
modal run --detach modal_app.py::train \
    --gpus 8 --train-seconds 18000 --objective flow
```

---

## Recipe

Held at the published DiT-L/2 settings so the baseline is comparable:

| | |
|---|---|
| model | DiT-L/2, 458 M params, patch 2, 24 layers, width 1024, 16 heads |
| latents | 32×32×4 → 256 tokens |
| batch | 256 global |
| optimizer | AdamW, lr 1e-4 constant, no warmup, no weight decay |
| EMA | 0.9999 |
| precision | bf16 autocast, fp32 master weights |
| label dropout | 10% (for classifier-free guidance) |

Do not scale the batch on the first run — if batch and objective change
together, the comparison to published numbers means nothing.
````

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ditflex"
version = "0.1.0"
description = "DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Julio Candanedo" }]

# torch is deliberately NOT pinned here: the correct build depends on the
# CUDA version of the host, and Modal installs it in the image definition.
# Install it first, then `pip install -e .`
dependencies = [
    "diffusers>=0.31",
    "transformers>=4.44",
    "safetensors>=0.4.5",
    "huggingface_hub>=0.26",
    "numpy>=1.26",
    "tqdm",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]
eval = [
    "scipy",              # FID: matrix sqrt of the covariance
    "torchvision",        # InceptionV3 features
    "pillow",
]
logging = [
    "wandb",
]
prepare = [                # only for scripts/prepare_latents.py
    "webdataset",
    "torchvision",
    "pillow",
]

[project.urls]
Repository = "https://github.com/jcandane/ditflex"

[tool.hatch.build.targets.wheel]
packages = ["src/ditflex"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]        # line length handled by formatter
```

### `run/modal_train.py`

```python
"""run/modal_train.py -- time-boxed DiT-L/2 training on Modal.

The training counterpart to tests/modal_ci.py. Launched detached so the
GitHub Actions job that starts it can exit after minutes while the run
continues:

    MODAL_GPUS=8 modal run --detach run/modal_train.py --train-seconds 7200 --objective ddpm

Long training is many short runs, not one long one: each invocation pulls
the latest checkpoint from the HF Hub (if any), trains until the wall-clock
budget is spent, saves, uploads, and exits. Chain invocations to accumulate
steps. Resume is exact because data sampling is stateless -- indices are
drawn from a generator seeded by (global_step, rank).

GPU kind and count are fixed when the Modal function is built, so they are
env vars (set by the workflow), not CLI flags:

    MODAL_GPU    B300 (default) | B200 | ...
    MODAL_GPUS   8 (default) | 2 for smoke

Requires the `huggingface` Modal secret (HF_TOKEN with write scope) for
pulling latents and pushing checkpoints. `wandb` secret is optional.

NOTE: this launcher is ready; the entrypoint it launches (ditflex.train)
is Phase 1 and does not exist yet. Until it does, this file is
scaffolding -- do not wire train.yml to real GPU hours before
tests/modal_ci.py is green and the overfit smoke passes.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "8"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

# The Modal function timeout must cover train_seconds PLUS checkpoint
# download (~12 min), upload (~12 min), and cold compile (2-5 min), or the
# run is killed mid-upload and the checkpoint is lost. Set a generous
# ceiling here; the real stepping budget is enforced inside train.py by
# --train-seconds (rank 0 checks the deadline every 500 steps and
# broadcasts the stop).
TIMEOUT_CEILING = 24 * 3600

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-train", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=TIMEOUT_CEILING,
    secrets=[
        modal.Secret.from_name("huggingface"),
        # modal.Secret.from_name("wandb"),  # enable when logging lands
    ],
)
def train(train_seconds: int = 7200, objective: str = "ddpm") -> int:
    import subprocess
    import sys

    import torch

    n_gpu = torch.cuda.device_count()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[modal] {n_gpu} GPUs:\n{result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True
    )

    # torchrun handles per-rank process spawn + env for single-node DDP.
    # nproc comes from what the container actually has, so the launcher
    # cannot disagree with the reservation.
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={n_gpu}",
        "--standalone",
        "-m",
        "ditflex.train",  # Phase 1: src/ditflex/train.py
        f"--train-seconds={train_seconds}",
        f"--objective={objective}",
    ]
    print(f"\n[modal] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(train_seconds: int = 7200, objective: str = "ddpm"):
    """
    Args:
        train_seconds: stepping budget (checkpoint I/O and compile are on top)
        objective:     ddpm | flow
    """
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")

    rc = train.remote(train_seconds=train_seconds, objective=objective)
    if rc != 0:
        raise SystemExit(rc)
```

### `scripts/overfit_smoke.py`

```python
#!/usr/bin/env python
""" DITMAP/scripts/overfit_smoke.py
Gate 3: overfit a tiny fixed batch. Loss must collapse toward zero.

This is the end-to-end test of the training path -- model construction,
processor swap, timestep conditioning, class conditioning, objective, and
optimizer -- on a problem small enough that failure to learn can only mean
something is wired wrong. A model that cannot memorise 128 samples will not
learn 1.28M.

It is NOT a test of generative quality. Loss going to ~0 is the pass
condition; the resulting model is worthless.

Run:
    python scripts/overfit_smoke.py                          # random latents, DDPM
    python scripts/overfit_smoke.py --objective flow
    python scripts/overfit_smoke.py --flex                   # via FlexAttention
    python scripts/overfit_smoke.py --latents ./dlatents/latents_0000.safetensors
    python scripts/overfit_smoke.py --small                  # 6-layer model, fast

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel

N_SAMPLES = 128
BATCH = 32
LATENT_SHAPE = (4, 32, 32)
N_CLASSES = 1000

# A fixed batch this small should drop by well over an order of magnitude.
# The absolute floor differs between objectives, so gate on the ratio.
PASS_RATIO = 0.10


def build_model(small: bool, flex: bool, device, dtype) -> DiTTransformer2DModel:
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
    if flex:
        from ditflex.attention import IdentityFlexSelfAttnProcessor
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


def loss_ddpm(model, x0, y, betas_cumprod):
    """eps-prediction against a linear-beta cosine-free schedule."""
    t = torch.randint(0, len(betas_cumprod), (x0.shape[0],), device=x0.device)
    ab = betas_cumprod[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    pred = model(hidden_states=xt, timestep=t, class_labels=y).sample[:, :4]
    return F.mse_loss(pred, eps)


def loss_flow(model, x0, y, _):
    """Rectified flow / linear interpolant: x_t = (1-t) x0 + t eps, target v = eps - x0."""
    t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device))   # logit-normal
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
    parser.add_argument("--flex", action="store_true", help="use FlexAttention processor")
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
    print(f"objective={args.objective}  flex={args.flex}  "
          f"layers={6 if args.small else 24}  steps={args.steps}")

    model = build_model(args.small, args.flex, device, torch.float32)
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
```

### `scripts/verify_identity.py`

```python
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
```

### `scripts/verify_latents.py`

```python
#!/usr/bin/env python
"""
Gate 2: validate the SD-VAE latents before spending GPU-hours on them.

The expensive failure this catches is double-scaling. The encoder already
applied scaling_factor=0.18215, so the stored latents have std ~1.0. If the
training code multiplies by 0.18215 again, nothing crashes -- the model just
trains on latents with the wrong variance and produces mysteriously poor FID
many hours later.

Run:
    python scripts/verify_latents.py                 # first shard only, fast
    python scripts/verify_latents.py --all           # every shard
    python scripts/verify_latents.py --local ./dlatents

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import list_repo_files, hf_hub_download
from safetensors import safe_open

REPO_ID = "sparsetrace/dlatentzz"
LATENT_SHAPE = (4, 32, 32)
FLAT_DIM = 4 * 32 * 32          # 4096
N_CLASSES = 1000
EXPECTED_TOTAL = 1_281_167      # ImageNet-1k train

# scaling_factor is pre-applied, so std should sit near 1. These bounds are
# loose enough for per-shard variation but far tighter than the ~5.5 you would
# see if the factor had NOT been applied, or the ~0.18 if applied twice.
STD_LO, STD_HI = 0.7, 1.4


def shard_paths(args) -> list[Path]:
    if args.local:
        paths = sorted(Path(args.local).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"no .safetensors under {args.local}")
        return paths if args.all else paths[:1]

    files = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset")
                   if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {REPO_ID}")
    if not args.all:
        files = files[:1]
    print(f"downloading {len(files)} shard(s) from {REPO_ID} ...")
    return [Path(hf_hub_download(REPO_ID, f, repo_type="dataset")) for f in files]


def check(cond: bool, msg: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return cond


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="check every shard")
    parser.add_argument("--local", type=str, default=None,
                        help="directory of local .safetensors instead of the Hub")
    args = parser.parse_args()

    paths = shard_paths(args)
    ok = True
    total_n = 0
    seen_labels = torch.zeros(N_CLASSES, dtype=torch.bool)

    for path in paths:
        print(f"\n{path.name}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            ok &= check({"latents", "labels"} <= keys,
                        f"keys present: {sorted(keys)}")
            if not {"latents", "labels"} <= keys:
                continue
            lat = f.get_tensor("latents")
            lab = f.get_tensor("labels")

        n = lat.shape[0]
        total_n += n

        ok &= check(lat.ndim == 2 and lat.shape[1] == FLAT_DIM,
                    f"latents shape {tuple(lat.shape)} == (N, {FLAT_DIM})")
        ok &= check(lat.dtype == torch.bfloat16,
                    f"latents dtype {lat.dtype} == bfloat16")
        ok &= check(lab.shape == (n,), f"labels shape {tuple(lab.shape)} == ({n},)")

        # Statistics on a subsample -- reading 300 MB per shard is enough.
        sub = lat[: min(n, 8192)].float()
        mean, std = sub.mean().item(), sub.std().item()
        ok &= check(STD_LO < std < STD_HI,
                    f"std {std:.4f} in ({STD_LO}, {STD_HI}) -- scaling applied exactly once")
        if not (STD_LO < std < STD_HI):
            if std > 3.0:
                print("         -> looks UNSCALED. Multiply by 0.18215 at load.")
            elif std < 0.4:
                print("         -> looks DOUBLE-scaled. Re-encode, or divide by 0.18215.")
        ok &= check(abs(mean) < 0.5, f"mean {mean:+.4f} near zero")
        ok &= check(torch.isfinite(sub).all().item(), "all finite (no NaN/Inf)")

        lo, hi = int(lab.min()), int(lab.max())
        ok &= check(0 <= lo and hi < N_CLASSES,
                    f"labels in [{lo}, {hi}] within [0, {N_CLASSES - 1}]")
        seen_labels[lab.long().clamp(0, N_CLASSES - 1).unique()] = True

        # The reshape the training code will perform.
        view = lat[:2].view(-1, *LATENT_SHAPE)
        ok &= check(view.shape == (2, *LATENT_SHAPE),
                    f"reshape to {(2, *LATENT_SHAPE)} works")

        print(f"         N={n:,}  mean={mean:+.4f}  std={std:.4f}")

    print(f"\ntotal N over {len(paths)} shard(s): {total_n:,}")
    if args.all:
        ok &= check(total_n == EXPECTED_TOTAL,
                    f"total {total_n:,} == ImageNet train {EXPECTED_TOTAL:,}")
        n_seen = int(seen_labels.sum())
        ok &= check(n_seen == N_CLASSES, f"{n_seen}/{N_CLASSES} classes present")
        gb = total_n * FLAT_DIM * 2 / 1024**3
        print(f"         resident size in bf16: {gb:.2f} GiB per GPU")
    else:
        print("         (run with --all to check totals and class coverage)")

    print()
    if ok:
        print("ALL CHECKS PASSED.")
        return 0
    print("FAILURES ABOVE -- do not start a long run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### `src/ditflex/__init__.py`

```python
""" src/ditflex/__init__.py
ditflex: DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions.
"""

from ditflex.attention import FlexSelfAttnProcessor, reference_self_attention

__all__ = ["FlexSelfAttnProcessor", "reference_self_attention"]
__version__ = "0.1.0"
```

### `tests/modal_ci.py`

```python
"""Modal runner for the dit-flex GPU gates and tests.

Adapted from the flashdiffusion CI runner, with one structural change: the
source is NOT cloned from GitHub inside the container. `modal run` uploads
the local checkout (the Actions checkout, in CI) into the image, so there
is no GH_TOKEN, no clone step, and the code under test is exactly the
code in the working tree -- including uncommitted changes when run locally.

Usage
-----
    # everything (verify_identity + pytest)
    modal run tests/modal_ci.py

    # include the torch.compile'd Flex path check
    modal run tests/modal_ci.py --compile-check

    # specific pytest file only
    modal run tests/modal_ci.py --test-file test_attention_identity.py

Environment
-----------
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET   auth (or ~/.modal.toml locally)
    MODAL_GPU     GPU target (default L40S -- the gates need a GPU with
                  Flex support, not a big one; use B300 via workflow input
                  for the pre-training smoke on the real hardware)
    TORCH_INDEX   torch wheel index (default cu128)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# This file lives in tests/. The mount MUST be the repo root, or the
# container is missing src/, run/, and pyproject.toml.
REPO_ROOT = Path(__file__).parent.parent

GPU_TYPE = os.environ.get("MODAL_GPU", "L40S")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
        "pytest>=8.0",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-ci", image=image)


@app.function(gpu=GPU_TYPE, timeout=1800)
def run_gates(test_files: list | None = None, compile_check: bool = False) -> int:
    import subprocess
    import sys

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True
    )

    rc = 0

    gate_cmd = [sys.executable, "tests/verify_identity.py"]
    if compile_check:
        gate_cmd.append("--compile")
    print(f"\n[modal] running: {' '.join(gate_cmd)}\n")
    rc = subprocess.run(gate_cmd, cwd="/repo").returncode or rc

    pytest_cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    pytest_cmd += test_files or ["tests"]
    print(f"\n[modal] running: {' '.join(pytest_cmd)}\n")
    rc = subprocess.run(pytest_cmd, cwd="/repo").returncode or rc

    return rc


@app.local_entrypoint()
def main(test_file: str = "", compile_check: bool = False):
    """
    Args:
        test_file:     specific pytest file (e.g. test_attention_identity.py), empty = all
        compile_check: also verify the torch.compile'd Flex path
    """
    files = None
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        files = [test_file]

    rc = run_gates.remote(test_files=files, compile_check=compile_check)
    if rc != 0:
        raise SystemExit(rc)
```

### `tests/test_attention_identity.py`

```python
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
```

### `tests/verify_identity.py`

```python
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
```

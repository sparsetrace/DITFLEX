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

HF_TOKEN (write scope: pulls latents AND pushes checkpoints) comes from
the launching environment -- a GitHub repo secret exported by train.yml,
or `export HF_TOKEN=...` locally -- and is forwarded into the container
via Secret.from_dict. No Modal-side secret needed.

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
# download (~12 min), upload (~12 min), and cold compile (~3 min) -- but
# NOT much more: a hung run bills until the timeout kills it. So the
# ceiling derives from the requested budget: the workflow exports
# MODAL_TRAIN_SECONDS and we add a 1-hour overhead allowance. A 2-hour
# dispatch can never bill more than ~3 hours, even if everything hangs.
_BUDGET = int(os.environ.get("MODAL_TRAIN_SECONDS", "7200"))
TIMEOUT_CEILING = _BUDGET + 3600

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
        "pillow",          # sample-grid PNG at end of each link
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
        modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")}),
        # add WANDB_API_KEY here the same way when logging lands
    ],
)
def train(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    lr: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
) -> int:
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
        "ditflex.train",
        f"--train-seconds={train_seconds}",
        f"--objective={objective}",
    ]
    if hub_repo:
        cmd.append(f"--hub-repo={hub_repo}")
    if max_steps > 0:
        cmd.append(f"--max-steps={max_steps}")
    if lr > 0.0:
        cmd.append(f"--lr={lr}")
    if wd >= 0.0:
        cmd.append(f"--wd={wd}")
    cmd.append(f"--clip={clip}")
    cmd.append(f"--spike-skip={spike_skip}")
    if seed_offset != 0:
        cmd.append(f"--seed-offset={seed_offset}")
    print(f"\n[modal] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    lr: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
):
    """
    Args:
        train_seconds: stepping budget (checkpoint I/O and compile are on top)
        objective:     ddpm | flow
        hub_repo:      checkpoint repo override. SMOKES MUST SET THIS to a
                       scratch repo -- a smoke that pushes to the real repo
                       would be silently resumed by the real run.
    """
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")

    rc = train.remote(
        train_seconds=train_seconds, objective=objective,
        hub_repo=hub_repo, max_steps=max_steps, lr=lr, wd=wd,
        clip=clip, spike_skip=spike_skip, seed_offset=seed_offset,
    )
    if rc != 0:
        raise SystemExit(rc)

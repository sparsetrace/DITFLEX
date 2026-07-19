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

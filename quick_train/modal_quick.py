"""quick_train/modal_quick.py -- loop-certification training smoke.

Runs the REAL training entrypoint (ditflex.train) on real latents for a
short wall-clock budget, with --no-push: nothing is written to any Hub
repo. This certifies the full training loop -- latents download and
GPU-resident load, compile-then-DDP wrap order, objective stepping, loss
decreasing, deadline broadcast -- without producing artifacts.

What it deliberately does NOT certify: the checkpoint chain
(save -> push -> pull -> exact resume). That is certified by the first
two short /run/ dispatches, whose steps count toward the real chain.

    MODAL_GPUS=2 modal run quick_train/modal_quick.py --train-seconds 600 --objective flow

Environment: MODAL_GPU (default B300), MODAL_GPUS (default 2), HF_TOKEN
(latents download), TORCH_INDEX (default cu129).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "2"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

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

app = modal.App("ditflex-quick-train", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=3 * 3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def quick(train_seconds: int = 600, objective: str = "flow", max_latent_files: int = 0) -> int:
    import subprocess
    import sys

    import torch

    n_gpu = torch.cuda.device_count()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[quick] {n_gpu} GPUs:\n{result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True
    )

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
        "--no-push",                      # the defining property of quick_train
    ]
    if max_latent_files > 0:
        cmd.append(f"--max-latent-files={max_latent_files}")
    print(f"\n[quick] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(train_seconds: int = 600, objective: str = "flow", max_latent_files: int = 0):
    """
    Args:
        train_seconds:    stepping budget
        objective:        ddpm | flow
        max_latent_files: 0 = all 32 shards (realistic); small N for speed
    """
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")
    rc = quick.remote(
        train_seconds=train_seconds, objective=objective, max_latent_files=max_latent_files
    )
    if rc != 0:
        raise SystemExit(rc)

"""train_diffusion/modal_train_dmap.py -- the DMAP-DiT chain.

Identical launch machinery to run/modal_train.py, with two pinned
differences: --qk-mode=dmap (EQ-sector attention: W_K tied to W_Q, every
score matrix symmetric, R == 0 by construction) and its own checkpoint
repo. The config-drift guard makes the separation load-bearing: a dmap
chain can never silently resume from an amap checkpoint or vice versa.

The experiment this trains, against the paper's Table 1: the baseline
chain learns R freely (drifting from ~1.0 at init toward the flow band
~0.78-0.86); this chain is pinned at R = 0. The difference in FID and
samples measures what the antisymmetric / non-equilibrium component of
attention is worth.

    MODAL_GPUS=2 modal run --detach train_diffusion/modal_train_dmap.py \
        --train-seconds 14400
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "2"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

_BUDGET = int(os.environ.get("MODAL_TRAIN_SECONDS", "14400"))
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

app = modal.App("ditflex-train-dmap", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=TIMEOUT_CEILING,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(
    train_seconds: int = 14400,
    objective: str = "flow",
    hub_repo: str = "sparsetrace/ditflex-L2-flow-dmap",
    dmap_alpha: float = 0.0,
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
        f"--hub-repo={hub_repo}",
        "--qk-mode=dmap",              # the defining property of this chain
        f"--dmap-alpha={dmap_alpha}",
    ]
    print(f"\n[modal] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(
    train_seconds: int = 14400,
    objective: str = "flow",
    hub_repo: str = "sparsetrace/ditflex-L2-flow-dmap",
    dmap_alpha: float = 0.0,
):
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")
    rc = train.remote(
        train_seconds=train_seconds, objective=objective,
        hub_repo=hub_repo, dmap_alpha=dmap_alpha,
    )
    if rc != 0:
        raise SystemExit(rc)

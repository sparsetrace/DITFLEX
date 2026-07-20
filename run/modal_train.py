"""run/modal_train.py -- detached Modal launcher for DiT/SiT training.

GitHub Actions starts this app and exits; the Modal function pulls the latest
healthy checkpoint, launches torchrun, trains for the requested budget, pushes
a new healthy checkpoint, and exits.

GPU reservation is configured by environment variables because Modal needs it
when the function is defined:

    MODAL_GPU=B200
    MODAL_GPUS=1
    MODAL_TRAIN_SECONDS=14400
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "8"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

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
        "pillow",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[
            ".git",
            "**/__pycache__",
            "*.egg-info",
            ".venv",
            ".ruff_cache",
            ".pytest_cache",
        ],
    )
)

app = modal.App("ditflex-train", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=TIMEOUT_CEILING,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    target_steps: int = 400_000,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    reset_lr_controller: bool = False,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
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
    if n_gpu <= 0:
        print("[modal] no CUDA devices visible")
        return 2

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"],
        check=True,
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
        f"--target-steps={target_steps}",
        f"--lr-policy={lr_policy}",
        f"--lr-min={lr_min}",
        f"--lr-hard-min={lr_hard_min}",
        f"--lr-backoff={lr_backoff}",
        f"--lr-min-scale={lr_min_scale}",
        f"--loss-rise-ratio={loss_rise_ratio}",
        f"--loss-emergency-ratio={loss_emergency_ratio}",
        f"--clip={clip}",
        f"--spike-skip={spike_skip}",
        f"--grad-ceiling={grad_ceiling}",
    ]
    if hub_repo:
        cmd.append(f"--hub-repo={hub_repo}")
    if max_steps > 0:
        cmd.append(f"--max-steps={max_steps}")
    if lr > 0.0:
        cmd.append(f"--lr={lr}")
    if wd >= 0.0:
        cmd.append(f"--wd={wd}")
    if seed_offset != 0:
        cmd.append(f"--seed-offset={seed_offset}")
    if reset_lr_controller:
        cmd.append("--reset-lr-controller")

    print(f"\n[modal] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    target_steps: int = 400_000,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    reset_lr_controller: bool = False,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
):
    if objective not in {"ddpm", "flow"}:
        raise SystemExit(f"unknown objective: {objective!r}")
    if lr_policy not in {"constant", "cosine", "adaptive"}:
        raise SystemExit(f"unknown lr_policy: {lr_policy!r}")

    rc = train.remote(
        train_seconds=train_seconds,
        objective=objective,
        hub_repo=hub_repo,
        max_steps=max_steps,
        target_steps=target_steps,
        lr=lr,
        lr_policy=lr_policy,
        lr_min=lr_min,
        lr_hard_min=lr_hard_min,
        lr_backoff=lr_backoff,
        lr_min_scale=lr_min_scale,
        loss_rise_ratio=loss_rise_ratio,
        loss_emergency_ratio=loss_emergency_ratio,
        reset_lr_controller=reset_lr_controller,
        wd=wd,
        clip=clip,
        spike_skip=spike_skip,
        seed_offset=seed_offset,
        grad_ceiling=grad_ceiling,
    )
    if rc != 0:
        raise SystemExit(rc)

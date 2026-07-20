"""Detached Modal supervisor for transactional DiT/SiT training.

One Modal container may launch several *fresh* torchrun processes.  Exit code
75 means the child detected retryable numerical instability and deliberately
discarded its candidate.  The supervisor then reloads the last promoted Hub
checkpoint with:

* a lower retry LR multiplier;
* a deterministic new data/objective seed offset;
* the same model, EMA, AdamW moments, and global step from the committed state.

The retry count is bounded.  Code errors and unrelated subprocess failures are
not hidden by a broad exception handler.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "8"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

_BUDGET = int(os.environ.get("MODAL_TRAIN_SECONDS", "7200"))
_MAX_RETRIES_ENV = int(os.environ.get("MODAL_MAX_RETRIES", "2"))
# Per attempt: checkpoint pull + compile + upload allowance.  The stepping
# budget itself is shared across retries by reading the child's retry marker.
TIMEOUT_CEILING = _BUDGET + 3600 * (_MAX_RETRIES_ENV + 1)

RETRY_EXIT_CODE = 75
RETRY_MARKER = Path("/tmp/ditflex_retry.json")
PROMOTION_MARKER = Path("/tmp/ditflex_promotion.json")

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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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
    resume_revision: str = "",
    resume_step: int = 0,
    auto_legacy_rollback: bool = True,
    legacy_suspect_ratio: float = 8.0,
    max_retries: int = 2,
    retry_seed_stride: int = 1_000_003,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    # Kept only so the existing v2 GitHub workflow remains callable.  V3 uses
    # committed-reference thresholds below rather than fast/slow EMA ratios.
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    health_loss_warn_ratio: float = 1.015,
    health_loss_retry_ratio: float = 1.025,
    health_loss_emergency_ratio: float = 1.05,
    health_grad_warn_ratio: float = 2.0,
    health_grad_retry_ratio: float = 4.0,
    health_grad_emergency_ratio: float = 8.0,
    commit_windows: int = 2,
    warning_patience: int = 2,
    reset_lr_controller: bool = False,
    grad_reference: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
) -> int:
    import subprocess
    import sys

    import torch

    if max_retries < 0:
        print("[modal] max_retries must be non-negative")
        return 2
    if not (0.0 < lr_backoff < 1.0):
        print("[modal] lr_backoff must lie in (0, 1)")
        return 2
    if train_seconds <= 0:
        print("[modal] train_seconds must be positive")
        return 2

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

    from ditflex.checkpoint import (
        resolve_revision_for_step,
        select_stable_resume_revision,
    )

    selected_revision = resume_revision.strip()
    selected_step: int | None = resume_step if resume_step > 0 else None
    if selected_revision and selected_step is not None:
        print("[modal] use only one of resume_revision or resume_step")
        return 2
    if selected_step is not None:
        if not hub_repo:
            print("[modal] resume_step requires hub_repo")
            return 2
        selected_revision = resolve_revision_for_step(hub_repo, selected_step)
        print(
            f"[modal] explicit anchor step {selected_step:,} -> "
            f"revision {selected_revision[:12]}"
        )
    elif not selected_revision and auto_legacy_rollback and hub_repo:
        selection = select_stable_resume_revision(
            hub_repo,
            suspect_ratio=legacy_suspect_ratio,
        )
        selected_revision = selection.revision or ""
        selected_step = selection.step
        print(f"[modal] resume selection: {selection.reason}")
        if selected_revision:
            print(
                f"[modal] using migration anchor step {selected_step:,} "
                f"revision {selected_revision[:12]}"
            )

    if loss_rise_ratio != 1.08 or loss_emergency_ratio != 1.35:
        print(
            "[modal] NOTE: v2 loss_rise_ratio/loss_emergency_ratio are deprecated; "
            "v3 uses health_loss_* committed-reference thresholds"
        )

    remaining_train_seconds = float(train_seconds)
    for attempt in range(max_retries + 1):
        if remaining_train_seconds < 1.0:
            print("[modal] retry budget exhausted before another attempt")
            return RETRY_EXIT_CODE

        RETRY_MARKER.unlink(missing_ok=True)
        PROMOTION_MARKER.unlink(missing_ok=True)

        attempt_factor = lr_backoff**attempt
        attempt_seed_offset = seed_offset + attempt * retry_seed_stride
        child_budget = max(1, int(remaining_train_seconds))

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={n_gpu}",
            "--standalone",
            "-m",
            "ditflex.train",
            f"--train-seconds={child_budget}",
            f"--objective={objective}",
            f"--target-steps={target_steps}",
            f"--attempt={attempt}",
            f"--attempt-lr-factor={attempt_factor}",
            f"--seed-offset={attempt_seed_offset}",
            f"--lr-policy={lr_policy}",
            f"--lr-min={lr_min}",
            f"--lr-hard-min={lr_hard_min}",
            f"--lr-min-scale={lr_min_scale}",
            f"--commit-windows={commit_windows}",
            f"--warning-patience={warning_patience}",
            f"--loss-warn-ratio={health_loss_warn_ratio}",
            f"--loss-retry-ratio={health_loss_retry_ratio}",
            f"--loss-emergency-ratio={health_loss_emergency_ratio}",
            f"--grad-warn-ratio={health_grad_warn_ratio}",
            f"--grad-retry-ratio={health_grad_retry_ratio}",
            f"--grad-emergency-ratio={health_grad_emergency_ratio}",
            f"--clip={clip}",
            f"--spike-skip={spike_skip}",
            f"--grad-ceiling={grad_ceiling}",
        ]
        if hub_repo:
            command.append(f"--hub-repo={hub_repo}")
        if selected_revision:
            command.append(f"--resume-revision={selected_revision}")
        if max_steps > 0:
            command.append(f"--max-steps={max_steps}")
        if lr > 0.0:
            command.append(f"--lr={lr}")
        if grad_reference > 0.0:
            command.append(f"--grad-reference={grad_reference}")
        if wd >= 0.0:
            command.append(f"--wd={wd}")
        if reset_lr_controller and attempt == 0:
            command.append("--reset-lr-controller")

        print(
            f"\n[modal] attempt {attempt}/{max_retries}: "
            f"lr_factor={attempt_factor:g} seed_offset={attempt_seed_offset} "
            f"budget={child_budget}s anchor="
            f"{selected_revision[:12] if selected_revision else 'latest'}\n"
            f"[modal] running: {' '.join(command)}\n"
        )
        started = time.time()
        result = subprocess.run(command, cwd="/repo")
        child_wall = time.time() - started
        if result.returncode == 0:
            print(f"[modal] attempt {attempt} completed successfully")
            return 0

        # torchrun commonly wraps a worker's exit code in ChildFailedError and
        # returns 1 itself.  The rank-0 atomic marker is therefore the source of
        # truth for a deliberate transactional retry.
        retry = _read_json(RETRY_MARKER)
        retry_requested = int(retry.get("exit_code", 0) or 0) == RETRY_EXIT_CODE
        if not retry_requested:
            print(
                f"[modal] child failed with non-retryable exit code "
                f"{result.returncode}; not masking the failure"
            )
            return result.returncode

        consumed = float(retry.get("elapsed_training_seconds", child_wall))
        remaining_train_seconds = max(0.0, remaining_train_seconds - consumed)
        print(
            f"[modal] retry requested: {retry.get('reason', 'no marker reason')}\n"
            f"[modal] stepping budget consumed={consumed:.1f}s, "
            f"remaining={remaining_train_seconds:.1f}s"
        )

        # If this attempt promoted healthy progress before a later failure,
        # retry from ordinary latest.  Otherwise preserve the explicit legacy
        # migration anchor instead of falling back to a suspect old latest.
        promotion = _read_json(PROMOTION_MARKER)
        promoted_step = int(promotion.get("step", 0) or 0)
        if promoted_step > 0 and (selected_step is None or promoted_step > selected_step):
            selected_revision = ""
            selected_step = promoted_step
            print(
                f"[modal] attempt promoted healthy step {promoted_step:,}; "
                "next retry will pull Hub latest"
            )

        if attempt >= max_retries:
            print("[modal] retry limit reached; last committed checkpoint remains untouched")
            return RETRY_EXIT_CODE

    return RETRY_EXIT_CODE


@app.local_entrypoint()
def main(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    target_steps: int = 400_000,
    resume_revision: str = "",
    resume_step: int = 0,
    auto_legacy_rollback: bool = True,
    legacy_suspect_ratio: float = 8.0,
    max_retries: int = 2,
    retry_seed_stride: int = 1_000_003,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    health_loss_warn_ratio: float = 1.015,
    health_loss_retry_ratio: float = 1.025,
    health_loss_emergency_ratio: float = 1.05,
    health_grad_warn_ratio: float = 2.0,
    health_grad_retry_ratio: float = 4.0,
    health_grad_emergency_ratio: float = 8.0,
    commit_windows: int = 2,
    warning_patience: int = 2,
    reset_lr_controller: bool = False,
    grad_reference: float = 0.0,
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

    return_code = train.remote(
        train_seconds=train_seconds,
        objective=objective,
        hub_repo=hub_repo,
        max_steps=max_steps,
        target_steps=target_steps,
        resume_revision=resume_revision,
        resume_step=resume_step,
        auto_legacy_rollback=auto_legacy_rollback,
        legacy_suspect_ratio=legacy_suspect_ratio,
        max_retries=max_retries,
        retry_seed_stride=retry_seed_stride,
        lr=lr,
        lr_policy=lr_policy,
        lr_min=lr_min,
        lr_hard_min=lr_hard_min,
        lr_backoff=lr_backoff,
        lr_min_scale=lr_min_scale,
        loss_rise_ratio=loss_rise_ratio,
        loss_emergency_ratio=loss_emergency_ratio,
        health_loss_warn_ratio=health_loss_warn_ratio,
        health_loss_retry_ratio=health_loss_retry_ratio,
        health_loss_emergency_ratio=health_loss_emergency_ratio,
        health_grad_warn_ratio=health_grad_warn_ratio,
        health_grad_retry_ratio=health_grad_retry_ratio,
        health_grad_emergency_ratio=health_grad_emergency_ratio,
        commit_windows=commit_windows,
        warning_patience=warning_patience,
        reset_lr_controller=reset_lr_controller,
        grad_reference=grad_reference,
        wd=wd,
        clip=clip,
        spike_skip=spike_skip,
        seed_offset=seed_offset,
        grad_ceiling=grad_ceiling,
    )
    if return_code != 0:
        raise SystemExit(return_code)

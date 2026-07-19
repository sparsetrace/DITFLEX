"""quick_train/modal_quick.py -- full dress rehearsal of the training chain.

Everything /run/ does, minus the 8 GPUs: real ditflex.train, real latents,
real checkpointing -- against the SCRATCH model repo (default
sparsetrace/quicktraindit), never the real per-objective repos.

Two legs in ONE dispatch, so the checkpoint chain is machine-verified:

    reset scratch repo (delete; fixed name, overwritten every dispatch)
    leg 1: fresh start, train --max-steps STEPS, save, push
    delete the local checkpoint dir      <- forces a genuine Hub pull
    leg 2: pull, resume, train --max-steps RESUME_STEPS, push
    verify: download state.json, assert step == STEPS + RESUME_STEPS

Green therefore proves: fresh-start -> save -> push -> pull -> exact
resume -> re-push, plus compile+DDP order, the objective stepping on real
data, and the deadline machinery -- the exact code path of /run/.

    MODAL_GPUS=2 modal run quick_train/modal_quick.py --steps 1000

Environment: MODAL_GPU (default B300), MODAL_GPUS (default 2), HF_TOKEN
(write scope: push + scratch-repo reset), TORCH_INDEX (default cu129).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "2"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

SCRATCH_REPO_DEFAULT = "sparsetrace/quicktraindit"
CKPT_DIR = "/tmp/ditflex_ckpt"          # must match ditflex.train.CKPT_DIR
TRAIN_SECONDS_CEILING = 7200            # per leg; --max-steps stops first

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

app = modal.App("ditflex-quick-train", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=4 * 3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def quick(
    steps: int = 1000,
    resume_steps: int = 200,
    objective: str = "flow",
    max_latent_files: int = 0,
    scratch_repo: str = SCRATCH_REPO_DEFAULT,
) -> int:
    import json
    import shutil
    import subprocess
    import sys

    import torch
    from huggingface_hub import HfApi, hf_hub_download

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

    # -- reset the scratch repo: fixed name, overwritten every dispatch,
    #    so leg 1 genuinely exercises the fresh-start path ---------------
    print(f"[quick] resetting scratch repo {scratch_repo}")
    HfApi().delete_repo(scratch_repo, repo_type="model", missing_ok=True)

    def leg(name: str, max_steps: int) -> int:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={n_gpu}", "--standalone",
            "-m", "ditflex.train",
            f"--train-seconds={TRAIN_SECONDS_CEILING}",
            f"--objective={objective}",
            f"--hub-repo={scratch_repo}",
            f"--max-steps={max_steps}",
        ]
        if max_latent_files > 0:
            cmd.append(f"--max-latent-files={max_latent_files}")
        print(f"\n[quick] {name}: {' '.join(cmd)}\n")
        return subprocess.run(cmd, cwd="/repo").returncode

    rc = leg("leg 1 (fresh start)", steps)
    if rc != 0:
        print("[quick] leg 1 FAILED")
        return rc

    # Force leg 2 to pull from the Hub, not find local files.
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print(f"[quick] deleted {CKPT_DIR} -- leg 2 must pull from the Hub")

    rc = leg("leg 2 (resume)", resume_steps)
    if rc != 0:
        print("[quick] leg 2 FAILED")
        return rc

    # -- machine-verify the chain ---------------------------------------
    state_path = hf_hub_download(
        scratch_repo, "state.json", repo_type="model", force_download=True
    )
    state = json.loads(Path(state_path).read_text())
    expected = steps + resume_steps
    print(f"\n[quick] scratch repo state: step={state['step']}  expected={expected}")
    for run in state.get("run_history", []):
        print(f"        run: steps {run['start_step']:,} -> {run['end_step']:,}  "
              f"({run['seconds']}s, world={run['world']}, {run['objective']})")

    if state["step"] != expected:
        print("[quick] CHAIN BROKEN -- resumed step count does not add up.")
        return 1
    if len(state.get("run_history", [])) != 2:
        print("[quick] CHAIN BROKEN -- expected exactly 2 runs in run_history.")
        return 1

    print(f"\n[quick] PASS -- fresh->save->push->pull->resume->push verified "
          f"at step {expected} on {scratch_repo}.")
    return 0


@app.local_entrypoint()
def main(
    steps: int = 1000,
    resume_steps: int = 200,
    objective: str = "flow",
    max_latent_files: int = 0,
    scratch_repo: str = SCRATCH_REPO_DEFAULT,
):
    """
    Args:
        steps:            leg-1 step count (fresh start)
        resume_steps:     leg-2 step count (after resume)
        objective:        ddpm | flow
        max_latent_files: 0 = all 32 shards (realistic); small N for speed
        scratch_repo:     overwritten every dispatch; NEVER a real run repo
    """
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")
    rc = quick.remote(
        steps=steps,
        resume_steps=resume_steps,
        objective=objective,
        max_latent_files=max_latent_files,
        scratch_repo=scratch_repo,
    )
    if rc != 0:
        raise SystemExit(rc)

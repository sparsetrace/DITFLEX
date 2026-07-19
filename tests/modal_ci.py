"""tests/modal_ci.py -- Modal runner for the ditflex gates and tests.
====

The source is NOT cloned from GitHub inside the container. `modal run`
uploads the local checkout (the Actions checkout, in CI), so there is no
GH_TOKEN, no clone step, and the code under test is exactly the code in
the working tree -- including uncommitted changes when run locally.

Gates, in order (each blocks the next from meaning anything):
  1. tests/verify_identity.py   Flex path vs fp64 math reference
  2. tests/verify_latents.py    first latent shard from the Hub
  3. pytest tests/
  4. (--smoke) tests/overfit_smoke.py, small model, both objectives

Usage
-----
    modal run tests/modal_ci.py
    modal run tests/modal_ci.py --compile-check
    modal run tests/modal_ci.py --smoke
    modal run tests/modal_ci.py --test-file test_attention_identity.py

Environment
-----------
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET   auth (or ~/.modal.toml locally)
    MODAL_GPU     GPU target. Default B300 -- the training hardware, so
                  the gates certify the kernels that will actually run.
    TORCH_INDEX   torch wheel index. Default cu129: B300 is SM103 /
                  compute_103, which requires CUDA >= 12.9.

Secrets: the `huggingface` Modal secret (HF_TOKEN) must exist -- the
latents gate pulls from the Hub. Create it once:
    modal secret create huggingface HF_TOKEN=hf_...
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# This file lives in tests/. The mount MUST be the repo root, or the
# container is missing src/, run/, and pyproject.toml.
REPO_ROOT = Path(__file__).parent.parent

GPU_TYPE = os.environ.get("MODAL_GPU", "B300")
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
        "pytest>=8.0",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-ci", image=image)


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_gates(
    test_files: list | None = None,
    compile_check: bool = False,
    smoke: bool = False,
) -> int:
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

    def run(cmd: list[str]) -> None:
        nonlocal rc
        print(f"\n[modal] running: {' '.join(cmd)}\n")
        rc = subprocess.run(cmd, cwd="/repo").returncode or rc

    gate_cmd = [sys.executable, "tests/verify_identity.py"]
    if compile_check:
        gate_cmd.append("--compile")
    run(gate_cmd)

    run([sys.executable, "tests/verify_latents.py"])

    pytest_cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    pytest_cmd += test_files or ["tests"]
    run(pytest_cmd)

    if smoke:
        for objective in ("ddpm", "flow"):
            run([sys.executable, "tests/overfit_smoke.py", "--small", "--objective", objective])

    return rc


@app.local_entrypoint()
def main(test_file: str = "", compile_check: bool = False, smoke: bool = False):
    """
    Args:
        test_file:     specific pytest file (e.g. test_attention_identity.py), empty = all
        compile_check: also verify the torch.compile'd Flex path
        smoke:         also run the small-model overfit smoke, both objectives
    """
    files = None
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        files = [test_file]

    rc = run_gates.remote(test_files=files, compile_check=compile_check, smoke=smoke)
    if rc != 0:
        raise SystemExit(rc)

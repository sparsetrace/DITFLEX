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

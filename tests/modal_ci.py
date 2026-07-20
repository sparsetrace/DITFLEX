"""Modal CI runner for ditflex correctness, stability, and GPU smoke gates.

The Actions checkout is mounted directly into the container, so the code under
CI is exactly the working tree being tested.  In addition to the existing
FlexAttention, latent, pytest, and overfit gates, this runner now verifies the
transactional training additions before any expensive smoke run:

* Ruff and bytecode compilation;
* committed-reference stability-controller tests;
* deterministic objective RNG tests;
* checkpoint validation/selection tests;
* importability of the Modal training supervisor.

Usage
-----
    modal run tests/modal_ci.py
    modal run tests/modal_ci.py --transactional-only
    modal run tests/modal_ci.py --compile-check
    modal run tests/modal_ci.py --smoke
    modal run tests/modal_ci.py --latents-all
    modal run tests/modal_ci.py --test-file test_stability.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_TYPE = os.environ.get("MODAL_GPU", "B300")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

TRANSACTIONAL_TESTS = [
    "tests/test_stability.py",
    "tests/test_objective_rng.py",
    "tests/test_checkpoint_selection.py",
    "tests/test_checkpoint_roundtrip.py",
]

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
        "ruff>=0.6",
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

app = modal.App("ditflex-ci", image=image)


@app.function(
    gpu=GPU_TYPE,
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_gates(
    test_files: list[str] | None = None,
    compile_check: bool = False,
    smoke: bool = False,
    latents_all: bool = False,
    transactional_only: bool = False,
) -> int:
    import subprocess
    import sys

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"[modal] GPU: {result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"],
        check=True,
    )

    first_failure = 0

    def run(cmd: list[str]) -> None:
        nonlocal first_failure
        print(f"\n[modal] running: {' '.join(cmd)}\n", flush=True)
        result = subprocess.run(cmd, cwd="/repo", check=False)
        if result.returncode != 0 and first_failure == 0:
            first_failure = result.returncode

    # Cheap source gates first.  These catch malformed supervisor/train changes
    # before spending time downloading latents or compiling GPU kernels.
    run([sys.executable, "-m", "ruff", "check", "src", "run", "tests"])
    run([sys.executable, "-m", "compileall", "-q", "src", "run", "tests"])
    run(
        [
            sys.executable,
            "-c",
            (
                "import run.modal_train as m; "
                "assert m.RETRY_EXIT_CODE == 75; "
                "assert m.RETRY_MARKER.name == 'ditflex_retry.json'; "
                "assert m.PROMOTION_MARKER.name == 'ditflex_promotion.json'"
            ),
        ]
    )

    # Always run the focused transactional tests, even when --test-file selects
    # another test.  They protect rollback/promotion behavior used in production.
    run([sys.executable, "-m", "pytest", "-v", "--tb=short", *TRANSACTIONAL_TESTS])

    if transactional_only:
        return first_failure

    gate_cmd = [sys.executable, "tests/verify_identity.py"]
    if compile_check:
        gate_cmd.append("--compile")
    run(gate_cmd)

    latents_cmd = [sys.executable, "tests/verify_latents.py"]
    if latents_all:
        latents_cmd.append("--all")
    run(latents_cmd)

    selected = test_files or ["tests"]
    run([sys.executable, "-m", "pytest", "-v", "--tb=short", *selected])

    if smoke:
        for objective in ("ddpm", "flow"):
            run(
                [
                    sys.executable,
                    "tests/overfit_smoke.py",
                    "--small",
                    "--objective",
                    objective,
                ]
            )
        run(
            [
                sys.executable,
                "tests/overfit_smoke.py",
                "--small",
                "--objective",
                "flow",
                "--qk-mode",
                "dmap",
            ]
        )
        run(
            [
                sys.executable,
                "tests/overfit_smoke.py",
                "--small",
                "--objective",
                "flow",
                "--qk-mode",
                "dmap",
                "--compile",
            ]
        )

    return first_failure


@app.local_entrypoint()
def main(
    test_file: str = "",
    compile_check: bool = False,
    smoke: bool = False,
    latents_all: bool = False,
    transactional_only: bool = False,
):
    """Run Modal CI gates.

    Args:
        test_file: Specific pytest file, or empty for the full suite.
        compile_check: Also verify the compiled FlexAttention path.
        smoke: Also run small-model overfit smoke tests.
        latents_all: Validate every latent shard instead of only shard zero.
        transactional_only: Run only static and transactional stability gates.
    """
    files = None
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        files = [test_file]

    rc = run_gates.remote(
        test_files=files,
        compile_check=compile_check,
        smoke=smoke,
        latents_all=latents_all,
        transactional_only=transactional_only,
    )
    if rc != 0:
        raise SystemExit(rc)

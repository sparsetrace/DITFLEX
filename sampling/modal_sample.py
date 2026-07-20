"""sampling/modal_sample.py -- on-demand sample grids from BOTH chains.

Pulls the latest checkpoint of each requested Hub repo, builds the
correct model variant from the checkpoint's own embedded config
(qk_mode decides builder), loads the EMA weights, renders the standard
fixed-seed 4x4 grid (same classes, same noise as the training-time
time-lapse), and returns PNG bytes. The workflow commits the PNGs into
/sampling/ in the GitHub repo.

    modal run sampling/modal_sample.py
    modal run sampling/modal_sample.py --repos sparsetrace/ditflex-L2-flow
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "B200")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31", "transformers>=4.44", "safetensors>=0.4.5",
        "huggingface_hub>=0.26", "numpy>=1.26", "pillow", "accelerate",
    )
    .add_local_dir(
        REPO_ROOT, remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-sampling", image=image)


@app.function(
    gpu=GPU_KIND,
    timeout=1800,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def sample_repo(repo: str, sample_steps: int = 50, cfg_scale: float = 4.0) -> tuple[int, bytes]:
    import io
    import json
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True)

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file

    from ditflex.config import Config
    from ditflex.model import build_model

    state = json.load(open(hf_hub_download(repo, "state.json")))
    step = int(state["step"])
    cfg_dict = state.get("config") or state.get("cfg")
    assert cfg_dict, "state.json lacks an embedded config"
    cfg = Config.from_dict(cfg_dict)
    print(f"[sample] {repo}: step {step:,}  qk_mode={cfg.model.qk_mode}")

    if cfg.model.qk_mode == "dmap":
        from ditflex.diffusion_model import build_dmap_model

        model = build_dmap_model(cfg.model)
    else:
        model = build_model(cfg.model)

    ema_sd = load_file(hf_hub_download(repo, "ema.safetensors"))
    missing, unexpected = model.load_state_dict(ema_sd, strict=False)
    n_params = sum(1 for _ in model.parameters())
    print(f"[sample] EMA loaded: {len(ema_sd)} tensors "
          f"(missing={len(missing)} buffers, unexpected={len(unexpected)})")
    assert len(unexpected) == 0, f"unexpected EMA keys: {unexpected[:5]}"
    assert len(ema_sd) >= n_params * 0.9, "EMA state dict suspiciously small"

    model = model.to(device="cuda", dtype=torch.float32).eval()

    # Fixed classes/seed: identical to the training-time time-lapse.
    try:
        from ditflex.sample import FIXED_CLASSES, FIXED_SEED
    except ImportError:
        FIXED_CLASSES = [207, 360, 387, 974, 88, 979, 417, 279,
                         972, 483, 21, 562, 933, 724, 985, 812]
        FIXED_SEED = 1234

    n = len(FIXED_CLASSES)
    g = torch.Generator(device="cpu").manual_seed(FIXED_SEED)
    x = torch.randn(n, cfg.model.in_channels, cfg.model.sample_size,
                    cfg.model.sample_size, generator=g).cuda()
    y = torch.tensor(FIXED_CLASSES, device="cuda")
    y_null = torch.full_like(y, cfg.model.num_classes)

    dt = 1.0 / sample_steps
    with torch.no_grad():
        for i in range(sample_steps):
            t = 1.0 - i * dt
            tt = torch.full((n,), t * 1000.0, device="cuda")
            v_c = model(hidden_states=x, timestep=tt, class_labels=y).sample[:, :4]
            v_u = model(hidden_states=x, timestep=tt, class_labels=y_null).sample[:, :4]
            v = v_u + cfg_scale * (v_c - v_u)
            x = x - dt * v

        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda().eval()
        imgs = vae.decode(x / 0.18215).sample

    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    side = int(n ** 0.5)
    px = imgs.shape[1]
    grid = np.zeros((side * px, side * px, 3), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, side)
        grid[r * px:(r + 1) * px, c * px:(c + 1) * px] = imgs[k]

    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    print(f"[sample] {repo}: grid rendered at step {step:,}")
    return step, buf.getvalue()


@app.local_entrypoint()
def main(
    repos: str = "sparsetrace/ditflex-L2-flow,sparsetrace/ditflex-L2-flow-dmap",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
):
    out_dir = Path(__file__).parent
    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        step, png = sample_repo.remote(repo, sample_steps=sample_steps, cfg_scale=cfg_scale)
        tag = repo.split("/")[-1].replace("ditflex-L2-", "")
        path = out_dir / f"{tag}_step_{step:07d}.png"
        path.write_bytes(png)
        print(f"[sample] wrote {path}")

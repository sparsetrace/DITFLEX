"""
sample_hhmap.py — Modal sampler for HHMAP checkpoints (or base+HHMAP).

    modal run HHMAP/sample_hhmap.py --step latest --weights ema
    modal run HHMAP/sample_hhmap.py --step 40000 --weights model --alpha 1.0

Loads SiT-XL/2 + HHMAP, restores the checkpoint (only W_D moved from AMAP), sets
the homotopy α (default = the checkpoint's saved α), and renders the fixed 4x4
grid so it is directly comparable to the AMAP/DMAP/HMAP grids. --alpha overrides
the operator point (e.g. sweep 0->1 from a single checkpoint to see the trade).
"""

from __future__ import annotations

import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "timm==1.0.19", "numpy<2", "huggingface_hub==0.26.2", "safetensors==0.4.5",
        "diffusers==0.31.0", "accelerate==1.1.1", "pillow", "torchdiffeq==0.2.5",
    )
    .env({"HF_HOME": "/cache/hf"})
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    .add_local_python_source("hhmap_attention", "hhmap_common")
)

app = modal.App("ditflex-hhmap-sample")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("HHMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=60 * 60,
              volumes={"/cache": ckpt_vol})
def run(repo: str, step: str, weights: str, sample_steps: int, cfg_scale: float,
        precision: str, alpha: float):
    import contextlib, torch
    import hhmap_common as C
    from hhmap_attention import set_alpha

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())

    os.chdir("/cache"); os.makedirs("/cache/samples", exist_ok=True)
    dev = "cuda"
    model, info = C.load_hhmap_checkpoint(dev, repo, step=step, weights=weights)
    if alpha >= 0.0:
        set_alpha(model, alpha)
        info = dict(info); info["alpha"] = alpha
    print(f"[hhmap] loaded {info}")

    tag = f"{info['step']}_{weights}_a{info.get('alpha', 1.0):.2f}"
    path = f"/cache/samples/hhmap_sample_{tag}.png"
    _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
    ckpt_vol.commit()
    from huggingface_hub import HfApi
    try:
        HfApi().upload_file(path_or_fileobj=path,
                            path_in_repo=f"samples/hhmap_sample_{tag}.png", repo_id=repo)
    except Exception as e:
        print(f"[hhmap] upload failed (non-fatal): {e!r}")
    return [(tag, png)]


@app.local_entrypoint()
def main(
    repo: str = "jcandane/HHMAP",
    step: str = "latest",
    weights: str = "ema",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    precision: str = "tf32",
    alpha: float = -1.0,   # <0 => use checkpoint's saved α; else override (0..1 sweep)
):
    grids = run.remote(repo, step, weights, sample_steps, cfg_scale, precision, alpha)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"; out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"hhmap_sample_{tag}.png"
        p.write_bytes(png)
        print(f"[hhmap] wrote {p}")

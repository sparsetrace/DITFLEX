"""
sample_fmap.py — render a 4x4 (16-image) grid from a FMAP checkpoint.

    modal run FMAP/sample_fmap.py                                  # latest ckpt
    modal run FMAP/sample_fmap.py --step 30000                     # a specific step
    modal run FMAP/sample_fmap.py --step base --weights model      # base + FMAP, un-finetuned
    modal run FMAP/sample_fmap.py --repo jcandane/FMAP --weights ema

Loads SiT-XL/2 + FMAP (flags read from the checkpoint's amap_config.json),
loads the requested weights (EMA by default, matching the ditflex sampler),
runs the official SiT transport ODE with CFG, decodes with SD-VAE (ft-ema),
and writes a 4x4 grid PNG that the workflow commits into AMAP/samples/.

Sampling is forward-only, so an L4 is plenty (default). The image matches
AMAP.py so Modal reuses the cached build.
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "timm==1.0.19",
        "numpy<2",
        "huggingface_hub==0.26.2",
        "safetensors==0.4.5",
        "diffusers==0.31.0",
        "accelerate==1.1.1",
        "pillow",
        "torchdiffeq==0.2.5",   # SiT transport ODE/SDE integrators
    )
    .env({"HF_HOME": "/cache/hf"})
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    .add_local_python_source("fmap_attention", "fmap_common")
)

app = modal.App("ditflex-fmap-sample")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("FMAP_SAMPLE_GPU", "L4")   # sampling is forward-only


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=60 * 60,
              volumes={"/cache": ckpt_vol})
def sample(repo: str, step: str, weights: str, sample_steps: int,
           cfg_scale: float) -> tuple[str, bytes]:
    import contextlib, torch
    import fmap_common as C

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir("/cache")
    os.makedirs("/cache/samples", exist_ok=True)
    dev = "cuda"

    model, info = C.load_fmap_checkpoint(dev, repo, step=step, weights=weights)
    ckpt_vol.commit()   # persist base ckpt + VAE cache
    tag = f"{repo.split('/')[-1]}_{info['step']}_{info['weights']}"
    print(f"[sample] {info}")

    amp = contextlib.nullcontext()
    path = f"/cache/samples/amap_{tag}.png"
    _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
    ckpt_vol.commit()
    print(f"[sample] grid rendered: {tag} ({len(png)//1024} KiB)")
    return tag, png


@app.local_entrypoint()
def main(
    repo: str = "jcandane/FMAP",
    step: str = "latest",      # int step | 'latest' | 'base'
    weights: str = "ema",      # 'ema' | 'model'
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
):
    tag, png = sample.remote(repo, step, weights, sample_steps, cfg_scale)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"fmap_{tag}.png"
    p.write_bytes(png)
    print(f"[sample] wrote {p}")

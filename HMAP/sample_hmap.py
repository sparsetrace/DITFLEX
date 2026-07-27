"""
sample_hmap.py — render a 4x4 (16-image) grid from an HMAP checkpoint.

    modal run HMAP/sample_hmap.py                                  # latest ckpt
    modal run HMAP/sample_hmap.py --step 30000                     # a specific step
    modal run HMAP/sample_hmap.py --step base --weights model      # base + HMAP, un-finetuned
    modal run HMAP/sample_hmap.py --repo jcandane/HMAP --weights ema

Loads SiT-XL/2 + HMAP (flags read from the checkpoint's hmap_config.json),
loads the requested weights (EMA by default, matching the ditflex sampler),
runs the official SiT transport ODE with CFG, decodes with SD-VAE (ft-ema),
and writes a 4x4 grid PNG that the workflow commits into HMAP/samples/.

Sampling is forward-only, so an L4 is plenty (default). The image matches
HMAP.py so Modal reuses the cached build.
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
    .add_local_python_source("hmap_attention", "hmap_common")
)

app = modal.App("ditflex-hmap-sample")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("HMAP_SAMPLE_GPU", "L4")   # sampling is forward-only


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=60 * 60,
              volumes={"/cache": ckpt_vol})
def sample(repo: str, step: str, weights: str, sample_steps: int,
           cfg_scale: float, alpha: float) -> tuple[str, bytes]:
    import contextlib, torch
    import hmap_common as C

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir("/cache")
    os.makedirs("/cache/samples", exist_ok=True)
    dev = "cuda"

    model, info = C.load_hmap_checkpoint(dev, repo, step=step, weights=weights)
    if alpha >= 0.0:                              # override the checkpoint's α
        from hmap_attention import set_alpha
        set_alpha(model, alpha); info["alpha"] = alpha
        print(f"[sample] α OVERRIDE -> {alpha:.3f}")
    ckpt_vol.commit()   # persist base ckpt + VAE cache
    tag = f"{repo.split('/')[-1]}_{info['step']}_{info['weights']}"
    print(f"[sample] {info}")

    amp = contextlib.nullcontext()
    path = f"/cache/samples/hmap_{tag}.png"
    _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
    ckpt_vol.commit()
    print(f"[sample] grid rendered: {tag} ({len(png)//1024} KiB)")
    return tag, png


@app.local_entrypoint()
def main(
    repo: str = "jcandane/HMAP",
    step: str = "latest",      # int step | 'latest' | 'base'
    weights: str = "ema",      # 'ema' | 'model'
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    alpha: float = -1.0,   # -1 = use checkpoint's α; else 0=AMAP .. 1=exact
):
    tag, png = sample.remote(repo, step, weights, sample_steps, cfg_scale, alpha)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"hmap_{tag}.png"
    p.write_bytes(png)
    print(f"[sample] wrote {p}")

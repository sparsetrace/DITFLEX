"""
sample0.py — step-0 sample grids for SiT-XL/2 with NO training.

    modal run AMAP/sample0.py --attn standard   # untouched original (baseline)
    modal run AMAP/sample0.py --attn amap       # AMAP (coupled) grafted, untrained

Builds SiT-XL/2, loads the 7M checkpoint, optionally applies the AMAP
operator, and renders one sample grid via amap_common.sample_grid — the SAME
seed / class panel / ODE integrator settings as AMAP.py's finetune previews,
so the three-way panel (standard step-0, AMAP step-0, AMAP step-N) is
pixel-comparable. Nothing is trained, no latents are downloaded, no
push-repo write access is required (the grid upload is best-effort).

Grids land in <push_repo>/samples/ as {attn}_step0000000.png and are also
returned to the local entrypoint, which writes them into AMAP/samples/ for
the workflow's commit step.
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
    .add_local_python_source("amap_attention", "amap_common")
)

app = modal.App("ditflex-sample0")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("AMAP_GPU", "B200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=60 * 60,
              volumes={"/cache": ckpt_vol})
def run(attn: str, precision: str, sample_steps: int, cfg_scale: float,
        push_repo: str, qk_rmsnorm: bool, learn_logit_scale: bool):
    import contextlib, torch
    import amap_common as C

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif precision == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif precision != "bf16":
        raise ValueError(f"precision must be tf32|highest|bf16, got {precision!r}")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())

    os.chdir("/cache")
    os.makedirs("/cache/samples", exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    if attn == "amap":
        from amap_attention import apply_amap, AMAPConfig
        # Keep the standard-forward reference BEFORE patching so the same
        # rel-shift diagnostic as AMAP.py's smoke stage is reported here too.
        x = torch.randn(2, 4, 32, 32, device=dev)
        t = torch.rand(2, device=dev)
        y = torch.randint(0, 1000, (2,), device=dev)
        with torch.no_grad(), amp:
            std_out = model(x, t, y)
        n_attn = apply_amap(model, AMAPConfig(qk_rmsnorm=qk_rmsnorm,
                                              learn_logit_scale=learn_logit_scale))
        with torch.no_grad(), amp:
            amap_out = model(x, t, y)
        shift = (amap_out - std_out).flatten().norm() / std_out.flatten().norm()
        print(f"[sample0] AMAP grafted: patched_attn={n_attn} "
              f"qk_rmsnorm={qk_rmsnorm} learn_logit_scale={learn_logit_scale} "
              f"rel-shift vs standard = {shift.item():.3f}")
        tag = "amap"
    elif attn == "standard":
        print("[sample0] untouched SiT-XL/2 — standard attention, no patch applied")
        tag = "standard"
    else:
        raise ValueError(f"attn must be standard|amap, got {attn!r}")

    print(f"[sample0] SiT-XL/2 params={n_params/1e6:.1f}M  precision={precision}  "
          f"sample_steps={sample_steps}  cfg={cfg_scale}")

    model.eval()
    name = f"{tag}_step0000000"
    path = f"/cache/samples/{name}.png"
    _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
    ckpt_vol.commit()
    print(f"[sample0] grid rendered: {path} ({len(png)//1024} KiB)")

    # Best-effort push; the grid is already safe on the Volume and in the
    # return value regardless.
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=path,
                            path_in_repo=f"samples/{name}.png", repo_id=push_repo)
        print(f"[sample0] uploaded -> {push_repo}/samples/{name}.png")
    except Exception as e:
        print(f"[sample0] HF upload failed (non-fatal, grid still returned): {e!r}")

    return [(name, png)]


@app.local_entrypoint()
def main(
    attn: str = "standard",     # standard | amap
    precision: str = "tf32",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    push_repo: str = "jcandane/AMAP",
    qk_rmsnorm: bool = False,       # amap only
    learn_logit_scale: bool = False,  # amap only
):
    grids = run.remote(attn, precision, sample_steps, cfg_scale, push_repo,
                       qk_rmsnorm, learn_logit_scale)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"{tag}.png"
        p.write_bytes(png)
        print(f"[sample0] wrote {p}")

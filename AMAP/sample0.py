"""
sample0.py — step-0 sample grids for SiT-XL/2 with NO training.

    modal run AMAP/sample0.py --attn standard   # untouched original (baseline)
    modal run AMAP/sample0.py --attn amap       # AMAP operator grafted, untrained
    modal run AMAP/sample0.py --attn kinetic    # weights PROJECTED to the PSD
                                                # cone, untouched standard forward

Three arms, same weights, same seed / class panel / ODE settings
(amap_common.sample_grid), pixel-comparable:

  standard : released checkpoint, released forward. Score = <q_i, k_j>
             = 1/2<m,m> - 1/2<n,n> + flux   (indefinite kinetic + circulation)

  amap     : operator swap (apply_amap), weights untouched.
             Score = 1/2<m,m> + flux        (NSD sector removed, flux KEPT)

  kinetic  : WEIGHT surgery, forward untouched. Per attention block the fused
             qkv rows are edited: W_Q, W_K <- (W_Q + W_K)/2 (biases likewise),
             so <q'_i, k'_j> = 1/2<m_i, m_j> exactly. Setting W_N = 0 makes W'
             symmetric, so the flux vanishes WITH the NSD sector: this is the
             pure PSD metric (DMAP kernel without the Doob tilt), NOT AMAP.
             Exact in SiT because no RoPE/QK-norm sits between qkv and the
             score (unlike nanochat).

  standard - kinetic  isolates (NSD sector + flux) jointly;
  amap - kinetic      isolates the flux alone;
  standard - amap     isolates the NSD sector alone.

Nothing is trained, no latents are downloaded, no push-repo write access is
required (grid upload is best-effort). Grids land in <push_repo>/samples/ as
{attn}_step0000000.png and are returned to the local entrypoint, which writes
them into AMAP/samples/ for the workflow's commit step.
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


def project_qkv_to_psd_cone(model) -> int:
    """Weight-space projection to the PSD cone: in every fused qkv Linear,
    W_Q, W_K <- (W_Q + W_K)/2 (and biases likewise), leaving W_V untouched.

    Under the UNTOUCHED standard forward this yields the score
    <q'_i, k'_j> = 1/2 <m_i, m_j>  (m = (q+k)/sqrt(2)) exactly: W_N = 0 kills
    both the NSD sector and (since W' is symmetric) the antisymmetric flux.
    Averaging whole matrices is per-head correct: head h occupies the same
    row block in W_Q and W_K, so blockwise and global averaging coincide.

    Returns the number of attention modules projected.
    """
    import torch
    n_proj = 0
    for name, mod in model.named_modules():
        qkv = getattr(mod, "qkv", None)
        if qkv is None or not hasattr(qkv, "weight"):
            continue
        W = qkv.weight
        if W.shape[0] != 3 * W.shape[1]:
            continue  # not a fused q,k,v projection
        D = W.shape[1]
        with torch.no_grad():
            avg_w = 0.5 * (W[:D] + W[D:2 * D])
            W[:D].copy_(avg_w)
            W[D:2 * D].copy_(avg_w)
            if qkv.bias is not None:
                b = qkv.bias
                avg_b = 0.5 * (b[:D] + b[D:2 * D])
                b[:D].copy_(avg_b)
                b[D:2 * D].copy_(avg_b)
        n_proj += 1
    return n_proj


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

    # Reference forward on the untouched model, for the rel-shift diagnostic
    # of the modified arms (same diagnostic as AMAP.py's smoke stage).
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    if attn == "standard":
        print("[sample0] untouched SiT-XL/2 — standard attention, no patch, "
              "no weight edits")
        tag = "standard"

    elif attn == "amap":
        from amap_attention import apply_amap, AMAPConfig
        n_attn = apply_amap(model, AMAPConfig(qk_rmsnorm=qk_rmsnorm,
                                              learn_logit_scale=learn_logit_scale))
        with torch.no_grad(), amp:
            out = model(x, t, y)
        shift = (out - std_out).flatten().norm() / std_out.flatten().norm()
        print(f"[sample0] AMAP operator grafted: patched_attn={n_attn} "
              f"qk_rmsnorm={qk_rmsnorm} learn_logit_scale={learn_logit_scale} "
              f"rel-shift vs standard = {shift.item():.3f}")
        tag = "amap"

    elif attn == "kinetic":
        n_attn = project_qkv_to_psd_cone(model)
        assert n_attn > 0, ("no fused qkv Linears found — SiT block structure "
                            "differs from the expected timm Attention layout")
        with torch.no_grad(), amp:
            out = model(x, t, y)
        shift = (out - std_out).flatten().norm() / std_out.flatten().norm()
        print(f"[sample0] PSD-cone weight projection: projected_qkv={n_attn} "
              f"(W_Q = W_K = (W_Q+W_K)/2; forward untouched) "
              f"rel-shift vs standard = {shift.item():.3f}")
        tag = "kinetic"

    else:
        raise ValueError(f"attn must be standard|amap|kinetic, got {attn!r}")

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
    attn: str = "standard",     # standard | amap | kinetic
    precision: str = "tf32",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    push_repo: str = "jcandane/AMAP",
    qk_rmsnorm: bool = False,       # amap arm only
    learn_logit_scale: bool = False,  # amap arm only
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

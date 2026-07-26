"""
AMAP.py — Modal ephemeral entrypoint for the AMAP experiment.

    modal run AMAP/AMAP.py --stage smoke
    modal run AMAP/AMAP.py --stage finetune --steps 2000 \
        --latents-repo <you>/dlatentzz --push-repo <you>/sit-xl2-amap

Stage `smoke`  : build SiT-XL/2, load the official 7M checkpoint, apply AMAP
                 (coupled), run a forward pass, and report the logit-scale shift
                 vs standard attention. No training, nothing pushed.
Stage `finetune`: the above + a short flow-matching finetune on YOUR latents
                 (same [N,4096] bf16 / 0.18215 layout as dlatentzz), then push
                 an AMAP checkpoint to `--push-repo`.

This is deliberately a self-contained ephemeral job. The long, transactional
finetune belongs in run/modal_train.py with qk_mode='amap' — see AMAP/README.md
for the two-line hook. Everything here reuses the existing qkv weights (no
surgery); the AMAP operator is a swapped forward, so the SiT state_dict loads
as-is and checkpoint keys are unchanged.
"""

from __future__ import annotations

import os

import modal

# --- image: SiT deps + AMAP; clone the official repo for models.py/download.py
# B200 is Blackwell (sm_100) -> needs torch built against CUDA 12.8 (cu128).
# torch/vision come from the pytorch cu128 index; everything else from PyPI.
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
    )
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    # AMAP modules travel with the job so we don't depend on the SiT checkout
    .add_local_python_source("amap_attention")
)

app = modal.App("ditflex-amap")

# persist the 2.7 GB SiT checkpoint across runs (find_model writes ./pretrained_models)
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

SIT_CKPT = "SiT-XL-2-256x256.pt"     # official 7M-step SiT-XL/2 (find_model)
HF_SECRET = modal.Secret.from_name("HF_TOKEN")   # provides HF_TOKEN env var
GPU = os.environ.get("AMAP_GPU", "B200")   # override per-run: AMAP_GPU=H200 modal run ...


def _build_sit_xl2():
    """SiT-XL/2 with the official architecture, weights loaded from the 7M ckpt."""
    import sys
    sys.path.insert(0, "/root/SiT")
    import torch
    from models import SiT_XL_2
    from download import find_model

    model = SiT_XL_2(input_size=32, in_channels=4)   # learn_sigma=True -> out 8
    state = find_model(SIT_CKPT)                      # downloads to pretrained_models/
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


def _load_latents(repo: str, max_shards: int | None = None):
    """Load dlatentzz-style latents: safetensors shards of [N,4096] bf16, scale
    0.18215 already applied. Returns a single [M,4,32,32] float tensor on CPU."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    api = HfApi()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.endswith(".safetensors")]
    files = sorted(files)[: max_shards or len(files)]
    chunks = []
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset")
        d = load_file(p)
        t = next(iter(d.values())) if len(d) == 1 else d[sorted(d)[0]]
        chunks.append(t.float())
    x = torch.cat(chunks, 0)                          # [M, 4096]
    std = x.std().item()
    assert 0.7 < std < 1.4, f"latent std {std:.3f}: expected ≈1.0 (0.18215 applied)"
    return x.reshape(-1, 4, 32, 32)


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str,
        latents_repo: str, qk_rmsnorm: bool, learn_logit_scale: bool,
        precision: str):
    import os, contextlib, torch, torch.nn.functional as F
    from amap_attention import apply_amap, AMAPConfig

    # Precision. Default tf32 matches the ditflex amap chain (fp32 activations,
    # TF32 tensor-core matmuls). PyTorch's own default is "highest" = plain fp32,
    # TF32 OFF — slower on B200 and different numerics from the real chain.
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

    os.chdir("/cache")   # find_model writes ./pretrained_models -> persisted volume
    dev = "cuda"
    torch.manual_seed(0)

    model = _build_sit_xl2().to(dev)
    ckpt_vol.commit()    # persist the downloaded checkpoint for next run
    n_params = sum(p.numel() for p in model.parameters())

    # ---- baseline (standard attention) logit stats for comparison ----
    B, N, C = 2, 256, 4
    x = torch.randn(B, C, 32, 32, device=dev)
    t = torch.rand(B, device=dev)
    y = torch.randint(0, 1000, (B,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    # ---- apply AMAP (coupled) ----
    cfg = AMAPConfig(qk_rmsnorm=qk_rmsnorm, learn_logit_scale=learn_logit_scale)
    n_attn = apply_amap(model, cfg)
    with torch.no_grad(), amp:
        amap_out = model(x, t, y)

    shift = (amap_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[amap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  precision={precision}")
    print(f"[amap] qk_rmsnorm={qk_rmsnorm} learn_logit_scale={learn_logit_scale}")
    print(f"[amap] output finite={torch.isfinite(amap_out).all().item()}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")
    print(f"[amap] (large shift is expected — this is what the finetune re-heals)")

    if stage == "smoke":
        print("[amap] smoke OK — nothing trained, nothing pushed.")
        return

    # ---- short flow-matching finetune (linear interpolant, velocity target) ----
    lat = _load_latents(latents_repo, max_shards=2).to(dev)   # a couple shards
    print(f"[amap] latents {tuple(lat.shape)} loaded from {latents_repo}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = 64
    model.train()
    for step in range(steps):
        idx = torch.randint(0, lat.shape[0], (bs,), device=dev)
        x1 = lat[idx]                                  # data latent
        x0 = torch.randn_like(x1)                      # noise
        tt = torch.rand(bs, device=dev)
        xt = (1 - tt[:, None, None, None]) * x0 + tt[:, None, None, None] * x1
        target = x1 - x0                               # linear-path velocity
        yy = torch.randint(0, 1000, (bs,), device=dev)
        with amp:
            pred = model(xt, tt, yy)[:, :4]                # velocity channels
            loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}")

    # ---- push AMAP checkpoint ----
    if push_repo:
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        import json, tempfile, os
        api = HfApi()
        api.create_repo(push_repo, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            save_file(model.state_dict(), os.path.join(d, "amap_sit_xl2.safetensors"))
            json.dump(
                {"qk_mode": "amap", "variant": "coupled", "base": SIT_CKPT,
                 "qk_rmsnorm": qk_rmsnorm, "learn_logit_scale": learn_logit_scale,
                 "precision": precision, "finetune_steps": steps, "lr": lr},
                open(os.path.join(d, "amap_config.json"), "w"), indent=2,
            )
            api.upload_folder(folder_path=d, repo_id=push_repo)
        print(f"[amap] pushed to {push_repo}")


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    steps: int = 500,
    lr: float = 1e-5,
    push_repo: str = "sparsetrace/AMAP",
    latents_repo: str = "sparsetrace/dlatentzz",
    qk_rmsnorm: bool = False,
    learn_logit_scale: bool = False,
    precision: str = "tf32",
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    run.remote(stage, steps, lr, push_repo, latents_repo,
               qk_rmsnorm, learn_logit_scale, precision)

"""
AMAP.py — Modal ephemeral entrypoint for the AMAP experiment.

    modal run AMAP/AMAP.py --stage smoke
    modal run AMAP/AMAP.py --stage sample
    modal run AMAP/AMAP.py --stage finetune --steps 2000 --sample-every 500

Stage `smoke`   : build SiT-XL/2, load the official 7M checkpoint, apply AMAP
                  (coupled), forward pass, report logit-scale shift vs standard
                  attention. Nothing trained or pushed.
Stage `sample`  : build + AMAP (or load a finetuned AMAP repo via --from-repo),
                  generate an image grid (official SiT transport ODE + SD-VAE),
                  save to the volume and push to --push-repo/samples/.
Stage `finetune`: official SiT flow-matching finetune on your latents
                  (transport.training_losses — linear path, velocity, t∈[0,1],
                  NOT the x1000 convention of the ditflex DiT-L/2 objective),
                  sampling every --sample-every steps, then push the checkpoint.

Self-contained ephemeral job. The long transactional finetune belongs in
run/modal_train.py with qk_mode='amap' (see AMAP/README.md). The AMAP operator
is a swapped forward reusing qkv, so the SiT state_dict loads as-is.
"""

from __future__ import annotations

import os

import modal

# --- image: SiT deps + AMAP; clone the official repo for models/transport/download
# B200 is Blackwell (sm_100) -> torch built against CUDA 12.8 (cu128).
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
        "diffusers==0.31.0",   # AutoencoderKL (SD-VAE) for decoding samples
        "accelerate==1.1.1",
    )
    .env({"HF_HOME": "/cache/hf"})   # persist VAE + hub cache on the volume
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    .add_local_python_source("amap_attention")
)

app = modal.App("ditflex-amap")

# persist the 2.7 GB SiT checkpoint + VAE across runs
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

SIT_CKPT = "SiT-XL-2-256x256.pt"     # official 7M-step SiT-XL/2 (find_model)
HF_SECRET = modal.Secret.from_name("HF_TOKEN")   # provides HF_TOKEN env var
GPU = os.environ.get("AMAP_GPU", "B200")   # override per-run: AMAP_GPU=H200 modal run ...

# Classic SiT demo classes for the sample grid (golden retriever, etc.)
SAMPLE_LABELS = [207, 360, 387, 974, 88, 979, 417, 279]


def _sit_path():
    import sys
    if "/root/SiT" not in sys.path:
        sys.path.insert(0, "/root/SiT")


def _build_sit_xl2():
    """SiT-XL/2 with the official architecture, weights from the 7M checkpoint."""
    _sit_path()
    from models import SiT_XL_2
    from download import find_model

    model = SiT_XL_2(input_size=32, in_channels=4)   # learn_sigma=True
    state = find_model(SIT_CKPT)                      # -> ./pretrained_models
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


def _load_latents(repo: str, max_shards: int | None = None):
    """dlatentzz-style latents: safetensors shards [N,4096] bf16, 0.18215 applied.
    Returns (latents [M,4,32,32] float, labels [M] long or None if absent)."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    api = HfApi()
    all_files = api.list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in all_files if f.endswith(".safetensors")
                    and "label" not in f.lower())
    shards = shards[: max_shards or len(shards)]
    chunks = []
    for f in shards:
        d = load_file(hf_hub_download(repo, f, repo_type="dataset"))
        chunks.append((next(iter(d.values())) if len(d) == 1 else d[sorted(d)[0]]).float())
    x = torch.cat(chunks, 0)
    std = x.std().item()
    assert 0.7 < std < 1.4, f"latent std {std:.3f}: expected ≈1.0 (0.18215 applied)"
    lat = x.reshape(-1, 4, 32, 32)

    # Optional labels: pair by index if the repo ships a labels file.
    labels = None
    label_files = [f for f in all_files if "label" in f.lower() and f.endswith(".safetensors")]
    if label_files:
        d = load_file(hf_hub_download(repo, sorted(label_files)[0], repo_type="dataset"))
        labels = (next(iter(d.values())) if len(d) == 1 else d[sorted(d)[0]]).long()[: lat.shape[0]]
    return lat, labels


def _sample_grid(model, dev, out_path, num_steps, cfg_scale, amp):
    """Official SiT ODE sample + SD-VAE decode -> saved grid PNG. Returns path."""
    _sit_path()
    import torch
    from transport import create_transport, Sampler
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image

    transport = create_transport("Linear", "velocity")
    sample_fn = Sampler(transport).sample_ode(num_steps=num_steps)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev)

    n = len(SAMPLE_LABELS)
    z = torch.randn(n, 4, 32, 32, device=dev)
    y = torch.tensor(SAMPLE_LABELS, device=dev)
    # classifier-free guidance: duplicate with null class
    z = torch.cat([z, z], 0)
    y = torch.cat([y, torch.tensor([1000] * n, device=dev)], 0)

    model.eval()
    with torch.no_grad(), amp:
        samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale)[-1]
        samples, _ = samples.chunk(2, dim=0)
        imgs = vae.decode(samples / 0.18215).sample
    model.train()
    save_image(imgs, out_path, nrow=4, normalize=True, value_range=(-1, 1))
    return out_path


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=2 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, latents_repo: str,
        qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        sample_every: int, sample_steps: int, cfg_scale: float, from_repo: str):
    import contextlib, json, tempfile, torch
    from amap_attention import apply_amap, AMAPConfig

    # precision (default tf32 matches the ditflex chain; torch default is fp32/no-tf32)
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

    model = _build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    # baseline logit stats
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    cfg = AMAPConfig(qk_rmsnorm=qk_rmsnorm, learn_logit_scale=learn_logit_scale)
    n_attn = apply_amap(model, cfg)

    # optionally load a previously-finetuned AMAP checkpoint (for `sample`)
    if from_repo:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(from_repo, "amap_sit_xl2.safetensors"))
        model.load_state_dict(sd, strict=False)
        print(f"[amap] loaded finetuned weights from {from_repo}")

    with torch.no_grad(), amp:
        amap_out = model(x, t, y)
    shift = (amap_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[amap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  precision={precision}")
    print(f"[amap] qk_rmsnorm={qk_rmsnorm} learn_logit_scale={learn_logit_scale}")
    print(f"[amap] output finite={torch.isfinite(amap_out).all().item()}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    def do_sample(tag):
        path = f"/cache/samples/amap_{tag}.png"
        _sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        if push_repo:
            from huggingface_hub import HfApi
            api = HfApi(); api.create_repo(push_repo, exist_ok=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=f"samples/amap_{tag}.png",
                            repo_id=push_repo)
        print(f"[amap] sample grid -> {path}" + (f" and {push_repo}/samples/" if push_repo else ""))
        return path

    if stage == "smoke":
        print("[amap] smoke OK — nothing trained, nothing pushed.")
        return
    if stage == "sample":
        do_sample("current")
        return

    # ---- finetune: official SiT flow-matching (transport.training_losses) ----
    _sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    lat, labels = _load_latents(latents_repo, max_shards=2)
    lat = lat.to(dev)
    if labels is None:
        print("[amap][warn] no labels file in latents repo -> UNCONDITIONAL finetune "
              "(null class). Real class-conditional finetune uses ditflex LatentStore "
              "in the supervisor.")
    else:
        labels = labels.to(dev)
        print(f"[amap] paired {labels.shape[0]:,} labels with latents")
    print(f"[amap] latents {tuple(lat.shape)} from {latents_repo}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = 64
    model.train()
    for step in range(steps):
        idx = torch.randint(0, lat.shape[0], (bs,), device=dev)
        x1 = lat[idx]
        yy = labels[idx] if labels is not None else torch.full((bs,), 1000, device=dev)
        with amp:
            loss = transport.training_losses(model, x1, dict(y=yy))["loss"].mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}")
        if sample_every > 0 and step > 0 and step % sample_every == 0:
            do_sample(f"step{step}")

    do_sample("final")

    # ---- push AMAP checkpoint ----
    if push_repo:
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        api = HfApi(); api.create_repo(push_repo, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            save_file(model.state_dict(), os.path.join(d, "amap_sit_xl2.safetensors"))
            json.dump(
                {"qk_mode": "amap", "variant": "coupled", "base": SIT_CKPT,
                 "objective": "SiT transport Linear/velocity (t in [0,1])",
                 "conditional": labels is not None,
                 "qk_rmsnorm": qk_rmsnorm, "learn_logit_scale": learn_logit_scale,
                 "precision": precision, "finetune_steps": steps, "lr": lr},
                open(os.path.join(d, "amap_config.json"), "w"), indent=2,
            )
            api.upload_folder(folder_path=d, repo_id=push_repo)
        print(f"[amap] pushed checkpoint to {push_repo}")


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
    sample_every: int = 0,
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    from_repo: str = "",
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    run.remote(stage, steps, lr, push_repo, latents_repo, qk_rmsnorm,
               learn_logit_scale, precision, sample_every, sample_steps,
               cfg_scale, from_repo)

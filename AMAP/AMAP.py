"""
AMAP.py — Modal ephemeral finetune of AMAP-on-SiT-XL/2.

    modal run AMAP/AMAP.py --stage smoke
    modal run AMAP/AMAP.py --stage finetune --steps 50000 \
        --save-every 10000 --sample-every 10000

Stage `smoke`   : build SiT-XL/2, load the 7M checkpoint, apply AMAP (coupled),
                  forward pass, report logit-scale shift vs standard attention.
Stage `finetune`: official SiT flow-matching finetune (transport.training_losses)
                  on your latents. `--steps N` trains N MORE steps: fresh runs go
                  0->N; if a checkpoint exists in --push-repo (resume=auto) it
                  continues start->start+N. EMA + periodic checkpoints/samples.

The AMAP operator is a swapped forward reusing qkv (no surgery), so the SiT
state_dict loads as-is. Shared build/EMA/sample helpers: amap_common.py.
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

app = modal.App("ditflex-amap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("AMAP_GPU", "B200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, latents_repo: str,
        qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str):
    import contextlib, json, tempfile, torch
    import amap_common as C
    from amap_attention import apply_amap, AMAPConfig

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

    # standard-attention forward (before AMAP) for the logit-shift diagnostic
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    # ---- resume detection (finetune only): if <push_repo> already has a
    # checkpoint, continue from it; AMAP flags come from the checkpoint so the
    # weights load cleanly. resume: "auto" (default) | "never" | "must". ----
    start_step, resume_sds = 0, None
    eff_qk_rmsnorm, eff_lls = qk_rmsnorm, learn_logit_scale
    if stage == "finetune" and resume != "never":
        s = C.latest_checkpoint_step(push_repo)
        if s is not None:
            ckcfg, model_sd, ema_sd = C.fetch_checkpoint(push_repo, s)
            start_step = s
            resume_sds = (model_sd, ema_sd)
            eff_qk_rmsnorm = bool(ckcfg.get("qk_rmsnorm", False))
            eff_lls = bool(ckcfg.get("learn_logit_scale", False))
            if (eff_qk_rmsnorm, eff_lls) != (qk_rmsnorm, learn_logit_scale):
                print(f"[amap] resume: using checkpoint's AMAP flags "
                      f"(qk_rmsnorm={eff_qk_rmsnorm}, learn_logit_scale={eff_lls}), "
                      f"overriding CLI")
            print(f"[amap] RESUMING from {push_repo}/checkpoints/step_{s:07d}")
        elif resume == "must":
            raise SystemExit(f"[amap] resume=must but no checkpoint in {push_repo}")
        else:
            print(f"[amap] no checkpoint in {push_repo} — fresh start from base SiT")

    amap_cfg = AMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = apply_amap(model, amap_cfg)
    with torch.no_grad(), amp:
        amap_out = model(x, t, y)
    shift = (amap_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[amap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  precision={precision}")
    print(f"[amap] qk_rmsnorm={eff_qk_rmsnorm} learn_logit_scale={eff_lls}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    if stage == "smoke":
        print("[amap] smoke OK — nothing trained, nothing pushed.")
        return []

    # ---- finetune: official SiT flow-matching (transport.training_losses) ----
    # PREFLIGHT: verify we can actually write checkpoints BEFORE spending compute.
    # (Reading dlatentzz needs only read access; creating <push_repo> needs write
    # rights in that namespace — catch a 403 in seconds, not after 10k steps.)
    from huggingface_hub import HfApi
    try:
        HfApi().create_repo(push_repo, exist_ok=True)
    except Exception as e:
        raise SystemExit(
            f"[amap] cannot write to '{push_repo}': {e}\n"
            f"       The Modal HF_TOKEN needs create/write rights in that namespace. "
            f"Use your own namespace (e.g. --push-repo jcandane/AMAP) or a token with "
            f"org-write access."
        )
    print(f"[amap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[amap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    if resume_sds is not None:
        model_sd, ema_sd = resume_sds
        _, unexpected = model.load_state_dict(model_sd, strict=False)
        assert not unexpected, f"unexpected keys on resume: {unexpected[:5]}"
    ema = C.EMA(model, decay=0.9999)
    if resume_sds is not None:
        ema.shadow = {k: v.to(dev).float() for k, v in resume_sds[1].items()}
        print(f"[amap] resumed model+EMA @ step {start_step:,} (optimizer reinitialized)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = 64
    grids: list[tuple[str, bytes]] = []

    def save_ckpt(step):
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        api = HfApi(); api.create_repo(push_repo, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            save_file(model.state_dict(), f"{d}/model.safetensors")
            save_file({k: v.contiguous() for k, v in ema.state_dict().items()},
                      f"{d}/ema.safetensors")
            json.dump(
                {"step": step, "qk_mode": "amap", "variant": "coupled",
                 "base": C.SIT_CKPT, "conditional": True,
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/amap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[amap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    def preview(tag):
        path = f"/cache/samples/amap_{tag}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        grids.append((tag, png))
        # push immediately (crash-resilient) — do NOT rely on the end-of-run return
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/amap_{tag}.png", repo_id=push_repo)
            print(f"[amap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[amap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps to train
    if steps <= 0:
        print(f"[amap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[amap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    model.train()
    for step in range(start_step + 1, end_step + 1):
        x1, yy = store.batch(step, 0, bs, base_seed=0)
        with amp:
            loss = transport.training_losses(model, x1, dict(y=yy))["loss"].mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); ema.update(model)

        if step % max(1, steps // 20) == 0 or step == start_step + 1:
            print(f"  step {step:6d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}")
        if save_every > 0 and step % save_every == 0:
            save_ckpt(step)
        if sample_every > 0 and step % sample_every == 0:
            preview(f"step{step:07d}")

    # final checkpoint + preview at end_step (skip if it exactly coincides with a
    # periodic save/sample already done, or if no new steps were trained)
    if end_step > start_step and (save_every <= 0 or end_step % save_every != 0):
        save_ckpt(end_step)
    if sample_every <= 0 or end_step % sample_every != 0:
        preview(f"step{end_step:07d}")
    return grids


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    steps: int = 500,
    lr: float = 1e-5,
    push_repo: str = "jcandane/AMAP",
    latents_repo: str = "sparsetrace/dlatentzz",
    qk_rmsnorm: bool = False,
    learn_logit_scale: bool = False,
    precision: str = "tf32",
    sample_every: int = 0,
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    save_every: int = 10000,
    max_shards: int = 0,   # 0 = all shards
    resume: str = "auto",  # auto | never | must
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    grids = run.remote(stage, steps, lr, push_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, precision, sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"amap_{tag}.png"
        p.write_bytes(png)
        print(f"[amap] wrote {p}")

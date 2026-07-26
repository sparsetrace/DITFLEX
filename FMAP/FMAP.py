"""
FMAP.py — Modal ephemeral finetune of FMAP-on-SiT-XL/2.

FMAP = SiT -> DMAP DIRECT. Identical operator to DMAP (the Mahalanobis
distance kernel, symmetric PSD core, no flux, R≡0):
    logit_ij = −½‖μ_i − μ_j‖²,  μ = R·W_M,  W_M = (W_Q+W_K)/√2
The ONLY difference from DMAP is the initialisation: FMAP folds BASE SiT
directly (no AMAP detour). This is the SiT2DMAP arm.

    modal run FMAP/FMAP.py --stage smoke
    modal run FMAP/FMAP.py --stage finetune --steps 21000 --save-every 10000

Checkpoint resolution for finetune (resume=auto):
    1. FMAP's own latest in --push-repo (jcandane/FMAP)  -> resume (folded)
    2. else base SiT-XL/2                                 -> FOLD, step 0  (no AMAP)

At step 0 we FOLD W_Q,W_K -> W_M = (W_Q+W_K)/√2 and drop W_N: the fused qkv
[3d,d] becomes wmv [2d,d], removing ~⅓ of the attention-projection params +
optimizer state. Exact (q,k never appear apart). `--steps N` trains N more
steps. Folded checkpoints are FMAP-only. Helpers: fmap_common.py; operator +
fold: fmap_attention.py.
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
        "torchdiffeq==0.2.5",
    )
    .env({"HF_HOME": "/cache/hf"})
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    .add_local_python_source("fmap_attention", "fmap_common")
)

app = modal.App("ditflex-fmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("FMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str,
        latents_repo: str, qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str):
    import contextlib, json, tempfile, torch
    import fmap_common as C
    from fmap_attention import install_folded_fmap, FMAPConfig

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
    print("[fmap] build: SiT->DMAP direct (fold base SiT) + UNCONDITIONAL step-0 snapshot")

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)

    # ---- checkpoint resolution: FMAP(folded) own -> base SiT fold (NO AMAP) ----
    # FMAP is the SiT->DMAP-direct arm: it folds base SiT's W_Q,W_K -> W_M with no
    # AMAP detour. (Operator is identical to DMAP; only the init differs.)
    start_step, warm, folded_sd, ema_sd = 0, None, None, None
    eff_qk_rmsnorm, eff_lls = qk_rmsnorm, learn_logit_scale
    if stage == "finetune" and resume != "never":
        s = C.latest_checkpoint_step(push_repo)
        if s is not None:
            ckcfg, folded_sd, ema_sd = C.fetch_checkpoint(push_repo, s)   # folded format
            start_step = s
            eff_qk_rmsnorm = bool(ckcfg.get("qk_rmsnorm", False))
            eff_lls = bool(ckcfg.get("learn_logit_scale", False))
            print(f"[fmap] RESUMING FMAP (folded) from {push_repo}/checkpoints/step_{s:07d}")
        elif resume == "must":
            raise SystemExit(f"[fmap] resume=must but no FMAP checkpoint in {push_repo}")
        else:
            print(f"[fmap] no FMAP checkpoint — folding BASE SiT directly (SiT->DMAP), step 0")

    with torch.no_grad(), amp:
        std_out = model(x, t, y)          # standard attention on the pre-fold weights

    # FOLD W_Q,W_K -> W_M (drop W_N). Exact for FMAP; ~1/3 fewer attn-proj params.
    dcfg = FMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = install_folded_fmap(model, dcfg, fold_weights=True)
    n_folded = sum(p.numel() for p in model.parameters())
    if folded_sd is not None:             # FMAP resume: load trained folded weights
        is_coupled = any(k.endswith(".qkv.weight") for k in folded_sd)
        if is_coupled:
            # folded-only program: a coupled (qkv) checkpoint is unambiguous —
            # just fold it and continue, no question asked.
            from fmap_attention import fold_state_dict
            folded_sd = fold_state_dict(folded_sd)
            if ema_sd is not None:
                ema_sd = fold_state_dict(ema_sd)
            print("[fmap] resume checkpoint was COUPLED (qkv) — auto-folded to wmv")
        _, unexp = model.load_state_dict(folded_sd, strict=False)
        if unexp:
            raise SystemExit(f"[fmap] unexpected keys resuming checkpoint: {unexp[:5]}")

    with torch.no_grad(), amp:
        fmap_out = model(x, t, y)
    shift = (fmap_out - std_out).flatten().norm() / std_out.flatten().norm()
    saved = 100.0 * (n_params - n_folded) / n_params
    print(f"[fmap] SiT-XL/2 {n_params/1e6:.1f}M -> {n_folded/1e6:.1f}M folded "
          f"(−{saved:.1f}%, W_N dropped)  attn={n_attn}  precision={precision}")
    print(f"[fmap] qk_rmsnorm={eff_qk_rmsnorm} learn_logit_scale={eff_lls}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    if stage == "smoke":
        print("[fmap] smoke OK — nothing trained, nothing pushed.")
        return []

    # PREFLIGHT: verify write access before spending compute.
    from huggingface_hub import HfApi
    try:
        HfApi().create_repo(push_repo, exist_ok=True)
    except Exception as e:
        raise SystemExit(
            f"[fmap] cannot write to '{push_repo}': {e}\n"
            f"       The Modal HF_TOKEN needs create/write rights in that namespace.")
    print(f"[fmap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[fmap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    # model weights are already loaded + folded above. EMA:
    ema = C.EMA(model, decay=0.9999)
    if ema_sd is not None:                 # FMAP folded resume carries a folded EMA
        shadow = ema.state_dict()
        for kk, vv in ema_sd.items():
            if kk in shadow:
                shadow[kk] = vv.to(dev).float()
        print(f"[fmap] loaded folded EMA @ step {start_step:,} (optimizer reinitialized)")
    else:
        print(f"[fmap] EMA snapshot from folded base SiT (SiT->DMAP direct); optimizer fresh")
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
                {"step": step, "qk_mode": "fmap", "variant": "folded",
                 "operator": "mahalanobis: -1/2||mu_i-mu_j||^2, mu=R·W_M, W_M=(W_Q+W_K)/sqrt2, R=0",
                 "folded": True, "attn_proj": "wmv [2d,d] (W_N dropped)",
                 "base": C.SIT_CKPT, "conditional": True, "init": "base-SiT (SiT->DMAP direct)",
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/fmap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[fmap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    def preview(tag):
        path = f"/cache/samples/fmap_{tag}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        grids.append((tag, png))
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/fmap_{tag}.png", repo_id=push_repo)
            print(f"[fmap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[fmap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps
    if steps <= 0:
        print(f"[fmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[fmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    # step-0 "before" snapshot — ALWAYS (unconditional), never fatal
    try:
        preview(f"step{start_step:07d}_start")
    except Exception as e:
        print(f"[fmap] step-0 snapshot failed (non-fatal): {e!r}")
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

    if end_step > start_step and (save_every <= 0 or end_step % save_every != 0):
        save_ckpt(end_step)
    if sample_every <= 0 or end_step % sample_every != 0:
        preview(f"step{end_step:07d}")
    return grids


@app.local_entrypoint()
def main(
    stage: str = "finetune",
    steps: int = 500,
    lr: float = 1e-5,
    push_repo: str = "jcandane/FMAP",
    latents_repo: str = "sparsetrace/dlatentzz",
    qk_rmsnorm: bool = False,
    learn_logit_scale: bool = False,
    precision: str = "tf32",
    sample_every: int = 0,
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    save_every: int = 10000,
    max_shards: int = 0,
    resume: str = "auto",
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
        p = out_dir / f"fmap_{tag}.png"
        p.write_bytes(png)
        print(f"[fmap] wrote {p}")

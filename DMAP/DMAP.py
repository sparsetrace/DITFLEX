"""
DMAP.py — Modal ephemeral finetune of DMAP-on-SiT-XL/2.

DMAP = the Mahalanobis distance-kernel half of AMAP (symmetric PSD core, no
flux, R≡0): logit_ij = −½‖μ_i − μ_j‖², μ = (q+k)/√2.

    modal run DMAP/DMAP.py --stage smoke
    modal run DMAP/DMAP.py --stage finetune --steps 21000 --save-every 10000

Checkpoint resolution for finetune (resume=auto):
    1. DMAP's own latest in --push-repo (jcandane/DMAP)  -> resume (folded)
    2. else AMAP's latest in --amap-repo (jcandane/AMAP)  -> WARM-START + FOLD (step 0)
    3. else base SiT-XL/2                                 -> fresh + FOLD

At step 0 we FOLD W_Q,W_K -> W_M = (W_Q+W_K)/√2 and drop W_N: the fused qkv
[3d,d] becomes wmv [2d,d], removing ~⅓ of the attention-projection params +
optimizer state. Exact for DMAP (q,k never appear apart). `--sample-at-start`
captures a grid on the folded weights before step 1 (the "before"). `--steps N`
trains N more steps. Folded checkpoints are DMAP-only. Helpers: dmap_common.py;
operator + fold: dmap_attention.py.
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
    .add_local_python_source("dmap_attention", "dmap_common")
)

app = modal.App("ditflex-dmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("DMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, amap_repo: str,
        latents_repo: str, qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str, sample_at_start: bool, fold_on_resume: bool):
    import contextlib, json, tempfile, torch
    import dmap_common as C
    from dmap_attention import install_folded_dmap, DMAPConfig

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

    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)

    # ---- checkpoint resolution: DMAP(folded) own -> AMAP warm-start -> base ----
    start_step, warm, folded_sd, ema_sd = 0, None, None, None
    eff_qk_rmsnorm, eff_lls = qk_rmsnorm, learn_logit_scale
    if stage == "finetune" and resume != "never":
        s = C.latest_checkpoint_step(push_repo)
        if s is not None:
            ckcfg, folded_sd, ema_sd = C.fetch_checkpoint(push_repo, s)   # folded format
            start_step = s
            eff_qk_rmsnorm = bool(ckcfg.get("qk_rmsnorm", False))
            eff_lls = bool(ckcfg.get("learn_logit_scale", False))
            print(f"[dmap] RESUMING DMAP (folded) from {push_repo}/checkpoints/step_{s:07d}")
        else:
            a = C.latest_checkpoint_step(amap_repo)
            if a is not None:
                _, am_sd, _ = C.fetch_checkpoint(amap_repo, a)
                _, unexp = model.load_state_dict(am_sd, strict=False)   # AMAP qkv -> model (pre-fold)
                warm = f"{amap_repo}/checkpoints/step_{a:07d}"
                print(f"[dmap] no DMAP checkpoint — WARM-STARTING from AMAP {warm}, "
                      f"folding W_Q,W_K->W_M (step reset to 0)")
                if unexp:
                    print(f"[dmap] (ignoring {len(unexp)} AMAP-only key(s), e.g. {unexp[:2]})")
            elif resume == "must":
                raise SystemExit(f"[dmap] resume=must but no DMAP ({push_repo}) "
                                 f"or AMAP ({amap_repo}) checkpoint found")
            else:
                print(f"[dmap] no DMAP or AMAP checkpoint — fresh start from base SiT, folding")

    with torch.no_grad(), amp:
        std_out = model(x, t, y)          # standard attention on the pre-fold weights

    # FOLD W_Q,W_K -> W_M (drop W_N). Exact for DMAP; ~1/3 fewer attn-proj params.
    dcfg = DMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = install_folded_dmap(model, dcfg, fold_weights=True)
    n_folded = sum(p.numel() for p in model.parameters())
    if folded_sd is not None:             # DMAP resume: load trained folded weights
        is_coupled = any(k.endswith(".qkv.weight") for k in folded_sd)
        if is_coupled:
            if fold_on_resume:
                from dmap_attention import fold_state_dict
                folded_sd = fold_state_dict(folded_sd)
                if ema_sd is not None:
                    ema_sd = fold_state_dict(ema_sd)
                print("[dmap] resume checkpoint is COUPLED (qkv) — folding it to wmv "
                      "(--fold-on-resume)")
            else:
                raise SystemExit(
                    f"[dmap] {push_repo}/checkpoints/step_{start_step:07d} is a COUPLED "
                    f"(qkv) checkpoint; this is the FOLDED program and can't resume it as-is.\n"
                    f"       Likely a stale/redirected repo. Fix one of:\n"
                    f"         • delete/rename that repo (or a fresh --push-repo) so DMAP "
                    f"warm-starts from AMAP,\n"
                    f"         • or pass --fold-on-resume to convert it to folded and continue.")
        _, unexp = model.load_state_dict(folded_sd, strict=False)
        if unexp:
            raise SystemExit(f"[dmap] unexpected keys resuming folded checkpoint: {unexp[:5]}")

    with torch.no_grad(), amp:
        dmap_out = model(x, t, y)
    shift = (dmap_out - std_out).flatten().norm() / std_out.flatten().norm()
    saved = 100.0 * (n_params - n_folded) / n_params
    print(f"[dmap] SiT-XL/2 {n_params/1e6:.1f}M -> {n_folded/1e6:.1f}M folded "
          f"(−{saved:.1f}%, W_N dropped)  attn={n_attn}  precision={precision}")
    print(f"[dmap] qk_rmsnorm={eff_qk_rmsnorm} learn_logit_scale={eff_lls}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    if stage == "smoke":
        print("[dmap] smoke OK — nothing trained, nothing pushed.")
        return []

    # PREFLIGHT: verify write access before spending compute.
    from huggingface_hub import HfApi
    try:
        HfApi().create_repo(push_repo, exist_ok=True)
    except Exception as e:
        raise SystemExit(
            f"[dmap] cannot write to '{push_repo}': {e}\n"
            f"       The Modal HF_TOKEN needs create/write rights in that namespace.")
    print(f"[dmap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[dmap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    # model weights are already loaded + folded above. EMA:
    ema = C.EMA(model, decay=0.9999)
    if ema_sd is not None:                 # DMAP folded resume carries a folded EMA
        shadow = ema.state_dict()
        for kk, vv in ema_sd.items():
            if kk in shadow:
                shadow[kk] = vv.to(dev).float()
        print(f"[dmap] loaded folded EMA @ step {start_step:,} (optimizer reinitialized)")
    elif warm:
        print(f"[dmap] EMA snapshot from AMAP warm-start (folded); optimizer fresh")
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
                {"step": step, "qk_mode": "dmap", "variant": "folded",
                 "operator": "mahalanobis: -1/2||mu_i-mu_j||^2, mu=R·W_M, W_M=(W_Q+W_K)/sqrt2, R=0",
                 "folded": True, "attn_proj": "wmv [2d,d] (W_N dropped)",
                 "base": C.SIT_CKPT, "conditional": True, "warm_start_from": warm,
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/dmap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[dmap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    def preview(tag):
        path = f"/cache/samples/dmap_{tag}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        grids.append((tag, png))
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/dmap_{tag}.png", repo_id=push_repo)
            print(f"[dmap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[dmap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps
    if steps <= 0:
        print(f"[dmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[dmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    if sample_at_start:
        preview(f"step{start_step:07d}_start")   # before any optimizer step (the "before")
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
    push_repo: str = "jcandane/DMAP",
    amap_repo: str = "jcandane/AMAP",   # warm-start source when no DMAP checkpoint
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
    sample_at_start: bool = True,    # snapshot step 0 before training (the "before")
    fold_on_resume: bool = False,    # convert a coupled (qkv) checkpoint to folded and continue
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    grids = run.remote(stage, steps, lr, push_repo, amap_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, precision, sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume, sample_at_start,
                       fold_on_resume)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"dmap_{tag}.png"
        p.write_bytes(png)
        print(f"[dmap] wrote {p}")

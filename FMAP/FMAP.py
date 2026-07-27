"""
FMAP.py — Modal ephemeral finetune of the ANNEALED AMAP->DMAP homotopy on SiT-XL/2.

FMAP starts as AMAP (Λ=1, symmetric Gram + antisymmetric flux) and adiabatically
anneals Λ: 1 -> 0 over the window [anneal_start, anneal_end], ending at pure DMAP
(Λ=0, the Mahalanobis distance kernel, no flux). The operator interpolates the
WHOLE thing so both endpoints are exact:

    L(Λ) = (1−½Λ)⟨m_i,m_j⟩ − (1−Λ)·½(‖m_i‖²+‖m_j‖²) + ½Λ(⟨q_i,k_j⟩−⟨k_i,q_j⟩)

    modal run FMAP/FMAP.py --stage smoke
    modal run FMAP/FMAP.py --stage finetune --steps 40000 \
        --anneal-start 0 --anneal-end 40000 --save-every 10000 --sample-every 10000

Coupled (reuses qkv, no fold — the flux needs q,k apart while Λ>0). At Λ=1 it is
exactly AMAP (sharp); at Λ=0 exactly DMAP. This arm measures the flux's value
smoothly: watch the Λ-vs-loss curve as it anneals. `--steps N` trains N MORE
steps; on resume the anneal window is read from the checkpoint so the schedule
is stable. Shared build/EMA/sample helpers: fmap_common.py; operator: fmap_attention.py.
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

app = modal.App("ditflex-fmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("FMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, latents_repo: str,
        qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str, anneal_start: int, anneal_end: int):
    import contextlib, json, tempfile, torch
    import fmap_common as C
    from fmap_attention import apply_fmap, FMAPConfig

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
    print("[fmap] build: annealed AMAP->DMAP (Λ 1->0) + UNCONDITIONAL step-0 snapshot")

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    # standard-attention forward (before FMAP) for the logit-shift diagnostic
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    # ---- resume detection (finetune only): if <push_repo> already has a
    # checkpoint, continue from it; FMAP flags come from the checkpoint so the
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
            # keep the schedule stable across resumes: adopt the checkpoint's window
            if "anneal_start" in ckcfg and "anneal_end" in ckcfg:
                if (anneal_start, anneal_end) != (ckcfg["anneal_start"], ckcfg["anneal_end"]):
                    print(f"[fmap] resume: adopting checkpoint anneal window "
                          f"[{ckcfg['anneal_start']}, {ckcfg['anneal_end']}] (overriding CLI)")
                anneal_start = int(ckcfg["anneal_start"])
                anneal_end = int(ckcfg["anneal_end"])
            if (eff_qk_rmsnorm, eff_lls) != (qk_rmsnorm, learn_logit_scale):
                print(f"[fmap] resume: using checkpoint's FMAP flags "
                      f"(qk_rmsnorm={eff_qk_rmsnorm}, learn_logit_scale={eff_lls}), "
                      f"overriding CLI")
            print(f"[fmap] RESUMING from {push_repo}/checkpoints/step_{s:07d}")
        elif resume == "must":
            raise SystemExit(f"[fmap] resume=must but no checkpoint in {push_repo}")
        else:
            print(f"[fmap] no checkpoint in {push_repo} — fresh start from base SiT")

    fmap_cfg = FMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = apply_fmap(model, fmap_cfg)
    with torch.no_grad(), amp:
        fmap_out = model(x, t, y)
    shift = (fmap_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[fmap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  precision={precision}")
    print(f"[fmap] qk_rmsnorm={eff_qk_rmsnorm} learn_logit_scale={eff_lls}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    if stage == "smoke":
        print("[fmap] smoke OK — nothing trained, nothing pushed.")
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
            f"[fmap] cannot write to '{push_repo}': {e}\n"
            f"       The Modal HF_TOKEN needs create/write rights in that namespace. "
            f"Use your own namespace (e.g. --push-repo jcandane/FMAP) or a token with "
            f"org-write access."
        )
    print(f"[fmap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[fmap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    if resume_sds is not None:
        model_sd, ema_sd = resume_sds
        _, unexpected = model.load_state_dict(model_sd, strict=False)
        assert not unexpected, f"unexpected keys on resume: {unexpected[:5]}"
    ema = C.EMA(model, decay=0.9999)
    ema_at_zero = False   # EMA is only valid once Λ has stopped moving (Λ=0)
    if resume_sds is not None:
        ema.shadow = {k: v.to(dev).float() for k, v in resume_sds[1].items()}
        print(f"[fmap] resumed model+EMA @ step {start_step:,} (optimizer reinitialized)")
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
            from fmap_attention import lambda_at as _lam
            json.dump(
                {"step": step, "qk_mode": "fmap", "variant": "coupled-annealed",
                 "operator": "L(Λ)=Λ·AMAP+(1−Λ)·DMAP; Λ=1 AMAP, Λ=0 DMAP(distance)",
                 "lambda": _lam(step, anneal_start, anneal_end),
                 "anneal_start": anneal_start, "anneal_end": anneal_end,
                 "base": C.SIT_CKPT, "conditional": True,
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
        # push immediately (crash-resilient) — do NOT rely on the end-of-run return
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/fmap_{tag}.png", repo_id=push_repo)
            print(f"[fmap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[fmap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps to train
    from fmap_attention import set_lambda, lambda_at
    print(f"[fmap] anneal window: Λ=1 (AMAP) until step {anneal_start:,}, "
          f"linear -> 0 (DMAP) by step {anneal_end:,}")
    if steps <= 0:
        print(f"[fmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[fmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    set_lambda(model, lambda_at(start_step, anneal_start, anneal_end))   # Λ for the "before" grid
    # step-0 "before" snapshot — ALWAYS (unconditional), never fatal
    try:
        preview(f"step{start_step:07d}_start")
    except Exception as e:
        print(f"[fmap] step-0 snapshot failed (non-fatal): {e!r}")
    model.train()
    for step in range(start_step + 1, end_step + 1):
        lam = lambda_at(step, anneal_start, anneal_end)
        set_lambda(model, lam)                    # Λ: 1 (AMAP) -> 0 (DMAP)
        x1, yy = store.batch(step, 0, bs, base_seed=0)
        with amp:
            loss = transport.training_losses(model, x1, dict(y=yy))["loss"].mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if lam == 0.0 and not ema_at_zero:
            # Λ just reached 0: the operator stops moving, so restart the EMA on
            # the FIXED DMAP operator. The prior EMA smeared across the anneal
            # (a moving minimum) and is not a valid solution — discard it.
            ema = C.EMA(model, decay=0.9999)
            ema_at_zero = True
            print(f"[fmap] Λ=0 reached at step {step:,}: EMA restarted on the fixed "
                  f"DMAP operator (previous EMA spanned the anneal and is discarded)")
        else:
            ema.update(model)

        if step % max(1, steps // 20) == 0 or step == start_step + 1:
            print(f"  step {step:6d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}  Λ={lam:.3f}")
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
    max_shards: int = 0,   # 0 = all shards
    resume: str = "auto",  # auto | never | must
    anneal_start: int = 0,      # hold Λ=1 (AMAP) until this step
    anneal_end: int = 40000,    # Λ hits 0 (pure DMAP) by this step
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    grids = run.remote(stage, steps, lr, push_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, precision, sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume,
                       anneal_start, anneal_end)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"fmap_{tag}.png"
        p.write_bytes(png)
        print(f"[fmap] wrote {p}")

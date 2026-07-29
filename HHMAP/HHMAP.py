"""
HHMAP.py — Modal finetune of the two-exact-potential Hodge homotopy on SiT-XL/2.

HHMAP freezes AMAP's ENTIRE operator (kinetic W_M, flux W_Q,W_K, values, proj,
MLP, conditioning) and the metric-induced tied potential w=‖m‖², and trains ONLY
a free exact potential W_D. One knob α (steps) anneals the flux OFF and both exact
potentials ON:

    logit = ½⟨m,m⟩[frozen W_M]
          + (1−α)·½(𝒲−𝒲ᵀ)[frozen flux]
          + α·½(w_i−w_j)   [frozen tied w=‖m‖², the DMAP/Coifman–Lafon potential]
          + α·½(φ_i−φ_j)   [FREE φ=diag(𝓡 W_D 𝓡ᵀ), the ONLY trainable tensor]

    α : 0 (exactly AMAP-40k) -> 1 (DMAP + a free learned exact potential)

The experiment: as the flux anneals away, can a single free exact potential W_D
absorb its reducible (exact) content, on top of DMAP's own w=‖m‖² potential? With
W_D the ONLY trainable tensor and Λ init 0, mass can move to exactly one place —
so "did the flux's exact content go into W_D" is directly measurable (watch ‖Λ‖).
If the α→1 loss stays flat (vs the original single-potential HMAP's rise), the
flux's reducible part WAS pure exact and W_D caught it; the residual is the true
coexact / irreducible circulation.

    modal run HHMAP/HHMAP.py --stage smoke
    modal run HHMAP/HHMAP.py --stage finetune --steps 40000 \
        --anneal-start 0 --anneal-end 40000 --save-every 10000 --sample-every 10000

Warm-starts from --amap-repo (jcandane/AMAP): loads AMAP qkv (kinetic+values) and
inits the FROZEN flux hmap_qk from AMAP's q,k so α=0 is exactly AMAP-40k. Only
wd_proj + _wd_lambda train. EMA restarts at α=1. Operator: hhmap_attention.py.
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
    .add_local_python_source("hhmap_attention", "hhmap_common")
)

app = modal.App("ditflex-hhmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("HHMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, latents_repo: str,
        qk_rmsnorm: bool, learn_logit_scale: bool, wd_rank: int,
        tied_potential: bool, free_potential: bool, precision: str,
        anneal_start: int, anneal_end: int, amap_repo: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str):
    import contextlib, json, tempfile, torch
    import hhmap_common as C
    from hhmap_attention import apply_hhmap, HHMAPConfig, freeze_except_wd

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
    print("[hhmap] build: frozen AMAP operator + two exact potentials (tied ‖m‖² "
          "+ free W_D), α anneals flux→0 & potentials→on; ONLY W_D trains; EMA restart at α=1")

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    # ---- resolution: HHMAP own checkpoint -> else warm-start from AMAP-40k ----
    start_step, resume_sds = 0, None
    eff_qk_rmsnorm, eff_lls, eff_wd_rank = qk_rmsnorm, learn_logit_scale, wd_rank
    eff_tied, eff_free = tied_potential, free_potential
    warm_qk_from_amap = None   # AMAP's q,k slices to init the FROZEN flux hmap_qk
    if stage == "finetune" and resume != "never":
        s = C.latest_checkpoint_step(push_repo)
        if s is not None:
            ckcfg, model_sd, ema_sd = C.fetch_checkpoint(push_repo, s)
            start_step = s
            resume_sds = (model_sd, ema_sd)
            eff_qk_rmsnorm = bool(ckcfg.get("qk_rmsnorm", False))
            eff_lls = bool(ckcfg.get("learn_logit_scale", False))
            eff_wd_rank = int(ckcfg.get("wd_rank", 0))
            eff_tied = bool(ckcfg.get("tied_potential", True))
            eff_free = bool(ckcfg.get("free_potential", True))
            if "anneal_start" in ckcfg and "anneal_end" in ckcfg:
                anneal_start, anneal_end = int(ckcfg["anneal_start"]), int(ckcfg["anneal_end"])
            print(f"[hhmap] RESUMING HHMAP from {push_repo}/checkpoints/step_{s:07d}")
        else:
            a = C.latest_checkpoint_step(amap_repo)
            if a is not None:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                folder = f"checkpoints/step_{a:07d}"
                try:
                    am_sd = load_file(hf_hub_download(amap_repo, f"{folder}/ema.safetensors"))
                    src = "ema"
                except Exception:
                    am_sd = load_file(hf_hub_download(amap_repo, f"{folder}/model.safetensors"))
                    src = "model"
                _, unexp = model.load_state_dict(am_sd, strict=False)   # AMAP qkv -> model
                warm_qk_from_amap = am_sd   # keep to init frozen hmap_qk from AMAP q,k
                print(f"[hhmap] WARM-START from AMAP {amap_repo}/{folder} ({src}); "
                      f"AMAP qkv -> kinetic+values, and -> FROZEN flux hmap_qk (step reset 0)")
                if unexp:
                    print(f"[hhmap] (ignoring {len(unexp)} AMAP-only key(s), e.g. {unexp[:2]})")
            elif resume == "must":
                raise SystemExit(f"[hhmap] resume=must but no HHMAP ({push_repo}) "
                                 f"or AMAP ({amap_repo}) checkpoint")
            else:
                print(f"[hhmap] no HHMAP or AMAP checkpoint — base SiT (α=0=AMAP-on-base)")

    hhmap_cfg = HHMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls,
                            wd_rank=eff_wd_rank,
                            tied_potential=eff_tied, free_potential=eff_free)
    n_attn = apply_hhmap(model, hhmap_cfg, alpha=0.0)  # creates frozen hmap_qk (from qkv) + free W_D

    # If warm-started from AMAP, hmap_qk was inited from the BASE qkv inside
    # apply_hhmap; that's already the AMAP qkv (we loaded it above), so the frozen
    # flux equals AMAP's. (AMAP is coupled — its flux comes from the same qkv.)
    n_train, n_froze = freeze_except_wd(model)
    print(f"[hhmap] frozen AMAP operator + tied potential; trainable={n_train/1e6:.2f}M "
          f"(W_D only) frozen={n_froze/1e6:.1f}M")
    with torch.no_grad(), amp:
        hh_out = model(x, t, y)
    shift = (hh_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[hhmap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  "
          f"precision={precision}  wd_rank={eff_wd_rank or 'full'}")
    print(f"[hhmap] rel-shift vs standard attn (α=0) = {shift.item():.3f}  "
          f"(should ≈ AMAP's shift)")

    if stage == "smoke":
        print("[hhmap] smoke OK — nothing trained, nothing pushed.")
        return []

    from huggingface_hub import HfApi
    try:
        HfApi().create_repo(push_repo, exist_ok=True)
    except Exception as e:
        raise SystemExit(
            f"[hhmap] cannot write to '{push_repo}': {e}\n"
            f"       Use your own namespace (e.g. --push-repo jcandane/HHMAP) or a "
            f"token with org-write access.")
    print(f"[hhmap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[hhmap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    if resume_sds is not None:
        model_sd, ema_sd = resume_sds
        _, unexpected = model.load_state_dict(model_sd, strict=False)
        assert not unexpected, f"unexpected keys on resume: {unexpected[:5]}"
    ema = C.EMA(model, decay=0.9999)
    if resume_sds is not None:
        ema.shadow = {k: v.to(dev).float() for k, v in resume_sds[1].items()}
        print(f"[hhmap] resumed model+EMA @ step {start_step:,} (optimizer reinitialized)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    bs = 64
    grids: list[tuple[str, bytes]] = []

    def save_ckpt(step):
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        from hhmap_attention import alpha_at as _aa
        api = HfApi(); api.create_repo(push_repo, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            save_file(model.state_dict(), f"{d}/model.safetensors")
            save_file({k: v.contiguous() for k, v in ema.state_dict().items()},
                      f"{d}/ema.safetensors")
            json.dump(
                {"step": step, "qk_mode": "hhmap", "variant": "two-potential-frozen",
                 "operator": ("½⟨m,m⟩[frozen W_M] + (1−α)·½(𝒲−𝒲ᵀ)[frozen flux] "
                              "+ α·½(w_i−w_j)[frozen tied w=‖m‖²] "
                              "+ α·½(φ_i−φ_j)[free φ=diag(𝓡 W_D 𝓡ᵀ)]"),
                 "alpha": _aa(step, anneal_start, anneal_end),
                 "anneal_start": anneal_start, "anneal_end": anneal_end,
                 "warm_start": amap_repo, "trainable": "wd_proj + _wd_lambda only",
                 "wd_rank": eff_wd_rank, "tied_potential": eff_tied,
                 "free_potential": eff_free, "base": C.SIT_CKPT, "conditional": True,
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/hhmap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[hhmap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    def preview(tag):
        path = f"/cache/samples/hhmap_{tag}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        grids.append((tag, png))
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/hhmap_{tag}.png", repo_id=push_repo)
            print(f"[hhmap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[hhmap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps
    from hhmap_attention import set_alpha, alpha_at

    def wd_mass():
        # mean ‖Λ‖ across wd potential heads — THE signal: mass moving into W_D.
        vals = []
        for mod in model.modules():
            if hasattr(mod, "_wd_lambda"):
                vals.append(mod._wd_lambda.detach().norm().item())
        return sum(vals) / max(len(vals), 1)

    print(f"[hhmap] homotopy: α 0(AMAP) -> 1(DMAP+free potential) over "
          f"[{anneal_start:,}, {anneal_end:,}]  (flux off, both potentials on)")
    if steps <= 0:
        print(f"[hhmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[hhmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    set_alpha(model, alpha_at(start_step, anneal_start, anneal_end))
    ema_at_one = False
    try:
        preview(f"step{start_step:07d}_start")
    except Exception as e:
        print(f"[hhmap] step-0 snapshot failed (non-fatal): {e!r}")
    model.train()
    for step in range(start_step + 1, end_step + 1):
        a = alpha_at(step, anneal_start, anneal_end)
        set_alpha(model, a)
        x1, yy = store.batch(step, 0, bs, base_seed=0)
        with amp:
            loss = transport.training_losses(model, x1, dict(y=yy))["loss"].mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        if a >= 1.0 and not ema_at_one:
            ema = C.EMA(model, decay=0.9999)
            ema_at_one = True
            print(f"[hhmap] α=1 reached at step {step:,}: EMA restarted on the fixed "
                  f"DMAP+free-potential operator")
        else:
            ema.update(model)

        if step % max(1, steps // 20) == 0 or step == start_step + 1:
            print(f"  step {step:6d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}"
                  f"  α={a:.3f}  ‖Λ‖(W_D mass)={wd_mass():.4f}")
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
    lr: float = 1e-4,     # W_D only, init 0 — can afford a higher LR than full-model
    push_repo: str = "jcandane/HHMAP",
    latents_repo: str = "sparsetrace/dlatentzz",
    qk_rmsnorm: bool = False,
    learn_logit_scale: bool = False,
    wd_rank: int = 0,     # 0 = full [C,C] P; r>0 = low-rank r-per-head
    tied_potential: bool = True,   # frozen DMAP w=‖m‖² channel (annealed in)
    free_potential: bool = True,   # trainable free W_D channel (the experiment)
    precision: str = "tf32",
    sample_every: int = 0,
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    save_every: int = 10000,
    max_shards: int = 0,
    resume: str = "auto",
    anneal_start: int = 0,
    anneal_end: int = 40000,
    amap_repo: str = "jcandane/AMAP",
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    if not (tied_potential or free_potential):
        raise SystemExit("at least one of --tied-potential / --free-potential must be on "
                         "(else α→1 is bare Gram with no exact sector)")
    grids = run.remote(stage, steps, lr, push_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, wd_rank, tied_potential, free_potential,
                       precision, anneal_start, anneal_end,
                       amap_repo, sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"hhmap_{tag}.png"
        p.write_bytes(png)
        print(f"[hhmap] wrote {p}")

"""
HMAP.py — Modal finetune of the HODGE homotopy on SiT-XL/2 (warm-started from AMAP).

HMAP freezes AMAP's kinetic sector (the metric W_M) and opens only the
antisymmetric sector, then adiabatically trades flux for the exact/Doob sector
with ONE knob α (γ = 1−α):

    logit = ½⟨m_i,m_j⟩[frozen W_M] + (1−α)·½(𝒲−𝒲ᵀ)[flux] + α·½(g1ᵀ−1gᵀ)[exact]
    m = (q₀+k₀)/√2 from FROZEN AMAP qkv;  𝒲=q_a k_aᵀ, g=diag(𝒲) from TRAINABLE hmap_qk

    α : 0 (exactly AMAP-40k) -> 1 (frozen kinetic + exact/Doob g, a DMAP-class op)

    modal run HMAP/HMAP.py --stage smoke
    modal run HMAP/HMAP.py --stage finetune --steps 40000 \
        --anneal-start 0 --anneal-end 40000 --save-every 10000 --sample-every 10000

Only `hmap_qk` (the antisymmetric generators q_a,k_a) trains; the kinetic, values,
proj, MLP, and conditioning are all frozen at AMAP-40k. Warm-starts from
`--amap-repo` (jcandane/AMAP). `--steps N` is additive; the anneal window is read
back from the checkpoint on resume so the schedule is stable. EMA restarts at α=1
(the prior EMA spans the moving operator and is discarded). Operator:
hmap_attention.py; shared helpers: hmap_common.py.
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

app = modal.App("ditflex-hmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("HMAP_GPU", "H200")


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(stage: str, steps: int, lr: float, push_repo: str, latents_repo: str,
        qk_rmsnorm: bool, learn_logit_scale: bool, precision: str,
        anneal_start: int, anneal_end: int, amap_repo: str,
        sample_every: int, sample_steps: int, cfg_scale: float, save_every: int,
        max_shards: int, resume: str):
    import contextlib, json, tempfile, torch
    import hmap_common as C
    from hmap_attention import apply_hmap, HMAPConfig

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
    print("[hmap] build: frozen-kinetic + α-homotopy AMAP->exact (γ=1−α) + EMA restart at α=1")

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    # standard-attention forward (before HMAP) for the logit-shift diagnostic
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)

    # ---- resolution: HMAP own checkpoint -> else warm-start from AMAP-40k ----
    # HMAP freezes the AMAP kinetic (W_M) and opens only the antisymmetric sector
    # (a separate trainable W_Q,W_K feeding flux + exact g). It therefore starts
    # from an AMAP checkpoint: α=0 is exactly that AMAP.
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
            if "anneal_start" in ckcfg and "anneal_end" in ckcfg:
                anneal_start, anneal_end = int(ckcfg["anneal_start"]), int(ckcfg["anneal_end"])
            print(f"[hmap] RESUMING HMAP from {push_repo}/checkpoints/step_{s:07d}")
        else:
            a = C.latest_checkpoint_step(amap_repo)
            if a is not None:
                # warm-start needs only AMAP's WEIGHTS, not its config (HMAP builds
                # its own). Pull safetensors directly — avoids *_config.json clash.
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
                print(f"[hmap] WARM-START from AMAP {amap_repo}/{folder} ({src}) "
                      f"(hmap_qk will init from these q,k; step reset to 0)")
                if unexp:
                    print(f"[hmap] (ignoring {len(unexp)} AMAP-only key(s), e.g. {unexp[:2]})")
            elif resume == "must":
                raise SystemExit(f"[hmap] resume=must but no HMAP ({push_repo}) "
                                 f"or AMAP ({amap_repo}) checkpoint found")
            else:
                print(f"[hmap] no HMAP or AMAP checkpoint — starting from base SiT (α=0=AMAP-on-base)")

    hmap_cfg = HMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = apply_hmap(model, hmap_cfg, alpha=0.0)   # hmap_qk inits from current qkv
    from hmap_attention import freeze_except_hmap
    n_train, n_froze = freeze_except_hmap(model)
    print(f"[hmap] frozen kinetic + open antisym: trainable={n_train/1e6:.1f}M "
          f"frozen={n_froze/1e6:.1f}M (only hmap_qk trains)")
    with torch.no_grad(), amp:
        hmap_out = model(x, t, y)
    shift = (hmap_out - std_out).flatten().norm() / std_out.flatten().norm()
    print(f"[hmap] SiT-XL/2 params={n_params/1e6:.1f}M  patched_attn={n_attn}  precision={precision}")
    print(f"[hmap] qk_rmsnorm={eff_qk_rmsnorm} learn_logit_scale={eff_lls}  "
          f"rel-shift vs standard attn = {shift.item():.3f}")

    if stage == "smoke":
        print("[hmap] smoke OK — nothing trained, nothing pushed.")
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
            f"[hmap] cannot write to '{push_repo}': {e}\n"
            f"       The Modal HF_TOKEN needs create/write rights in that namespace. "
            f"Use your own namespace (e.g. --push-repo jcandane/HMAP) or a token with "
            f"org-write access."
        )
    print(f"[hmap] push target OK: {push_repo}")

    C.sit_path()
    from transport import create_transport
    transport = create_transport("Linear", "velocity")

    store = C.LatentStore.from_hub(latents_repo, device=dev, max_files=(max_shards or None))
    print(f"[hmap] latents resident: {len(store):,}  "
          f"labels [{int(store.labels.min())},{int(store.labels.max())}]  from {latents_repo}")

    if resume_sds is not None:
        model_sd, ema_sd = resume_sds
        _, unexpected = model.load_state_dict(model_sd, strict=False)
        assert not unexpected, f"unexpected keys on resume: {unexpected[:5]}"
    ema = C.EMA(model, decay=0.9999)
    if resume_sds is not None:
        ema.shadow = {k: v.to(dev).float() for k, v in resume_sds[1].items()}
        print(f"[hmap] resumed model+EMA @ step {start_step:,} (optimizer reinitialized)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
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
            from hmap_attention import alpha_at as _aa
            json.dump(
                {"step": step, "qk_mode": "hmap", "variant": "coupled-frozen-kinetic",
                 "operator": "½⟨m,m⟩[frozen W_M] + (1−α)·½(𝒲−𝒲ᵀ)[flux] + α·½(g1ᵀ−1gᵀ)[exact], g=diag(𝒲)",
                 "alpha": _aa(step, anneal_start, anneal_end),
                 "anneal_start": anneal_start, "anneal_end": anneal_end,
                 "warm_start": amap_repo, "trainable": "hmap_qk only (kinetic frozen)",
                 "base": C.SIT_CKPT, "conditional": True,
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/hmap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[hmap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    def preview(tag):
        path = f"/cache/samples/hmap_{tag}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        ckpt_vol.commit()
        grids.append((tag, png))
        # push immediately (crash-resilient) — do NOT rely on the end-of-run return
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"samples/hmap_{tag}.png", repo_id=push_repo)
            print(f"[hmap] preview grid '{tag}' -> {push_repo}/samples/ ({len(png)//1024} KiB)")
        except Exception as e:
            print(f"[hmap] preview '{tag}' rendered but upload failed (non-fatal): {e!r}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps to train
    from hmap_attention import set_alpha, alpha_at

    def wqk_gap():
        # mean ‖W_Q − W_K‖ across hmap_qk projections — watch the flux generator
        import torch as _t
        gaps = []
        for mod in model.modules():
            if hasattr(mod, "hmap_qk"):
                Cq = mod.hmap_qk.weight.shape[1]
                wq, wk = mod.hmap_qk.weight[:Cq], mod.hmap_qk.weight[Cq:]
                gaps.append((wq - wk).norm().item())
        return sum(gaps) / max(len(gaps), 1)

    print(f"[hmap] homotopy: α 0(AMAP) -> 1(exact/DMAP-class) over "
          f"[{anneal_start:,}, {anneal_end:,}]  (γ = 1−α)")
    if steps <= 0:
        print(f"[hmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[hmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}")
    set_alpha(model, alpha_at(start_step, anneal_start, anneal_end))
    ema_at_one = False   # EMA valid only once α has stopped moving (α=1)
    # step-0 "before" snapshot — ALWAYS (unconditional), never fatal
    try:
        preview(f"step{start_step:07d}_start")
    except Exception as e:
        print(f"[hmap] step-0 snapshot failed (non-fatal): {e!r}")
    model.train()
    for step in range(start_step + 1, end_step + 1):
        a = alpha_at(step, anneal_start, anneal_end)
        set_alpha(model, a)                       # α: 0 (AMAP) -> 1 (exact)
        x1, yy = store.batch(step, 0, bs, base_seed=0)
        with amp:
            loss = transport.training_losses(model, x1, dict(y=yy))["loss"].mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if a >= 1.0 and not ema_at_one:
            # α reached 1: operator stops moving -> restart EMA on the fixed
            # exact operator (the prior EMA spanned the homotopy; discard it).
            ema = C.EMA(model, decay=0.9999)
            ema_at_one = True
            print(f"[hmap] α=1 reached at step {step:,}: EMA restarted on the fixed "
                  f"exact operator (previous EMA spanned the homotopy)")
        else:
            ema.update(model)

        if step % max(1, steps // 20) == 0 or step == start_step + 1:
            print(f"  step {step:6d}  loss {loss.item():.4f}  grad {gnorm.item():.2f}"
                  f"  α={a:.3f}  |Wq-Wk|={wqk_gap():.3f}")
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
    push_repo: str = "jcandane/HMAP",
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
    anneal_start: int = 0,       # α held 0 (AMAP) until this step
    anneal_end: int = 40000,     # α reaches 1 (exact/DMAP-class) by this step
    amap_repo: str = "jcandane/AMAP",   # warm-start source (frozen kinetic + init hmap_qk)
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    grids = run.remote(stage, steps, lr, push_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, precision, anneal_start, anneal_end, amap_repo,
                       sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"hmap_{tag}.png"
        p.write_bytes(png)
        print(f"[hmap] wrote {p}")

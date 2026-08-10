"""
DMAP.py — Modal ephemeral finetune of DMAP-on-SiT-XL/2.

DMAP = the symmetric/reversible sector of AMAP: the Mahalanobis distance
kernel with its metric-induced potential.

    logit_ij = −½‖μ_i − μ_j‖²,  μ = R·W_M,  W_M = (W_Q+W_K)/√2
             = μ_i·μ_j − ½‖μ_i‖² − ½‖μ_j‖²

The −½‖μ_j‖² column term IS the potential (Coifman–Lafon self-affinity of
the metric W_M W_Mᵀ); the −½‖μ_i‖² row term is constant per row and dies in
the row-softmax. The wmv bias shifts μ, hence the potential's center
(tier-1 linear tilt). No separate potential weight.

FOLD ≡ TIED INIT. Folding W_Q,W_K → W_M is identical to tying W_K = W_Q at
their √2-normalized average: W_N and the flux Ω are deleted, ~⅓ of the
attention-projection params + optimizer state removed. NOTE the asymmetry
with the forward surgery: attention→AMAP *adds* a PSD correction to an
intact score (near-lossless); AMAP→DMAP *deletes* a sector the trained
network co-adapted with. Expect a nonzero step-0 wound — measured below
globally and per block — and a recovery phase. That is the experiment.

Step-0 inference is UNCONDITIONAL: every invocation renders (a) a pre-fold
grid on the source weights when warm-starting, and (b) a post-fold grid
before step 1, both tagged with provenance (fromAMAP / fromBASE / resume).
Smoke renders (b) too (nothing pushed).

    modal run DMAP/DMAP.py --stage smoke
    modal run DMAP/DMAP.py --stage finetune --steps 100000 --save-every 10000

Checkpoint resolution for finetune (resume=auto):
    1. DMAP's own latest in --push-repo   -> resume (folded)
    2. else AMAP's latest in --amap-repo  -> WARM-START + FOLD (step 0)
    3. else base SiT-XL/2                 -> fresh + FOLD

Protocol note: LR default is 1e-4 CONSTANT (DiT/SiT convention) to keep the
matched comparison with the AMAP arm honest. Change it only in lockstep
with the AMAP workflow. Helpers: dmap_common.py; operator + fold:
dmap_attention.py.
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
        max_shards: int, resume: str):
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
    print("[dmap] build: fold(=tied W_Q=W_K init) + warm-start + step-0 grids")

    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()
    n_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 1000, (2,), device=dev)

    grids: list[tuple[str, bytes]] = []

    def render(tag, push: bool):
        """Sample a grid; append to grids; optionally push. Never fatal."""
        path = f"/cache/samples/dmap_{tag}.png"
        try:
            _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
            ckpt_vol.commit()
            grids.append((tag, png))
            print(f"[dmap] grid '{tag}' rendered ({len(png)//1024} KiB)")
            if push:
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=path,
                                    path_in_repo=f"samples/dmap_{tag}.png",
                                    repo_id=push_repo)
                print(f"[dmap] grid '{tag}' -> {push_repo}/samples/")
        except Exception as e:
            print(f"[dmap] grid '{tag}' failed (non-fatal): {e!r}")

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
                      f"tying/folding W_Q,W_K -> W_M (step reset to 0)")
                if unexp:
                    print(f"[dmap] (ignoring {len(unexp)} AMAP-only key(s), e.g. {unexp[:2]})")
            elif resume == "must":
                raise SystemExit(f"[dmap] resume=must but no DMAP ({push_repo}) "
                                 f"or AMAP ({amap_repo}) checkpoint found")
            else:
                print(f"[dmap] no DMAP or AMAP checkpoint — fresh start from base SiT, folding")

    provenance = "resume" if folded_sd is not None else ("fromAMAP" if warm else "fromBASE")

    # ---- pre-fold reference forward + per-block taps (source model) ----
    per_block_src: dict[str, torch.Tensor] = {}
    per_block_new: dict[str, torch.Tensor] = {}

    def _tap(store):
        hooks = []
        for name, m in model.named_modules():
            if hasattr(m, "num_heads") and hasattr(m, "scale") and (
                    hasattr(m, "qkv") or hasattr(m, "wmv")):
                def h(_m, _i, o, _name=name):
                    store[_name] = o.detach().float()
                hooks.append(m.register_forward_hook(h))
        return hooks

    hooks = _tap(per_block_src)
    with torch.no_grad(), amp:
        std_out = model(x, t, y)          # standard attention on the pre-fold weights
    for h in hooks:
        h.remove()

    # pre-fold "before" grid on the source weights (only meaningful when not resuming)
    if stage == "finetune" and folded_sd is None:
        render(f"prefold_{provenance}_step{start_step:07d}", push=False)

    # ---- FOLD W_Q,W_K -> W_M (drop W_N). Exact fold == tied init. ----
    dcfg = DMAPConfig(qk_rmsnorm=eff_qk_rmsnorm, learn_logit_scale=eff_lls)
    n_attn = install_folded_dmap(model, dcfg, fold_weights=True)
    n_folded = sum(p.numel() for p in model.parameters())
    if folded_sd is not None:             # DMAP resume: load trained folded weights
        is_coupled = any(k.endswith(".qkv.weight") for k in folded_sd)
        if is_coupled:
            from dmap_attention import fold_state_dict
            folded_sd = fold_state_dict(folded_sd)
            if ema_sd is not None:
                ema_sd = fold_state_dict(ema_sd)
            print("[dmap] resume checkpoint was COUPLED (qkv) — auto-folded to wmv")
        _, unexp = model.load_state_dict(folded_sd, strict=False)
        if unexp:
            raise SystemExit(f"[dmap] unexpected keys resuming checkpoint: {unexp[:5]}")

    hooks = _tap(per_block_new)
    with torch.no_grad(), amp:
        dmap_out = model(x, t, y)
    for h in hooks:
        h.remove()

    # ---- surgery diagnostics: global + per-block rel-shift ----
    shift = (dmap_out - std_out).flatten().norm() / std_out.flatten().norm()
    saved = 100.0 * (n_params - n_folded) / n_params
    print(f"[dmap] SiT-XL/2 {n_params/1e6:.1f}M -> {n_folded/1e6:.1f}M folded "
          f"(−{saved:.1f}%, W_N dropped)  attn={n_attn}  precision={precision}")
    print(f"[dmap] provenance={provenance}  qk_rmsnorm={eff_qk_rmsnorm} "
          f"learn_logit_scale={eff_lls}")
    print(f"[dmap] SURGERY WOUND global rel-shift = {shift.item():.4f}")
    block_shift = {}
    for name in per_block_src:
        if name in per_block_new:
            a_, b_ = per_block_src[name], per_block_new[name]
            block_shift[name] = ((b_ - a_).norm() / (a_.norm() + 1e-12)).item()
    if block_shift:
        worst = sorted(block_shift.items(), key=lambda kv: -kv[1])[:5]
        print("[dmap] per-block rel-shift (attention outputs, worst 5): "
              + ", ".join(f"{k.split('.')[-2]}={v:.3f}" for k, v in worst))

    if stage == "smoke":
        render(f"smoke_{provenance}_step{start_step:07d}", push=False)
        print("[dmap] smoke OK — nothing trained, nothing pushed.")
        return grids

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

    ema = C.EMA(model, decay=0.9999)
    if ema_sd is not None:
        shadow = ema.state_dict()
        for kk, vv in ema_sd.items():
            if kk in shadow:
                shadow[kk] = vv.to(dev).float()
        print(f"[dmap] loaded folded EMA @ step {start_step:,} (optimizer reinitialized)")
    elif warm:
        print(f"[dmap] EMA snapshot from AMAP warm-start (folded); optimizer fresh")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = 64

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
                 "operator": "mahalanobis: -1/2||mu_i-mu_j||^2, mu=R·W_M, "
                             "W_M=(W_Q+W_K)/sqrt2 (== tied W_Q=W_K); potential = "
                             "column term 1/2||mu_j||^2 (metric self-affinity)",
                 "folded": True, "attn_proj": "wmv [2d,d] (W_N dropped)",
                 "base": C.SIT_CKPT, "conditional": True,
                 "provenance": provenance, "warm_start_from": warm,
                 "surgery_rel_shift": float(shift.item()),
                 "qk_rmsnorm": eff_qk_rmsnorm, "learn_logit_scale": eff_lls,
                 "precision": precision, "lr": lr,
                 "objective": "SiT transport Linear/velocity (t in [0,1])"},
                open(f"{d}/dmap_config.json", "w"), indent=2)
            api.upload_folder(folder_path=d, repo_id=push_repo,
                              path_in_repo=f"checkpoints/step_{step:07d}")
        print(f"[dmap] saved checkpoint step {step:,} -> {push_repo}/checkpoints/step_{step:07d}")

    end_step = start_step + steps   # steps = N ADDITIONAL steps
    if steps <= 0:
        print(f"[dmap] steps={steps} <= 0; nothing to train — sampling current weights.")
    else:
        print(f"[dmap] training {steps:,} more steps: {start_step:,} -> {end_step:,}  "
              f"(constant lr={lr:g})")

    # step-0 post-fold snapshot — ALWAYS, provenance-tagged
    render(f"step{start_step:07d}_{provenance}_postfold", push=True)

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
            render(f"step{step:07d}", push=True)

    if end_step > start_step and (save_every <= 0 or end_step % save_every != 0):
        save_ckpt(end_step)
    if sample_every <= 0 or end_step % sample_every != 0:
        render(f"step{end_step:07d}", push=True)
    return grids


@app.local_entrypoint()
def main(
    stage: str = "finetune",
    steps: int = 500,
    lr: float = 1e-4,                   # PROTOCOL: constant 1e-4, matched with AMAP arm
    push_repo: str = "jcandane/DMAP",
    amap_repo: str = "jcandane/AMAP",   # warm-start source when no DMAP checkpoint
    latents_repo: str = "sparsetrace/dlatentzz",
    qk_rmsnorm: bool = False,
    learn_logit_scale: bool = False,
    precision: str = "tf32",
    sample_every: int = 10000,
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    save_every: int = 10000,
    max_shards: int = 0,
    resume: str = "auto",
):
    if stage == "finetune" and not latents_repo:
        raise SystemExit("finetune needs --latents-repo <your-hf-latents-dataset>")
    grids = run.remote(stage, steps, lr, push_repo, amap_repo, latents_repo, qk_rmsnorm,
                       learn_logit_scale, precision, sample_every, sample_steps,
                       cfg_scale, save_every, max_shards, resume)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"dmap_{tag}.png"
        p.write_bytes(png)
        print(f"[dmap] wrote {p}")

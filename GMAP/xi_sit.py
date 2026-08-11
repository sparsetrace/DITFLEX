"""
GMAP/xi_sit.py — frozen anti-attention (Girsanov) scan on SiT-XL/2: NO training.

    modal run GMAP/xi_sit.py                          # base SiT, standard variant
    modal run GMAP/xi_sit.py --base amap              # AMAP-finetuned ckpt (EMA)
    modal run GMAP/xi_sit.py --base amap --ckpt-step 50000 --weights model

Loads ONE set of weights, patches GMAP (the (c_sym, c_flux)-coupled operator),
then sweeps rays through the two-coupling plane, rendering the FIXED
sample-grid panel (amap_common.FIXED_CLASSES / FIXED_SEED — pixel-comparable
to every sample0/AMAP grid) and computing a PAIRED frozen flow-matching loss
(same deterministic latent batches at every point) per point:

  xi        : (2-t, t)   anti-attention diagonal. Bidirectional SiT has no
              causal mask, so t -> 0 is a literal Girsanov annealing to a
              reversible (detailed-balance) kernel.
  fluxcut   : (1, t)     pinned metric, flux -> 0 — the frozen coexact dial
  symheat   : (c, 1)     pinned flux, metric coupling 1 -> 2
  fluxboost : (1, t>1)   flux over-driven past trained coupling (r > 1)

Base arms:
  --base standard : released 7M SiT-XL/2, variant="standard"
                    (sym = indefinite 1/2(qk + qk^T); (1,1) == released model)
  --base amap     : AMAP finetuned checkpoint from --ckpt-repo (EMA by
                    default), variant="amap" (sym = PSD Gram; (1,1) == the
                    AMAP operator those weights were trained under)

Nothing trains. The (1,1) grid is the baseline; a rel-shift diagnostic
asserts (1,1) matches the unpatched forward of the same weights bit-for-bit
(standard base) / matches the AMAP forward (amap base).

REQUIRES amap_common.py present in this folder (copy the one file from
AMAP/ unchanged — it is pure library code).

Outputs: grids gmap_<base>_<ray>_t<t>.png + scan JSON/CSV with the FM-loss
column, pushed to <push_repo>/samples/gmap/ and /probes/, and returned to the
local entrypoint which writes GMAP/samples/ for the workflow's commit step.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

_HERE = Path(__file__).parent


def _find_module(name: str) -> str:
    """Locate a helper module: GMAP/ first, then ../AMAP/ (no copy needed)."""
    for cand in (_HERE / name, _HERE.parent / "AMAP" / name):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"{name} not found next to xi_sit.py or in ../AMAP/ — "
        f"GMAP needs gmap_attention.py locally and amap_common.py "
        f"(local copy or the AMAP/ original)."
    )


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
    .add_local_file(_find_module("gmap_attention.py"), "/root/gmap_attention.py")
    .add_local_file(_find_module("amap_common.py"), "/root/amap_common.py")
)

app = modal.App("ditflex-gmap")
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
GPU = os.environ.get("AMAP_GPU", "B200")


def _ray_points(rays: str, spacing: float):
    """Build the (ray, t, c_sym, c_flux) list. (1,1) kept once per ray for
    free replicates, mirroring the nanochat scan."""
    import numpy as np
    want = [r.strip() for r in rays.split(",") if r.strip()]
    pts = []
    lo = [round(float(x), 4) for x in np.arange(0.0, 1.0 + 1e-9, spacing)]
    hi = [round(float(x), 4) for x in np.arange(1.0, 2.0 + 1e-9, spacing)]
    for r in want:
        if r == "xi":
            pts += [("xi", t, 2.0 - t, t) for t in lo]
        elif r == "fluxcut":
            pts += [("fluxcut", t, 1.0, t) for t in lo]
        elif r == "symheat":
            pts += [("symheat", c, c, 1.0) for c in hi]
        elif r == "fluxboost":
            pts += [("fluxboost", t, 1.0, t) for t in hi if t > 1.0]
        else:
            raise ValueError(f"unknown ray {r!r}")
    return pts


@app.function(image=image, gpu=GPU, secrets=[HF_SECRET], timeout=6 * 60 * 60,
              volumes={"/cache": ckpt_vol})
def run(base: str, ckpt_repo: str, ckpt_step: str, weights: str,
        rays: str, spacing: float, precision: str,
        sample_steps: int, cfg_scale: float,
        fm_batches: int, fm_bs: int, latents_repo: str, max_shards: int,
        push_repo: str):
    import contextlib, csv, json, sys, torch
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")     # add_local_file mounts land in /root
    import amap_common as C
    from gmap_attention import apply_gmap, GMAPConfig

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

    # ---- weights + reference forward (pre-patch) ------------------------------
    model = C.build_sit_xl2().to(dev)
    ckpt_vol.commit()

    x = torch.randn(2, 4, 32, 32, device=dev)
    t0 = torch.rand(2, device=dev)
    y0 = torch.randint(0, 1000, (2,), device=dev)

    if base == "standard":
        variant, ckpt_flags, origin, sd = "standard", {}, "base-7M", None
        with torch.no_grad(), amp:
            ref_out = model(x, t0, y0)          # released forward, released weights
    elif base == "amap":
        variant, sd = "amap", None
        resolved = C.resolve_checkpoint_step(ckpt_repo, ckpt_step)
        if resolved is None:
            ckpt_flags, origin = {}, f"{ckpt_repo}:base+amap-operator"
            print("[gmap] base=amap with step='base': Karpathy-analogue — "
                  "un-finetuned SiT weights under the AMAP operator")
        else:
            cfg_dict, model_sd, ema_sd = C.fetch_checkpoint(ckpt_repo, resolved)
            ckpt_flags = {k: bool(cfg_dict.get(k, False))
                          for k in ("qk_rmsnorm", "learn_logit_scale")}
            sd = ema_sd if weights == "ema" else model_sd
            origin = f"{ckpt_repo}:step_{resolved:07d}:{weights}"
        ref_out = None                           # reference built after patch
    else:
        raise ValueError(f"base must be standard|amap, got {base!r}")

    cfg = GMAPConfig(variant=variant, **ckpt_flags) if base == "amap" else \
          GMAPConfig(variant=variant)
    cfg = apply_gmap(model, cfg)
    if base == "amap" and sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        if missing:
            print(f"[gmap] note: {len(missing)} keys missing on load "
                  f"(expected only if flags mismatch): {missing[:3]}")
    n_attn = model._gmap_n_attn
    print(f"[gmap] base={base} origin={origin} variant={variant} "
          f"patched_attn={n_attn} precision={precision}")

    # ---- (1,1) identity check -------------------------------------------------
    cfg.c_sym, cfg.c_flux = 1.0, 1.0
    with torch.no_grad(), amp:
        out11 = model(x, t0, y0)
    if base == "standard":
        shift = (out11 - ref_out).flatten().norm() / ref_out.flatten().norm()
        print(f"[gmap] (1,1) vs released forward rel-shift = {shift.item():.2e} "
              f"(must be ~0: GMAP standard at (1,1) IS standard attention)")
        assert shift.item() < 1e-4, "GMAP (1,1) failed to reproduce standard attention"
    else:
        print("[gmap] (1,1) is the AMAP operator on these weights (baseline arm)")

    # ---- frozen paired FM-loss machinery --------------------------------------
    store = None
    if fm_batches > 0:
        C.sit_path()
        from transport import create_transport
        transport = create_transport("Linear", "velocity")
        store = C.LatentStore.from_hub(latents_repo, device=dev,
                                       max_files=(max_shards or None))
        print(f"[gmap] FM-loss column: {fm_batches} paired batches of {fm_bs} "
              f"({len(store):,} latents resident)")

        def fm_loss():
            tot = 0.0
            for i in range(fm_batches):
                x1, yy = store.batch(i, 0, fm_bs, base_seed=777)   # PAIRED
                torch.manual_seed(10_000 + i)   # pin t/noise draws per batch
                with torch.no_grad(), amp:
                    tot += transport.training_losses(
                        model, x1, dict(y=yy))["loss"].mean().item()
            return tot / fm_batches
    else:
        def fm_loss():
            return None

    # ---- the scan -------------------------------------------------------------
    points = _ray_points(rays, spacing)
    print(f"[gmap] scanning {len(points)} points on rays [{rays}] "
          f"spacing {spacing}")
    from huggingface_hub import HfApi
    api = HfApi()
    rows, grids = [], []
    model.eval()
    for ray, tcoord, cs, cf in points:
        cfg.c_sym, cfg.c_flux = float(cs), float(cf)
        name = f"gmap_{base}_{ray}_t{tcoord:.2f}"
        path = f"/cache/samples/{name}.png"
        _, png = C.sample_grid(model, dev, path, sample_steps, cfg_scale, amp)
        fm = fm_loss()
        rows.append(dict(base=base, ray=ray, t=tcoord, c_sym=cs, c_flux=cf,
                         fm_loss=fm))
        grids.append((name, png))
        fm_s = f"{fm:.5f}" if fm is not None else "n/a"
        print(f"[gmap] {ray:9s} t={tcoord:.2f} (c_sym={cs:.2f}, c_flux={cf:.2f})"
              f" | fm_loss {fm_s} | grid {len(png)//1024} KiB", flush=True)
        try:
            api.upload_file(path_or_fileobj=path,
                            path_in_repo=f"samples/gmap/{name}.png",
                            repo_id=push_repo)
        except Exception as e:
            print(f"[gmap] grid upload failed (non-fatal): {e!r}")
    ckpt_vol.commit()
    cfg.c_sym, cfg.c_flux = 1.0, 1.0

    # ---- archive scan table ---------------------------------------------------
    result = {"base": base, "origin": origin, "variant": variant,
              "rays": rays, "spacing": spacing, "sample_steps": sample_steps,
              "cfg_scale": cfg_scale, "fm_batches": fm_batches,
              "paired": True, "scan": rows}
    jpath, cpath = "/cache/samples/gmap_scan.json", "/cache/samples/gmap_scan.csv"
    json.dump(result, open(jpath, "w"), indent=2)
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    try:
        for p, dest in [(jpath, f"probes/gmap_scan_{base}.json"),
                        (cpath, f"probes/gmap_scan_{base}.csv")]:
            api.upload_file(path_or_fileobj=p, path_in_repo=dest,
                            repo_id=push_repo)
        print(f"[gmap] scan table -> {push_repo}/probes/")
    except Exception as e:
        print(f"[gmap] table upload failed (non-fatal): {e!r}")
    return grids


@app.local_entrypoint()
def main(
    base: str = "standard",             # standard | amap
    ckpt_repo: str = "jcandane/AMAP",   # amap base only
    ckpt_step: str = "latest",          # amap base: latest | base | <int>
    weights: str = "ema",               # amap base: ema | model
    rays: str = "xi,fluxcut,symheat,fluxboost",
    spacing: float = 0.25,
    precision: str = "tf32",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
    fm_batches: int = 4,                # 0 disables the FM-loss column
    fm_bs: int = 64,
    latents_repo: str = "sparsetrace/dlatentzz",
    max_shards: int = 1,                # 1 shard is plenty for a paired column
    push_repo: str = "jcandane/AMAP",
):
    grids = run.remote(base, ckpt_repo, ckpt_step, weights, rays, spacing,
                       precision, sample_steps, cfg_scale, fm_batches, fm_bs,
                       latents_repo, max_shards, push_repo)
    from pathlib import Path
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    for tag, png in (grids or []):
        p = out_dir / f"{tag}.png"
        p.write_bytes(png)
        print(f"[gmap] wrote {p}")

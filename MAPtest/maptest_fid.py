"""
MAPtest/maptest_fid.py — FID across SiT-XL/2, AMAP, and DMAP (same size),
all sampled with the official SiT transport sampler against ONE shared
reference. This is the quantitative headline for the AMAP/DMAP study: the
expected ordering is FID(SiT) ≤ FID(AMAP) ≤ FID(DMAP), i.e. full attention ≥
symmetric+flux ≥ symmetric-only.

RUN (all three, one job so they share a reference):
    modal run MAPtest/maptest_fid.py \
        --models "sit,amap:jcandane/AMAP,dmap:jcandane/DMAP" \
        --num-samples 50000 --latents-repo sparsetrace/dlatentzz

=============================================================================
THREE THINGS THAT DECIDE WHETHER YOUR NUMBER MEANS ANYTHING  (unchanged truths)
=============================================================================
1. REFERENCE. ref_mode="latents" builds stats from YOUR VAE-decoded dlatentzz
   latents — self-consistent, factors out VAE error, IDEAL for SiT-vs-AMAP-vs-
   DMAP, but NOT comparable to any published SiT FID. To compare SiT to the
   paper's ~2.06, pass --ref-stats-url <ADM VIRTUAL_imagenet256_labeled.npz>
   and use cfg_scale=1.5, num_samples=50000.
2. SAMPLE COUNT. 50k is the convention and the bias only cancels between
   models at identical N. All three models here use the same N, sampler, cfg,
   seed policy, and reference — so the DIFFERENCES are trustworthy even if the
   absolute latents-mode numbers aren't publication figures.
3. EMA. AMAP/DMAP checkpoints are sampled from EMA by default (weights="ema");
   SiT uses the released 7M weights. Sampling raw understates a model.
=============================================================================
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MAPTEST_GPU", "H200")
TIMEOUT = int(os.environ.get("MAPTEST_SECONDS", str(8 * 3600)))  # 3 models x 50k is hours

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "timm==1.0.19", "numpy<2", "scipy>=1.11",
        "huggingface_hub==0.26.2", "safetensors==0.4.5",
        "diffusers==0.31.0", "accelerate==1.1.1",
        "pytorch-fid>=0.3.0", "pillow", "tqdm", "torchdiffeq==0.2.5",
    )
    .env({"HF_HOME": "/cache/hf"})
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT /root/SiT")
    # mount the repo so AMAP/ and DMAP/ helpers are importable (like modal_fid.py)
    .add_local_dir(
        REPO_ROOT, remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv",
                ".ruff_cache", ".pytest_cache", "**/samples/*", "**/logo/*"],
        copy=True,
    )
)

app = modal.App("maptest-fid", image=image)
ckpt_vol = modal.Volume.from_name("sit-ckpts", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("HF_TOKEN")
VAE_ID = "stabilityai/sd-vae-ft-ema"
VAE_SCALE = 0.18215


# =============================================================================
# ADAPTER — SiT / AMAP / DMAP. The only place this knows the codebase.
# =============================================================================
def _add_paths():
    import sys
    for p in ("/repo/AMAP", "/repo/DMAP", "/root/SiT"):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_model(kind: str, repo: str | None, step: str, weights: str, dev):
    """Return (model, label). kind in {sit, amap, dmap}.
    sit  -> released SiT-XL/2 7M, full attention (the upper bound).
    amap -> SiT + AMAP (coupled), weights from `repo`.
    dmap -> SiT + folded DMAP, weights from `repo`.
    """
    _add_paths()
    kind = kind.lower()
    if kind == "sit":
        import amap_common as A
        model = A.build_sit_xl2().to(dev).eval()
        return model, "sit-7M(full-attn)"
    if kind == "amap":
        import amap_common as A
        model, info = A.load_amap_checkpoint(dev, repo, step=step, weights=weights)
        return model.eval(), f"amap@{info['step']}-{info['weights']}"
    if kind == "dmap":
        import dmap_common as D
        model, info = D.load_dmap_checkpoint(dev, repo, step=step, weights=weights)
        return model.eval(), f"dmap@{info['step']}-{info['weights']}"
    raise ValueError(f"unknown model kind {kind!r} (expected sit|amap|dmap)")


def make_sampler(steps: int):
    """One official SiT transport ODE sampler, reused for all three models."""
    _add_paths()
    from transport import create_transport, Sampler
    return Sampler(create_transport("Linear", "velocity")).sample_ode(num_steps=steps)


def sample_batch(sample_fn, model, labels, cfg_scale: float, seed: int, dev):
    """[B,4,32,32] scaled latents via CFG transport sampling (SiT-standard)."""
    import torch
    B = labels.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(B, 4, 32, 32, generator=g).to(dev)
    y = labels.to(dev)
    z = torch.cat([z, z], 0)                                   # CFG duplicate
    y = torch.cat([y, torch.full((B,), 1000, device=dev)], 0)  # null class
    with torch.no_grad():
        out = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale)[-1]
    cond, _ = out.chunk(2, 0)
    return cond


def _parse_models(spec: str):
    """'sit,amap:jcandane/AMAP:latest:ema,dmap:jcandane/DMAP' -> list of
    (kind, repo|None, step, weights)."""
    out = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        p = item.split(":")
        kind = p[0].lower()
        repo = p[1] if len(p) > 1 and p[1] else None
        step = p[2] if len(p) > 2 and p[2] else "latest"
        weights = p[3] if len(p) > 3 and p[3] else "ema"
        if kind in ("amap", "dmap") and not repo:
            raise ValueError(f"model '{item}' needs a repo, e.g. {kind}:jcandane/{kind.upper()}")
        out.append((kind, repo, step, weights))
    return out


# =============================================================================
# reference-side helpers (reused from evaluation/modal_fid.py, generic)
# =============================================================================
def _load_latent_store(latents_repo: str, n: int, num_classes: int, seed: int):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(repo_id=latents_repo, repo_type="dataset"))
    shards = sorted(path.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {path} (repo_type='dataset'?)")
    print(f"[fid] latent store: {len(shards)} shard(s)")

    lat_parts, lab_parts = [], []
    for s in shards:
        d = load_file(str(s))
        lk = next((k for k in ("latents", "latent", "x", "data") if k in d), None)
        if lk is None:
            raise KeyError(f"{s.name}: no latent tensor; keys={list(d)}")
        lat_parts.append(d[lk])
        bk = next((k for k in ("labels", "label", "y", "classes") if k in d), None)
        if bk is not None:
            lab_parts.append(d[bk])
    lat = torch.cat(lat_parts) if len(lat_parts) > 1 else lat_parts[0]
    lab = (torch.cat(lab_parts) if len(lab_parts) > 1 else lab_parts[0]) if lab_parts else None
    print(f"[fid] {lat.shape[0]:,} latents, labels={'yes' if lab is not None else 'no'}")
    if lat.shape[0] < n:
        raise ValueError(f"store has {lat.shape[0]} latents, need {n}")

    g = torch.Generator().manual_seed(seed)
    if lab is not None:
        per = n // num_classes
        idx = []
        for c in range(num_classes):
            pool = (lab == c).nonzero(as_tuple=True)[0]
            if len(pool) < per:
                raise ValueError(f"class {c}: {len(pool)} available, need {per}")
            idx.append(pool[torch.randperm(len(pool), generator=g)[:per]])
        idx = torch.cat(idx)[torch.randperm(n, generator=g)]
        print(f"[fid] reference: class-balanced, {per}/class")
    else:
        idx = torch.randperm(lat.shape[0], generator=g)[:n]
        print("[fid] reference: uniform random (no labels -> not class-balanced)")
    return lat, idx


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    import numpy as np
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + off).dot(sigma2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _stats(features):
    import numpy as np
    return features.mean(axis=0), np.cov(features, rowvar=False)


@app.function(gpu=GPU_KIND, cpu=8.0, memory=32768, timeout=TIMEOUT,
              secrets=[HF_SECRET], volumes={"/cache": ckpt_vol})
def evaluate(models: str, num_samples: int, batch_size: int, sample_steps: int,
             cfg_scale: float, ref_stats_url: str, ref_mode: str,
             latents_repo: str, seed: int) -> str:
    if not ref_stats_url and ref_mode == "latents" and not latents_repo:
        raise ValueError("ref_mode='latents' needs --latents-repo, or pass --ref-stats-url")
    if num_samples % 1000 != 0:
        print(f"[fid] WARNING: num_samples={num_samples} not a multiple of 1000; "
              "classes can't be exactly balanced")
    specs = _parse_models(models)
    os.chdir("/cache")

    # --- ABI smoke test FIRST (torchvision C++ ops vs torch) ---
    import torch, torchvision
    print(f"[fid] torch {torch.__version__} / torchvision {torchvision.__version__}")
    torchvision.ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)
    print("[fid] torchvision C++ ops OK")

    import numpy as np
    from diffusers import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    from tqdm import tqdm

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(dev).eval()

    @torch.no_grad()
    def feats_from_latents(lat):
        if lat.dim() == 2:
            lat = lat.view(lat.shape[0], 4, 32, 32)
        img = vae.decode(lat / VAE_SCALE).sample
        img = (img.clamp(-1, 1) + 1.0) / 2.0
        return inception(img)[0].squeeze(-1).squeeze(-1).cpu().numpy()

    # ---- shared reference ----
    if ref_stats_url:
        import urllib.request
        print(f"[fid] downloading reference stats: {ref_stats_url}")
        urllib.request.urlretrieve(ref_stats_url, "/tmp/ref.npz")
        ref = np.load("/tmp/ref.npz")
        mu_ref, sigma_ref = ref["mu"], ref["sigma"]
        ref_desc = ref_stats_url
    else:
        print(f"[fid] reference from real latents (n={num_samples})")
        all_lat, idx = _load_latent_store(latents_repo, num_samples, 1000, seed)
        if all_lat.dim() == 2:                       # CHW vs HWC layout probe
            probe = all_lat[idx[:256]].float()
            def _nc(x):
                u = x[..., :, :-1].reshape(-1); v = x[..., :, 1:].reshape(-1)
                u = u - u.mean(); v = v - v.mean()
                return float(u @ v / (u.norm() * v.norm() + 1e-12))
            chw = _nc(probe.view(-1, 4, 32, 32))
            hwc = _nc(probe.view(-1, 32, 32, 4).permute(0, 3, 1, 2))
            print(f"[fid] layout probe: CHW={chw:.3f} HWC={hwc:.3f} -> "
                  f"{'CHW (correct)' if chw > hwc else 'HWC — FIX RESHAPE'}")
        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc="ref"):
            sl = idx[i:i + batch_size]
            feats[i:i + len(sl)] = feats_from_latents(
                all_lat[sl].to(device=dev, dtype=torch.float32))
        mu_ref, sigma_ref = _stats(feats)
        ref_desc = f"VAE-decoded latents from {latents_repo} (NOT publication-comparable)"
        del feats, all_lat
    ckpt_vol.commit()

    # ---- per-model generation ----
    sample_fn = make_sampler(sample_steps)
    results = {"config": {"num_samples": num_samples, "sample_steps": sample_steps,
                          "cfg_scale": cfg_scale, "seed": seed, "reference": ref_desc,
                          "gpu": GPU_KIND}, "fid": {}}

    for kind, repo, step, weights in specs:
        model, label = load_model(kind, repo, step, weights, dev)
        ckpt_vol.commit()
        print(f"[fid] === {label} ({kind}{':' + repo if repo else ''}) ===")

        torch.manual_seed(seed)
        labels_all = torch.arange(num_samples, device=dev) % 1000
        labels_all = labels_all[torch.randperm(num_samples, device=dev)]

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc=label):
            lab = labels_all[i:i + batch_size]
            lat = sample_batch(sample_fn, model, lab, cfg_scale,
                               seed * 1_000_003 + i, dev)
            feats[i:i + len(lab)] = feats_from_latents(lat)

        mu, sigma = _stats(feats)
        score = frechet_distance(mu, sigma, mu_ref, sigma_ref)
        results["fid"][label] = round(score, 4)
        print(f"[fid] {label}: FID = {score:.4f}")
        del model, feats
        torch.cuda.empty_cache()

    out = json.dumps(results, indent=2)
    print(out)
    return out


@app.local_entrypoint()
def main(
    models: str = "sit,amap:jcandane/AMAP,dmap:jcandane/DMAP",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,          # SiT best-FID guidance
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "sparsetrace/dlatentzz",
    seed: int = 0,
):
    payload = evaluate.remote(
        models=models, num_samples=num_samples, batch_size=batch_size,
        sample_steps=sample_steps, cfg_scale=cfg_scale, ref_stats_url=ref_stats_url,
        ref_mode=ref_mode, latents_repo=latents_repo, seed=seed)
    out = Path(__file__).parent / "fid_results.json"
    out.write_text(payload)
    print(f"[fid] wrote {out}")

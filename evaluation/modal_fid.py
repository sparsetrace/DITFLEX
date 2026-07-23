"""FID evaluation for the ditflex chains.

Streams: sample latents -> VAE decode -> InceptionV3 pool features ->
Frechet distance against reference statistics. Images are never written to
disk; only the 2048-d features are retained (50k x 2048 fp32 = 410 MB).

RUN:
    modal run evaluation/modal_fid.py --repos "sparsetrace/ditflex-L2-flow-dmap" \
        --num-samples 50000 --ref-stats-url <url-or-empty>

=============================================================================
THREE THINGS THAT DECIDE WHETHER YOUR NUMBER MEANS ANYTHING
=============================================================================

1. REFERENCE STATISTICS. FID is a distance to a reference distribution; the
   number is meaningless without saying which.
     * ref_stats_url = ADM's VIRTUAL_imagenet256_labeled.npz  -> comparable
       to published DiT/SiT numbers (they all use the ADM eval suite).
     * ref_mode = "latents"  -> statistics computed from YOUR OWN VAE-decoded
       latents. This measures the generative model only, factoring out VAE
       reconstruction error. Self-consistent and ideal for dmap-vs-amap, but
       NOT comparable to any published figure. Say which one you used.

2. SAMPLE COUNT. FID is biased upward at small N and the bias does not
   cancel between models unless N is identical. 50,000 is the convention.
   10,000 is defensible for internal comparison; 1,000 is a smoke test and
   should never be reported as "FID".

3. EMA WEIGHTS. Under the constant-LR recipe the raw weights orbit the
   minimum and the EMA sits in it. Sampling raw weights understates the
   model. See the ADAPTER section: confirm load_checkpoint() pulls EMA.
=============================================================================
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "H100")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

# 50k samples x 50 Euler steps x 2 (CFG) = 5e6 forward passes. Budget hours,
# not minutes, and size the timeout accordingly.
TIMEOUT = int(os.environ.get("MODAL_FID_SECONDS", "14400"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "accelerate>=0.34",     # else diffusers falls back to slow VAE loading
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "scipy>=1.11",
        "pytorch-fid>=0.3.0",   # canonical ported InceptionV3 weights
        "pillow",
        "tqdm",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv",
                ".ruff_cache", ".pytest_cache"],
        copy=True,   # install-time visibility; see modal_sample.py notes
    )
)

app = modal.App("ditflex-fid", image=image)

VAE_ID = "stabilityai/sd-vae-ft-ema"   # the DiT/SiT standard decoder
VAE_SCALE = 0.18215


# =============================================================================
# ADAPTER -- wire these two to sampling/sample.py. They are the only places
# this script needs to know your codebase, and they are deliberately isolated.
# =============================================================================

def load_checkpoint(repo: str, device):
    """Return (model, model_config) with EMA weights loaded, in eval mode.

    Mirror whatever sampling/sample.py does. The shape is:
        cfg  = ModelConfig(**checkpoint_config_dict)
        model = build_dmap_model(cfg) if cfg.qk_mode == "dmap" else build_model(cfg)
        model.load_state_dict(ema_state_dict)   # <-- EMA, not raw
    """
    import sys
    sys.path.insert(0, "/repo/src")
    from ditflex.config import ModelConfig                      # noqa: F401
    from ditflex.model import build_model                       # noqa: F401
    from ditflex.diffusion_model import build_dmap_model        # noqa: F401

    raise NotImplementedError(
        "Wire to sampling/sample.py's checkpoint loader. Confirm it selects "
        "the EMA shadow weights, not the raw ones."
    )


def sample_latents(model, labels, *, steps: int, cfg_scale: float, cfg, device):
    """Return a [B, 4, 32, 32] latent batch. Mirror sample.py's integrator.

    Flow/rectified-flow Euler, matching the `flow` objective:
        x = randn; for t in linspace(0,1,steps+1)[:-1]:
            v = model(x, t, labels)                       # velocity
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x + v * dt
    Use the SAME integrator and step count for every chain you compare, or
    the comparison measures the sampler rather than the model.
    """
    raise NotImplementedError("Wire to sampling/sample.py's sampler.")


# =============================================================================



def _load_latent_store(latents_repo: str, n: int, num_classes: int, seed: int):
    """Sample n latents (class-balanced when labels are present) from an HF store.

    Handles sharded stores: the previous version grabbed the FIRST shard via
    rglob(), which silently sampled a fraction of the dataset.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(repo_id=latents_repo, repo_type="dataset"))
    shards = sorted(path.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no .safetensors under {path} -- check latents_repo and repo_type "
            f"(this branch assumes repo_type='dataset')"
        )
    print(f"[fid] latent store: {len(shards)} shard(s)")

    lat_parts, lab_parts = [], []
    for s in shards:
        d = load_file(str(s))
        lk = next((k for k in ("latents", "latent", "x", "data") if k in d), None)
        if lk is None:
            raise KeyError(f"{s.name}: no latent tensor found; keys = {list(d)}")
        lat_parts.append(d[lk])
        bk = next((k for k in ("labels", "label", "y", "classes") if k in d), None)
        if bk is not None:
            lab_parts.append(d[bk])

    lat = torch.cat(lat_parts) if len(lat_parts) > 1 else lat_parts[0]
    lab = None
    if lab_parts:
        lab = torch.cat(lab_parts) if len(lab_parts) > 1 else lab_parts[0]
    print(f"[fid] {lat.shape[0]:,} latents, labels={'yes' if lab is not None else 'no'}")

    if lat.shape[0] < n:
        raise ValueError(f"store has {lat.shape[0]} latents, need {n}")

    g = torch.Generator().manual_seed(seed)
    if lab is not None:
        # Class-balanced, matching the generation side exactly.
        per = n // num_classes
        idx = []
        for c in range(num_classes):
            pool = (lab == c).nonzero(as_tuple=True)[0]
            if len(pool) < per:
                raise ValueError(f"class {c}: {len(pool)} available, need {per}")
            idx.append(pool[torch.randperm(len(pool), generator=g)[:per]])
        idx = torch.cat(idx)
        idx = idx[torch.randperm(len(idx), generator=g)]
        print(f"[fid] reference: class-balanced, {per}/class")
    else:
        idx = torch.randperm(lat.shape[0], generator=g)[:n]
        print("[fid] reference: uniform random (no labels found -- NOT class-balanced, "
              "which biases FID against the class-balanced generation side)")
    return lat, idx


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Standard FID formula (Heusel et al. 2017)."""
    import numpy as np
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # Numerically singular product; nudge both covariances.
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * np.trace(covmean))


def _stats(features):
    import numpy as np
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@app.function(
    gpu=GPU_KIND,
    cpu=8.0,
    memory=32768,   # latent store is ~10 GiB in CPU RAM during reference pass
    timeout=TIMEOUT,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def evaluate(
    repos: str = "sparsetrace/ditflex-L2-flow-dmap",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    seed: int = 0,
) -> str:
    import numpy as np
    import torch
    
    # Fail before spending GPU time on a misconfigured dispatch.
    if not ref_stats_url and ref_mode == "latents" and not latents_repo:
        raise ValueError(
            "ref_mode='latents' requires --latents-repo (the HF dataset repo "
            "holding the SD-VAE latents). Either set it, or pass --ref-stats-url "
            "pointing at a precomputed reference .npz."
        )
    if num_samples % 1000 != 0:
        print(f"[fid] WARNING: num_samples={num_samples} is not a multiple of 1000, "
              f"so classes cannot be exactly balanced.")


    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import AutoencoderKL
    from tqdm import tqdm
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    # --- ABI smoke test: forces torchvision's C++ extension to load ---------
    # torchvision built against a different torch than the CUDA wheel fails
    # here with "operator torchvision::nms does not exist" -- cheap to check
    # now, expensive to discover after an hour of sampling.
    import torchvision
    print(f"[fid] torch {torch.__version__} / torchvision {torchvision.__version__}")
    torchvision.ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)
    print("[fid] torchvision C++ ops OK")
    from pytorch_fid.inception import InceptionV3



    

    # ---- Inception (block 3 = 2048-d pool features, the FID standard) ----
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).eval()

    @torch.no_grad()
    def features_from_latents(lat):
        """[B,4,32,32] latents -> [B,2048] Inception pool features."""
        img = vae.decode(lat / VAE_SCALE).sample          # [-1, 1]
        img = (img.clamp(-1, 1) + 1.0) / 2.0              # [0, 1]
        # InceptionV3(resize_input=True, normalize_input=True) handles the
        # 299x299 bilinear resize and the [-1,1] rescale internally. Do NOT
        # pre-resize -- resize implementation is a known source of FID drift
        # between codebases (cf. clean-fid).
        return inception(img)[0].squeeze(-1).squeeze(-1).cpu().numpy()

    # ---- reference statistics -------------------------------------------
    if ref_stats_url:
        import urllib.request
        print(f"[fid] downloading reference stats: {ref_stats_url}")
        urllib.request.urlretrieve(ref_stats_url, "/tmp/ref.npz")
        ref = np.load("/tmp/ref.npz")
        mu_ref, sigma_ref = ref["mu"], ref["sigma"]
        ref_desc = ref_stats_url
    elif ref_mode == "latents":
        print(f"[fid] computing reference stats from real latents (n={num_samples})")
        all_lat, idx = _load_latent_store(latents_repo, num_samples, 1000, seed)

        # --- layout check: (C,H,W) vs (H,W,C) is silent if wrong ------------
        if all_lat.dim() == 2:
            probe = all_lat[idx[:256]].float()

            def _nbr_corr(x):                      # x: [B,C,H,W]
                u = x[..., :, :-1].reshape(-1)
                v = x[..., :, 1:].reshape(-1)
                u = u - u.mean(); v = v - v.mean()
                return float(u @ v / (u.norm() * v.norm() + 1e-12))

            chw = _nbr_corr(probe.view(-1, 4, 32, 32))
            hwc = _nbr_corr(probe.view(-1, 32, 32, 4).permute(0, 3, 1, 2))
            print(f"[fid] layout probe: neighbour corr  CHW={chw:.3f}  HWC={hwc:.3f}")
            print(f"[fid] -> latents are {'CHW (correct)' if chw > hwc else 'HWC -- FIX THE RESHAPE'}")
            print(f"[fid] per-channel mean {probe.view(-1,4,32,32).mean((0,2,3)).tolist()}")
            print(f"[fid] per-channel std  {probe.view(-1,4,32,32).std((0,2,3)).tolist()}")
        # ---------------------------------------------------------------------

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc="ref"):
            sl = idx[i : i + batch_size]
            lat = all_lat[sl].to(device=device, dtype=torch.float32)
            feats[i : i + len(sl)] = features_from_latents(lat)
        mu_ref, sigma_ref = _stats(feats)
        ref_desc = (f"VAE-decoded latents from {latents_repo} "
                    f"(same decoder as generation; NOT comparable to published FID)")
        del feats, all_lat
    else:
        raise ValueError("supply ref_stats_url, or ref_mode='latents' with latents_repo")

    # ---- per-chain generation --------------------------------------------
    results = {"config": {
        "num_samples": num_samples, "sample_steps": sample_steps,
        "cfg_scale": cfg_scale, "seed": seed, "reference": ref_desc,
        "gpu": GPU_KIND,
    }, "fid": {}}

    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        print(f"[fid] === {repo} ===")
        model, cfg = load_checkpoint(repo, device)

        torch.manual_seed(seed)
        # Class-balanced: exactly num_samples/1000 per class, then shuffled.
        labels_all = torch.arange(num_samples, device=device) % cfg.num_classes
        labels_all = labels_all[torch.randperm(num_samples, device=device)]

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc=repo.split("/")[-1]):
            lab = labels_all[i : i + batch_size]
            with torch.no_grad():
                lat = sample_latents(
                    model, lab, steps=sample_steps, cfg_scale=cfg_scale,
                    cfg=cfg, device=device,
                )
                feats[i : i + len(lab)] = features_from_latents(lat)

        mu, sigma = _stats(feats)
        score = frechet_distance(mu, sigma, mu_ref, sigma_ref)
        results["fid"][repo] = round(score, 4)
        print(f"[fid] {repo}: FID = {score:.4f}")

        del model, feats
        torch.cuda.empty_cache()

    out = json.dumps(results, indent=2)
    print(out)
    return out


@app.local_entrypoint()
def main(
    repos: str = "sparsetrace/ditflex-L2-flow-dmap",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    seed: int = 0,
):
    payload = evaluate.remote(
        repos=repos, num_samples=num_samples, batch_size=batch_size,
        sample_steps=sample_steps, cfg_scale=cfg_scale,
        ref_stats_url=ref_stats_url, ref_mode=ref_mode,
        latents_repo=latents_repo, seed=seed,
    )
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fid_results.json").write_text(payload)
    print("[fid] wrote evaluation/fid_results.json")

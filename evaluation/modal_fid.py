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
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
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
    """Return (model, ModelConfig) with EMA weights loaded, in eval mode.

    Filenames are discovered rather than assumed: the first run prints the
    repo's file list, so if the guesses below miss, the log tells you exactly
    what to put in CFG_NAMES / EMA_NAMES / RAW_NAMES.
    """
    import dataclasses
    import json
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    sys.path.insert(0, "/repo/src")
    from ditflex.config import ModelConfig
    from ditflex.diffusion_model import build_dmap_model
    from ditflex.model import build_model

    CFG_NAMES = ("config.json", "model_config.json", "ditflex_config.json")
    EMA_NAMES = ("ema.safetensors", "ema_model.safetensors", "model_ema.safetensors")
    RAW_NAMES = ("model.safetensors", "diffusion_pytorch_model.safetensors",
                 "pytorch_model.safetensors")

    path = Path(snapshot_download(repo_id=repo, repo_type="model"))
    listing = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    print(f"[fid] {repo} contains: {listing}")

    # ---- config ----
    cfg_file = next((path / n for n in CFG_NAMES if (path / n).exists()), None)
    if cfg_file is None:
        raise FileNotFoundError(
            f"no config json among {CFG_NAMES}; repo has {listing}. "
            f"Add the correct name to CFG_NAMES."
        )
    raw_cfg = json.loads(cfg_file.read_text())
    model_cfg = raw_cfg.get("model", raw_cfg)        # full Config or flat ModelConfig
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    dropped = set(model_cfg) - known
    if dropped:
        print(f"[fid] ignoring non-ModelConfig keys: {sorted(dropped)}")
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in known})
    print(f"[fid] cfg: qk_mode={cfg.qk_mode} qk_norm={cfg.qk_norm} "
          f"dmap_alpha={cfg.dmap_alpha} num_classes={cfg.num_classes}")

    # ---- weights: EMA strongly preferred ----
    wfile = next((path / n for n in EMA_NAMES if (path / n).exists()), None)
    used_ema = wfile is not None
    if wfile is None:
        wfile = next((path / n for n in RAW_NAMES if (path / n).exists()), None)
    if wfile is None:
        raise FileNotFoundError(
            f"no weights among {EMA_NAMES + RAW_NAMES}; repo has {listing}."
        )
    state = load_file(str(wfile))
    if not used_ema:                                  # EMA may live inside the same file
        if any(k.startswith("ema.") for k in state):
            state = {k[4:]: v for k, v in state.items() if k.startswith("ema.")}
            used_ema = True
    if used_ema:
        print(f"[fid] weights: {wfile.name} (EMA)")
    else:
        print(f"[fid] *** WARNING: {wfile.name} looks like RAW weights, not EMA.  ***")
        print("[fid] *** Under constant-LR the raw weights orbit the minimum while ***")
        print("[fid] *** the EMA sits in it; FID will be inflated. Check the repo.  ***")

    # ---- build + load ----
    model = build_dmap_model(cfg) if cfg.qk_mode == "dmap" else build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[fid] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"[fid]   missing[:5]    = {missing[:5]}")
        if unexpected:
            print(f"[fid]   unexpected[:5] = {unexpected[:5]}")
        if len(missing) > 10:
            raise RuntimeError("too many missing keys -- wrong weights file or config")
    return model.to(device).eval(), cfg


def sample_latents(model, labels, *, steps: int, cfg_scale: float, cfg,
                   device, objective: str, seed: int):
    """Return [B,4,32,32] scaled latents using the SAME sampler as sample.py.

    NOTE the per-batch `seed`. sample_flow/sample_ddim seed their initial noise
    from this argument (default FIXED_SEED), so calling them with a constant
    seed would draw every batch from the same noise tensors: diversity would
    collapse and FID would be badly inflated, with no error raised anywhere.
    """
    import sys

    sys.path.insert(0, "/repo/src")
    from ditflex.sample import sample_ddim, sample_flow

    sampler = sample_flow if objective == "flow" else sample_ddim
    return sampler(
        model, labels.cpu(),
        num_classes=cfg.num_classes,
        cfg_scale=cfg_scale,
        ode_steps=steps,
        seed=seed,
        device=device,
    )



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
    objective: str = "flow",
    seed: int = 0,
) -> str:
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

    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    from tqdm import tqdm

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

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
        "cfg_scale": cfg_scale, "seed": seed, "objective": objective,
        "reference": ref_desc,
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
                # Per-batch seed -- see the note in sample_latents().
                lat = sample_latents(
                    model, lab, steps=sample_steps, cfg_scale=cfg_scale,
                    cfg=cfg, device=device, objective=objective,
                    seed=seed * 1_000_003 + i,
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
    objective: str = "flow",
    seed: int = 0,
):
    payload = evaluate.remote(
        repos=repos, num_samples=num_samples, batch_size=batch_size,
        sample_steps=sample_steps, cfg_scale=cfg_scale,
        ref_stats_url=ref_stats_url, ref_mode=ref_mode,
        latents_repo=latents_repo, objective=objective, seed=seed,
    )
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fid_results.json").write_text(payload)
    print("[fid] wrote evaluation/fid_results.json")

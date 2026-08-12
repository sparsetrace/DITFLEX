"""FID evaluation for the ditflex chains.

Streams: sample latents -> VAE decode -> InceptionV3 pool features ->
Frechet distance against reference statistics. Images are never written to
disk; only the 2048-d features are retained (50k x 2048 fp32 = 410 MB).

RUN:
    modal run evaluation/modal_fid.py \
        --repos "org/sit-regular,org/sit-amap,org/sit-dmap" \
        --num-samples 50000 --bootstrap-reps 100 \
        --ref-stats-url <ADM-imagenet256-stats-url>

Output for each model includes the ordinary FID-50k plus bootstrap uncertainty:
    FID = 2.63 +/- 0.08 (95% bootstrap-normal CI)
The JSON also retains bootstrap standard error and percentile CI endpoints.

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
TIMEOUT = int(os.environ.get("MODAL_FID_SECONDS", "28800"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    # torchvision MUST come from the same index as torch: pytorch-fid pulls it
    # in, and a PyPI CPU wheel against a CUDA torch fails with
    # "operator torchvision::nms does not exist".
    .pip_install("torch", "torchvision", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "accelerate>=0.34",     # else diffusers falls back to slow VAE loading
        "diffusers>=0.31",
        # NOT transformers: diffusers only needs it for single-file loaders,
        # and importing it drags in transformers.AutoImageProcessor ->
        # torchvision.io, which is where the ABI mismatch explodes.
        # AutoencoderKL loads fine without it.
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "scipy>=1.11",
        "pytorch-fid>=0.3.0",   # canonical ported InceptionV3 weights
        "pillow",
        "tqdm",
        "torchdiffeq>=0.2.4",
        "timm>=1.0",
    )
    # Pre-cache the canonical pytorch-fid Inception weights during the image
    # build. Runtime downloads from GitHub are occasionally reset, which can
    # otherwise kill a long FID job before model evaluation starts.
    .run_commands(
        "mkdir -p /root/.cache/torch/hub/checkpoints && "
        "curl -fL --retry 12 --retry-delay 5 --retry-all-errors "
        "--connect-timeout 30 --max-time 900 "
        "-o /root/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth "
        "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth && "
        "test -s /root/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth"
    )
    .run_commands("git clone --depth 1 https://github.com/willisma/SiT.git /opt/SiT")
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

def _find_model_cfg(obj, known: set):
    """Depth-first search for the nested dict carrying the most ModelConfig
    fields. Checkpoint state files bury the model config at varying depths;
    this finds it without hardcoding a path."""
    best, best_score, best_path = None, 0, ""
    stack = [(obj, "")]
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            score = len(known & set(cur))
            if score > best_score:
                best, best_score, best_path = cur, score, path or "<root>"
            stack.extend((v, f"{path}.{k}" if path else k)
                         for k, v in cur.items() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend((v, f"{path}[]") for v in cur if isinstance(v, (dict, list)))
    return best, best_score, best_path


def _parse_repo_spec(spec: str):
    """Parse `repo`, `repo@latest`, or `repo@path/to/checkpoint`.

    `@latest` selects the numerically largest checkpoints/step_XXXXXXXX folder
    without assuming that the repository root itself contains the newest weights.
    """
    import re
    from huggingface_hub import HfApi

    if "@" not in spec:
        return spec, None
    repo_id, selector = spec.split("@", 1)
    selector = selector.strip().strip("/")
    if selector != "latest":
        return repo_id, selector or None

    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    steps = []
    for name in files:
        m = re.match(r"^checkpoints/(step_(\d+))(/|$)", name)
        if m:
            steps.append((int(m.group(2)), f"checkpoints/{m.group(1)}"))
    if not steps:
        print(f"[fid] {repo_id}@latest: no checkpoints/step_* folders; using repo root")
        return repo_id, None
    step, subfolder = max(steps)
    print(f"[fid] {repo_id}@latest -> {subfolder} (step={step:,})")
    return repo_id, subfolder


def _load_official_sit(device):
    """Load the official Xie et al. SiT-XL/2 ImageNet-256 checkpoint."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, "/opt/SiT")
    from download import find_model
    from models import SiT_models

    model = SiT_models["SiT-XL/2"](
        input_size=32,
        num_classes=1000,
        learn_sigma=True,
    ).to(device)
    state = find_model("SiT-XL-2-256x256.pt")
    model.load_state_dict(state, strict=True)
    model.eval()
    cfg = SimpleNamespace(num_classes=1000, qk_mode="regular", backend="official_sit")
    print("[fid] loaded official willisma/SiT SiT-XL/2 ImageNet-256 checkpoint")
    return model, cfg, "official_sit"


def load_checkpoint(repo: str, device):
    """Return (model, config-like object, backend) with evaluation weights loaded.

    Specs:
      * official-sit
      * jcandane/AMAP@latest
      * jcandane/DMAP@checkpoints/step_0080000
      * any ordinary ditflex Hugging Face repo
    """
    if repo.lower() in {"official-sit", "sit-official", "xie-sit", "sit-xl/2"}:
        return _load_official_sit(device)

    import dataclasses
    import json
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    sys.path.insert(0, "/repo/src")
    from ditflex.config import ModelConfig
    from ditflex.diffusion_model import build_dmap_model
    from ditflex.model import build_model

    repo_id, subfolder = _parse_repo_spec(repo)
    CFG_NAMES = (
        "state.json", "config.json", "model_config.json", "ditflex_config.json",
        "amap_config.json", "dmap_config.json",
    )
    EMA_NAMES = ("ema.safetensors", "ema_model.safetensors", "model_ema.safetensors")
    RAW_NAMES = ("model.safetensors", "diffusion_pytorch_model.safetensors", "pytorch_model.safetensors")

    allow_patterns = None
    if subfolder:
        allow_patterns = [f"{subfolder}/*"]
    path = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,
        ignore_patterns=["archive/*", "samples/*", "**/optim.safetensors"],
    ))
    base = path / subfolder if subfolder else path
    listing = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    print(f"[fid] {repo}: selected {base.relative_to(path) if base != path else '<root>'}")
    print(f"[fid] downloaded files: {listing}")

    cfg_file = next((base / n for n in CFG_NAMES if (base / n).exists()), None)
    if cfg_file is None:
        # Future DAC checkpoints may use another descriptive *_config.json name.
        # Restrict the fallback to the selected checkpoint directory so an
        # unrelated repository-level config cannot be loaded accidentally.
        config_candidates = sorted(base.glob("*_config.json"))
        if len(config_candidates) == 1:
            cfg_file = config_candidates[0]
            print(f"[fid] config fallback: {cfg_file.name}")
        elif len(config_candidates) > 1:
            raise FileNotFoundError(
                f"multiple *_config.json files in {base}; cannot choose safely: "
                f"{[x.name for x in config_candidates]}"
            )
        else:
            raise FileNotFoundError(
                f"no config json among {CFG_NAMES} (or unique *_config.json) "
                f"in selected checkpoint {base}; files={listing}"
            )
    raw_cfg = json.loads(cfg_file.read_text())
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    model_cfg, score, where = _find_model_cfg(raw_cfg, known)

    # `amap_config.json` / `dmap_config.json` are conversion metadata in the
    # structured checkpoints, not necessarily a serialized ModelConfig.  The
    # converted models retain the repository's canonical SiT backbone, so use
    # the dataclass defaults as that backbone and overlay every compatible
    # field supplied by the checkpoint metadata.  Generic configs with a real
    # ModelConfig-like block still take the original path below.
    structured_name = cfg_file.name.lower() in {"amap_config.json", "dmap_config.json"}
    if score >= 3:
        cfg_kwargs = {k: v for k, v in model_cfg.items() if k in known}
        cfg = ModelConfig(**cfg_kwargs)
        print(f"[fid] model config from {cfg_file.name}:{where} ({score}/{len(known)} fields)")
    elif structured_name:
        try:
            default_cfg = ModelConfig()
        except TypeError as exc:
            raise KeyError(
                f"{cfg_file.name} is structured conversion metadata ({score}/{len(known)} ModelConfig fields), "
                "but ModelConfig() has no usable defaults for reconstructing the SiT backbone. "
                f"Constructor error: {exc}"
            ) from exc

        cfg_kwargs = dataclasses.asdict(default_cfg)

        # Collect compatible fields anywhere in the metadata.  This is more
        # permissive than requiring a particular nesting convention, while the
        # state-dict checks below guard against silently constructing the wrong
        # architecture.
        found = {}
        stack = [raw_cfg]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in known and not isinstance(v, (dict, list)):
                        found[k] = v
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(obj, list):
                stack.extend(v for v in obj if isinstance(v, (dict, list)))
        cfg_kwargs.update(found)

        # The filename is authoritative for the structured attention family.
        # AMAP uses the ordinary builder with qk_mode='amap'; DMAP uses its
        # dedicated tied-Q/K builder.
        if "qk_mode" in known:
            cfg_kwargs["qk_mode"] = "amap" if cfg_file.name.lower().startswith("amap_") else "dmap"
        cfg = ModelConfig(**{k: v for k, v in cfg_kwargs.items() if k in known})
        print(
            f"[fid] {cfg_file.name}: structured metadata ({score}/{len(known)} direct ModelConfig fields); "
            f"using canonical ModelConfig defaults + {len(found)} compatible checkpoint override(s)"
        )
    else:
        raise KeyError(
            f"{cfg_file.name}: no ModelConfig-like block found (best match {score}/{len(known)})"
        )

    print(f"[fid] cfg: qk_mode={getattr(cfg, 'qk_mode', None)} num_classes={cfg.num_classes}")

    wfile = next((base / n for n in EMA_NAMES if (base / n).exists()), None)
    used_ema = wfile is not None
    if wfile is None:
        wfile = next((base / n for n in RAW_NAMES if (base / n).exists()), None)
    if wfile is None:
        raise FileNotFoundError(f"no evaluation weights among {EMA_NAMES + RAW_NAMES} in {base}")
    state = load_file(str(wfile))
    if not used_ema and any(k.startswith("ema.") for k in state):
        state = {k[4:]: v for k, v in state.items() if k.startswith("ema.")}
        used_ema = True
    print(f"[fid] weights: {wfile.relative_to(path)} ({'EMA' if used_ema else 'RAW'})")
    if not used_ema:
        print("[fid] WARNING: checkpoint does not look like EMA weights; verify before paper reporting")

    qk_mode = str(getattr(cfg, "qk_mode", "regular")).lower()
    if qk_mode == "dmap":
        model = build_dmap_model(cfg)
        builder = "build_dmap_model"
    else:
        model = build_model(cfg)
        builder = "build_model"
    print(f"[fid] architecture dispatch: qk_mode={qk_mode!r} -> {builder}")
    missing, unexpected = model.load_state_dict(state, strict=False)

    tied_k = [k for k in missing if ".to_k." in k]
    other_missing = [k for k in missing if ".to_k." not in k]
    if tied_k and qk_mode != "dmap":
        raise RuntimeError(
            f"{len(tied_k)} to_k tensors missing but qk_mode={qk_mode!r}; wrong checkpoint/config?"
        )
    if tied_k:
        print(f"[fid] {len(tied_k)} to_k tensors absent -- expected for DMAP")
    if other_missing or unexpected:
        print(f"[fid] load_state_dict: {len(other_missing)} other missing, {len(unexpected)} unexpected")
        if other_missing:
            print(f"[fid] missing[:8] = {other_missing[:8]}")
        if unexpected:
            print(f"[fid] unexpected[:8] = {unexpected[:8]}")
        if len(other_missing) > 10 or len(unexpected) > 10:
            raise RuntimeError(
                "too many state-dict mismatches -- reconstructed backbone/config does not match checkpoint"
            )

    return model.to(device).eval(), cfg, "ditflex"


def sample_latents(model, labels, *, steps: int, cfg_scale: float, cfg,
                   device, objective: str, seed: int, backend: str):
    """Return [B,4,32,32] scaled latents under a common 50-step Euler/CFG protocol."""
    import torch
    import sys

    if backend == "official_sit":
        if objective != "flow":
            raise ValueError("official-sit adapter currently supports objective='flow' only")
        sys.path.insert(0, "/opt/SiT")
        from transport import create_transport, Sampler

        # Official SiT defaults: Linear path, velocity prediction.  We choose
        # Euler with `steps` so the baseline matches the custom chains' step count.
        transport = create_transport("Linear", "velocity", None, None, None)
        sample_fn = Sampler(transport).sample_ode(
            sampling_method="euler",
            num_steps=steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
        )
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn((len(labels), 4, 32, 32), generator=g, device=device)
        y = labels.to(device)
        z = torch.cat([z, z], dim=0)
        y_null = torch.full_like(y, 1000)
        y_cfg = torch.cat([y, y_null], dim=0)
        out = sample_fn(z, model.forward_with_cfg, y=y_cfg, cfg_scale=cfg_scale)[-1]
        out, _ = out.chunk(2, dim=0)
        return out

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
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
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


def bootstrap_fid(features, mu_ref, sigma_ref, *, observed_score: float, reps: int, seed: int,
                  ref_features=None, confidence: float = 0.95):
    """Bootstrap sampling uncertainty of an FID estimate.

    If ref_features is None, the reference statistics are treated as fixed
    (the right choice for published/precomputed ADM ImageNet statistics).
    If ref_features is supplied, both generated and reference samples are
    independently bootstrapped (the right choice for ref_mode='latents').

    Returns the bootstrap standard error, a symmetric normal-approximation
    half-width (z * SE), and a RECENTERED percentile interval around the
    observed FID. Raw percentile bootstrap FIDs are upward-shifted because FID
    itself is a biased finite-sample plug-in estimator and bootstrap samples
    contain duplicates. Their spread is useful for uncertainty; their raw
    location is not a CI for the observed FID. The raw bootstrap mean/bias are
    retained as diagnostics.
    """
    import numpy as np

    if reps <= 0:
        return None
    if reps < 20:
        print(f"[fid] WARNING: bootstrap_reps={reps} is too small for a stable 95% CI")

    rng = np.random.default_rng(seed)
    n = features.shape[0]
    nr = None if ref_features is None else ref_features.shape[0]
    values = np.empty(reps, dtype=np.float64)

    for b in range(reps):
        ig = rng.integers(0, n, size=n)
        mu_g, sigma_g = _stats(features[ig])
        if ref_features is None:
            mu_r, sigma_r = mu_ref, sigma_ref
        else:
            ir = rng.integers(0, nr, size=nr)
            mu_r, sigma_r = _stats(ref_features[ir])
        values[b] = frechet_distance(mu_g, sigma_g, mu_r, sigma_r)
        if (b + 1) % max(1, reps // 10) == 0 or b + 1 == reps:
            print(f"[fid] bootstrap {b + 1}/{reps}")

    alpha = 1.0 - confidence
    boot_mean = float(values.mean())
    boot_bias = boot_mean - float(observed_score)
    centered = values - boot_mean
    qlo, qhi = np.quantile(centered, [alpha / 2.0, 1.0 - alpha / 2.0])
    lo = float(observed_score + qlo)
    hi = float(observed_score + qhi)
    se = values.std(ddof=1) if reps > 1 else 0.0
    # 1.95996 is the two-sided 95% standard-normal quantile.
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else None
    if z is None:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
    return {
        "reps": int(reps),
        "confidence": float(confidence),
        "std_error": float(se),
        "normal_half_width": float(z * se),
        "centered_percentile_low": lo,
        "centered_percentile_high": hi,
        "bootstrap_mean_raw": boot_mean,
        "bootstrap_bias_raw": float(boot_bias),
    }


@app.function(
    gpu=GPU_KIND,
    cpu=8.0,
    memory=32768,   # latent store is ~10 GiB in CPU RAM during reference pass
    timeout=TIMEOUT,
    secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
)
def evaluate(
    repos: str = "jcandane/AMAP@latest,jcandane/DMAP@checkpoints/step_0080000",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    objective: str = "flow",
    seed: int = 0,
    bootstrap_reps: int = 100,
    inference_precision: str = "bf16",
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

    # --- ABI smoke test FIRST ----------------------------------------------
    # torchvision built against a different torch than the CUDA wheel fails
    # with "operator torchvision::nms does not exist". Both diffusers (via
    # transformers.AutoImageProcessor) and pytorch_fid import torchvision, so
    # this MUST run before either or the failure surfaces as an opaque
    # "Could not import module 'AutoImageProcessor'" from inside their lazy
    # import machinery.
    import torch
    import torchvision

    print(f"[fid] torch {torch.__version__} / torchvision {torchvision.__version__}")
    torchvision.ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)
    print("[fid] torchvision C++ ops OK")
    # -----------------------------------------------------------------------

    import numpy as np
    from diffusers import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    from tqdm import tqdm

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    precision = inference_precision.lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("inference_precision must be one of: bf16, fp16, fp32")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    use_amp = amp_dtype is not None
    print(f"[fid] model inference precision: {precision} "
          f"({'autocast' if use_amp else 'full fp32'})")

    # ---- Inception (block 3 = 2048-d pool features, the FID standard) ----
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).eval()

    @torch.no_grad()
    def features_from_latents(lat):
        """latents -> [B,2048] Inception pool features.

        Accepts flat [B,4096] (how the store keeps them) or shaped
        [B,4,32,32] (what the sampler returns). Layout is channel-major,
        confirmed by the neighbour-correlation probe below.
        """
        if lat.dim() == 2:
            lat = lat.view(lat.shape[0], 4, 32, 32)
        img = vae.decode(lat / VAE_SCALE).sample          # [-1, 1]
        img = (img.clamp(-1, 1) + 1.0) / 2.0              # [0, 1]
        # InceptionV3(resize_input=True, normalize_input=True) handles the
        # 299x299 bilinear resize and the [-1,1] rescale internally. Do NOT
        # pre-resize -- resize implementation is a known source of FID drift
        # between codebases (cf. clean-fid).
        return inception(img)[0].squeeze(-1).squeeze(-1).cpu().numpy()

    # ---- reference statistics -------------------------------------------
    # Keep reference features only when they are estimated from a finite
    # latent sample; this lets the bootstrap include reference-side sampling
    # uncertainty.  Precomputed ADM statistics are treated as fixed.
    ref_features = None
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

        # --- layout probe: (C,H,W) vs (H,W,C) is silent if wrong ------------
        # Images are spatially smooth, channels are not, so the correct
        # reshape has the higher horizontal-neighbour correlation. A wrong
        # layout decodes without error and yields meaningless FID.
        if all_lat.dim() == 2:
            probe = all_lat[idx[:256]].float()

            def _nbr_corr(x):                          # x: [B,C,H,W]
                u = x[..., :, :-1].reshape(-1)
                v = x[..., :, 1:].reshape(-1)
                u = u - u.mean()
                v = v - v.mean()
                return float(u @ v / (u.norm() * v.norm() + 1e-12))

            chw = _nbr_corr(probe.view(-1, 4, 32, 32))
            hwc = _nbr_corr(probe.view(-1, 32, 32, 4).permute(0, 3, 1, 2))
            print(f"[fid] layout probe: neighbour corr  CHW={chw:.3f}  HWC={hwc:.3f}")
            print(f"[fid] -> latents are "
                  f"{'CHW (correct)' if chw > hwc else 'HWC -- FIX THE RESHAPE'}")
            shaped = probe.view(-1, 4, 32, 32)
            print(f"[fid] per-channel mean {shaped.mean((0, 2, 3)).tolist()}")
            print(f"[fid] per-channel std  {shaped.std((0, 2, 3)).tolist()}")
            print("[fid] std ~1 => latents are pre-scaled, so dividing by "
                  f"{VAE_SCALE} before decode is correct")
        # ---------------------------------------------------------------------

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc="ref"):
            sl = idx[i : i + batch_size]
            lat = all_lat[sl].to(device=device, dtype=torch.float32)
            feats[i : i + len(sl)] = features_from_latents(lat)
        mu_ref, sigma_ref = _stats(feats)
        ref_features = feats
        ref_desc = (f"VAE-decoded latents from {latents_repo} "
                    f"(same decoder as generation; NOT comparable to published FID)")
        del all_lat
    else:
        raise ValueError("supply ref_stats_url, or ref_mode='latents' with latents_repo")

    # ---- per-chain generation --------------------------------------------
    results = {"config": {
        "num_samples": num_samples, "sample_steps": sample_steps,
        "cfg_scale": cfg_scale, "seed": seed, "objective": objective,
        "reference": ref_desc,
        "gpu": GPU_KIND,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_confidence": 0.95,
        "inference_precision": precision,
    }, "fid": {}}

    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        print(f"[fid] === {repo} ===")
        model, cfg, backend = load_checkpoint(repo, device)

        torch.manual_seed(seed)
        # Class-balanced: exactly num_samples/1000 per class, then shuffled.
        labels_all = torch.arange(num_samples, device=device) % cfg.num_classes
        labels_all = labels_all[torch.randperm(num_samples, device=device)]

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc=repo.split("/")[-1]):
            lab = labels_all[i : i + batch_size]
            with torch.inference_mode():
                # Run only the generative model under AMP.  Keep VAE/Inception/FID
                # feature extraction in FP32 for a stable, standard FID pipeline.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    lat = sample_latents(
                        model, lab, steps=sample_steps, cfg_scale=cfg_scale,
                        cfg=cfg, device=device, objective=objective,
                        seed=seed * 1_000_003 + i, backend=backend,
                    )
                lat = lat.float()
                feats[i : i + len(lab)] = features_from_latents(lat)

        mu, sigma = _stats(feats)
        score = frechet_distance(mu, sigma, mu_ref, sigma_ref)
        boot = bootstrap_fid(
            feats, mu_ref, sigma_ref, observed_score=score, reps=bootstrap_reps,
            seed=seed + 17_171, ref_features=ref_features, confidence=0.95,
        )
        entry = {"score": round(score, 4), "num_samples": num_samples}
        if boot is not None:
            entry["bootstrap"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in boot.items()}
            pm = boot["normal_half_width"]
            print(f"[fid] {repo}: FID = {score:.4f} +/- {pm:.4f} "
                  f"(95% bootstrap-normal CI)")
            print(f"[fid] {repo}: centered-percentile 95% bootstrap CI = "
                  f"[{boot['centered_percentile_low']:.4f}, {boot['centered_percentile_high']:.4f}]")
            print(f"[fid] {repo}: raw bootstrap mean = {boot['bootstrap_mean_raw']:.4f} "
                  f"(plug-in bootstrap bias diagnostic = {boot['bootstrap_bias_raw']:+.4f})")
        else:
            print(f"[fid] {repo}: FID = {score:.4f} (bootstrap disabled)")
        results["fid"][repo] = entry

        del model, feats
        torch.cuda.empty_cache()

    out = json.dumps(results, indent=2)
    print(out)
    return out


@app.local_entrypoint()
def main(
    repos: str = "jcandane/AMAP@latest,jcandane/DMAP@checkpoints/step_0080000",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    objective: str = "flow",
    seed: int = 0,
    bootstrap_reps: int = 100,
    inference_precision: str = "bf16",
):
    payload = evaluate.remote(
        repos=repos, num_samples=num_samples, batch_size=batch_size,
        sample_steps=sample_steps, cfg_scale=cfg_scale,
        ref_stats_url=ref_stats_url, ref_mode=ref_mode,
        latents_repo=latents_repo, objective=objective, seed=seed,
        bootstrap_reps=bootstrap_reps, inference_precision=inference_precision,
    )
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fid_results.json").write_text(payload)
    print("[fid] wrote evaluation/fid_results.json")

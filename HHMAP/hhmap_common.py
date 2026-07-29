"""
Shared HHMAP helpers used by both HHMAP.py (finetune) and sample_hhmap.py.

Pure library code — no `modal` import; heavy imports are inside functions so it
ships via add_local_python_source and imports on the GitHub runner without torch.

Checkpoint layout in the Hub repo (e.g. jcandane/HHMAP):

    checkpoints/step_0010000/model.safetensors    # raw finetuned weights (only W_D moved)
    checkpoints/step_0010000/ema.safetensors      # EMA (full state_dict)
    checkpoints/step_0010000/hhmap_config.json    # step + α window + flags + base
    samples/hhmap_*.png
"""

from __future__ import annotations

SIT_CKPT = "SiT-XL-2-256x256.pt"   # official 7M-step SiT-XL/2 (find_model)

FIXED_CLASSES = [207, 360, 387, 974, 88, 979, 417, 279,
                 972, 483, 21, 562, 933, 724, 985, 812]
FIXED_SEED = 1234


def sit_path():
    import sys
    if "/root/SiT" not in sys.path:
        sys.path.insert(0, "/root/SiT")


def build_sit_xl2():
    """SiT-XL/2 with the official architecture, base 7M weights loaded."""
    sit_path()
    from models import SiT_XL_2
    from download import find_model

    model = SiT_XL_2(input_size=32, in_channels=4)   # learn_sigma=True
    state = find_model(SIT_CKPT)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


def make_hhmap_config(cfg_dict: dict):
    """HHMAPConfig from a plain dict (checkpoint's hhmap_config.json)."""
    from hhmap_attention import HHMAPConfig
    return HHMAPConfig(
        qk_rmsnorm=bool(cfg_dict.get("qk_rmsnorm", False)),
        learn_logit_scale=bool(cfg_dict.get("learn_logit_scale", False)),
        wd_rank=int(cfg_dict.get("wd_rank", 0)),
    )


class EMA:
    """EMA over the FULL state_dict (params + buffers), so ema.safetensors is a
    complete, strict-loadable checkpoint."""

    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        import torch
        with torch.no_grad():
            for k, v in model.state_dict().items():
                s = self.shadow[k]
                if v.dtype.is_floating_point:
                    s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


def _batch_seed(base_seed: int, global_step: int, rank: int) -> int:
    return (((base_seed + 1) * 1_000_000_007 + global_step) * 8192 + rank) % (2**63 - 1)


class LatentStore:
    """Compact port of ditflex.latents.LatentStore for the ephemeral finetune."""

    def __init__(self, latents, labels, latent_shape=(4, 32, 32), num_classes=1000):
        self.latents = latents
        self.labels = labels.long()
        self.latent_shape = tuple(latent_shape)
        sub = latents[: min(len(latents), 8192)].float()
        std = sub.std().item()
        assert 0.7 < std < 1.4, (
            f"latent std {std:.4f}: expected ~1.0 (0.18215 applied exactly once)")
        lo, hi = int(self.labels.min()), int(self.labels.max())
        assert 0 <= lo and hi < num_classes, f"labels [{lo},{hi}] vs num_classes {num_classes}"

    def __len__(self):
        return self.latents.shape[0]

    def batch(self, global_step, rank, batch_size, base_seed=0):
        import torch
        g = torch.Generator().manual_seed(_batch_seed(base_seed, global_step, rank))
        idx = torch.randint(0, len(self), (batch_size,), generator=g).to(self.latents.device)
        return self.latents[idx].view(-1, *self.latent_shape).float(), self.labels[idx]

    @classmethod
    def from_hub(cls, repo_id="sparsetrace/dlatentzz", device="cuda",
                 max_files=None, num_classes=1000):
        import torch
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors import safe_open

        files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset")
                       if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors in {repo_id}")
        files = files[:max_files] if max_files else files
        lat, lab = [], []
        for f in files:
            p = hf_hub_download(repo_id, f, repo_type="dataset")
            with safe_open(p, framework="pt", device="cpu") as sf:
                lat.append(sf.get_tensor("latents"))
                lab.append(sf.get_tensor("labels"))
        latents = torch.cat(lat, 0).to(device)
        labels = torch.cat(lab, 0).to(device)
        return cls(latents, labels, num_classes=num_classes)


def sample_grid(model, dev, out_path, num_steps=50, cfg_scale=4.0, amp=None):
    """Official SiT ODE sample + SD-VAE (ft-ema) decode -> 4x4 grid PNG."""
    import contextlib
    sit_path()
    import numpy as np
    import torch
    from transport import create_transport, Sampler
    from diffusers.models import AutoencoderKL
    from PIL import Image

    amp = amp or contextlib.nullcontext()
    transport = create_transport("Linear", "velocity")
    sample_fn = Sampler(transport).sample_ode(num_steps=num_steps)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev)

    n = len(FIXED_CLASSES)
    g = torch.Generator(device="cpu").manual_seed(FIXED_SEED)
    z = torch.randn(n, 4, 32, 32, generator=g).to(dev)
    y = torch.tensor(FIXED_CLASSES, device=dev)
    z = torch.cat([z, z], 0)
    y = torch.cat([y, torch.full((n,), 1000, device=dev)], 0)

    was_training = model.training
    model.eval()
    with torch.no_grad(), amp:
        samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale)[-1]
        samples, _ = samples.chunk(2, dim=0)
        imgs = vae.decode(samples / 0.18215).sample
    if was_training:
        model.train()

    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    side = int(n ** 0.5)
    px = imgs.shape[1]
    grid = np.zeros((side * px, side * px, 3), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, side)
        grid[r * px:(r + 1) * px, c * px:(c + 1) * px] = imgs[k]
    Image.fromarray(grid).save(out_path)
    with open(out_path, "rb") as f:
        return out_path, f.read()


def latest_checkpoint_step(repo: str) -> int | None:
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo)
    except Exception:
        return None
    found = []
    for f in files:
        if f.startswith("checkpoints/step_"):
            try:
                found.append(int(f.split("/")[1].split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(found) if found else None


def fetch_checkpoint(repo: str, step: int):
    """Download a checkpoint's config + weights. Returns (cfg_dict, model_sd, ema_sd)."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    folder = f"checkpoints/step_{step:07d}"
    cfg = {}
    for name in ("hhmap_config.json", "hmap_config.json", "amap_config.json",
                 "config.json"):
        try:
            cfg = json.load(open(hf_hub_download(repo, f"{folder}/{name}")))
            break
        except Exception:
            continue
    model_sd = load_file(hf_hub_download(repo, f"{folder}/model.safetensors"))
    ema_sd = load_file(hf_hub_download(repo, f"{folder}/ema.safetensors"))
    return cfg, model_sd, ema_sd


def resolve_checkpoint_step(repo: str, step) -> int | None:
    if step in (None, "base", ""):
        return None
    if step == "latest":
        s = latest_checkpoint_step(repo)
        if s is None:
            raise FileNotFoundError(f"{repo} has no checkpoints/step_* folders")
        return s
    return int(step)


def load_hhmap_checkpoint(dev, repo: str, step="latest", weights: str = "ema"):
    """Build SiT-XL/2 + HHMAP and load a finetuned checkpoint (or base+HHMAP if
    step is 'base'/None). weights: 'ema' or 'model'. Returns (model, info)."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from hhmap_attention import apply_hhmap, HHMAPConfig, set_alpha

    resolved = resolve_checkpoint_step(repo, step)
    model = build_sit_xl2().to(dev)

    if resolved is None:
        apply_hhmap(model, HHMAPConfig(), alpha=0.0)   # α=0 = AMAP endpoint
        return model, {"repo": repo, "step": "base", "weights": "base", "alpha": 0.0}

    folder = f"checkpoints/step_{resolved:07d}"
    cfg_dict = json.load(open(hf_hub_download(repo, f"{folder}/hhmap_config.json")))
    apply_hhmap(model, make_hhmap_config(cfg_dict))   # adds wd_proj/_wd_lambda before load

    fname = "ema.safetensors" if weights == "ema" else "model.safetensors"
    sd = load_file(hf_hub_download(repo, f"{folder}/{fname}"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    a = float(cfg_dict.get("alpha", 1.0))
    set_alpha(model, a)
    model = model.to(dev)
    return model, {"repo": repo, "step": resolved, "weights": weights,
                   "alpha": a, "config": cfg_dict}

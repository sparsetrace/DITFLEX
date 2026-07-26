"""
Shared AMAP helpers used by both AMAP.py (finetune) and sample_amap.py (sampler).

No `modal` import here — this is pure library code so it ships cleanly via
add_local_python_source and imports on the GitHub runner without torch (all
heavy imports are inside functions).

Checkpoint layout in the Hub repo (e.g. sparsetrace/AMAP):

    checkpoints/step_0010000/model.safetensors   # raw finetuned weights
    checkpoints/step_0010000/ema.safetensors     # EMA (full state_dict)
    checkpoints/step_0010000/amap_config.json     # step + AMAP flags + base
    samples/amap_*.png
"""

from __future__ import annotations

SIT_CKPT = "SiT-XL-2-256x256.pt"   # official 7M-step SiT-XL/2 (find_model)

# Fixed classes/seed identical to ditflex sampling/modal_sample.py so AMAP
# grids are directly comparable to the chain grids (4x4, same noise).
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
    state = find_model(SIT_CKPT)                      # -> ./pretrained_models
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


def make_amap_config(cfg_dict: dict):
    """AMAPConfig from a plain dict (checkpoint's amap_config.json)."""
    from amap_attention import AMAPConfig
    return AMAPConfig(
        qk_rmsnorm=bool(cfg_dict.get("qk_rmsnorm", False)),
        learn_logit_scale=bool(cfg_dict.get("learn_logit_scale", False)),
    )


class EMA:
    """Exponential moving average over the FULL state_dict (params + buffers),
    so ema.safetensors is a complete, strict-loadable checkpoint."""

    def __init__(self, model, decay: float = 0.9999):
        import torch
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


def load_latents(repo: str, max_shards: int | None = None):
    """dlatentzz-style latents: shards [N,4096] bf16, 0.18215 applied.
    Returns (latents [M,4,32,32] float, labels [M] long or None)."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    api = HfApi()
    all_files = api.list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in all_files if f.endswith(".safetensors")
                    and "label" not in f.lower())
    shards = shards[: max_shards or len(shards)]
    chunks = []
    for f in shards:
        d = load_file(hf_hub_download(repo, f, repo_type="dataset"))
        chunks.append((next(iter(d.values())) if len(d) == 1 else d[sorted(d)[0]]).float())
    x = torch.cat(chunks, 0)
    std = x.std().item()
    assert 0.7 < std < 1.4, f"latent std {std:.3f}: expected ≈1.0 (0.18215 applied)"
    lat = x.reshape(-1, 4, 32, 32)

    labels = None
    label_files = [f for f in all_files if "label" in f.lower() and f.endswith(".safetensors")]
    if label_files:
        d = load_file(hf_hub_download(repo, sorted(label_files)[0], repo_type="dataset"))
        labels = (next(iter(d.values())) if len(d) == 1 else d[sorted(d)[0]]).long()[: lat.shape[0]]
    return lat, labels


def sample_grid(model, dev, out_path, num_steps=50, cfg_scale=4.0, amp=None):
    """Official SiT ODE sample + SD-VAE (ft-ema) decode -> 4x4 grid PNG.
    Matches ditflex modal_sample.py output. Returns (out_path, png_bytes)."""
    import contextlib
    import io
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
    z = torch.cat([z, z], 0)                                   # CFG duplicate
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


def resolve_checkpoint_step(repo: str, step) -> int | None:
    """Return an int step for a repo checkpoint. step may be an int, a numeric
    string, 'latest', or 'base'. 'base'/None -> None (un-finetuned base+AMAP)."""
    if step in (None, "base", ""):
        return None
    if step == "latest":
        from huggingface_hub import HfApi
        api = HfApi()
        found = []
        for f in api.list_repo_files(repo):
            if f.startswith("checkpoints/step_"):
                try:
                    found.append(int(f.split("/")[1].split("_")[1]))
                except (IndexError, ValueError):
                    pass
        if not found:
            raise FileNotFoundError(f"{repo} has no checkpoints/step_* folders")
        return max(found)
    return int(step)


def load_amap_checkpoint(dev, repo: str, step="latest", weights: str = "ema"):
    """Build SiT-XL/2 + AMAP and load a finetuned checkpoint (or base+AMAP if
    step is 'base'/None). weights: 'ema' or 'model'. Returns (model, info)."""
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from amap_attention import apply_amap, AMAPConfig

    resolved = resolve_checkpoint_step(repo, step)
    model = build_sit_xl2().to(dev)

    if resolved is None:
        apply_amap(model, AMAPConfig())
        return model, {"repo": repo, "step": "base", "weights": "base"}

    folder = f"checkpoints/step_{resolved:07d}"
    cfg_dict = json.load(open(hf_hub_download(repo, f"{folder}/amap_config.json")))
    apply_amap(model, make_amap_config(cfg_dict))   # before load (adds params if any)

    fname = "ema.safetensors" if weights == "ema" else "model.safetensors"
    sd = load_file(hf_hub_download(repo, f"{folder}/{fname}"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model = model.to(dev)
    return model, {"repo": repo, "step": resolved, "weights": weights, "config": cfg_dict}

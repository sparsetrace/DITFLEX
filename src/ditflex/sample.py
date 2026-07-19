"""src/ditflex/sample.py -- generate images from a trained checkpoint.

Called by train.py after the final save+push of every chain link (and
usable standalone). Uses a FIXED seed and FIXED class set, so the
samples/ folder on the checkpoint repo becomes a time-lapse: the same 16
noise tensors, decoded at every link, sharpening as the chain grows.

Samplers:
  flow: Euler integration of the learned velocity field from t=1 (noise)
        to t=0 (data), matching the training interpolant
        x_t = (1-t) x0 + t eps  =>  dx/dt = v = eps - x0.
        Timesteps passed as t * 1000.0 (float), exactly as trained.
  ddpm: deterministic DDIM on eps-prediction over the same linear-beta
        schedule the objective trains against.

Both use classifier-free guidance via the null class (index num_classes),
with cond/uncond batched into one forward. Decode goes through the SAME
VAE that encoded the dataset (stabilityai/sd-vae-ft-ema), dividing by the
0.18215 the encoder multiplied.

Runs on the raw (uncompiled) model in eager mode: ~100 forwards of
DiT-L/2 at batch 32 is well under a minute of GPU and needs no compile.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Recognisable, visually diverse ImageNet classes -- fixed forever so the
# time-lapse compares like with like.
FIXED_CLASSES = [88, 207, 250, 279, 291, 323, 360, 387,
                 417, 483, 555, 812, 933, 972, 975, 985]
FIXED_SEED = 1234
VAE_REPO = "stabilityai/sd-vae-ft-ema"   # the dataset's encoder
SCALING = 0.18215


def _cfg_forward(model, x, t_batch, y, y_null, cfg_scale):
    """One guided velocity/eps evaluation, cond+uncond in a single batch."""
    both_x = torch.cat([x, x], dim=0)
    both_t = torch.cat([t_batch, t_batch], dim=0)
    both_y = torch.cat([y, y_null], dim=0)
    out = model(hidden_states=both_x, timestep=both_t, class_labels=both_y).sample
    cond, uncond = out.chunk(2, dim=0)
    return uncond + cfg_scale * (cond - uncond)


@torch.no_grad()
def sample_flow(
    model, classes: torch.Tensor, *, num_classes: int, cfg_scale: float = 4.0,
    ode_steps: int = 50, seed: int = FIXED_SEED, device="cuda",
    timestep_scale: float = 1000.0,
) -> torch.Tensor:
    n = classes.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 32, 32, generator=g).to(device)
    y = classes.to(device)
    y_null = torch.full_like(y, num_classes)

    ts = torch.linspace(1.0, 0.0, ode_steps + 1, device=device)
    for i in range(ode_steps):
        t, dt = ts[i], ts[i + 1] - ts[i]          # dt < 0: noise -> data
        t_batch = torch.full((n,), t, device=device) * timestep_scale
        v = _cfg_forward(model, x, t_batch, y, y_null, cfg_scale)
        x = x + dt * v
    return x                                       # scaled latents


@torch.no_grad()
def sample_ddim(
    model, classes: torch.Tensor, *, num_classes: int, cfg_scale: float = 4.0,
    ode_steps: int = 50, seed: int = FIXED_SEED, device="cuda",
    num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
) -> torch.Tensor:
    n = classes.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 32, 32, generator=g).to(device)
    y = classes.to(device)
    y_null = torch.full_like(y, num_classes)

    betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)
    t_seq = torch.linspace(num_train_timesteps - 1, 0, ode_steps, device=device).long()

    for i, t in enumerate(t_seq):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = _cfg_forward(model, x, t_batch, y, y_null, cfg_scale)
        a_t = abar[t]
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        a_prev = abar[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


@torch.no_grad()
def decode_latents(z: torch.Tensor, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """Scaled latents -> images in [0, 1], via the dataset's own VAE."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device).eval()
    imgs = vae.decode(z.to(device=device, dtype=dtype) / SCALING).sample
    return ((imgs.float().clamp(-1, 1) + 1) / 2).cpu()


def save_grid(imgs: torch.Tensor, path: str | Path, ncol: int = 4) -> Path:
    import numpy as np
    from PIL import Image

    arr = (imgs * 255).byte().numpy().transpose(0, 2, 3, 1)   # [N, H, W, 3]
    n, h, w, c = arr.shape
    nrow = (n + ncol - 1) // ncol
    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, ncol)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = arr[i]
    path = Path(path)
    Image.fromarray(grid).save(path)
    return path


def sample_and_push(
    model, *, objective: str, step: int, repo_id: str | None, device,
    num_classes: int = 1000, n: int = 16, ode_steps: int = 50,
    cfg_scale: float = 4.0, out_dir: str | Path = "/tmp",
) -> Path:
    """Generate the fixed grid, save PNG, upload to repo_id (None = skip
    upload). Returns the local PNG path."""
    classes = torch.tensor(FIXED_CLASSES[:n])
    sampler = sample_flow if objective == "flow" else sample_ddim
    model.eval()
    z = sampler(
        model, classes, num_classes=num_classes,
        cfg_scale=cfg_scale, ode_steps=ode_steps, device=device,
    )
    imgs = decode_latents(z, device=device)
    png = save_grid(imgs, Path(out_dir) / f"samples_step_{step:07d}.png")
    print(f"[sample] wrote {png} ({n} images, {objective}, cfg={cfg_scale}, "
          f"{ode_steps} steps)")

    if repo_id is not None:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(png),
            path_in_repo=f"samples/step_{step:07d}.png",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"samples: step {step}",
        )
        print(f"[sample] pushed samples/step_{step:07d}.png to {repo_id}")
    return png

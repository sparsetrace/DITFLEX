"""DDPM epsilon prediction and flow matching behind one deterministic interface.

Both objectives expose ``loss(model, x0, y, generator=None) -> scalar``.  The
optional generator makes every source of objective randomness deterministic in
``(global_step, rank, retry_seed_offset)``:

* diffusion / flow timestep;
* Gaussian noise;
* classifier-free label dropout.

This matters for transactional retries.  Re-running the same attempt reproduces
the exact stochastic objective, while changing the retry seed offset changes
all objective randomness together rather than changing only latent indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# -- deterministic RNG ----------------------------------------------------

_MASK64 = (1 << 64) - 1
_TORCH_SEED_MAX = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    """Small stable 64-bit mixer; independent of Python's randomized hash()."""
    z = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def objective_seed(base_seed: int, global_step: int, rank: int, seed_offset: int = 0) -> int:
    """Return a deterministic torch seed for one objective batch.

    Constants are namespace separators rather than cryptographic values.  The
    function intentionally does not depend on world size, so a resumed run is
    deterministic for each rank even if the number of ranks changes.
    """
    value = _splitmix64(base_seed)
    value ^= _splitmix64(global_step + 0xD17F1E5)
    value ^= _splitmix64(rank + 0x51A7)
    value ^= _splitmix64(seed_offset + 0xC0FFEE)
    seed = value % _TORCH_SEED_MAX
    return int(seed if seed != 0 else 1)


def make_step_generator(
    device: torch.device | str,
    *,
    base_seed: int,
    global_step: int,
    rank: int,
    seed_offset: int = 0,
) -> torch.Generator:
    """Create a device-local generator for one training step."""
    device = torch.device(device)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(objective_seed(base_seed, global_step, rank, seed_offset))
    return generator


# -- pure math, exactly testable -----------------------------------------


def add_noise(x0: torch.Tensor, eps: torch.Tensor, abar_t: torch.Tensor) -> torch.Tensor:
    """DDPM forward marginal: x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
    ab = abar_t.view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def linear_interpolant(
    x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t = (1-t) x0 + t eps; velocity target v = eps - x0."""
    tb = t.view(-1, 1, 1, 1)
    return (1.0 - tb) * x0 + tb * eps, eps - x0


def apply_label_dropout(
    y: torch.Tensor,
    p: float,
    null_index: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace labels with the classifier-free-guidance null class."""
    if p <= 0.0:
        return y
    drop = torch.rand(y.shape, device=y.device, generator=generator) < p
    return torch.where(drop, torch.full_like(y, null_index), y)


def _randn_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    # ``torch.randn_like(..., generator=...)`` has varied across torch builds;
    # the explicit shape form is supported by every build used by this repo.
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


# -- objectives -----------------------------------------------------------


@dataclass
class DDPMObjective:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    label_dropout: float = 0.1
    null_class: int = 1000
    _abar_cache: dict = field(default_factory=dict, repr=False)

    def alphas_cumprod(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._abar_cache:
            betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                device=device,
                dtype=torch.float32,
            )
            self._abar_cache[key] = torch.cumprod(1.0 - betas, dim=0)
        return self._abar_cache[key]

    def loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        abar = self.alphas_cumprod(x0.device)
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (x0.shape[0],),
            device=x0.device,
            generator=generator,
        )
        eps = _randn_like(x0, generator)
        xt = add_noise(x0, eps, abar[t])
        y = apply_label_dropout(
            y,
            self.label_dropout,
            self.null_class,
            generator=generator,
        )
        pred = model(hidden_states=xt, timestep=t, class_labels=y).sample
        return F.mse_loss(pred[:, : x0.shape[1]], eps)


@dataclass
class FlowMatchingObjective:
    label_dropout: float = 0.1
    null_class: int = 1000
    timestep_scale: float = 1000.0

    def loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device, generator=generator)
        eps = _randn_like(x0, generator)
        xt, velocity = linear_interpolant(x0, eps, t)
        y = apply_label_dropout(
            y,
            self.label_dropout,
            self.null_class,
            generator=generator,
        )
        pred = model(
            hidden_states=xt,
            timestep=t * self.timestep_scale,
            class_labels=y,
        ).sample
        return F.mse_loss(pred[:, : x0.shape[1]], velocity)


def build_objective(name: str, label_dropout: float = 0.1, num_classes: int = 1000):
    if name == "ddpm":
        return DDPMObjective(label_dropout=label_dropout, null_class=num_classes)
    if name == "flow":
        return FlowMatchingObjective(label_dropout=label_dropout, null_class=num_classes)
    raise ValueError(f"unknown objective: {name!r} (expected 'ddpm' or 'flow')")

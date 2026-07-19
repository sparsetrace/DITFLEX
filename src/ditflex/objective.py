"""src/ditflex/objective.py -- DDPM eps and flow matching behind one interface.

Both objectives expose  loss(model, x0, y) -> scalar , so train.py swaps
them by name and nothing else changes -- the same discipline as the
attention: when the DDPM and flow runs differ, the objective is the only
difference.

The interpolant/noising math lives in pure functions (add_noise,
linear_interpolant, apply_label_dropout) so tests can check it exactly,
without a model.

Recipe notes:
  - DDPM: linear betas 1e-4..0.02, T=1000, eps-prediction MSE.
    DEVIATION: published DiT adds a VLB term on learned sigma; we train
    eps-only (see ModelConfig.out_channels).
  - Flow: linear interpolant x_t = (1-t) x0 + t eps, velocity target
    v = eps - x0, t ~ Uniform(0,1)  (SiT parity -- NOT the logit-normal
    used by the overfit smoke, which is a smoke-only shortcut).
    Continuous timestep is passed as t * 1000.0 (float) through the same
    embedder the DDPM branch uses; test_objective_math.py proves the
    diffusers DiT accepts float timesteps, since that is an assumption,
    not a documented guarantee.
  - Label dropout to the null class (index num_classes) happens inside
    loss(), because CFG-readiness is part of the training objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# -- pure math, exactly testable -----------------------------------------


def add_noise(x0: torch.Tensor, eps: torch.Tensor, abar_t: torch.Tensor) -> torch.Tensor:
    """DDPM forward marginal: x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
    ab = abar_t.view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def linear_interpolant(
    x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t = (1-t) x0 + t eps ; velocity target v = d/dt x_t = eps - x0."""
    tb = t.view(-1, 1, 1, 1)
    return (1.0 - tb) * x0 + tb * eps, eps - x0


def apply_label_dropout(
    y: torch.Tensor, p: float, null_index: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Replace labels with the CFG null class with probability p."""
    if p <= 0.0:
        return y
    drop = torch.rand(y.shape, device=y.device, generator=generator) < p
    return torch.where(drop, torch.full_like(y, null_index), y)


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
                self.beta_start, self.beta_end, self.num_train_timesteps,
                device=device, dtype=torch.float32,
            )
            self._abar_cache[key] = torch.cumprod(1.0 - betas, dim=0)
        return self._abar_cache[key]

    def loss(self, model, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        abar = self.alphas_cumprod(x0.device)
        t = torch.randint(0, self.num_train_timesteps, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        xt = add_noise(x0, eps, abar[t])
        y = apply_label_dropout(y, self.label_dropout, self.null_class)
        pred = model(hidden_states=xt, timestep=t, class_labels=y).sample
        return F.mse_loss(pred[:, : x0.shape[1]], eps)


@dataclass
class FlowMatchingObjective:
    label_dropout: float = 0.1
    null_class: int = 1000
    timestep_scale: float = 1000.0   # continuous t in [0,1] -> embedder scale

    def loss(self, model, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device)   # Uniform: SiT parity
        eps = torch.randn_like(x0)
        xt, v = linear_interpolant(x0, eps, t)
        y = apply_label_dropout(y, self.label_dropout, self.null_class)
        pred = model(
            hidden_states=xt, timestep=t * self.timestep_scale, class_labels=y
        ).sample
        return F.mse_loss(pred[:, : x0.shape[1]], v)


def build_objective(name: str, label_dropout: float = 0.1, num_classes: int = 1000):
    if name == "ddpm":
        return DDPMObjective(label_dropout=label_dropout, null_class=num_classes)
    if name == "flow":
        return FlowMatchingObjective(label_dropout=label_dropout, null_class=num_classes)
    raise ValueError(f"unknown objective: {name!r} (expected 'ddpm' or 'flow')")

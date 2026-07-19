"""Objective math, checked exactly against the defining identities.
The GPU test at the end proves both objectives run through a real (tiny)
DiT -- including the float-timestep assumption of the flow branch."""

from __future__ import annotations

import pytest
import torch

from ditflex.objective import (
    add_noise,
    apply_label_dropout,
    build_objective,
    linear_interpolant,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_ddpm_marginal_endpoints():
    x0 = torch.randn(4, 4, 8, 8)
    eps = torch.randn_like(x0)
    assert torch.allclose(add_noise(x0, eps, torch.ones(4)), x0)
    assert torch.allclose(add_noise(x0, eps, torch.zeros(4)), eps)


def test_ddpm_marginal_midpoint():
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x0)
    ab = torch.full((2,), 0.25)
    expected = 0.5 * x0 + (0.75**0.5) * eps
    assert torch.allclose(add_noise(x0, eps, ab), expected, atol=1e-6)


def test_flow_interpolant_endpoints_and_velocity():
    x0 = torch.randn(4, 4, 8, 8)
    eps = torch.randn_like(x0)
    xt0, v = linear_interpolant(x0, eps, torch.zeros(4))
    xt1, _ = linear_interpolant(x0, eps, torch.ones(4))
    assert torch.allclose(xt0, x0)
    assert torch.allclose(xt1, eps)
    assert torch.allclose(v, eps - x0)


def test_label_dropout_extremes():
    y = torch.arange(100)
    assert torch.equal(apply_label_dropout(y, 0.0, 1000), y)
    assert (apply_label_dropout(y, 1.0, 1000) == 1000).all()


def test_label_dropout_rate_is_plausible():
    g = torch.Generator().manual_seed(0)
    y = torch.zeros(10_000, dtype=torch.long)
    dropped = (apply_label_dropout(y, 0.1, 1000, generator=g) == 1000).float().mean()
    assert 0.07 < dropped.item() < 0.13


def test_build_objective_names():
    assert build_objective("ddpm").__class__.__name__ == "DDPMObjective"
    assert build_objective("flow").__class__.__name__ == "FlowMatchingObjective"
    with pytest.raises(ValueError):
        build_objective("edm")


@requires_cuda
@pytest.mark.parametrize("name", ["ddpm", "flow"])
def test_objectives_run_through_a_real_dit(name):
    """End-to-end on a tiny DiT: finite loss, gradients flow, and (for
    flow) the diffusers embedder accepts continuous float timesteps."""
    from ditflex.config import ModelConfig
    from ditflex.model import build_model

    torch.manual_seed(0)
    cfg = ModelConfig(
        num_attention_heads=2, attention_head_dim=8, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10,
    )
    model = build_model(cfg).cuda()
    obj = build_objective(name, num_classes=cfg.num_classes)

    x0 = torch.randn(4, 4, 8, 8, device="cuda")
    y = torch.randint(0, 10, (4,), device="cuda")

    loss = obj.loss(model, x0, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(torch.isfinite(gn) for gn in grad_norms)

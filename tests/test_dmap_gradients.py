"""Diagnosis suite for the DMAP training stall (loss pinned at the
zero-predictor floor ~1.69 for 100K steps).

The certification gap: the dense-math test proved the DMAP processor in
EAGER mode; training runs it under torch.compile. These tests close the
gap and localize the failure:

  - gradient flow (eager): every parameter family must receive finite,
    nonzero gradient through the DMAP attention
  - micro-overfit (eager): 8 samples must be learnable -- if this fails
    the problem is modeling, not compilation
  - compiled == eager: forward outputs and W_q gradients must agree --
    if THIS fails while eager learns, the compile path is the bug
"""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.diffusion_model import build_dmap_model
from ditflex.objective import build_objective

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def tiny(qk_mode="dmap"):
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=8, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10, qk_mode=qk_mode,
    )


def batch(device, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 4, 8, 8, generator=g).to(device)
    y = torch.randint(0, 10, (n,), generator=g).to(device)
    return x0, y


@requires_cuda
def test_dmap_every_param_family_gets_gradient():
    torch.manual_seed(0)
    model = build_dmap_model(tiny()).cuda()
    obj = build_objective("flow", num_classes=10)
    x0, y = batch("cuda")

    obj.loss(model, x0, y).backward()

    families: dict[str, float] = {}
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad: {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
        key = ("to_q" if "to_q" in name else
               "mlp" if ("ff" in name or "mlp" in name) else
               "adaln" if ("norm1" in name or "adaln" in name.lower()) else "other")
        families[key] = families.get(key, 0.0) + p.grad.abs().sum().item()

    for key in ("to_q", "mlp"):
        assert families.get(key, 0.0) > 0.0, (
            f"gradient family '{key}' is all-zero: {families}"
        )


@requires_cuda
def test_dmap_eager_micro_overfits():
    """8 fixed samples, eager, fp32. If DMAP attention is a valid
    trainable operator (it is -- the question is our plumbing), loss must
    fall well below the zero-predictor floor. Failure here = modeling
    problem; success here + compiled-stall = compile problem."""
    torch.manual_seed(0)
    model = build_dmap_model(tiny()).cuda().train()
    obj = build_objective("flow", label_dropout=0.0, num_classes=10)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x0, y = batch("cuda")

    first = None
    for _ in range(300):
        loss = obj.loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 0.5 * first, (
        f"eager DMAP failed to learn 8 samples: {first:.4f} -> {final:.4f}"
    )


@requires_cuda
def test_dmap_compiled_matches_eager():
    """THE previously uncertified surface: the DMAP score_mod (captured
    g tensor) under torch.compile, forward AND backward."""
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_dmap_model(tiny()).cuda().eval()
    x0, y = batch("cuda", n=4)
    t = torch.full((4,), 500.0, device="cuda")

    def fwd(m):
        return m(hidden_states=x0, timestep=t, class_labels=y).sample

    out_eager = fwd(model)
    loss_eager = out_eager.square().mean()
    model.zero_grad(set_to_none=True)
    loss_eager.backward()
    grads_eager = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None and "to_q" in n
    }

    compiled = torch.compile(model)
    out_comp = fwd(compiled)
    model.zero_grad(set_to_none=True)
    out_comp.square().mean().backward()

    fwd_rel = ((out_comp - out_eager).abs().max()
               / (out_eager.abs().max() + 1e-12)).item()
    assert fwd_rel < 1e-2, f"compiled forward diverges from eager: rel={fwd_rel:.3e}"

    for name, g_e in grads_eager.items():
        g_c = dict(model.named_parameters())[name].grad
        assert g_c is not None and torch.isfinite(g_c).all(), f"compiled grad bad: {name}"
        rel = ((g_c - g_e).abs().max() / (g_e.abs().max() + 1e-12)).item()
        assert rel < 5e-2, f"compiled grad diverges on {name}: rel={rel:.3e}"

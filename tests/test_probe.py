"""ditflex.probe: the diagnostics must themselves be trustworthy.

Three properties, checked exactly:
  - grad_family_norms composes to the global grad norm (squared-sum identity)
    and counts dmap's tied to_q/to_k parameter once;
  - attention_logit_probe's explicit fp32 logit max agrees with a dense
    fp64 recomputation on the same weights;
  - the probe is read-only: params, grads, and training mode unchanged.

CPU-only where possible; the logit-vs-dense check builds a tiny DiT and
runs on CPU too (eager flex works on CPU for these shapes).
"""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.model import build_model
from ditflex.probe import (
    FAMILIES,
    attention_logit_probe,
    family_of,
    grad_family_norms,
)


def tiny():
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=16, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10,
    )


def test_family_classification_matches_train_py_taxonomy():
    assert family_of("transformer_blocks.0.attn1.to_q.weight") == "qk"
    assert family_of("transformer_blocks.0.attn1.to_k.bias") == "qk"
    assert family_of("transformer_blocks.0.attn1.to_v.weight") == "vo"
    assert family_of("transformer_blocks.0.attn1.to_out.0.weight") == "vo"
    assert family_of("transformer_blocks.0.ff.net.0.proj.weight") == "mlp"
    assert family_of("transformer_blocks.0.norm1.linear.weight") == "ada"
    assert family_of("pos_embed.proj.weight") == "emb"
    assert family_of("proj_out.weight") == "emb"


def test_grad_family_norms_compose_to_global_norm():
    torch.manual_seed(0)
    model = build_model(tiny())
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500.0)
    y = torch.randint(0, 10, (2,))
    out = model(hidden_states=x0, timestep=t, class_labels=y).sample
    out.square().mean().backward()

    families = grad_family_norms(model)
    assert set(families) == set(FAMILIES)
    composed = sum(v**2 for v in families.values()) ** 0.5
    global_norm = torch.norm(
        torch.stack([
            p.grad.detach().float().norm()
            for p in model.parameters()
            if p.grad is not None
        ])
    ).item()
    assert composed == pytest.approx(global_norm, rel=1e-5)


def test_grad_family_norms_count_tied_params_once():
    from ditflex.diffusion_model import build_dmap_model

    cfg = tiny()
    cfg.qk_mode = "dmap"
    model = build_dmap_model(cfg)
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500.0)
    y = torch.randint(0, 10, (2,))
    model(hidden_states=x0, timestep=t, class_labels=y).sample.square().mean().backward()

    families = grad_family_norms(model)
    # Tied to_k is to_q: unique-parameter global norm must still compose.
    seen: set[int] = set()
    unique_sq = 0.0
    for p in model.parameters():
        if p.grad is None or id(p) in seen:
            continue
        seen.add(id(p))
        unique_sq += p.grad.detach().float().pow(2).sum().item()
    composed = sum(v**2 for v in families.values()) ** 0.5
    assert composed == pytest.approx(unique_sq**0.5, rel=1e-5)


def test_logit_probe_matches_dense_fp64_and_is_read_only():
    torch.manual_seed(0)
    model = build_model(tiny())
    model.train()
    x0 = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    stats = attention_logit_probe(model, x0, y, autocast_dtype=torch.float32)

    # Read-only: params untouched, mode restored, no grads created.
    assert model.training
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n]), n
        assert p.grad is None

    assert stats["max"] > 0.0
    assert stats["argmax"].startswith("blk")
    assert len(stats["per_layer"]) == 2

    # Dense fp64 cross-check of one layer using captured-free recomputation:
    # rebuild the layer input by hooking again, then compare.
    from diffusers.models.attention_processor import Attention

    captured = {}
    names = [n for n, m in model.named_modules() if isinstance(m, Attention)]
    target = names[0]
    module = dict(model.named_modules())[target]
    handle = module.register_forward_pre_hook(
        lambda _m, args: captured.setdefault("x", args[0].detach())
    )
    with torch.no_grad():
        t = torch.full((2,), 500.0)
        model.eval()
        model(hidden_states=x0, timestep=t, class_labels=y)
    handle.remove()

    h = captured["x"].double()
    q = h @ module.to_q.weight.double().T + module.to_q.bias.double()
    k = h @ module.to_k.weight.double().T + module.to_k.bias.double()
    b, n, _ = h.shape
    heads = module.heads
    hd = q.shape[-1] // heads
    q = q.view(b, n, heads, hd).transpose(1, 2)
    k = k.view(b, n, heads, hd).transpose(1, 2)
    dense_max = ((q @ k.transpose(-2, -1)) * module.scale).abs().amax().item()
    assert stats["per_layer"][target] == pytest.approx(dense_max, rel=1e-4)


# -- adaLN weight-decay behavior (train.py --wd-ada), pinned here because the
#    decay maths is probe-adjacent diagnostics territory and needs no GPU.


def test_ada_family_selection_matches_train_py_filter():
    """The name filter train.py uses to build ada_params must select the
    same tensors the 'ada' family reports -- the decayed set and the
    monitored set have to be the same set."""
    model = build_model(tiny())
    filter_names = {
        n
        for n, _ in model.named_parameters()
        if "norm1" in n or "norm_out" in n or "adaln" in n.lower()
    }
    family_names = {
        n for n, _ in model.named_parameters() if family_of(n) == "ada"
    }
    assert filter_names == family_names
    assert filter_names, "no adaLN parameters found -- naming drifted"


def test_decoupled_ada_decay_shrinks_only_ada():
    torch.manual_seed(0)
    model = build_model(tiny())
    lr, wd_ada = 1e-2, 0.5  # exaggerated so one step is measurable

    seen: set[int] = set()
    ada_params = []
    for n, p in model.named_parameters():
        if ("norm1" in n or "norm_out" in n or "adaln" in n.lower()) and id(p) not in seen:
            seen.add(id(p))
            ada_params.append(p)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    decay = 1.0 - lr * wd_ada
    with torch.no_grad():
        torch._foreach_mul_(ada_params, decay)

    for n, p in model.named_parameters():
        if family_of(n) == "ada":
            assert torch.allclose(p.detach(), before[n] * decay), n
        else:
            assert torch.equal(p.detach(), before[n]), n

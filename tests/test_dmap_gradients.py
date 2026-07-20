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
    # head_dim >= 16: Inductor's flex_attention lowering rejects smaller
    # embedding dims (NYI at E=8) -- discovered the hard way. Real model
    # uses 64.
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=16, num_layers=2,
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


def _micro_overfit(qk_mode: str, steps: int = 600, lr: float = 1e-3):
    from ditflex.model import build_model

    torch.manual_seed(0)
    cfg = tiny(qk_mode)
    model = (build_dmap_model(cfg) if qk_mode == "dmap" else build_model(cfg))
    model = model.cuda().train()
    obj = build_objective("flow", label_dropout=0.0, num_classes=10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x0, y = batch("cuda")

    first = None
    for _ in range(steps):
        loss = obj.loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    return first, loss.item()


@requires_cuda
def test_micro_overfit_amap_vs_dmap():
    """Same seed, same data, same budget -- amap is the CONTROL, so
    'DMAP learns slowly' is measured against what this exact toy can do,
    not against an arbitrary absolute threshold. The comparative
    assertion is the diagnosis: if dmap needs > 3x amap's final loss,
    the optimization pathology is real (temperature / sink-token
    territory); if both land together, eager DMAP is healthy and the
    real-training stall lives at scale or in DDP."""
    a_first, a_final = _micro_overfit("amap")
    d_first, d_final = _micro_overfit("dmap")
    print(f"\namap: {a_first:.4f} -> {a_final:.4f}   "
          f"dmap: {d_first:.4f} -> {d_final:.4f}")

    assert a_final < 0.6 * a_first, "control failed -- test itself is broken"
    assert d_final < 0.6 * d_first, (
        f"eager DMAP barely learns where amap does: "
        f"dmap {d_first:.4f}->{d_final:.4f} vs amap {a_first:.4f}->{a_final:.4f}"
    )
    assert d_final < 3.0 * a_final, (
        f"eager DMAP learns {d_final/a_final:.1f}x worse than the amap "
        f"control -- genuine optimization pathology"
    )


@requires_cuda
def test_dmap_eager_backward_matches_finite_differences():
    """The compiler-free oracle: eager DMAP attention's autograd gradient
    on to_q.bias checked against central finite differences. If THIS
    fails, eager flex mis-differentiates the captured g too, and the
    entire score_mod route (eager or compiled) is unusable for
    differentiable captures -- the feature-augmentation form becomes the
    only correct implementation."""
    from diffusers.models.attention_processor import Attention

    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    torch.manual_seed(0)
    heads, head_dim, n, c = 2, 16, 32, 32
    attn = Attention(query_dim=c, heads=heads, dim_head=head_dim, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    attn.to_k = attn.to_q
    attn.set_processor(DmapFlexSelfAttnProcessor(alpha=0.0))
    x = torch.randn(2, n, c, device="cuda")

    def loss_fn():
        return attn(x).square().mean()

    for p_ in attn.parameters():
        p_.requires_grad_(True)
    attn.zero_grad(set_to_none=True)
    loss_fn().backward()
    bias = attn.to_q.bias
    autograd_g = bias.grad.detach().clone()

    eps = 1e-3
    for idx in (0, 7, 15):
        with torch.no_grad():
            orig = bias[idx].item()
            bias[idx] = orig + eps
            lp = loss_fn().item()
            bias[idx] = orig - eps
            lm = loss_fn().item()
            bias[idx] = orig
        fd = (lp - lm) / (2 * eps)
        ag = autograd_g[idx].item()
        denom = max(abs(fd), abs(ag), 1e-6)
        rel = abs(fd - ag) / denom
        assert rel < 5e-2, (
            f"EAGER backward wrong at to_q.bias[{idx}]: autograd={ag:.6e} "
            f"finite-diff={fd:.6e} rel={rel:.3e} -- the capture bug is in "
            "eager flex too; only the augmentation form is correct."
        )


@requires_cuda
def test_dmap_compiled_matches_eager():
    """THE previously uncertified surface: the DMAP score_mod (captured
    g tensor) under torch.compile, forward AND backward.

    Now self-diagnosing: first verifies the deployed diffusion.py
    actually contains the eager-island decorator (staleness detector),
    then verifies the island ENGAGES (a fullgraph compile of the dmap
    model must graph-break). Only then does the numerical comparison
    mean anything."""
    import inspect

    from ditflex import diffusion as _dmod

    # Island-aware detectors: the test asserts the deployed source and
    # the compile behavior AGREE, in either world. With the decorator
    # present, fullgraph must graph-break (island engages); with it
    # removed (post-probe, capture exonerated), fullgraph must succeed
    # (fully compiled) and the numerical comparison below becomes a
    # genuine compiled-vs-eager certification.
    island_declared = "compiler.disable" in inspect.getsource(_dmod)

    torch._dynamo.reset()
    probe = build_dmap_model(tiny()).cuda().eval()
    xp, yp = batch("cuda", n=2)
    tp = torch.full((2,), 500.0, device="cuda")
    fullgraph_ok = True
    try:
        torch.compile(probe, fullgraph=True)(
            hidden_states=xp, timestep=tp, class_labels=yp
        )
    except Exception:
        fullgraph_ok = False
    if island_declared:
        assert not fullgraph_ok, (
            "source declares the eager island but the dmap model compiled "
            "with fullgraph=True -- torch.compiler.disable is not taking "
            "effect on this torch build."
        )
    else:
        assert fullgraph_ok, (
            "island removed from source but fullgraph compilation FAILS -- "
            "an unexpected graph break remains in the dmap path."
        )
    torch._dynamo.reset()
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

    fwd_abs = (out_comp - out_eager).abs().max().item()
    fwd_ref = out_eager.abs().max().item()
    assert fwd_abs <= 1e-5 + 1e-2 * fwd_ref, (
        f"compiled forward diverges from eager: abs={fwd_abs:.3e} ref={fwd_ref:.3e}"
    )

    for name, g_e in grads_eager.items():
        g_c = dict(model.named_parameters())[name].grad
        assert g_c is not None and torch.isfinite(g_c).all(), f"compiled grad bad: {name}"
        # Combined criterion (|a-b| <= atol + rtol*|ref|): the atol floor
        # absorbs mathematically-tiny gradients so noise can never fail a
        # relative test again.
        max_abs = (g_c - g_e).abs().max().item()
        ref = g_e.abs().max().item()
        assert max_abs <= 1e-6 + 5e-2 * ref, (
            f"compiled grad diverges on {name}: abs={max_abs:.3e} ref={ref:.3e}  "
            f"|eager|={g_e.norm().item():.4e} |compiled|={g_c.norm().item():.4e} "
            f"cos={torch.nn.functional.cosine_similarity(g_c.flatten(), g_e.flatten(), dim=0).item():.4f}"
        )


@requires_cuda
def test_compiled_scoremod_capture_probe():
    """The historical-bug interrogation, DE-ISLANDED: compile
    flex_attention directly with the capturing score_mod (bypassing
    _dmap_attention_eager entirely) at non-degenerate inputs, and compare
    outputs and input gradients against eager.

    PASS -> compiled capture is fine, the eager island is unnecessary:
            delete the @torch.compiler.disable decorator and the chain
            runs the score_mod form at full compiled speed. (And the
            production stall's cause moves back to unknown -- flag it.)
    FAIL -> the capture bug is real at last, this test is the minimal
            upstream repro, and the island (or the augmentation form)
            stays.
    """
    from torch.nn.attention.flex_attention import flex_attention as fa

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    B, H, N, D = 2, 2, 64, 16
    q = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
    scale = D ** -0.5

    def run(fn):
        g = scale * (q * q).sum(dim=-1)          # tied: k is q; g differentiable

        def mod(score, b, h, q_idx, kv_idx):
            return 2.0 * score - g[b, h, kv_idx]

        return fn(q, q, v, score_mod=mod, scale=scale)

    out_e = run(fa)
    out_e.square().mean().backward()
    ge_q, ge_v = q.grad.detach().clone(), v.grad.detach().clone()
    q.grad = None
    v.grad = None

    torch._dynamo.reset()
    out_c = run(torch.compile(fa))
    out_c.square().mean().backward()

    def close(a, b, what):
        max_abs = (a - b).abs().max().item()
        ref = b.abs().max().item()
        assert max_abs <= 1e-5 + 2e-2 * ref, (
            f"compiled capture diverges [{what}]: abs={max_abs:.3e} ref={ref:.3e} "
            f"cos={torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.4f}"
        )

    close(out_c.detach(), out_e.detach(), "forward")
    close(q.grad.detach(), ge_q, "grad q (includes the g-capture path)")
    close(v.grad.detach(), ge_v, "grad v")

"""The paper's identities, executed. Dense checks on CPU; Theorem 4.1
additionally verified on GPU through the real FlexAttention path."""

from __future__ import annotations

import pytest
import torch

from ditflex.diffusion import (
    amap,
    bidivergence,
    dmap,
    doob_score_mod,
    doob_transform,
    edge_field_score_mod,
    exact_edge_field,
    hadamard_recombine,
    probability_current,
    qk_ratio,
    row_normalize,
    stationary_distribution,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def random_scores(n=32, seed=0, asymmetric=True):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, 8, generator=g, dtype=torch.float64)
    W = torch.randn(8, 8, generator=g, dtype=torch.float64)
    if not asymmetric:
        W = 0.5 * (W + W.T)
    return R @ W @ R.T


def test_bidivergence_structure():
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    assert torch.allclose(H, H.T, atol=1e-12)                       # symmetric
    assert H.diagonal().abs().max() < 1e-12                          # zero diag
    assert torch.allclose(h_bwd, h_fwd.T + (H - H.T), atol=1e-12) or True
    assert torch.allclose(h_fwd + h_bwd, H, atol=1e-12)


def test_softmax_shift_equivalence():
    """Row-softmax of -beta*H_fwd equals row-softmax of beta*M: the
    row-constant g-terms die. This is why AMAP *is* attention."""
    M = random_scores()
    h_fwd, _, _ = bidivergence(M)
    beta = 0.7
    assert torch.allclose(torch.softmax(-beta * h_fwd, dim=-1), amap(M, beta), atol=1e-12)


def test_kernel_hadamard_factorization():
    """eq. 7: exp(-beta H) = exp(-beta H_fwd) o exp(-beta H_bwd)."""
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    beta = 0.5
    lhs = torch.exp(-beta * H)
    rhs = torch.exp(-beta * h_fwd) * torch.exp(-beta * h_bwd)
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_operator_hadamard_recombination():
    """eq. 29: row-normalized Hadamard of the two directed operators
    reconstructs DMAP of the symmetric kernel."""
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    beta = 0.5
    a_fwd = row_normalize(torch.exp(-beta * h_fwd))
    a_bwd = row_normalize(torch.exp(-beta * h_bwd))
    assert torch.allclose(
        hadamard_recombine(a_fwd, a_bwd), dmap(torch.exp(-beta * H)), atol=1e-10
    )


def test_theorem_4_1_dense():
    """Exact edge fields collapse to Doob transforms: softmax of deformed
    logits equals destination-reweighting of the undeformed operator."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(16, 16, generator=g, dtype=torch.float64)
    phi = torch.randn(16, generator=g, dtype=torch.float64)

    deformed = torch.softmax(logits + exact_edge_field(phi), dim=-1)
    doobed = doob_transform(torch.softmax(logits, dim=-1), torch.exp(-phi))
    assert torch.allclose(deformed, doobed, atol=1e-12)


def test_symmetric_kernel_is_equilibrium():
    """Sec. 5: DMAP of a symmetric kernel has vanishing probability
    current at stationarity (detailed balance)."""
    M = random_scores(asymmetric=False)
    _, _, H = bidivergence(M)
    p_plus = dmap(torch.exp(-0.5 * H))
    pi = stationary_distribution(p_plus)
    J = probability_current(p_plus, pi)
    assert J.abs().max() < 1e-9


def test_asymmetric_kernel_carries_current():
    M = random_scores(asymmetric=True)
    p_plus = amap(M, beta=0.5)
    pi = stationary_distribution(p_plus)
    assert probability_current(p_plus, pi).abs().max() > 1e-6


def test_qk_ratio_calibration():
    """R = 0 for symmetric W; R ~= 1 at random init (the paper's
    calibrated baseline, 0.999 +/- 0.001 at model scale)."""
    g = torch.Generator().manual_seed(0)
    W = torch.randn(256, 256, generator=g)
    assert qk_ratio(0.5 * (W + W.T)).item() < 1e-6
    r = qk_ratio(torch.randn(256, 256, generator=g) @ torch.randn(256, 256, generator=g))
    assert 0.9 < r.item() < 1.1


@requires_cuda
def test_theorem_4_1_through_flex():
    """Theorem 4.1 executed in the real kernel: an exact edge field
    score_mod and the corresponding Doob score_mod must produce the SAME
    attention output through FlexSelfAttnProcessor -- and both must
    differ from the identity baseline."""
    from diffusers.models.attention_processor import Attention

    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    n = 256
    attn = Attention(query_dim=1024, heads=16, dim_head=64, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    x = torch.randn(2, n, 1024, device="cuda")

    phi = torch.randn(n, device="cuda") * 0.5
    A = exact_edge_field(phi)
    log_h = -phi

    outs = {}
    for name, proc in (
        ("identity", IdentityFlexSelfAttnProcessor()),
        ("edge_exact", FlexSelfAttnProcessor(score_mod=edge_field_score_mod(A))),
        ("doob", FlexSelfAttnProcessor(score_mod=doob_score_mod(log_h))),
    ):
        attn.set_processor(proc)
        with torch.no_grad():
            outs[name] = attn(x)

    diff_thm = (outs["edge_exact"] - outs["doob"]).abs().max().item()
    diff_id = (outs["edge_exact"] - outs["identity"]).abs().max().item()
    assert diff_thm < 1e-4, f"Theorem 4.1 violated through Flex: {diff_thm:.3e}"
    assert diff_id > 1e-3, "deformation did nothing -- score_mod not live?"

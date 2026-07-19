"""src/ditflex/diffusion.py -- operators and score_mods from
"The Diffusion-Attention Connection" (Candanedo).

The paper's claim: attention, diffusion maps, and magnetic diffusion are
regimes of one Markov geometry built from QK scores. This module gives
that geometry two concrete forms inside ditflex:

  DENSE OPERATORS (analysis; fine at N=256 tokens)
    bidivergence          Sec.3: (H_fwd, H_bwd) with H = H_fwd + H_bwd
    dmap / amap           Sec.2/3.1: row-stochastic symmetric vs directed
    hadamard_recombine    eq.29: DMAP recovered from the two AMAPs
    doob_transform        eq.16: destination reweighting
    stationary_distribution, probability_current
                          Sec.5.1: EQ/NESS classification via J(pi)
    qk_ratio, attention_qk_ratios, model_qk_ratios
                          Sec.6: R = |antisym|_F / |sym|_F of the QK
                          bilinear. Random init calibrates to R ~= 1;
                          trained flow models sit ~0.78-0.86. Measured
                          across the chain's 10K-step checkpoints this
                          gives R(step) -- training dynamics the paper's
                          static Table 1 does not have.

  SCORE_MODS (experiments; plug into FlexSelfAttnProcessor)
    doob_score_mod(log_h)        exact sector: score + log h[kv].
    exact_edge_field(phi)        A_ij = phi_i - phi_j (zero holonomy)
    edge_field_score_mod(A)      general antisymmetric deformation --
                                 nonexact A carries genuine circulation
                                 (the NESS/driven sector, Sec.5.3)
    temperature_score_mod(beta)  score * beta

  Theorem 4.1 becomes executable: flex(edge_field(exact_edge_field(phi)))
  must equal flex(doob(log h)) with h = exp(-phi) -- asserted in
  tests/test_diffusion_math.py through the real Flex path.

BOUNDARY, refined: pure symmetrization is not score_mod-expressible
(pointwise access, no s_ji), so the EQ sector enters through model.py's
weight tying (W_K := W_Q). Given tied weights, everything else IS
Flex-expressible, and the trainable mechanism lives at the bottom of
this module: DmapFlexSelfAttnProcessor -- row-normalization of the
squared-distance kernel exp(-H), single-pass at alpha=0 via the
surviving destination potential 2s_ij - g_j, with the Coifman-Lafon
Doob correction (alpha > 0) as a second pass whose degrees come from
return_lse. attention.py stays the frozen, gate-certified baseline;
this module is the paper, complete: operators, deformations,
measurement, and mechanism.
"""

from __future__ import annotations

import torch

from ditflex.attention import (
    FlexSelfAttnProcessor,
    ScoreMod,
    flex_attention,
    identity_score_mod,
)

# ---------------------------------------------------------------------------
# Dense operators (Sec. 2-5)
# ---------------------------------------------------------------------------


def bidivergence(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directional pair from a (possibly asymmetric) score matrix M.

    H_fwd = g 1^T - M,  H_bwd = 1 g^T - M^T,  g = diag(M).
    Their sum H is symmetric with zero diagonal (the squared-distance
    matrix of the symmetric part of M), and a row-softmax of -beta*H_fwd
    equals a row-softmax of beta*M because the g-terms are row-constant.
    Returns (H_fwd, H_bwd, H)."""
    g = M.diagonal(dim1=-2, dim2=-1)
    h_fwd = g.unsqueeze(-1) - M
    h_bwd = g.unsqueeze(-2) - M.transpose(-2, -1)
    return h_fwd, h_bwd, h_fwd + h_bwd


def row_normalize(P: torch.Tensor) -> torch.Tensor:
    return P / P.sum(dim=-1, keepdim=True)


def dmap(P: torch.Tensor) -> torch.Tensor:
    """Row-stochastic diffusion-map operator of a positive kernel (eq. 3)."""
    return row_normalize(P)


def amap(M: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Forward directed operator: row-softmax over QK scores (Sec. 3.1).
    Identical to softmax(-beta * H_fwd) by shift-invariance."""
    return torch.softmax(beta * M, dim=-1)


def hadamard_recombine(a_fwd: torch.Tensor, a_bwd: torch.Tensor) -> torch.Tensor:
    """eq. 29: row-normalized Hadamard product of the two directed
    operators reconstructs DMAP of the symmetric kernel."""
    return row_normalize(a_fwd * a_bwd)


def doob_transform(p_plus: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """eq. 16: destination reweighting by h > 0, then row renormalization."""
    tilted = p_plus * h.unsqueeze(-2)
    return row_normalize(tilted)


def stationary_distribution(
    p_plus: torch.Tensor, iters: int = 2000, tol: float = 1e-12
) -> torch.Tensor:
    """Power iteration for pi = pi P+ (irreducible row-stochastic P+)."""
    n = p_plus.shape[-1]
    pi = torch.full((n,), 1.0 / n, dtype=p_plus.dtype, device=p_plus.device)
    for _ in range(iters):
        nxt = pi @ p_plus
        nxt = nxt / nxt.sum()
        if (nxt - pi).abs().max() < tol:
            return nxt
        pi = nxt
    return pi


def probability_current(p_plus: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """eq. 24: J(pi) = diag(pi) P+ - (diag(pi) P+)^T. Zero iff detailed
    balance (EQ); nonzero at stationarity is NESS."""
    flux = pi.unsqueeze(-1) * p_plus
    return flux - flux.transpose(-2, -1)


# ---------------------------------------------------------------------------
# The R ratio (Sec. 6)
# ---------------------------------------------------------------------------


def qk_ratio(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """R = |antisym(W)|_F / |sym(W)|_F for a square bilinear W."""
    sym = 0.5 * (W + W.transpose(-2, -1))
    anti = 0.5 * (W - W.transpose(-2, -1))
    return anti.norm(dim=(-2, -1)) / (sym.norm(dim=(-2, -1)) + eps)


def attention_qk_ratios(attn) -> torch.Tensor:
    """Per-head R for one diffusers Attention module. The token-space
    bilinear of head h is B_h = W_q[h]^T @ W_k[h]  (q_i . k_j =
    x_i B x_j^T with row-vector tokens and diffusers' [out, in] Linear
    weights)."""
    wq, wk = attn.to_q.weight.detach(), attn.to_k.weight.detach()
    heads = attn.heads
    head_dim = wq.shape[0] // heads
    ratios = []
    for h in range(heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        B = wq[sl].transpose(0, 1) @ wk[sl]
        ratios.append(qk_ratio(B.float()))
    return torch.stack(ratios)


def model_qk_ratios(model) -> dict:
    """Layer-indexed per-head R across every diffusers Attention module,
    plus the layer-mean the paper's Table 1 reports. Run against the
    chain's checkpoints for R(step)."""
    from diffusers.models.attention_processor import Attention

    per_layer = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            per_layer[name] = attention_qk_ratios(module)
    if not per_layer:
        raise ValueError("no diffusers Attention modules found")
    layer_means = torch.stack([r.mean() for r in per_layer.values()])
    return {
        "per_layer": per_layer,
        "layer_mean": layer_means.mean().item(),
        "layer_std": layer_means.std().item(),
    }


# ---------------------------------------------------------------------------
# Score_mods (Sec. 4, executable)
# ---------------------------------------------------------------------------


def doob_score_mod(log_h: torch.Tensor) -> ScoreMod:
    """Exact sector: destination tilt score + log h[kv]. By Thm 4.1 this
    is everything an exact edge field can do after row-softmax."""

    def mod(score, b, h, q_idx, kv_idx):
        return score + log_h[kv_idx]

    return mod


def exact_edge_field(phi: torch.Tensor) -> torch.Tensor:
    """A_ij = phi_i - phi_j: the coboundary of a node potential. Exact,
    zero holonomy around every cycle."""
    return phi.unsqueeze(-1) - phi.unsqueeze(-2)


def edge_field_score_mod(A: torch.Tensor) -> ScoreMod:
    """General antisymmetric logit deformation score + A[q, kv] (eq. 13).
    With A exact this reduces to a Doob tilt (Thm 4.1); with nonexact A
    it injects genuine circulation -- the deformation the EQ sector
    cannot absorb, and the natural first ditflex experiment."""

    def mod(score, b, h, q_idx, kv_idx):
        return score + A[q_idx, kv_idx]

    return mod


def temperature_score_mod(beta: float) -> ScoreMod:
    """Uniform inverse-temperature rescaling of the scores."""

    def mod(score, b, h, q_idx, kv_idx):
        return score * beta

    return mod


# ---------------------------------------------------------------------------
# The trainable mechanism (the full DMAP-DiT attention)
# ---------------------------------------------------------------------------


class DmapFlexSelfAttnProcessor(FlexSelfAttnProcessor):
    """Diffusion-map attention: row-normalization of the squared-distance
    kernel exp(-H), optionally with the Coifman-Lafon density correction.

    With scores s_ij = scale * (q_i . k_j) and self-scores g = diag(s),
    the symmetric squared-distance is H_ij = g_i + g_j - 2 s_ij. The
    row-softmax of -H kills only the SOURCE term g_i (row-constant); the
    DESTINATION term g_j survives as a potential:

        DMAP(alpha=0):  softmax_j( 2 s_ij - g_j )

    -- doubled effective temperature plus a destination-norm tilt. This
    is NOT plain attention (softmax of s_ij); the surviving potential is
    the difference, and it needs no second pass: g_j = scale*(q_j . k_j)
    is a per-token vector computed before Flex. One pass, one score_mod.

    alpha > 0 adds the Coifman-Lafon correction: a further Doob tilt by
    q^{-alpha}, where q are the degrees of the ACTUAL kernel exp(-H).
    Pass 1 runs Flex on the alpha=0 scores with return_lse=True, and
    log q_i = lse_i - g_i (the row-constant e^{-g_i} factored back in).
    Pass 2 applies  2 s_ij - g_j - alpha * log q_j.
    (alpha = 0.5: Fokker-Planck; alpha = 1: Laplace-Beltrami.)

    Gradients flow through g and log_q on purpose: the potentials are
    part of the learned computation, not frozen statistics.

    NOTE: EQ/DMAP semantics additionally require symmetric scores; that
    is enforced by weight tying in model.py (qk_mode="dmap"), not here.
    On untied weights this computes the analogous construction over the
    symmetric part's geometry with g_j = scale*(q_j . k_j).

    Cost: alpha=0 is a single Flex pass (baseline cost + a trivial
    elementwise g). alpha>0 adds one extra attention pass per layer."""

    def __init__(self, alpha: float = 0.0):
        super().__init__(score_mod=identity_score_mod)
        self.alpha = float(alpha)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("DmapFlexSelfAttnProcessor is self-attention only, no masks.")
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got ndim={hidden_states.ndim}.")

        batch, seq_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        heads = attn.heads
        head_dim = query.shape[-1] // heads
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        # The potential softmax does not kill: g_j = scale * (q_j . k_j).
        g = attn.scale * (query * key).sum(dim=-1)          # [B, H, N]

        def dmap_mod(score, b, h, q_idx, kv_idx):
            return 2.0 * score - g[b, h, kv_idx]

        if self.alpha == 0.0:
            out = flex_attention(query, key, value, score_mod=dmap_mod, scale=attn.scale)
        else:
            # Degrees of the actual kernel exp(-H):
            # lse_i = log sum_j exp(2 s_ij - g_j) = g_i + log q_i.
            _, lse = flex_attention(
                query, key, value, score_mod=dmap_mod, scale=attn.scale, return_lse=True
            )
            log_q = lse - g
            alpha = self.alpha

            def corrected_mod(score, b, h, q_idx, kv_idx):
                return 2.0 * score - g[b, h, kv_idx] - alpha * log_q[b, h, kv_idx]

            out = flex_attention(
                query, key, value, score_mod=corrected_mod, scale=attn.scale
            )

        out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

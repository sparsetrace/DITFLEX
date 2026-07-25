"""
AMAP weight surgery.

Turns a trained SiT/DiT attention block into the AMAP parametrisation:

    S_ij = 1/2 <m_i, m_j>              # symmetric, PSD  (from W_M)
         + 1/2 ( <p_i, q_j> - <q_i, p_j> )   # antisymmetric directed field R (P, Q)

where, per attention block, from the fused qkv projection we take the row
blocks W_Q, W_K, W_V and set

    W_M = (W_Q + W_K) / sqrt(2)        # symmetric-PSD generator   (kept)
    W_N = (W_Q - W_K) / sqrt(2)        # negative-definite generator (DISCARDED)

The symmetric sector of standard attention is  1/2 W_M W_M^T - 1/2 W_N W_N^T
(indefinite); AMAP keeps only the +1/2 W_M W_M^T (PSD) half. The antisymmetric
sector 1/2 (W - W^T) = 1/2 (W_N W_M^T - W_M W_N^T) depends on W_N, so once W_N is
gone a nonzero directed field must be an *independent* parameter R = (P, Q):

    r_init="zero"    -> P, Q small-normal near-zero; the directed field regrows
                        during finetuning ("back to health"). Truly discards W_N.
    r_init="antisym" -> P = W_Q, Q = W_K, so R starts at the SiT's exact
                        antisymmetric part 1/2 (W - W^T); only the negative-
                        definite symmetric piece -1/2 W_N W_N^T is dropped.

This module has NO torch-cuda / modal / hub dependency: it operates on a plain
state_dict (dict[str, Tensor]) so it runs and unit-tests on CPU. AMAP.py wraps
it for Modal B200 + Hugging Face.

Output key convention (per matched block, `<attn>` = matched attn prefix):
    <attn>.wm.weight     [dim, dim]      symmetric-PSD generator W_M
    <attn>.wm.bias       [dim]           (b_Q + b_K)/sqrt(2)   (if qkv had bias)
    <attn>.v.weight      [dim, dim]      W_V, unchanged
    <attn>.v.bias        [dim]           b_V, unchanged        (if bias)
    <attn>.R.p.weight    [dim, dim]      directed field proj P (no bias)
    <attn>.R.q.weight    [dim, dim]      directed field proj Q (no bias)
The original `<attn>.qkv.*` keys are removed. `<attn>.proj.*` and every non-attn
key pass through untouched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

# Torch is the only import; kept lazy-friendly so `--help`/inventory work even
# in a minimal env, but surgery itself needs it.
import torch
from torch import Tensor

SQRT2 = math.sqrt(2.0)

# Matches the fused qkv weight of a transformer block and captures the attn
# prefix, e.g. "blocks.7.attn" from "blocks.7.attn.qkv.weight". Override via
# SurgeryConfig.qkv_weight_pattern if your keys differ (print an inventory
# first with `dry_run` to check).
DEFAULT_QKV_PATTERN = r"^(?P<attn>.*\.attn)\.qkv\.weight$"


@dataclass
class SurgeryConfig:
    num_heads: int
    r_init: str = "zero"                 # "zero" | "antisym"
    r_zero_std: float = 0.02             # std for near-zero directed field init
    qkv_weight_pattern: str = DEFAULT_QKV_PATTERN
    seed: int = 0
    # If your qk-norm checkpoint carries per-head q_norm/k_norm, surgery can't
    # cleanly fold two norms into one W_M projection. Default: refuse and tell
    # you to start from a pre-qk-norm (pure-recipe) checkpoint. Set True to drop
    # them with a warning instead.
    drop_qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.r_init not in ("zero", "antisym"):
            raise ValueError(f"r_init must be 'zero' or 'antisym', got {self.r_init!r}")


@dataclass
class SurgeryReport:
    matched_blocks: list[str] = field(default_factory=list)
    had_bias: bool = False
    dropped_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    residual_check: dict[str, float] = field(default_factory=dict)


def inventory(state_dict: dict[str, Tensor], pattern: str = DEFAULT_QKV_PATTERN) -> list[dict]:
    """Report attention blocks the surgery would touch. Use before writing."""
    rx = re.compile(pattern)
    out = []
    for k, v in state_dict.items():
        m = rx.match(k)
        if not m:
            continue
        attn = m.group("attn")
        bias_k = f"{attn}.qkv.bias"
        out.append(
            {
                "attn": attn,
                "qkv_weight_shape": tuple(v.shape),
                "has_bias": bias_k in state_dict,
                "qk_norm": any(
                    f"{attn}.{n}" in state_dict for n in ("q_norm.weight", "k_norm.weight")
                ),
            }
        )
    return out


def _split_qkv(w: Tensor, dim: int) -> tuple[Tensor, Tensor, Tensor]:
    if w.shape[0] != 3 * dim or w.shape[1] != dim:
        raise ValueError(
            f"expected fused qkv weight [3*dim, dim] = [{3*dim}, {dim}], got {tuple(w.shape)}"
        )
    q, k, v = w[:dim], w[dim : 2 * dim], w[2 * dim :]
    return q, k, v


def _antisym_residual(wq: Tensor, wk: Tensor, wm: Tensor, p: Tensor, q: Tensor) -> float:
    """
    Sanity number: how close is  1/2 W_M W_M^T + antisym(P,Q)  to the SiT's
    PSD-projected attention  1/2 W_M W_M^T + 1/2 (W - W^T) ?
    In r_init='antisym' this residual is ~0 (P=W_Q, Q=W_K reproduce it exactly).
    In r_init='zero' it equals the norm of the (discarded) directed field.
    Everything is fp64 to keep the check meaningful.
    """
    wq, wk, wm, p, q = (t.double() for t in (wq, wk, wm, p, q))
    target_antisym = 0.5 * (wq @ wk.t() - wk @ wq.t())
    got_antisym = 0.5 * (p @ q.t() - q @ p.t())
    num = (got_antisym - target_antisym).norm().item()
    den = target_antisym.norm().item() + 1e-12
    return num / den


def run_surgery(
    state_dict: dict[str, Tensor],
    cfg: SurgeryConfig,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Tensor], SurgeryReport]:
    """Return (new_state_dict, report). Pure; does not mutate the input dict."""
    rx = re.compile(cfg.qkv_weight_pattern)
    gen = torch.Generator().manual_seed(cfg.seed)
    out: dict[str, Tensor] = {}
    report = SurgeryReport()

    # Which attn prefixes are we transforming?
    attn_prefixes: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        m = rx.match(k)
        if m:
            attn_prefixes[m.group("attn")] = v
    if not attn_prefixes:
        raise ValueError(
            f"no qkv weights matched pattern {cfg.qkv_weight_pattern!r}; "
            f"run inventory() to inspect keys"
        )

    # Keys owned by matched blocks that we rewrite rather than pass through.
    owned: set[str] = set()
    for attn in attn_prefixes:
        owned |= {f"{attn}.qkv.weight", f"{attn}.qkv.bias"}
        for n in ("q_norm.weight", "k_norm.weight", "q_norm.bias", "k_norm.bias"):
            owned.add(f"{attn}.{n}")

    for attn, qkv_w in sorted(attn_prefixes.items()):
        dim = qkv_w.shape[1]
        head_dim = dim // cfg.num_heads
        if dim % cfg.num_heads:
            raise ValueError(f"{attn}: dim {dim} not divisible by num_heads {cfg.num_heads}")

        wq, wk, wv = _split_qkv(qkv_w, dim)
        wm = (wq + wk) / SQRT2
        out[f"{attn}.wm.weight"] = wm.clone()
        out[f"{attn}.v.weight"] = wv.clone()

        bias_k = f"{attn}.qkv.bias"
        if bias_k in state_dict:
            report.had_bias = True
            b = state_dict[bias_k]
            bq, bk, bv = b[:dim], b[dim : 2 * dim], b[2 * dim :]
            out[f"{attn}.wm.bias"] = ((bq + bk) / SQRT2).clone()
            out[f"{attn}.v.bias"] = bv.clone()

        # qk-norm handling
        if any(f"{attn}.{n}" in state_dict for n in ("q_norm.weight", "k_norm.weight")):
            if cfg.drop_qk_norm:
                report.warnings.append(f"{attn}: dropped q_norm/k_norm (drop_qk_norm=True)")
                for n in ("q_norm.weight", "k_norm.weight", "q_norm.bias", "k_norm.bias"):
                    if f"{attn}.{n}" in state_dict:
                        report.dropped_keys.append(f"{attn}.{n}")
            else:
                raise ValueError(
                    f"{attn} carries qk-norm params; surgery can't fold two head-norms "
                    f"into one W_M. Start from a pre-qk-norm checkpoint (README: "
                    f"'pre-migration checkpoints are the pure-recipe artifact'), or pass "
                    f"drop_qk_norm=True to discard them."
                )

        # Directed field R = (P, Q)
        if cfg.r_init == "antisym":
            p, q = wq.clone(), wk.clone()
        else:  # "zero": near-zero, but nonzero so gradients flow into R
            p = torch.empty_like(wq).normal_(0.0, cfg.r_zero_std, generator=gen)
            q = torch.empty_like(wk).normal_(0.0, cfg.r_zero_std, generator=gen)
        out[f"{attn}.R.p.weight"] = p
        out[f"{attn}.R.q.weight"] = q

        report.matched_blocks.append(attn)
        report.residual_check[attn] = _antisym_residual(wq, wk, wm, p, q)

    # Pass through everything not owned by a transformed block.
    for k, v in state_dict.items():
        if k in owned:
            continue
        out[k] = v

    log(
        f"[surgery] r_init={cfg.r_init}  blocks={len(report.matched_blocks)}  "
        f"bias={report.had_bias}  heads={cfg.num_heads}"
    )
    if report.matched_blocks:
        rc = report.residual_check[report.matched_blocks[0]]
        log(f"[surgery] antisym residual (block 0) = {rc:.3e} "
            f"({'≈0 expected' if cfg.r_init == 'antisym' else 'norm of discarded field'})")
    for w in report.warnings:
        log(f"[surgery][warn] {w}")
    return out, report


def build_amap_config(source_config: dict | None, cfg: SurgeryConfig, provenance: dict) -> dict:
    """Config.json for the AMAP checkpoint: source config + AMAP markers."""
    c = dict(source_config or {})
    c["qk_mode"] = "amap"
    c["amap"] = {
        "r_init": cfg.r_init,
        "num_heads": cfg.num_heads,
        "score": "0.5*<m_i,m_j> + 0.5*(<p_i,q_j> - <q_i,p_j>), scaled 1/sqrt(head_dim)",
        "params": {"symmetric": "attn.wm", "value": "attn.v", "directed_field": "attn.R.{p,q}"},
        "discarded": "W_N = (W_Q - W_K)/sqrt2  (negative-definite symmetric sector)",
        **provenance,
    }
    return c


# --------------------------------------------------------------------------- #
# Self-test: run `python surgery.py` to verify the algebra on a toy state_dict.
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    dim, heads, n_blocks = 32, 4, 3
    sd: dict[str, Tensor] = {}
    for i in range(n_blocks):
        sd[f"blocks.{i}.attn.qkv.weight"] = torch.randn(3 * dim, dim)
        sd[f"blocks.{i}.attn.qkv.bias"] = torch.randn(3 * dim)
        sd[f"blocks.{i}.attn.proj.weight"] = torch.randn(dim, dim)
        sd[f"blocks.{i}.mlp.fc1.weight"] = torch.randn(4 * dim, dim)  # passthrough
    sd["x_embedder.weight"] = torch.randn(dim, dim)  # passthrough

    inv = inventory(sd)
    assert len(inv) == n_blocks and all(b["has_bias"] for b in inv), inv

    # antisym mode must reproduce the true antisymmetric part exactly.
    out_a, rep_a = run_surgery(sd, SurgeryConfig(num_heads=heads, r_init="antisym"))
    assert max(rep_a.residual_check.values()) < 1e-10, rep_a.residual_check

    # Verify the symmetric-PSD reconstruction against the true attention's
    # symmetric sector minus the discarded negative-definite part.
    for i in range(n_blocks):
        w = sd[f"blocks.{i}.attn.qkv.weight"]
        wq, wk = w[:dim].double(), w[dim:2 * dim].double()
        wm = out_a[f"blocks.{i}.attn.wm.weight"].double()
        wn = (wq - wk) / SQRT2
        sym_true = 0.5 * (wq @ wk.t() + wk @ wq.t())            # indefinite
        sym_amap = 0.5 * (wm @ wm.t())                          # PSD kept
        # sym_true == sym_amap - 1/2 W_N W_N^T
        recon = sym_amap - 0.5 * (wn @ wn.t())
        rel = (recon - sym_true).norm() / (sym_true.norm() + 1e-12)
        assert rel < 1e-5, rel  # fp32 surgery vs fp64 reference; algebra exact in amap_attention.py
        # PSD check on the kept symmetric generator
        eig = torch.linalg.eigvalsh(sym_amap)
        assert eig.min() > -1e-9, eig.min()

    # zero mode: R present, near-zero, passthroughs intact, qkv gone.
    out_z, _ = run_surgery(sd, SurgeryConfig(num_heads=heads, r_init="zero", r_zero_std=0.02))
    for i in range(n_blocks):
        assert f"blocks.{i}.attn.qkv.weight" not in out_z
        assert f"blocks.{i}.attn.wm.weight" in out_z
        assert f"blocks.{i}.attn.R.p.weight" in out_z
        assert out_z[f"blocks.{i}.attn.R.p.weight"].std().item() < 0.1
    assert "x_embedder.weight" in out_z and "blocks.0.mlp.fc1.weight" in out_z

    print("selftest OK: antisym exact, PSD verified, discarded term = -1/2 W_N W_N^T, passthrough intact")


if __name__ == "__main__":
    _selftest()

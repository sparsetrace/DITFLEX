# HMAP  (Hodge MAP — the flux↔exact homotopy)

HMAP is the **Hodge completion**: it warm-starts from AMAP-40k, **freezes the
kinetic sector** (the metric W_M), and opens only the **antisymmetric sector**,
then adiabatically trades the flux for the exact/Doob sector with ONE knob
α (γ = 1−α):

    logit_ij = ½⟨m_i,m_j⟩            [frozen kinetic, W_M — W_N dropped]
             + (1−α)·½(𝒲 − 𝒲ᵀ)_ij    [flux  𝒜_AMAP, circulation]
             + α·½(g_i − g_j)          [exact 𝒜_exact, Doob;  g_i = q_a_i·k_a_i]

    α : 0  →  exactly AMAP-40k (kinetic + flux)
    α : 1  →  frozen kinetic + exact/Doob potential g  (a Coifman–Lafon, DMAP-class op)

The exact coboundary ½(g_i−g_j) collapses under softmax to the per-key Doob tilt
−g_j (the g_i part washes out). AMAP/DMAP are the curl-only / gradient-only faces
of this one operator; HMAP interpolates between them along the single homotopy α.

## Two-projection freeze

Freezing the metric while opening the flux requires two projections:
- **frozen** `qkv` (AMAP-40k) → q₀,k₀ (kinetic m=(q₀+k₀)/√2) and v (values);
- **trainable** `hmap_qk` [C,2C], init from qkv's q,k → q_a,k_a (flux + g).

`freeze_except_hmap` freezes everything except `hmap_qk`, so this is a *tiny*
finetune — the geometry is pinned at AMAP-40k and only the antisymmetric
generators move. No `‖W_Q−W_K‖` penalty (it competes with flow-matching); the
schedule alone drives the trade. `|Wq-Wk|` is logged so you can watch the flux
generator without penalizing it.

## Run

    modal run HMAP/HMAP.py --stage smoke
    modal run HMAP/HMAP.py --stage finetune --amap-repo jcandane/AMAP \
        --anneal-start 0 --anneal-end 40000 --steps 40000 \
        --save-every 10000 --sample-every 10000

Warm-starts from `--amap-repo`; α anneals 0→1 over `[anneal_start, anneal_end]`.
`--steps N` additive; anneal window read back from the checkpoint on resume. EMA
restarts at α=1 (the prior EMA spans the moving operator — same lesson as FMAP).
Per-checkpoint config records α and the window; sampling uses the saved α, or
override:

    modal run HMAP/sample_hmap.py --step latest --weights ema        # its native α
    modal run HMAP/sample_hmap.py --step 40000 --alpha 0.0           # AMAP operator
    modal run HMAP/sample_hmap.py --step 40000 --alpha 1.0           # exact operator

## What it measures

Watch the loss along α: 0→1. If it stays flat, the flux was reducible to the
exact/Doob sector at fixed geometry (the "sleight of hand" 𝒜_AMAP → 𝒜_exact holds
— consistent with 400k-DMAP sharpness). If it rises as α→1, that rise is the
irreducible circulation the potential can't absorb. Either way the α-vs-loss
curve quantifies the exact/circulating split — the empirical backbone of the
Hodge story. (See also the Hodge diagnostic on AMAP-40k: divergence → g,
PCA of the residual → circulation.)

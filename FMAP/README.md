# FMAP  (annealed AMAP → DMAP homotopy)

The **annealing arm**: start as AMAP (sharp) and adiabatically walk to DMAP by
scheduling a coefficient **Λ: 1 → 0**. This measures the antisymmetric flux's
value *smoothly*, rather than as two separate endpoint models.

Because AMAP's symmetric sector (½⟨m_i,m_j⟩) and DMAP's (the distance kernel
⟨m_i,m_j⟩ − ½‖m_i‖² − ½‖m_j‖²) differ, FMAP interpolates the **whole** operator so
both endpoints are exact:

    L(Λ) = (1−½Λ)⟨m_i,m_j⟩ − (1−Λ)·½(‖m_i‖²+‖m_j‖²) + ½Λ(⟨q_i,k_j⟩−⟨k_i,q_j⟩)
    m = (q+k)/√2

    Λ = 1  ->  L = ½⟨m_i,m_j⟩ + ½·flux            = AMAP  (sharp)
    Λ = 0  ->  L = −½‖m_i−m_j‖²                    = DMAP  (distance kernel)

Coupled (reuses qkv, **no fold** while Λ>0 — the flux needs q,k apart). Λ is a
per-module scalar set each step from `lambda_at(step, start, end)`: held at 1
until `anneal_start`, linear to 0 by `anneal_end`, then pure DMAP.

## Files (borrowed from /AMAP/)

- `fmap_attention.py` — annealed operator (`apply_fmap`, `set_lambda`,
  `lambda_at`). Self-test verifies L(1)=AMAP, L(0)=DMAP exactly.
- `fmap_common.py` — build/EMA/LatentStore/sample/checkpoint (Λ-aware loader).
- `FMAP.py` — Modal finetune; sets Λ each step, logs it, saves it per checkpoint.
- `sample_fmap.py` — Modal L4 sampler (sets Λ from the checkpoint config).

## Run

    # start from base SiT as AMAP (Λ=1), anneal to pure DMAP by step 40000
    modal run FMAP/FMAP.py --stage finetune --steps 40000 \
        --anneal-start 0 --anneal-end 40000 --save-every 10000 --sample-every 10000

- `--anneal-start` : hold Λ=1 (pure AMAP) until this step — set >0 for an AMAP
  warm-up so it sharpens like AMAP *before* annealing (e.g. 10000).
- `--anneal-end`   : Λ reaches 0 (pure DMAP) here.
- `--steps N` is additive; on resume the anneal window is read from the
  checkpoint so the schedule doesn't jump. Each checkpoint records its Λ.

Sampling a checkpoint uses the Λ it was saved at (0 => pure DMAP):

    modal run FMAP/sample_fmap.py --step latest --weights ema

## Reading it

The headline is the **Λ-vs-loss curve** as it anneals. If loss stays flat while
Λ: 1→0, the flux is unnecessary — pure DMAP is as good as AMAP, smoothly. If
loss rises as Λ→0, that rise *is* the measured value of the flux. Either way
it's a stronger statement than comparing the two endpoint models in isolation.

## Note

FMAP is coupled throughout; it only becomes fold-able (to W_M, dropping W_N)
*after* Λ reaches exactly 0 — a separate consolidation step, not done here.
This is a different arm from /DMAP/ (folded, warm-started) and /AMAP/ (Λ≡1).

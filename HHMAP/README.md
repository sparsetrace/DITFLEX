# HHMAP  (Hodge-Hodge MAP — two exact potentials)

The confirming experiment. Where **HMAP** gave the exact sector one channel
(`g = diag(𝒲)`, tied to the flux projection — only the diagonal-sourced 𝒜_w),
**HHMAP** frees a full exact potential `W_D` and freezes everything else, to test
cleanly whether the flux's reducible content is pure exact.

## Operator

    logit_ij = ½⟨m_i,m_j⟩                    [FROZEN kinetic, W_M from AMAP]
             + (1−α)·½(𝒲 − 𝒲ᵀ)_ij            [FROZEN flux, W_Q,W_K — annealed OFF]
             + α·½(w_i − w_j)                  [FROZEN tied w=‖m‖² — the DMAP potential]
             + α·½(φ_i − φ_j)                  [FREE φ=diag(𝓡 W_D 𝓡ᵀ) — the only trainable]

`m=(q₀+k₀)/√2`, `𝒲=q_a k_aᵀ` (frozen), `w_i=‖m_i‖²`, `φ_i=r_iᵀΛr_i` with `r=𝓡 P`.
`W_D = P diag(Λ) Pᵀ` is **symmetric-indefinite** (Λ mixed sign), init 0.

## Endpoints

- **α=0** — exactly AMAP-40k: `½⟨m,m⟩ + ½(𝒲−𝒲ᵀ)` (Gram + flux, no potential).
- **α=1** — DMAP + a free potential: `½⟨m,m⟩ + ½(w_i−w_j) + ½(φ_i−φ_j)`. The
  kinetic + tied `‖m‖²` is the DMAP distance kernel; `W_D` adds a free exact
  potential on top.

Both potentials ride α (AMAP has no one-body potential, so scheduling `w` in as
the flux anneals out keeps the deformation smooth — no bolt-on the frozen decoder
never saw). The exact coboundaries `½(w_i−w_j)`, `½(φ_i−φ_j)` collapse under
softmax to per-key Doob tilts `−w_j`, `−φ_j`.

## What it tests

As the flux anneals away, can a single free exact potential `W_D` absorb its
reducible content, **on top of** DMAP's own `‖m‖²` potential? `W_D` is the ONLY
trainable tensor and Λ inits 0, so mass can move to exactly one place — watch
`‖Λ‖` grow. If the α→1 loss stays **flat** (vs HMAP's rise), the flux's reducible
part was pure exact and `W_D` caught the part HMAP's single `g`-channel missed
(the off-diagonal-sourced 𝒜_v). Whatever residual remains is the genuine
**coexact** / irreducible circulation — the honest lower bound on the flux's
non-reducible content.

This is the operationalization of the Hodge decomposition:
`𝒜_flux = 𝒜_exact (⊃ 𝒜_w tied + 𝒜_v free W_D) ⊕ 𝒜_coexact (irreducible)`.

## Trainable surface

**Only** `W_D` (`wd_proj` + `_wd_lambda`). Kinetic, flux (`hmap_qk`), values,
proj, MLP, conditioning, and the tied `w`-potential are ALL frozen at AMAP-40k.
`~+2%` params (the `wd_proj` head, `[C,C]` or low-rank via `--wd-rank`).

## Run

    # warm-start from AMAP-40k, anneal 0->1 over 40k, only W_D trains
    modal run HHMAP/HHMAP.py --stage finetune --amap-repo jcandane/AMAP \
        --steps 40000 --anneal-start 0 --anneal-end 40000 \
        --save-every 10000 --sample-every 10000

    # sweep α on a trained checkpoint (see the trade at fixed weights)
    modal run HHMAP/sample_hhmap.py --step latest --weights ema --alpha 0.94
    modal run HHMAP/sample_hhmap.py --step latest --weights ema --alpha 1.0

## Files (mirrors /HMAP/)

- `hhmap_attention.py` — operator (`apply_hhmap`, `freeze_except_wd`, `alpha_at`,
  `set_alpha`); self-test verifies α=0==AMAP, α=1==DMAP+free-φ, only W_D trainable.
- `hhmap_common.py` — build/EMA/sampler/checkpoint helpers.
- `HHMAP.py` — Modal finetune (warm-start AMAP, freeze all but W_D, anneal).
- `sample_hhmap.py` — sampler with `--alpha` override for the α-sweep.
- `.github/workflows/HHMAP.yml`, `HHMAP-sample.yml`.

## The comparison this arm buys

- **HHMAP α→1 loss flat vs HMAP α→1 rise** — confirms the residual HMAP saw was
  under-parametrized *exact* content (the missing 𝒜_v channel), not irreducible
  circulation.
- **‖Λ‖ growth as α→1** — direct measurement of flux-exact-content migrating into
  the free potential.
- **residual at α=1** — the honest irreducible-coexact number, once BOTH exact
  channels (tied w + free W_D) are available.

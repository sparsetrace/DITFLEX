# DMAP (folded)

Attention as a **Mahalanobis distance kernel** — the symmetric PSD core of AMAP
with the antisymmetric flux removed (R ≡ 0). This variant **folds at step 0**:
because DMAP uses q,k only through μ = (q+k)/√2 = R·W_M, the direction
W_N = (W_Q−W_K)/√2 is dead weight, so the fused `qkv [3d,d]` is replaced by a
slimmer fused `wmv [2d,d] = [W_M ; W_V]` and the forward projects μ directly.

    μ = R·W_M
    logit_ij = −½‖μ_i − μ_j‖²   (× d_h^-½)          # ≤ 0, zero diagonal

The fold W_M = (W_Q+W_K)/√2 is **exact** for DMAP (the loss never depended on
W_N), so folding at step 0 loses no reachable solution while dropping ~⅓ of the
attention-projection params and optimizer state. It is a **one-way door**: a
folded checkpoint can't become AMAP or standard attention again.

## Files

- `dmap_attention.py` — `install_folded_dmap` (qkv→wmv fold + distance-kernel
  forward) and `fold_state_dict` (fold an AMAP checkpoint for warm-start).
  Self-test verifies the fold is exact and the state_dict is ⅓ smaller.
- `dmap_common.py` — shared build/EMA/LatentStore/sample/checkpoint helpers.
- `DMAP.py` — Modal B200 finetune with the warm-start chain below.
- `sample_dmap.py` — Modal L4 sampler (4×4 grid from any folded DMAP checkpoint).

## Checkpoint resolution (finetune, resume=auto)

1. **DMAP's own** latest in `--push-repo` (`jcandane/DMAP`, folded) → resume it.
2. else **AMAP's** latest in `--amap-repo` (`jcandane/AMAP`) → load its qkv,
   **fold W_Q,W_K→W_M**, step 0.
3. else **base SiT-XL/2** → fold its qkv, step 0.

`--steps N` trains N MORE steps. Checkpoints carry `attn.wmv.*` (not `qkv`).

## Run

    modal run DMAP/DMAP.py --stage smoke                              # fold + report size/shift
    modal run DMAP/DMAP.py --stage finetune --steps 40000 \
        --save-every 10000 --sample-every 10000 --sample-at-start
    modal run DMAP/sample_dmap.py --step latest --weights ema
    modal run DMAP/sample_dmap.py --step base                        # base SiT folded, un-healed

`--sample-at-start` renders a grid from the warm-started (folded, un-finetuned)
weights before step 1 — the "before" for the before/after comparison.

Workflows: `DMAP.yml` (finetune, B200) and `DMAP-sample.yml` (sampling, L4).
Needs `MODAL_TOKEN_ID/SECRET` + the Modal `HF_TOKEN` secret. Grids commit into
`DMAP/samples/`; checkpoints go to `jcandane/DMAP/checkpoints/step_*`.

## Notes

- Folded checkpoints are DMAP-only and can't be sampled by AMAP's sampler — keep
  the coupled AMAP checkpoints if you want AMAP↔DMAP weight comparisons.
- DMAP's rel-shift vs standard attention is large by design (distance kernel ≠
  dot-product), so expect a big step-1 loss bump, then healing. Logits are
  bounded above by construction, so it should be at least as stable as AMAP.

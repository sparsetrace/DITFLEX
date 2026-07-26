# DMAP

Attention as a **Mahalanobis distance kernel** — the symmetric PSD core of AMAP
with the antisymmetric flux removed (R ≡ 0). Sibling of `/AMAP/`.

    μ_i      = (q_i + k_i)/√2                    # R·W_M, W_M = (W_Q+W_K)/√2
    logit_ij = −½‖μ_i − μ_j‖²   (× d_h^-½)       # law of cosines, ≤ 0, zero diagonal

Softmax of a negative squared distance is a heat/diffusion kernel, bounded above
by construction — the README's `−|q_i−q_j|² + const`. That's why the dmap chain
never needed the stabilisers directed attention did. AMAP is this plus the flux
term `+½(q_i·k_j − k_i·q_j)`; DMAP drops it.

Both reuse the SiT qkv (no surgery, state_dict unchanged), so DMAP can be
**warm-started directly from an AMAP checkpoint**.

## Files (mirror /AMAP/)

- `dmap_attention.py` — the distance-kernel forward; `apply_dmap` monkeypatches
  timm `Attention`. Self-test verifies logit = −½‖μ_i−μ_j‖² (zero diag, ≤0, symmetric).
- `dmap_common.py` — shared build/EMA/LatentStore/sample/checkpoint helpers.
- `DMAP.py` — Modal B200 finetune with the warm-start chain below.
- `sample_dmap.py` — Modal L4 sampler (4×4 grid from any DMAP checkpoint).

## Checkpoint resolution (finetune, resume=auto)

1. **DMAP's own** latest in `--push-repo` (`jcandane/DMAP`) → resume it.
2. else **AMAP's** latest in `--amap-repo` (`jcandane/AMAP`) → warm-start, step 0.
3. else **base SiT-XL/2** → fresh.

`--steps N` trains N MORE steps from the resolved start. AMAP→DMAP is a one-time
warm start; after DMAP's first checkpoint it self-resumes. Provenance
(`warm_start_from`) is recorded in each `dmap_config.json`.

## Run

    modal run DMAP/DMAP.py --stage finetune --steps 21000 --save-every 10000 --sample-every 7000
    modal run DMAP/sample_dmap.py --step latest --weights ema
    modal run DMAP/sample_dmap.py --step base            # base + DMAP, before healing

Workflows: `DMAP.yml` (finetune, B200) and `DMAP-sample.yml` (sampling, L4).
Needs `MODAL_TOKEN_ID/SECRET` + the Modal `HF_TOKEN` secret. Grids commit into
`DMAP/samples/`; checkpoints go to `jcandane/DMAP/checkpoints/step_*`.

## Note on the shift

DMAP's rel-shift vs standard attention is large by design — a distance kernel is
very different from dot-product attention — so expect a bigger step-1 loss bump
than AMAP's 0.458, then healing. Because the logits are bounded above, it should
be at least as stable as AMAP (no large positive diagonal to tame).

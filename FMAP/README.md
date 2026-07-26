# FMAP  (SiT → DMAP direct)

The **SiT2DMAP** arm: fold **base SiT** directly into the Mahalanobis distance
kernel, no AMAP detour.

**The operator is identical to DMAP** — folded distance kernel, symmetric PSD
core, no flux (R ≡ 0):

    μ = R·W_M,   W_M = (W_Q + W_K)/√2,   logit_ij = −½‖μ_i − μ_j‖²

The *only* difference from `/DMAP/` is initialisation: DMAP folds from an **AMAP**
checkpoint (SiT→AMAP→DMAP); FMAP folds from **base SiT** (SiT→DMAP). Same
equation, no new ansatz — this arm isolates whether the AMAP detour helps.

Note: going SiT→DMAP directly *drops* W_N and the antisymmetric flux together
(the flux = ½(W_N W_Mᵀ − W_M W_Nᵀ) needs W_N) — it doesn't "cancel" them. FMAP
answers the empirical question: does the distance kernel reach the same quality
from base SiT as it does via AMAP, given equal training?

## Files (mirror /DMAP/)

- `fmap_attention.py` — folded distance-kernel op (`install_folded_fmap`,
  `fold_state_dict`). Self-test verifies the fold is exact, ⅓ smaller attn proj.
- `fmap_common.py` — shared build/EMA/LatentStore/sample/checkpoint helpers.
- `FMAP.py` — Modal finetune; resolves FMAP-own checkpoint → else folds base SiT.
- `sample_fmap.py` — Modal L4 sampler (4×4 grid from any folded FMAP checkpoint).

## Checkpoint resolution (finetune, resume=auto)

1. FMAP's own latest in `--push-repo` (`jcandane/FMAP`, folded) → resume.
2. else **base SiT-XL/2** → fold, step 0.   (no AMAP)

`--steps N` trains N more steps. Checkpoints carry `attn.wmv.*`, push to
`jcandane/FMAP`. Step-0 grid is snapshotted unconditionally.

## Run

    modal run FMAP/FMAP.py --stage finetune --steps 50000 --save-every 10000 --sample-every 10000
    modal run FMAP/sample_fmap.py --step latest --weights ema
    modal run FMAP/sample_fmap.py --step base            # base SiT folded, un-healed

## The comparison

Add FMAP to MAPtest to get the full picture:

    modal run MAPtest/maptest_fid.py \
      --models "sit,amap:jcandane/AMAP,dmap:jcandane/DMAP,fmap:jcandane/FMAP"

(MAPtest currently loads sit/amap/dmap; adding an `fmap` kind is a one-line
adapter change — say the word.) FMAP vs DMAP at matched steps = "does the AMAP
warm-start matter?"; FMAP vs AMAP = "flux vs no-flux, both from SiT."

# AMAP

Attention with a PSD-forced symmetric sector plus the standard antisymmetric
flux field — a magnetic-Laplacian-shaped attention operator.

    W_AMAP = ½ W_M Wᴹᵀ + ½ (W − Wᵀ),   W_M = (W_Q + W_K)/√2,   W = W_Q W_Kᵀ
           = W + ½ W_N W_Nᵀ             (W_N = (W_Q − W_K)/√2)

So AMAP = standard attention with the negative-definite symmetric piece
−½ W_N W_Nᵀ removed; the flux sector is untouched. Per head, in token space:

    m = (q + k)/√2
    logit_ij = ( ½⟨m_i,m_j⟩  +  ½(⟨q_i,k_j⟩ − ⟨k_i,q_j⟩) ) · d_h^-½
               └ PSD Gram ┘     └────── directed flux ──────┘

## Files

- `amap_attention.py` — eager AMAP forward (N≈256, no Flash/Flex). `apply_amap`
  monkeypatches `forward` onto timm `Attention` modules; **no new params, no
  state_dict changes** (7M SiT weights load as-is, transactional resume intact).
  Self-test verifies eager == explicit bilinear form and the ½WₙWₙᵀ identity.
- `AMAP.py` — Modal B200 entrypoint: load 7M SiT-XL/2 → apply AMAP → smoke or
  short flow-matching finetune on your HF latents → push checkpoint.
- `surgery.py` — the **decoupled arm** (later): materialises an independent W_M
  so the PSD kernel and the flux field train separately. Not used by the coupled
  run. Self-test verifies the PSD/antisym decomposition.

## Two arms

- **coupled** (build now): reuse `qkv`; W_M and flux share q,k, so one weight set
  is constrained by both sectors. `apply_amap` only.
- **decoupled** (later): `surgery.py` adds W_M := (W_Q+W_K)/√2 as its own
  parameter; the flux field keeps q,k. More capacity, needs the migration path.

## Run

    modal run AMAP.py --stage smoke
    modal run AMAP.py --stage finetune --steps 2000 \
        --latents-repo <you>/dlatentzz --push-repo <you>/sit-xl2-amap

Or dispatch `.github/workflows/AMAP.yml` (needs `MODAL_TOKEN_ID/SECRET` repo
secrets and the Modal `huggingface` secret with `HF_TOKEN`).

## Scale note

½⟨m_i,m_i⟩ is a large positive diagonal → AMAP logits are more self-biased and
higher-scale than the ⟨q,k⟩ the SiT trained at (this project's 240K–344K failure
axis). If the finetune won't settle, `--qk-rmsnorm` and/or `--learn-logit-scale`
bound it. Off by default so the first run is the faithful operator.

## Stage-2 hook (long finetune in the real supervisor)

In `run/modal_train.py`, after the model is built and weights loaded:

    if cfg.qk_mode == "amap":
        from AMAP.amap_attention import apply_amap, AMAPConfig
        apply_amap(model, AMAPConfig(qk_rmsnorm=cfg.qk_rmsnorm))

then drive it with the existing transactional chain (stability gates, adaptive
LR, promotion markers).

# dit-flex

DiT-L/2 on ImageNet-256 latents, with self-attention routed through PyTorch
FlexAttention so the attention score function is a swappable component.

Baselines: **DiT** (Peebles & Xie, 2023) for the DDPM objective, **SiT**
(Ma et al., 2024) for flow matching — same architecture, objective swapped,
directly comparable at DiT-L/2.

Two chains train in parallel, each with its own Hub checkpoint repo:

| chain | attention | repo | status |
|---|---|---|---|
| **amap** (baseline) | directed QK, R learned freely | `sparsetrace/ditflex-L2-flow` | ~344K / 400K; qk-norm migration at 344K |
| **dmap** (experiment) | W_K ≡ W_Q, symmetric scores, R ≡ 0 | `sparsetrace/ditflex-L2-flow-dmap` | ~117K / 400K; stable, no interventions |

Training is **time-boxed, transactional, and chained**: each job pulls the
last *committed healthy* checkpoint from the Hub, trains a candidate, and
promotes it only after consecutive stability windows pass loss and
gradient-distribution gates. A bad candidate exits with code 75 and is never
uploaded; the Modal supervisor retries from committed latest with a lower LR
factor and a fresh deterministic seed stream. Long training is many short
runs, not one long one.

---

## Stability findings (the 240K–344K arc)

This section records what actually happened, because the interventions in
this repo exist as responses to it.

**The failure.** From ~240K the amap chain developed a gradient-spike
instability with a distinctive signature: *loss perfectly flat* (~0.77)
while pre-clip gradient norms drifted up in slow motion — median 8 → 68 →
102 → 250 → 1000+ across ~60K steps — punctuated by spike storms that
tripped skip-guards and, at 273K, a transactional retry cascade.

**The diagnosis** (via `src/ditflex/probe.py`, opt-in rank-0 diagnostics):

1. **adaLN modulation weights grow without bound** under the published
   recipe (wd = 0). The `ada` family reached |w| ≈ 4900 of |w|_total ≈ 4930
   — an order of magnitude heavier than every other family combined — and
   carried **~99% of every gradient spike** (per-family attribution on each
   skipped step names it every time).
2. **Downstream, block-1's QK logits explode.** The probe measured them at
   3.4e6 → 8.6e6 → 16.2e6 over 270K → 334K. A softmax at that magnitude is
   an exactly one-hot, discontinuous switch: near-zero gradient almost
   everywhere, enormous gradient at flip boundaries — which is precisely
   the flat-loss-plus-spikes signature.
3. The two compose: adaLN's scale modulations amplify the tokens feeding
   attention; attention logits inherit the growth; spikes route back
   through adaLN.

**What helped, in order of leverage:**

* **QK-norm** (per-head `RMSNorm(head_dim, eps=1e-6)` on Q and K) — the
  structural fix, adopted at the 344K migration. Bounds per-head logits by
  construction; the saturated head's function survives (a logit gap of ~30
  is already functionally one-hot) while the flip-boundary cliffs do not.
* **tf32 precision** (fp32 activations, TF32 tensor-core matmuls — the
  published DiT/SiT numerics) — `--precision tf32`, now the default.
  Observed calmer gradient behavior than the previous bf16-autocast
  configuration at the same LR, at ~2× activation memory and lower
  throughput (~2.3 vs ~4 steps/s at batch 256). Not sufficient alone: the
  logit growth continued under tf32, confirming the pathology is
  architectural, not numerical.
* **adaLN-only decoupled weight decay** (`--wd-ada`, default 0.01 on the
  amap chain) — a targeted restoring force, `p *= 1 − lr·wd_ada` on the
  adaLN family only, applied outside the optimizer so checkpoints stay
  compatible. Measurably shrinks |w|_ada, but at safe doses it loses the
  race against episode-timescale logit growth — background hygiene, not
  the cure. Kept at 0.01 post-migration.
* **Adaptive LR backoff** (the v4 stability controller) — kept the chain
  alive and learning throughout (samples improved 270K → 344K), at the
  cost of running at ~28% of the scheduled LR. The controller's
  bounded-growth health reference also *normalized* the drift over many
  promotions (reference 38.5 → 320); treat a slowly ratcheting reference
  as a red flag, not adaptation.

**What the dmap chain shows.** Under the identical recipe, the R ≡ 0 chain
exhibits none of this: grad p90/median ≈ 1.1, zero skips, flat probe
logits (effective logits are bounded above by construction:
`−|q_i−q_j|² + const`). Its adaLN family is just as heavy — so adaLN
growth alone is not sufficient; the directed-attention chain's use of it
is part of the mechanism. This is itself a datapoint for the R-ratio
experiment. The dmap chain is deliberately **not** given qk-norm: untied
norms would break R ≡ 0, and a tied norm flattens the destination
potential `g_j` that defines the DMAP kernel. Pre-committed trigger: if
its probe shows logit growth or grad-median ratcheting in the 200–280K
range, a DMAP-appropriate intervention gets designed then, as its own arm.

**Comparability note for any writeup.** The chains are no longer
recipe-identical: amap carries {tf32 from ~275K, wd_ada = 0.01, qk-norm
from 344K}; dmap carries {tf32 from ~117K}. The honest framing is "each
arm run under the minimal stabilization it required, deviations tabulated
per arm"; per-run settings are recorded in every checkpoint's
`run_history[*].effective`.

---

## Known deviations from the published DiT/SiT recipe

* `out_channels = 4` (no learned-sigma channels; MSE-only objectives).
* Latents are posterior **mode**, not sampled; no horizontal-flip pass;
  torchvision Resize+CenterCrop rather than ADM `center_crop_arr`.
* **qk-norm** on the amap chain from step 344K (see above). Pre-migration
  checkpoints (≤ 344K, first revision) are the pure-recipe artifact.
* **wd_ada = 0.01** on the amap chain (adaLN-only decoupled decay).
* LR followed the adaptive controller, not constant 1e-4, from ~250K on
  the amap chain (retry backoffs; exact trajectory in `run_history`).
* dmap chain: W_K ≡ W_Q (~25M fewer params), DMAP logit modification —
  these ARE the experiment, not incidental deviations.

---

## Repo structure

```
DITFLEX/
├── .github/workflows/
│   ├── tests.yml                # CPU+GPU gates on push (Modal CI)
│   ├── quick-train.yml          # 2-GPU dress rehearsal of the full chain
│   ├── train.yml                # amap chain: pulls latest, transactional
│   ├── train-diffusion.yml      # dmap chain: same supervisor, qk-mode pinned
│   ├── train-recovery-270k.yml  # pinned-step bounded recovery segments
│   ├── recover-checkpoint.yml   # restore a healthy step as Hub latest
│   └── sampling.yml             # fixed-seed grids from both chains
├── run/
│   ├── modal_train.py           # THE transactional supervisor (both chains)
│   ├── migrate_qknorm.py        # one-shot 344K qk-norm migration CLI
│   └── recover_checkpoint.py
├── src/ditflex/
│   ├── attention.py             # Flex processor; qk-norm applied pre-kernel
│   ├── model.py                 # baseline builder (+ install_qk_norms)
│   ├── diffusion.py             # DMAP operators & score_mods (the paper)
│   ├── diffusion_model.py       # dmap builder (refuses qk_norm)
│   ├── migrate.py               # name-keyed checkpoint migration core
│   ├── probe.py                 # opt-in diagnostics: grad families, logits
│   ├── stability.py             # v4 controller: windows, references, retry
│   ├── train.py                 # transactional loop; --precision, --wd-ada,
│   │                            #   --qk-norm, --probe-attn-logits
│   ├── checkpoint.py / ema.py / latents.py / objective.py / sample.py
│   └── config.py / distributed.py
└── tests/                       # incl. verify_identity (both attention
                                 #   configs), test_migrate_qknorm, test_probe
```

---

## Operational runbook

**Routine links.** Dispatch `train` (amap) or `train-diffusion` (dmap) with
defaults. Both route through `run/modal_train.py`: stable resume selection,
bounded transactional retries, adaptive LR, promotion markers. amap
defaults: tf32, wd_ada 0.01, probe on, qk_norm **false until the 344K
migration is pushed, true after**. dmap defaults: tf32, wd_ada 0, probe on.

**The 344K qk-norm migration** (one-time, amap only):

```bash
# full local rehearsal, uploads nothing:
python run/migrate_qknorm.py --repo sparsetrace/ditflex-L2-flow --step 344000 --dry-run
# then for real:
python run/migrate_qknorm.py --repo sparsetrace/ditflex-L2-flow --step 344000 --push
```

The migration remaps the index-keyed AdamW state **by parameter name**
(inserting norm params shifts `named_parameters()` order — an
index-preserving load would attach moments to the wrong tensors), extends
the EMA shadow, embeds `qk_norm: true` in the config, resets the stability
reference (the pre-norm reference was contaminated by the divergence), and
pushes under an unmistakable commit message. Then:

1. **Warmup** — `train-recovery-270k`: `resume_step=344000`,
   `qk_norm=true`, `reset_lr_controller=true`, `lr=0.00001`,
   `max_steps=5000`. Expect a loss bump at step one (ones-init RMSNorms
   rescale Q/K), recovery within a few hundred steps, and the probe's
   blk1 logit line reading double digits instead of 1.6e7.
2. **Final stretch** — `train`: `qk_norm=true`, `reset_lr_controller=true`
   (the warmup's base LR differs; the controller refuses silent LR
   changes), `lr=0`, defaults otherwise → full cosine to 400K.

**Reading the probe.** `[probe] attn logits` healthy range is ~5–30 per
head; three-digit values are worth watching, sustained growth is the
alarm. `[probe] ... (SPIKE)` lines print raw pre-clip per-family norms —
the dominant family IS the spike's address. Turn the probe off
(`probe=false`) for routine links once trends are boring.

**Recovery.** Every promotion is a Hub commit; `recover-checkpoint.yml`
(dry-run by default) restores any historical step as latest. The
`stability window` log lines plus `run_history` in `state.json` are the
forensic record.

---

## Data notes

Latents: `sparsetrace/dlatentzz` — 32 safetensors shards, ~10.5 GB, 1.28M
ImageNet-1k train images, `[N, 4096]` bf16, **scaling factor 0.18215
already applied** (std ≈ 1.0 asserted on every load; ≈ 5.5 means unscaled,
≈ 0.18 means double-scaled). Encoded with `posterior.mode()`, no flips.
The full tensor lives on every GPU; batches are fancy-indexed with seeds
that are pure functions of `(base_seed, step, rank)` — resume is exact and
survives world-size changes. No DataLoader anywhere.

## Recipe (amap chain, as originally launched)

| | |
|---|---|
| model | DiT-L/2, 458M params, patch 2, 24 layers, width 1024, 16 heads |
| latents | 32×32×4 → 256 tokens |
| batch | 256 global |
| optimizer | AdamW, lr 1e-4, no warmup, wd 0 |
| EMA | 0.9999 |
| precision | bf16 autocast originally; **tf32 from ~275K** |
| label dropout | 10% (CFG) |
| stabilization | see "Stability findings" — wd_ada 0.01, qk-norm from 344K |

## Secrets

GitHub repo secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (launch only).
Modal secret `huggingface` → `HF_TOKEN` with write scope (latents pull,
checkpoint push). `modal run` uploads the checkout, so Modal never clones
from GitHub. Never commit tokens.

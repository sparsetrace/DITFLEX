# dit-flex

DiT-L/2 on ImageNet-256 latents, with self-attention routed through PyTorch
FlexAttention so the attention score function is a swappable component.

Baselines: **DiT** (Peebles & Xie, 2023) for the DDPM objective, **SiT**
(Ma et al., 2024) for flow matching — SiT is the same architecture with the
objective swapped, so both are directly comparable at the DiT-L/2 config.

Training runs are **time-boxed and chained**: each job trains for a fixed wall
clock, pushes a checkpoint to the HF Hub, and exits. The next job resumes from
it. Long training is many short runs, not one long one.

---

## Status / TODO

Ordered. Do not skip ahead — each step makes the next one interpretable.

### Phase 0 — correctness gates
- [x] Encode ImageNet-256 → SD-VAE latents, upload to HF
- [x] Verify latent reconstruction (decode → image looks right)
- [ ] `scripts/verify_identity.py` — identity FlexAttention vs default Diffusers
      processor, assert max abs diff < 1e-4 in bf16. **Blocks everything.**
- [ ] `scripts/verify_latents.py` — load from Hub, assert shape `[N, 4096]`,
      `std ≈ 1.0` (scaling factor already applied — see Data Notes), labels in
      `[0, 999]`, N matches manifest
- [ ] `scripts/overfit_smoke.py` — 128 samples, loss → ~0 in a few hundred steps

### Phase 1 — plumbing
- [ ] `config.py` dataclasses, JSON round-trip test
- [ ] `latents.py` — GPU-resident store, rank-offset deterministic sampling
- [ ] `objective.py` — DDPM eps and flow matching behind one interface
- [ ] `checkpoint.py` — save/load/resume, HF Hub push/pull
- [ ] `train.py` — DDP, `torch.compile`, time-boxed loop
- [ ] `modal_app.py` — the only Modal-aware file

### Phase 2 — CI and launch
- [ ] `.github/workflows/test.yml` — CPU unit tests on push
- [ ] `.github/workflows/smoke.yml` — manual, 2 GPU / 10 min, full path
      (download → train → checkpoint → upload) end to end
- [ ] `.github/workflows/run.yml` — manual, 8 GPU / 5 h, detached

### Phase 3 — first real run
- [ ] 2×B300 smoke: 1000 steps, confirm loss decreasing and checkpoint round-trips
- [ ] 8×B300 × 5 h, DDPM objective, batch 256 — matches published DiT-L/2 recipe
- [ ] Chain runs to 400K steps
- [ ] `eval.py` + cached ImageNet reference stats → FID vs published DiT-L/2

### Phase 4 — the actual experiment
- [ ] Flow matching objective, same config → compare against SiT-L/2
- [ ] The score_mod modification
- [ ] Horizontal-flip latents (see Data Notes) if chasing published numbers

---

## Repo structure

```
dit-flex/
├── README.md
├── pyproject.toml
├── .github/workflows/
│   ├── test.yml                 # CPU unit tests, on push
│   ├── smoke.yml                # manual: 2 GPU, ~10 min, full path
│   └── run.yml                  # manual: 8 GPU, ~5 h, --detach
├── modal_app.py                 # the ONLY Modal-aware file
├── src/ditflex/
│   ├── config.py                # dataclasses -> JSON -> checkpoint
│   ├── distributed.py           # rank/world/init, rank0-only helpers
│   ├── attention.py             # IdentityFlexSelfAttnProcessor + score_mods
│   ├── model.py                 # build DiT-L/2, swap processors
│   ├── latents.py               # GPU-resident store; NO DataLoader
│   ├── objective.py             # DDPM eps | flow matching
│   ├── ema.py
│   ├── train.py                 # loop, compile, DDP, time-box
│   ├── checkpoint.py            # save/load/resume + HF Hub
│   ├── sample.py                # DDIM / ODE sampling + CFG
│   └── eval.py                  # FID vs cached reference stats
├── scripts/
│   ├── prepare_latents.py       # cleaned from imagenet-processing.ipynb
│   ├── verify_identity.py
│   ├── verify_latents.py
│   └── overfit_smoke.py
└── tests/
    ├── test_attention_identity.py
    ├── test_latents_shapes.py
    └── test_config_roundtrip.py
```

`torch.compile` lives in `train.py`, not `model.py` — tests and
`verify_identity.py` need the uncompiled model.

---

## Secrets

The important thing is **where** each lives. GitHub only launches; Modal does
the work and needs the data credentials.

### GitHub repository secrets
Used by the workflow launcher only.

| Secret | Purpose |
|---|---|
| `MODAL_TOKEN_ID` | authenticate `modal run` from CI |
| `MODAL_TOKEN_SECRET` | same |

`GITHUB_TOKEN` is injected automatically and does not need to be created.
The repo being private does not require an extra token: `actions/checkout`
uses the automatic token, and `modal run` uploads the checked-out source
into the container, so Modal never clones from GitHub.

### Modal secrets
Created with `modal secret create <name> KEY=value`. Referenced by name in
`modal_app.py`.

| Modal secret | Keys | Purpose |
|---|---|---|
| `huggingface` | `HF_TOKEN` | pull latents dataset, push checkpoints — **needs write scope** |
| `wandb` *(optional)* | `WANDB_API_KEY` | run logging |

One token with write access is simplest. If you want least privilege, split
into a read token for `sparsetrace/dlatentzz` and a write token scoped to the
checkpoint repo, as `HF_TOKEN_READ` / `HF_TOKEN_WRITE`.

### Local development
`.env` (gitignored), or just export:

```bash
export HF_TOKEN=hf_...
modal token new          # writes ~/.modal.toml, no env var needed
```

**Never** commit tokens. The source notebook had `HF_TOKEN` read from env —
keep that pattern in `scripts/prepare_latents.py`.

---

## Data notes

Latents: `sparsetrace/dlatentzz` — 32 safetensors files, ~10.5 GB total,
1.28 M ImageNet train images.

Four properties of the encoding that the training code must respect:

1. **Flat storage.** Shape is `[N, 4096]`, not `[N, 4, 32, 32]`.
   `latents.py` must `.view(-1, 4, 32, 32)`.
2. **Scaling factor already applied.** `z = z * 0.18215` happened at encode
   time. **Do not apply it again.** Assert `std ≈ 1.0` on load — if you see
   `≈ 5.5`, something is double-scaling.
3. **Deterministic latents.** Encoded with `posterior.mode()`, not
   `.sample()`. DiT samples the posterior each epoch; we froze the mean.
   Deviation from the published recipe — acceptable, but state it in any writeup.
4. **No horizontal flips.** DiT trains with random h-flip before the VAE.
   Latents cannot be flipped directly (the conv VAE is not exactly
   equivariant), so matching the reference recipe requires encoding a second
   flipped pass (+10.5 GB, trivial at 96 GB/GPU).

`dtype` is bf16 on disk, cast to fp32 or bf16 at load depending on the
training precision.

---

## Design decisions

**No DataLoader, no DistributedSampler.** The full 10.5 GB tensor lives on
each GPU. Each rank draws indices from a generator seeded by
`(global_step, rank)` — stateless, so resume is exact and survives a change
of world size.

**DDP, not FSDP.** DiT-L is 458 M params; weights + EMA + AdamW states are
~7.3 GB in fp32. Sharding buys nothing at this scale and costs complexity.

**Fixed shapes everywhere.** Fixed batch, fixed 256-token sequence,
`drop_last=True` → one `torch.compile`, no recompiles.

**Compile inner, then DDP-wrap.** Test the other order once; this interaction
has been version-sensitive.

**Save uncompiled, unwrapped state dicts.** Strip `_orig_mod.` and `module.`
prefixes before writing, or checkpoints will not load into a bare model for
sampling.

---

## Checkpointing

Runs are time-boxed. The loop checks a deadline every 500 steps (rank 0
decides, broadcast to all — avoids clock drift), stops cleanly, saves, uploads.

Budget: ~7.3 GB per checkpoint, ~12 min up and ~12 min down at 100 MB/s.
A 5 h job is ~4.5 h of training. `torch.compile` costs another 2–5 min on a
cold container.

Hub layout (`sparsetrace/dit-flex-L2`, model repo):

```
state.json              # step, wall-clock, config, git sha, run_history
model.safetensors       # fp32 weights
ema.safetensors         # fp32 EMA (0.9999)
optim.safetensors       # AdamW m, v
archive/step_0200000/   # periodic, EMA + state only, kept forever
```

Top level is always "latest" and is overwritten each run; HF repos are git,
so prior revisions remain recoverable. `run_history` in `state.json` records
each run's step range and duration — worth having when a loss discontinuity
turns out to align with a run boundary.

---

## Quickstart

```bash
# gates
python scripts/verify_identity.py
python scripts/verify_latents.py
python scripts/overfit_smoke.py

# short run on Modal
modal run modal_app.py::train --gpus 2 --train-seconds 600 --objective ddpm

# real run, detached (survives the CI job exiting)
modal run --detach modal_app.py::train \
    --gpus 8 --train-seconds 18000 --objective flow
```

---

## Recipe

Held at the published DiT-L/2 settings so the baseline is comparable:

| | |
|---|---|
| model | DiT-L/2, 458 M params, patch 2, 24 layers, width 1024, 16 heads |
| latents | 32×32×4 → 256 tokens |
| batch | 256 global |
| optimizer | AdamW, lr 1e-4 constant, no warmup, no weight decay |
| EMA | 0.9999 |
| precision | bf16 autocast, fp32 master weights |
| label dropout | 10% (for classifier-free guidance) |

Do not scale the batch on the first run — if batch and objective change
together, the comparison to published numbers means nothing.

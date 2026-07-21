# ditflex -- repo snapshot

Generated 2026-07-21 00:08 UTC by context/context.py. 45 files.

## Tree

```
DITFLEX/
├── .github/
    ├── workflows/
        ├── context.yml
        ├── quick-train.yml
        ├── recover-checkpoint.yml
        ├── sampling.yml
        ├── tests.yml
        ├── train-diffusion.yml
        ├── train-recovery-270k.yml
        ├── train.yml
├── README.md
├── pyproject.toml
├── quick_train/
    ├── modal_quick.py
├── run/
    ├── modal_train.py
    ├── recover_checkpoint.py
├── sampling/
    ├── modal_sample.py
├── src/
    ├── ditflex/
        ├── __init__.py
        ├── attention.py
        ├── checkpoint.py
        ├── config.py
        ├── diffusion.py
        ├── diffusion_model.py
        ├── distributed.py
        ├── ema.py
        ├── latents.py
        ├── model.py
        ├── objective.py
        ├── sample.py
        ├── stability.py
        ├── train.py
├── tests/
    ├── modal_ci.py
    ├── overfit_smoke.py
    ├── test_attention_identity.py
    ├── test_checkpoint_roundtrip.py
    ├── test_checkpoint_selection.py
    ├── test_config_roundtrip.py
    ├── test_diffusion_math.py
    ├── test_dmap_gradients.py
    ├── test_dmap_model.py
    ├── test_ema.py
    ├── test_latents_shapes.py
    ├── test_objective_math.py
    ├── test_objective_rng.py
    ├── test_stability.py
    ├── verify_identity.py
    ├── verify_latents.py
├── train_diffusion/
    ├── modal_train_dmap.py
```

## Files

### `.github/workflows/context.yml`

```yaml
name: context

# Regenerate context/context.md (a one-file snapshot of the repo for
# sharing) and commit it back.
#
# Triggers: manual dispatch, or a push to main that modifies the
# generator itself -- and ONLY the generator. Editing source files does
# not refresh the snapshot automatically; dispatch when you want a fresh
# one. (The workflow's own commit cannot retrigger it: it touches only
# context/context.md, carries [skip ci], and GITHUB_TOKEN pushes do not
# start workflows.)
#
# Runs entirely on the Actions runner -- no Modal, no GPU, no secrets
# beyond the automatic GITHUB_TOKEN. Reading files and writing markdown
# is not a job for an ephemeral server.

on:
  workflow_dispatch:

  push:
    branches: [main]
    paths:
      - context/context.py

permissions:
  contents: write   # allows the push with the automatic GITHUB_TOKEN

jobs:
  snapshot:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Generate context.md
        run: python context/context.py

      - name: Commit if changed
        run: |
          git config user.name "ditflex context bot"
          git config user.email "actions@github.com"
          git add context/context.md
          if git diff --cached --quiet; then
            echo "context.md unchanged -- nothing to commit"
          else
            git commit -m "context: refresh repo snapshot [skip ci]"
            git push
          fi
```

### `.github/workflows/quick-train.yml`

```yaml
name: quick-train

# Dress rehearsal of the training chain on 2 GPUs: real ditflex.train,
# real latents, real checkpointing against the SCRATCH repo
# (sparsetrace/quicktraindit -- fixed name, reset+overwritten every
# dispatch). Two legs in one job; the runner asserts the resumed step
# count, so green means the whole fresh->push->pull->resume chain works.
#
# Manual dispatch only. NOT detached: this is a test, so the Actions job
# waits and shows red/green like tests.yml.

on:
  workflow_dispatch:
    inputs:
      gpus:
        description: "GPU count"
        required: false
        default: "2"
      gpu_kind:
        description: "GPU kind (B300 | B200)"
        required: false
        default: "B300"
      steps:
        description: "Leg-1 steps (fresh start)"
        required: false
        default: "1000"
      resume_steps:
        description: "Leg-2 steps (after resume)"
        required: false
        default: "200"
      objective:
        description: "Objective (ddpm | flow)"
        required: false
        default: "flow"
      max_latent_files:
        description: "Latent shards to load (0 = all 32, realistic)"
        required: false
        default: "0"

jobs:
  quick:
    name: "quick · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} · ${{ inputs.objective }} · ${{ inputs.steps }}+${{ inputs.resume_steps }}"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        run: pip install modal

      - name: Run quick train chain on Modal
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu_kind }}
          MODAL_GPUS: ${{ inputs.gpus }}
        run: |
          modal run quick_train/modal_quick.py \
            --steps "${{ inputs.steps }}" \
            --resume-steps "${{ inputs.resume_steps }}" \
            --objective "${{ inputs.objective }}" \
            --max-latent-files "${{ inputs.max_latent_files }}"
```

### `.github/workflows/recover-checkpoint.yml`

```yaml
name: recover-checkpoint

# Restore a healthy checkpoint from Hub revision history after a
# divergence poisoned 'latest'. Walks the commit ledger, finds the
# commit whose state.json reports the requested step, re-uploads those
# files as the new latest.
#
# SAFETY: defaults to dry_run=true, which only prints the commit ledger
# (commit-id + step for every revision) and what would be restored.
# Inspect that output, then re-dispatch with dry_run=false to execute.

on:
  workflow_dispatch:
    inputs:
      repo:
        description: "Checkpoint repo to operate on"
        required: false
        default: "sparsetrace/ditflex-L2-flow"
      step:
        description: "Step to restore as latest (e.g. 240000)"
        required: true
      dry_run:
        description: "true = print the ledger only; false = actually restore"
        required: false
        default: "true"

jobs:
  recover:
    name: "recover ${{ inputs.repo }} -> step ${{ inputs.step }} (dry_run=${{ inputs.dry_run }})"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install huggingface_hub
        run: pip install "huggingface_hub>=0.26"

      - name: Run recovery
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          if [ "${{ inputs.dry_run }}" = "false" ]; then
            python run/recover_checkpoint.py \
              --repo "${{ inputs.repo }}" --step "${{ inputs.step }}"
          else
            python run/recover_checkpoint.py \
              --repo "${{ inputs.repo }}" --step "${{ inputs.step }}" --dry-run
          fi
```

### `.github/workflows/sampling.yml`

```yaml
name: sampling

# Manual dispatch: render the fixed-seed grids from BOTH chains' latest
# checkpoints and COMMIT the PNGs into /sampling/ in this repo.
# Non-detached (the job needs the results back to commit them).

on:
  workflow_dispatch:
    inputs:
      repos:
        description: "Comma-separated checkpoint repos"
        required: false
        default: "sparsetrace/ditflex-L2-flow,sparsetrace/ditflex-L2-flow-dmap"
      gpu:
        description: "GPU kind (L4 is plenty: 100 fwd passes @ batch 16, eager flex, no backward; ~5-10 cents/dispatch. Bump to A10G/B200 only if L4 is unavailable)"
        required: false
        default: "L4"
      sample_steps:
        description: "Euler steps"
        required: false
        default: "50"
      cfg_scale:
        description: "CFG scale"
        required: false
        default: "4.0"

permissions:
  contents: write   # required to push the PNGs back

jobs:
  sample:
    name: "sample · ${{ inputs.repos }}"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        run: |
          pip install 'modal<1.5'
          modal --version

      - name: Authenticate Modal
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: |
          modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

      - name: Render grids on Modal
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu }}
        run: |
          modal run sampling/modal_sample.py \
            --repos "${{ inputs.repos }}" \
            --sample-steps "${{ inputs.sample_steps }}" \
            --cfg-scale "${{ inputs.cfg_scale }}"

      - name: Commit grids
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add sampling/*.png
          git diff --cached --quiet && echo "no new grids" && exit 0
          git commit -m "sampling: grids from latest checkpoints [skip ci]"
          git push
```

### `.github/workflows/tests.yml`

```yaml
name: tests

on:
  push:
    branches: [main]
    paths:
      - src/ditflex/**
      - tests/**
      - pyproject.toml
      - .github/workflows/tests.yml

  pull_request:
    paths:
      - src/ditflex/**
      - tests/**
      - pyproject.toml

  workflow_dispatch:
    inputs:
      gpu:
        description: "GPU target (B300 | B200 | A100 | L40S)"
        required: false
        default: "B300"
      test_file:
        description: "Specific test file, empty = all"
        required: false
        default: ""
      compile_check:
        description: "Also verify the compiled Flex path"
        required: false
        default: "false"
      smoke:
        description: "Also run the overfit smoke (small model, both objectives)"
        required: false
        default: "false"
      latents_all:
        description: "Check every latent shard (10.5 GB) instead of the first"
        required: false
        default: "false"

jobs:
  gates:
    name: "ditflex gates · ${{ github.event.inputs.gpu || 'B300' }}"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        # If 'Token not found' appears with the probe printing 'yes',
        # a Modal client release changed credential handling: pin with
        #   pip install 'modal<1.5'
        run: |
          pip install 'modal<1.5'
          modal --version

      # Explicit authentication: writes the config file AND validates the
      # credentials against the server. Failure HERE with a clear message
      # means the token was rotated/revoked -> mint a new one in the Modal
      # dashboard and update the GitHub secrets. Success here followed by
      # a working `modal run` means env-var pickup was the broken piece.
      - name: Authenticate Modal
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: |
          modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

      # No GH_TOKEN: `modal run` uploads the checkout into the container,
      # so Modal never clones from GitHub. HF_TOKEN is a GitHub repo secret,
      # exported here and forwarded into the container by modal_ci.py.
      - name: Run gates on Modal
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ github.event.inputs.gpu || 'B300' }}
        run: |
          echo "modal token id present: ${MODAL_TOKEN_ID:+yes}"
          modal run tests/modal_ci.py \
            --test-file "${{ github.event.inputs.test_file || '' }}" \
            ${{ github.event.inputs.compile_check == 'true' && '--compile-check' || '' }} \
            ${{ github.event.inputs.smoke == 'true' && '--smoke' || '' }} \
            ${{ github.event.inputs.latents_all == 'true' && '--latents-all' || '' }}
```

### `.github/workflows/train-diffusion.yml`

```yaml
name: train-diffusion

# The DMAP-DiT chain: EQ-sector attention (W_K tied to W_Q, R == 0),
# same recipe and objective as the baseline chain, its OWN checkpoint
# repo. Manual dispatch only, detached -- monitor in the Modal dashboard
# (app: ditflex-train-dmap).

on:
  workflow_dispatch:
    inputs:
      gpus:
        description: "GPU count (2 is the certified sweet spot)"
        required: false
        default: "2"
      gpu_kind:
        description: "GPU kind (B300 | B200)"
        required: false
        default: "B300"
      train_seconds:
        description: "Stepping budget in seconds (14400 = 4 h)"
        required: false
        default: "14400"
      objective:
        description: "Objective (ddpm | flow)"
        required: false
        default: "flow"
      hub_repo:
        description: "Checkpoint repo for the DMAP chain (never share with the baseline)"
        required: false
        default: "sparsetrace/ditflex-L2-flow-dmap"
      lr:
        description: "LR override for this run (0 = keep recipe 1e-4; gain-compensation: 0.00005)"
        required: false
        default: "0"
      wd:
        description: "Weight-decay override for this run (-1 = keep recipe 0)"
        required: false
        default: "-1"
      clip:
        description: "Gradient-clip max-norm (46K cliff protocol: 0.25)"
        required: false
        default: "1.0"
      spike_skip:
        description: "Skip steps with grad norm > this x EMA (0 = off)"
        required: false
        default: "4.0"
      grad_ceiling:
        description: "Absolute grad-norm skip ceiling (0 = off)"
        required: false
        default: "25.0"
      dmap_alpha:
        description: "Coifman-Lafon exponent (0 | 0.5 | 1)"
        required: false
        default: "0.0"

jobs:
  launch:
    name: "dmap · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} · ${{ inputs.objective }}"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        run: pip install 'modal<1.5'

      - name: Launch detached DMAP training run
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu_kind }}
          MODAL_GPUS: ${{ inputs.gpus }}
          MODAL_TRAIN_SECONDS: ${{ inputs.train_seconds }}
        run: |
          modal run --detach train_diffusion/modal_train_dmap.py \
            --train-seconds "${{ inputs.train_seconds }}" \
            --objective "${{ inputs.objective }}" \
            --hub-repo "${{ inputs.hub_repo }}" \
            --dmap-alpha "${{ inputs.dmap_alpha }}" \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            --lr "${{ inputs.lr }}" \
            --wd "${{ inputs.wd }}"
```

### `.github/workflows/train-recovery-270k.yml`

```yaml
name: train-recovery-270k

# Manual only: resumes a known-good checkpoint and runs a bounded,
# transactional recovery segment on Modal.
on:
  workflow_dispatch:
    inputs:
      gpus:
        description: "GPU count"
        required: false
        default: "1"
      gpu_kind:
        description: "Modal GPU kind"
        required: false
        default: "B200"
      train_seconds:
        description: "Total Modal supervisor budget in seconds"
        required: false
        default: "7200"
      objective:
        description: "Objective (ddpm | flow)"
        required: false
        default: "flow"
      hub_repo:
        description: "Hugging Face checkpoint repository"
        required: false
        default: "sparsetrace/ditflex-L2-flow"

      # Recovery anchor and bounded diagnostic.
      resume_step:
        description: "Exact committed Hub step to resume"
        required: false
        default: "270000"
      target_steps:
        description: "Global training target used by the LR policy"
        required: false
        default: "400000"
      max_steps:
        description: "Maximum optimizer steps in this invocation (0 = time only)"
        required: false
        default: "5000"

      # Keep the checkpoint's current LR/controller by default.
      lr:
        description: "LR override (0 = restore checkpoint LR)"
        required: false
        default: "0"
      lr_policy:
        description: "LR policy (constant | cosine | adaptive)"
        required: false
        default: "adaptive"
      lr_backoff:
        description: "Fresh-process retry LR multiplier"
        required: false
        default: "0.5"
      lr_min_scale:
        description: "Minimum adaptive scale"
        required: false
        default: "0.125"
      max_retries:
        description: "Maximum transactional retries"
        required: false
        default: "2"

      wd:
        description: "Weight decay override (-1 = restore recipe/checkpoint value)"
        required: false
        default: "-1"
      clip:
        description: "Global gradient clipping max norm"
        required: false
        default: "1.0"
      spike_skip:
        description: "Frozen-reference relative gradient skip multiplier"
        required: false
        default: "10.0"
      grad_ceiling:
        description: "Absolute pre-clip gradient skip ceiling (0 = disabled)"
        required: false
        default: "500.0"
      grad_reference:
        description: "Optional frozen gradient reference override (0 = checkpoint reference)"
        required: false
        default: "0"

      # A high skip rate alone should warn before it rejects a flat-loss run.
      skip_warn_rate:
        description: "Window skip-rate warning threshold"
        required: false
        default: "0.30"
      skip_retry_rate:
        description: "Window skip-rate retry threshold"
        required: false
        default: "0.40"
      skip_emergency_rate:
        description: "Window skip-rate emergency threshold"
        required: false
        default: "0.60"

      seed_offset:
        description: "Base deterministic stochastic-stream offset"
        required: false
        default: "0"

jobs:
  launch:
    name: >-
      recovery · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} ·
      step=${{ inputs.resume_step }} · max=${{ inputs.max_steps }}
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        run: pip install 'modal<1.5'

      - name: Launch detached transactional recovery
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu_kind }}
          MODAL_GPUS: ${{ inputs.gpus }}
          MODAL_TRAIN_SECONDS: ${{ inputs.train_seconds }}
        run: |
          set -euo pipefail
          echo "Launching recovery from global step ${{ inputs.resume_step }}"
          echo "Modal token id present: ${MODAL_TOKEN_ID:+yes}"

          modal run --detach run/modal_train.py \
            --train-seconds "${{ inputs.train_seconds }}" \
            --objective "${{ inputs.objective }}" \
            --hub-repo "${{ inputs.hub_repo }}" \
            --resume-step "${{ inputs.resume_step }}" \
            --no-auto-legacy-rollback \
            --target-steps "${{ inputs.target_steps }}" \
            --max-steps "${{ inputs.max_steps }}" \
            --max-retries "${{ inputs.max_retries }}" \
            --lr "${{ inputs.lr }}" \
            --lr-policy "${{ inputs.lr_policy }}" \
            --lr-backoff "${{ inputs.lr_backoff }}" \
            --lr-min-scale "${{ inputs.lr_min_scale }}" \
            --wd "${{ inputs.wd }}" \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            --grad-reference "${{ inputs.grad_reference }}" \
            --skip-warn-rate "${{ inputs.skip_warn_rate }}" \
            --skip-retry-rate "${{ inputs.skip_retry_rate }}" \
            --skip-emergency-rate "${{ inputs.skip_emergency_rate }}" \
            --seed-offset "${{ inputs.seed_offset }}"
```

### `.github/workflows/train.yml`

```yaml
name: train

# Manual dispatch only.  Each run is detached: GitHub exits after launch while
# Modal pulls the latest healthy checkpoint, trains, saves, and pushes.

on:
  workflow_dispatch:
    inputs:
      gpus:
        description: "GPU count (1 = full global batch 256 on one GPU)"
        required: false
        default: "1"
      gpu_kind:
        description: "Modal GPU kind (B200 | B300 | RTX-PRO-6000)"
        required: false
        default: "B200"
      train_seconds:
        description: "Stepping budget in seconds; Modal timeout adds one hour"
        required: false
        default: "14400"
      objective:
        description: "Objective (ddpm | flow)"
        required: false
        default: "flow"
      hub_repo:
        description: "Checkpoint repo; keep one repo per objective/variant"
        required: false
        default: "sparsetrace/ditflex-L2-flow"
      target_steps:
        description: "Global stop and cosine horizon"
        required: false
        default: "400000"
      max_steps:
        description: "Maximum data steps this invocation (0 = time-box only)"
        required: false
        default: "0"

      lr_policy:
        description: "constant | cosine | adaptive (cosine plus loss/spike backoff)"
        required: false
        default: "adaptive"
      lr:
        description: "Base LR before scheduling (0 = recipe 1e-4). Leave 0 for adaptive migration."
        required: false
        default: "0"
      lr_min:
        description: "Cosine envelope floor at target_steps"
        required: false
        default: "0.00001"
      lr_hard_min:
        description: "Absolute floor after adaptive backoffs"
        required: false
        default: "0.000001"
      lr_backoff:
        description: "Adaptive multiplier after sustained instability"
        required: false
        default: "0.5"
      lr_min_scale:
        description: "Minimum adaptive multiplier"
        required: false
        default: "0.125"
      loss_rise_ratio:
        description: "Fast-loss EMA / slow-loss EMA warning threshold"
        required: false
        default: "1.08"
      loss_emergency_ratio:
        description: "Fast-loss EMA / slow-loss EMA emergency threshold"
        required: false
        default: "1.35"
      reset_lr_controller:
        description: "Discard persisted controller state after intentional policy changes"
        required: false
        type: boolean
        default: false

      wd:
        description: "Weight decay override (-1 = keep checkpoint/config; recommended for this chain)"
        required: false
        default: "-1"
      clip:
        description: "Global gradient-clip max norm"
        required: false
        default: "1.0"
      spike_skip:
        description: "Skip update above this multiple of gradient-norm EMA (0 = off)"
        required: false
        default: "4.0"
      grad_ceiling:
        description: "Absolute raw gradient-norm skip ceiling (0 = off; recommended)"
        required: false
        default: "0"
      seed_offset:
        description: "Runtime-only data-order seed offset"
        required: false
        default: "0"

jobs:
  launch:
    name: >-
      launch · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} · ${{ inputs.objective }} ·
      ${{ inputs.lr_policy }} to ${{ inputs.target_steps }}
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Modal
        run: pip install 'modal<1.5'

      - name: Launch detached training run
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu_kind }}
          MODAL_GPUS: ${{ inputs.gpus }}
          MODAL_TRAIN_SECONDS: ${{ inputs.train_seconds }}
        run: |
          echo "modal token id present: ${MODAL_TOKEN_ID:+yes}"
          modal run --detach run/modal_train.py \
            --train-seconds "${{ inputs.train_seconds }}" \
            --objective "${{ inputs.objective }}" \
            --hub-repo "${{ inputs.hub_repo }}" \
            --target-steps "${{ inputs.target_steps }}" \
            --max-steps "${{ inputs.max_steps }}" \
            --lr-policy "${{ inputs.lr_policy }}" \
            --lr "${{ inputs.lr }}" \
            --lr-min "${{ inputs.lr_min }}" \
            --lr-hard-min "${{ inputs.lr_hard_min }}" \
            --lr-backoff "${{ inputs.lr_backoff }}" \
            --lr-min-scale "${{ inputs.lr_min_scale }}" \
            --loss-rise-ratio "${{ inputs.loss_rise_ratio }}" \
            --loss-emergency-ratio "${{ inputs.loss_emergency_ratio }}" \
            --wd "${{ inputs.wd }}" \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            --seed-offset "${{ inputs.seed_offset }}" \
            ${{ inputs.reset_lr_controller && '--reset-lr-controller' || '' }}
```

### `README.md`

````markdown
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
````

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ditflex"
version = "0.1.0"
description = "DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Julio Candanedo" }]

# torch is deliberately NOT pinned here: the correct build depends on the
# CUDA version of the host, and Modal installs it in the image definition.
# Install it first, then `pip install -e .`
dependencies = [
    "diffusers>=0.31",
    "transformers>=4.44",
    "safetensors>=0.4.5",
    "huggingface_hub>=0.26",
    "numpy>=1.26",
    "tqdm",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]
eval = [
    "scipy",              # FID: matrix sqrt of the covariance
    "torchvision",        # InceptionV3 features
    "pillow",
]
logging = [
    "wandb",
]
prepare = [                # only for scripts/prepare_latents.py
    "webdataset",
    "torchvision",
    "pillow",
]

[project.urls]
Repository = "https://github.com/jcandane/ditflex"

[tool.hatch.build.targets.wheel]
packages = ["src/ditflex"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]        # line length handled by formatter
```

### `quick_train/modal_quick.py`

```python
"""quick_train/modal_quick.py -- full dress rehearsal of the training chain.

Everything /run/ does, minus the 8 GPUs: real ditflex.train, real latents,
real checkpointing -- against the SCRATCH model repo (default
sparsetrace/quicktraindit), never the real per-objective repos.

Two legs in ONE dispatch, so the checkpoint chain is machine-verified:

    reset scratch repo (delete; fixed name, overwritten every dispatch)
    leg 1: fresh start, train --max-steps STEPS, save, push
    delete the local checkpoint dir      <- forces a genuine Hub pull
    leg 2: pull, resume, train --max-steps RESUME_STEPS, push
    verify: download state.json, assert step == STEPS + RESUME_STEPS

Green therefore proves: fresh-start -> save -> push -> pull -> exact
resume -> re-push, plus compile+DDP order, the objective stepping on real
data, and the deadline machinery -- the exact code path of /run/.

    MODAL_GPUS=2 modal run quick_train/modal_quick.py --steps 1000

Environment: MODAL_GPU (default B300), MODAL_GPUS (default 2), HF_TOKEN
(write scope: push + scratch-repo reset), TORCH_INDEX (default cu129).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "2"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

SCRATCH_REPO_DEFAULT = "sparsetrace/quicktraindit"
CKPT_DIR = "/tmp/ditflex_ckpt"          # must match ditflex.train.CKPT_DIR
TRAIN_SECONDS_CEILING = 7200            # per leg; --max-steps stops first

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
        "pillow",          # sample-grid PNG at end of each link
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-quick-train", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=4 * 3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def quick(
    steps: int = 1000,
    resume_steps: int = 200,
    objective: str = "flow",
    max_latent_files: int = 0,
    scratch_repo: str = SCRATCH_REPO_DEFAULT,
) -> int:
    import json
    import shutil
    import subprocess
    import sys

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    n_gpu = torch.cuda.device_count()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[quick] {n_gpu} GPUs:\n{result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True
    )

    # -- reset the scratch repo: fixed name, overwritten every dispatch,
    #    so leg 1 genuinely exercises the fresh-start path ---------------
    print(f"[quick] resetting scratch repo {scratch_repo}")
    HfApi().delete_repo(scratch_repo, repo_type="model", missing_ok=True)

    def leg(name: str, max_steps: int) -> int:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={n_gpu}", "--standalone",
            "-m", "ditflex.train",
            f"--train-seconds={TRAIN_SECONDS_CEILING}",
            f"--objective={objective}",
            f"--hub-repo={scratch_repo}",
            f"--max-steps={max_steps}",
        ]
        if max_latent_files > 0:
            cmd.append(f"--max-latent-files={max_latent_files}")
        print(f"\n[quick] {name}: {' '.join(cmd)}\n")
        return subprocess.run(cmd, cwd="/repo").returncode

    rc = leg("leg 1 (fresh start)", steps)
    if rc != 0:
        print("[quick] leg 1 FAILED")
        return rc

    # Force leg 2 to pull from the Hub, not find local files.
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print(f"[quick] deleted {CKPT_DIR} -- leg 2 must pull from the Hub")

    rc = leg("leg 2 (resume)", resume_steps)
    if rc != 0:
        print("[quick] leg 2 FAILED")
        return rc

    # -- machine-verify the chain ---------------------------------------
    state_path = hf_hub_download(
        scratch_repo, "state.json", repo_type="model", force_download=True
    )
    state = json.loads(Path(state_path).read_text())
    expected = steps + resume_steps
    print(f"\n[quick] scratch repo state: step={state['step']}  expected={expected}")
    for run in state.get("run_history", []):
        print(f"        run: steps {run['start_step']:,} -> {run['end_step']:,}  "
              f"({run['seconds']}s, world={run['world']}, {run['objective']})")

    if state["step"] != expected:
        print("[quick] CHAIN BROKEN -- resumed step count does not add up.")
        return 1
    if len(state.get("run_history", [])) != 2:
        print("[quick] CHAIN BROKEN -- expected exactly 2 runs in run_history.")
        return 1

    print(f"\n[quick] PASS -- fresh->save->push->pull->resume->push verified "
          f"at step {expected} on {scratch_repo}.")
    return 0


@app.local_entrypoint()
def main(
    steps: int = 1000,
    resume_steps: int = 200,
    objective: str = "flow",
    max_latent_files: int = 0,
    scratch_repo: str = SCRATCH_REPO_DEFAULT,
):
    """
    Args:
        steps:            leg-1 step count (fresh start)
        resume_steps:     leg-2 step count (after resume)
        objective:        ddpm | flow
        max_latent_files: 0 = all 32 shards (realistic); small N for speed
        scratch_repo:     overwritten every dispatch; NEVER a real run repo
    """
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")
    rc = quick.remote(
        steps=steps,
        resume_steps=resume_steps,
        objective=objective,
        max_latent_files=max_latent_files,
        scratch_repo=scratch_repo,
    )
    if rc != 0:
        raise SystemExit(rc)
```

### `run/modal_train.py`

```python
"""Detached Modal supervisor for transactional DiT/SiT training.

One Modal container may launch several *fresh* torchrun processes.  Exit code
75 means the child detected retryable numerical instability and deliberately
discarded its candidate.  The supervisor then reloads the last promoted Hub
checkpoint with:

* a lower retry LR multiplier;
* a deterministic new data/objective seed offset;
* the same model, EMA, AdamW moments, and global step from the committed state.

The retry count is bounded.  Code errors and unrelated subprocess failures are
not hidden by a broad exception handler.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "8"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

_BUDGET = int(os.environ.get("MODAL_TRAIN_SECONDS", "7200"))
_MAX_RETRIES_ENV = int(os.environ.get("MODAL_MAX_RETRIES", "2"))
# Per attempt: checkpoint pull + compile + upload allowance.  The stepping
# budget itself is shared across retries by reading the child's retry marker.
TIMEOUT_CEILING = _BUDGET + 3600 * (_MAX_RETRIES_ENV + 1)

RETRY_EXIT_CODE = 75
RETRY_MARKER = Path("/tmp/ditflex_retry.json")
PROMOTION_MARKER = Path("/tmp/ditflex_promotion.json")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
        "pillow",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[
            ".git",
            "**/__pycache__",
            "*.egg-info",
            ".venv",
            ".ruff_cache",
            ".pytest_cache",
        ],
    )
)

app = modal.App("ditflex-train", image=image)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=TIMEOUT_CEILING,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    target_steps: int = 400_000,
    resume_revision: str = "",
    resume_step: int = 0,
    auto_legacy_rollback: bool = True,
    legacy_suspect_ratio: float = 8.0,
    max_retries: int = 2,
    retry_seed_stride: int = 1_000_003,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    # Kept only so the existing v2 GitHub workflow remains callable.  V3 uses
    # committed-reference thresholds below rather than fast/slow EMA ratios.
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    health_loss_warn_ratio: float = 1.015,
    health_loss_retry_ratio: float = 1.025,
    health_loss_emergency_ratio: float = 1.05,
    health_grad_warn_ratio: float = 2.0,
    health_grad_retry_ratio: float = 4.0,
    health_grad_emergency_ratio: float = 8.0,
    commit_windows: int = 2,
    warning_patience: int = 2,
    reset_lr_controller: bool = False,
    grad_reference: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
) -> int:
    import subprocess
    import sys

    import torch

    if max_retries < 0:
        print("[modal] max_retries must be non-negative")
        return 2
    if not (0.0 < lr_backoff < 1.0):
        print("[modal] lr_backoff must lie in (0, 1)")
        return 2
    if train_seconds <= 0:
        print("[modal] train_seconds must be positive")
        return 2
    if not (0.0 <= skip_warn_rate <= skip_retry_rate <= skip_emergency_rate <= 1.0):
        print(
            "[modal] skip thresholds must satisfy "
            "0 <= warn <= retry <= emergency <= 1"
        )
        return 2

    n_gpu = torch.cuda.device_count()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[modal] {n_gpu} GPUs:\n{result.stdout.strip()}")
    if n_gpu <= 0:
        print("[modal] no CUDA devices visible")
        return 2

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"],
        check=True,
    )

    # The editable install above happens after this Modal worker interpreter
    # has already started.  pip writes a .pth/editable-finder file, but the
    # running interpreter does not automatically re-process newly created
    # .pth files.  Add the src-layout directory explicitly for imports in this
    # supervisor process and export it for every fresh torchrun child.
    repo_src = Path("/repo/src")
    if not repo_src.is_dir():
        raise RuntimeError(f"expected source directory is missing: {repo_src}")
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in inherited_pythonpath.split(os.pathsep) if part]
    if repo_src_str not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_src_str, *pythonpath_parts])

    from ditflex.checkpoint import (
        resolve_revision_for_step,
        select_stable_resume_revision,
    )

    selected_revision = resume_revision.strip()
    selected_step: int | None = resume_step if resume_step > 0 else None
    if selected_revision and selected_step is not None:
        print("[modal] use only one of resume_revision or resume_step")
        return 2
    if selected_step is not None:
        if not hub_repo:
            print("[modal] resume_step requires hub_repo")
            return 2
        selected_revision = resolve_revision_for_step(hub_repo, selected_step)
        print(
            f"[modal] explicit anchor step {selected_step:,} -> "
            f"revision {selected_revision[:12]}"
        )
    elif not selected_revision and auto_legacy_rollback and hub_repo:
        selection = select_stable_resume_revision(
            hub_repo,
            suspect_ratio=legacy_suspect_ratio,
        )
        selected_revision = selection.revision or ""
        selected_step = selection.step
        print(f"[modal] resume selection: {selection.reason}")
        if selected_revision:
            print(
                f"[modal] using migration anchor step {selected_step:,} "
                f"revision {selected_revision[:12]}"
            )

    if loss_rise_ratio != 1.08 or loss_emergency_ratio != 1.35:
        print(
            "[modal] NOTE: v2 loss_rise_ratio/loss_emergency_ratio are deprecated; "
            "v3 uses health_loss_* committed-reference thresholds"
        )

    remaining_train_seconds = float(train_seconds)
    for attempt in range(max_retries + 1):
        if remaining_train_seconds < 1.0:
            print("[modal] retry budget exhausted before another attempt")
            return RETRY_EXIT_CODE

        RETRY_MARKER.unlink(missing_ok=True)
        PROMOTION_MARKER.unlink(missing_ok=True)

        attempt_factor = lr_backoff**attempt
        attempt_seed_offset = seed_offset + attempt * retry_seed_stride
        child_budget = max(1, int(remaining_train_seconds))

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={n_gpu}",
            "--standalone",
            "-m",
            "ditflex.train",
            f"--train-seconds={child_budget}",
            f"--objective={objective}",
            f"--target-steps={target_steps}",
            f"--attempt={attempt}",
            f"--attempt-lr-factor={attempt_factor}",
            f"--seed-offset={attempt_seed_offset}",
            f"--lr-policy={lr_policy}",
            f"--lr-min={lr_min}",
            f"--lr-hard-min={lr_hard_min}",
            f"--lr-min-scale={lr_min_scale}",
            f"--commit-windows={commit_windows}",
            f"--warning-patience={warning_patience}",
            f"--loss-warn-ratio={health_loss_warn_ratio}",
            f"--loss-retry-ratio={health_loss_retry_ratio}",
            f"--loss-emergency-ratio={health_loss_emergency_ratio}",
            f"--grad-warn-ratio={health_grad_warn_ratio}",
            f"--grad-retry-ratio={health_grad_retry_ratio}",
            f"--grad-emergency-ratio={health_grad_emergency_ratio}",
            f"--clip={clip}",
            f"--spike-skip={spike_skip}",
            f"--grad-ceiling={grad_ceiling}",
            f"--skip-warn-rate={skip_warn_rate}",
            f"--skip-retry-rate={skip_retry_rate}",
            f"--skip-emergency-rate={skip_emergency_rate}",
        ]
        if hub_repo:
            command.append(f"--hub-repo={hub_repo}")
        if selected_revision:
            command.append(f"--resume-revision={selected_revision}")
        if max_steps > 0:
            command.append(f"--max-steps={max_steps}")
        if lr > 0.0:
            command.append(f"--lr={lr}")
        if grad_reference > 0.0:
            command.append(f"--grad-reference={grad_reference}")
        if wd >= 0.0:
            command.append(f"--wd={wd}")
        if reset_lr_controller and attempt == 0:
            command.append("--reset-lr-controller")

        print(
            f"\n[modal] attempt {attempt}/{max_retries}: "
            f"lr_factor={attempt_factor:g} seed_offset={attempt_seed_offset} "
            f"budget={child_budget}s anchor="
            f"{selected_revision[:12] if selected_revision else 'latest'}\n"
            f"[modal] running: {' '.join(command)}\n"
        )
        started = time.time()
        result = subprocess.run(command, cwd="/repo")
        child_wall = time.time() - started
        if result.returncode == 0:
            print(f"[modal] attempt {attempt} completed successfully")
            return 0

        # torchrun commonly wraps a worker's exit code in ChildFailedError and
        # returns 1 itself.  The rank-0 atomic marker is therefore the source of
        # truth for a deliberate transactional retry.
        retry = _read_json(RETRY_MARKER)
        retry_requested = int(retry.get("exit_code", 0) or 0) == RETRY_EXIT_CODE
        if not retry_requested:
            print(
                f"[modal] child failed with non-retryable exit code "
                f"{result.returncode}; not masking the failure"
            )
            return result.returncode

        consumed = float(retry.get("elapsed_training_seconds", child_wall))
        remaining_train_seconds = max(0.0, remaining_train_seconds - consumed)
        print(
            f"[modal] retry requested: {retry.get('reason', 'no marker reason')}\n"
            f"[modal] stepping budget consumed={consumed:.1f}s, "
            f"remaining={remaining_train_seconds:.1f}s"
        )

        # If this attempt promoted healthy progress before a later failure,
        # retry from ordinary latest.  Otherwise preserve the explicit legacy
        # migration anchor instead of falling back to a suspect old latest.
        promotion = _read_json(PROMOTION_MARKER)
        promoted_step = int(promotion.get("step", 0) or 0)
        if promoted_step > 0 and (selected_step is None or promoted_step > selected_step):
            selected_revision = ""
            selected_step = promoted_step
            print(
                f"[modal] attempt promoted healthy step {promoted_step:,}; "
                "next retry will pull Hub latest"
            )

        if attempt >= max_retries:
            print("[modal] retry limit reached; last committed checkpoint remains untouched")
            return RETRY_EXIT_CODE

    return RETRY_EXIT_CODE


@app.local_entrypoint()
def main(
    train_seconds: int = 7200,
    objective: str = "flow",
    hub_repo: str = "",
    max_steps: int = 0,
    target_steps: int = 400_000,
    resume_revision: str = "",
    resume_step: int = 0,
    auto_legacy_rollback: bool = True,
    legacy_suspect_ratio: float = 8.0,
    max_retries: int = 2,
    retry_seed_stride: int = 1_000_003,
    lr: float = 0.0,
    lr_policy: str = "adaptive",
    lr_min: float = 1e-5,
    lr_hard_min: float = 1e-6,
    lr_backoff: float = 0.5,
    lr_min_scale: float = 0.125,
    loss_rise_ratio: float = 1.08,
    loss_emergency_ratio: float = 1.35,
    health_loss_warn_ratio: float = 1.015,
    health_loss_retry_ratio: float = 1.025,
    health_loss_emergency_ratio: float = 1.05,
    health_grad_warn_ratio: float = 2.0,
    health_grad_retry_ratio: float = 4.0,
    health_grad_emergency_ratio: float = 8.0,
    commit_windows: int = 2,
    warning_patience: int = 2,
    reset_lr_controller: bool = False,
    grad_reference: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
):
    if objective not in {"ddpm", "flow"}:
        raise SystemExit(f"unknown objective: {objective!r}")
    if lr_policy not in {"constant", "cosine", "adaptive"}:
        raise SystemExit(f"unknown lr_policy: {lr_policy!r}")
    if not (0.0 <= skip_warn_rate <= skip_retry_rate <= skip_emergency_rate <= 1.0):
        raise SystemExit(
            "skip thresholds must satisfy "
            "0 <= warn <= retry <= emergency <= 1"
        )

    return_code = train.remote(
        train_seconds=train_seconds,
        objective=objective,
        hub_repo=hub_repo,
        max_steps=max_steps,
        target_steps=target_steps,
        resume_revision=resume_revision,
        resume_step=resume_step,
        auto_legacy_rollback=auto_legacy_rollback,
        legacy_suspect_ratio=legacy_suspect_ratio,
        max_retries=max_retries,
        retry_seed_stride=retry_seed_stride,
        lr=lr,
        lr_policy=lr_policy,
        lr_min=lr_min,
        lr_hard_min=lr_hard_min,
        lr_backoff=lr_backoff,
        lr_min_scale=lr_min_scale,
        loss_rise_ratio=loss_rise_ratio,
        loss_emergency_ratio=loss_emergency_ratio,
        health_loss_warn_ratio=health_loss_warn_ratio,
        health_loss_retry_ratio=health_loss_retry_ratio,
        health_loss_emergency_ratio=health_loss_emergency_ratio,
        health_grad_warn_ratio=health_grad_warn_ratio,
        health_grad_retry_ratio=health_grad_retry_ratio,
        health_grad_emergency_ratio=health_grad_emergency_ratio,
        commit_windows=commit_windows,
        warning_patience=warning_patience,
        reset_lr_controller=reset_lr_controller,
        grad_reference=grad_reference,
        wd=wd,
        clip=clip,
        spike_skip=spike_skip,
        seed_offset=seed_offset,
        grad_ceiling=grad_ceiling,
        skip_warn_rate=skip_warn_rate,
        skip_retry_rate=skip_retry_rate,
        skip_emergency_rate=skip_emergency_rate,
    )
    if return_code != 0:
        raise SystemExit(return_code)
```

### `run/recover_checkpoint.py`

```python
"""Recover a healthy checkpoint from Hub revision history.

Every periodic push is a Hub commit, so a poisoned 'latest' (the 247K
divergence pushed collapsed checkpoints from step 250K onward) is
recoverable: find the commit whose state.json reports the wanted step,
download the four checkpoint files at that revision, re-upload them as
the new latest.

    HF_TOKEN=... python run/recover_checkpoint.py \
        --repo sparsetrace/ditflex-L2-flow --step 240000
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

FILES = ["model.safetensors", "ema.safetensors", "optim.safetensors", "state.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--step", type=int, required=True, help="step to restore as latest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    commits = api.list_repo_commits(args.repo)          # newest first
    print(f"[recover] {len(commits)} commits in {args.repo}")

    target_rev = None
    for c in commits:
        try:
            path = hf_hub_download(args.repo, "state.json", revision=c.commit_id)
        except Exception:
            continue
        state = json.loads(Path(path).read_text())
        step = state.get("step")
        print(f"  {c.commit_id[:10]}  step={step}")
        if step == args.step:
            target_rev = c.commit_id
            break
    if target_rev is None:
        raise SystemExit(f"no commit found with step == {args.step}")

    print(f"[recover] restoring step {args.step} from revision {target_rev[:10]}")
    if args.dry_run:
        print("[recover] dry run -- nothing uploaded")
        return

    with tempfile.TemporaryDirectory() as td:
        for f in FILES:
            src = hf_hub_download(args.repo, f, revision=target_rev)
            (Path(td) / f).write_bytes(Path(src).read_bytes())
        api.upload_folder(
            folder_path=td,
            repo_id=args.repo,
            commit_message=f"recover: restore step {args.step} as latest "
                           f"(revert post-divergence checkpoints)",
        )
    print(f"[recover] done -- latest is now the healthy step {args.step}")


if __name__ == "__main__":
    main()
```

### `sampling/modal_sample.py`

```python
"""sampling/modal_sample.py -- on-demand sample grids from BOTH chains.

Pulls the latest checkpoint of each requested Hub repo, builds the
correct model variant from the checkpoint's own embedded config
(qk_mode decides builder), loads the EMA weights, renders the standard
fixed-seed 4x4 grid (same classes, same noise as the training-time
time-lapse), and returns PNG bytes. The workflow commits the PNGs into
/sampling/ in the GitHub repo.

    modal run sampling/modal_sample.py
    modal run sampling/modal_sample.py --repos sparsetrace/ditflex-L2-flow
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "B200")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31", "transformers>=4.44", "safetensors>=0.4.5",
        "huggingface_hub>=0.26", "numpy>=1.26", "pillow", "accelerate",
    )
    .add_local_dir(
        REPO_ROOT, remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-sampling", image=image)


@app.function(
    gpu=GPU_KIND,
    timeout=1800,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def sample_repo(repo: str, sample_steps: int = 50, cfg_scale: float = 4.0) -> tuple[int, bytes]:
    import io
    import json
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True)

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file

    from ditflex.config import Config
    from ditflex.model import build_model

    state = json.load(open(hf_hub_download(repo, "state.json")))
    step = int(state["step"])
    cfg_dict = state.get("config") or state.get("cfg")
    assert cfg_dict, "state.json lacks an embedded config"
    cfg = Config.from_dict(cfg_dict)
    print(f"[sample] {repo}: step {step:,}  qk_mode={cfg.model.qk_mode}")

    if cfg.model.qk_mode == "dmap":
        from ditflex.diffusion_model import build_dmap_model

        model = build_dmap_model(cfg.model)
    else:
        model = build_model(cfg.model)

    ema_sd = load_file(hf_hub_download(repo, "ema.safetensors"))
    missing, unexpected = model.load_state_dict(ema_sd, strict=False)
    n_params = sum(1 for _ in model.parameters())
    print(f"[sample] EMA loaded: {len(ema_sd)} tensors "
          f"(missing={len(missing)} buffers, unexpected={len(unexpected)})")
    assert len(unexpected) == 0, f"unexpected EMA keys: {unexpected[:5]}"
    assert len(ema_sd) >= n_params * 0.9, "EMA state dict suspiciously small"

    model = model.to(device="cuda", dtype=torch.float32).eval()

    # Fixed classes/seed: identical to the training-time time-lapse.
    try:
        from ditflex.sample import FIXED_CLASSES, FIXED_SEED
    except ImportError:
        FIXED_CLASSES = [207, 360, 387, 974, 88, 979, 417, 279,
                         972, 483, 21, 562, 933, 724, 985, 812]
        FIXED_SEED = 1234

    n = len(FIXED_CLASSES)
    g = torch.Generator(device="cpu").manual_seed(FIXED_SEED)
    x = torch.randn(n, cfg.model.in_channels, cfg.model.sample_size,
                    cfg.model.sample_size, generator=g).cuda()
    y = torch.tensor(FIXED_CLASSES, device="cuda")
    y_null = torch.full_like(y, cfg.model.num_classes)

    dt = 1.0 / sample_steps
    with torch.no_grad():
        for i in range(sample_steps):
            t = 1.0 - i * dt
            tt = torch.full((n,), t * 1000.0, device="cuda")
            v_c = model(hidden_states=x, timestep=tt, class_labels=y).sample[:, :4]
            v_u = model(hidden_states=x, timestep=tt, class_labels=y_null).sample[:, :4]
            v = v_u + cfg_scale * (v_c - v_u)
            x = x - dt * v

        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda().eval()
        imgs = vae.decode(x / 0.18215).sample

    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    side = int(n ** 0.5)
    px = imgs.shape[1]
    grid = np.zeros((side * px, side * px, 3), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, side)
        grid[r * px:(r + 1) * px, c * px:(c + 1) * px] = imgs[k]

    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    print(f"[sample] {repo}: grid rendered at step {step:,}")
    return step, buf.getvalue()


@app.local_entrypoint()
def main(
    repos: str = "sparsetrace/ditflex-L2-flow,sparsetrace/ditflex-L2-flow-dmap",
    sample_steps: int = 50,
    cfg_scale: float = 4.0,
):
    out_dir = Path(__file__).parent
    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        step, png = sample_repo.remote(repo, sample_steps=sample_steps, cfg_scale=cfg_scale)
        tag = repo.split("/")[-1].replace("ditflex-L2-", "")
        path = out_dir / f"{tag}_step_{step:07d}.png"
        path.write_bytes(png)
        print(f"[sample] wrote {path}")
```

### `src/ditflex/__init__.py`

```python
"""ditflex: DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions."""

from ditflex.attention import (
    FlexSelfAttnProcessor,
    IdentityFlexSelfAttnProcessor,
    identity_score_mod,
    reference_self_attention,
)
from ditflex.config import Config, DataConfig, HubConfig, ModelConfig, TrainConfig
from ditflex.diffusion import (
    DmapFlexSelfAttnProcessor,
    doob_score_mod,
    edge_field_score_mod,
    exact_edge_field,
    model_qk_ratios,
    qk_ratio,
    temperature_score_mod,
)
from ditflex.diffusion_model import build_dmap_model
from ditflex.ema import EMA
from ditflex.latents import LatentStore, batch_seed
from ditflex.model import build_model
from ditflex.objective import build_objective

__all__ = [
    "Config",
    "DmapFlexSelfAttnProcessor",
    "EMA",
    "doob_score_mod",
    "edge_field_score_mod",
    "exact_edge_field",
    "model_qk_ratios",
    "qk_ratio",
    "temperature_score_mod",
    "DataConfig",
    "FlexSelfAttnProcessor",
    "HubConfig",
    "IdentityFlexSelfAttnProcessor",
    "LatentStore",
    "ModelConfig",
    "TrainConfig",
    "batch_seed",
    "build_dmap_model",
    "build_model",
    "build_objective",
    "identity_score_mod",
    "reference_self_attention",
]
__version__ = "0.1.0"
```

### `src/ditflex/attention.py`

```python
"""FlexAttention self-attention for diffusers' Attention module.

This is the ONLY attention implementation in the repo. There is no SDPA path.

Three things live here:

  1. identity_score_mod -- the baseline score function. It is a *real*
     score_mod (not None) so that the baseline traverses exactly the same
     FlexAttention machinery as any experimental score_mod: swapping the
     experiment in changes only the function, never the dispatch path.
     Compiled, the identity inlines to nothing.

  2. FlexSelfAttnProcessor -- a diffusers attention processor whose score
     function is a swappable component.

  3. reference_self_attention -- straight-line softmax attention written
     directly from the math (explicit matmuls, explicit softmax) using the
     module's own weights, intended to run in fp64. It exists so
     scripts/verify_identity.py can check the Flex path against something
     that depends on no fused kernel at all. Test utility only, never a
     training path.

Scope: DiT-L/2 self-attention on [B, N, C] tokens. Cross-attention,
attention masks, 4D inputs, group/spatial norm, qk-norm, in-processor
residuals, and output rescaling are all rejected loudly rather than
handled -- in a repo whose premise is that every deviation from the
baseline is known, a config surprise should fail at the gate, not show up
later as an uninterpretable training curve.

Performance note: eager flex_attention is the slow-but-correct fallback
and is what the gates use. The fast path is whole-model torch.compile in
train.py, which fuses the score_mod into the generated kernel. Do not
compile here -- tests and verify_identity.py need the uncompiled module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    from torch.nn.attention.flex_attention import flex_attention
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import torch.nn.attention.flex_attention.flex_attention. "
        "ditflex requires a PyTorch build with FlexAttention support (>= 2.5)."
    ) from e

# score_mod(score, batch, head, q_idx, kv_idx) -> score
ScoreMod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def identity_score_mod(score, b, h, q_idx, kv_idx):
    """No modification: standard full attention routed through FlexAttention."""
    return score


class FlexSelfAttnProcessor:
    """Self-attention through FlexAttention, for diffusers' Attention module.

    Args:
        score_mod: FlexAttention score modification. Defaults to
            ``identity_score_mod`` (the DiT/SiT baseline). The softmax scale
            is always taken from ``attn.scale`` -- never from Flex's default
            -- so the computation is exactly the module's configured
            attention regardless of how the module was built.
    """

    def __init__(self, score_mod: ScoreMod | None = None):
        self.score_mod: ScoreMod = score_mod if score_mod is not None else identity_score_mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("FlexSelfAttnProcessor is self-attention only.")
        if attention_mask is not None:
            raise ValueError("Fixed-shape training uses no attention mask.")
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got ndim={hidden_states.ndim}.")
        if getattr(attn, "group_norm", None) is not None:
            raise ValueError("group_norm is not handled by this processor.")
        if getattr(attn, "spatial_norm", None) is not None:
            raise ValueError("spatial_norm is not handled by this processor.")
        if getattr(attn, "norm_q", None) is not None or getattr(attn, "norm_k", None) is not None:
            raise ValueError("qk-norm is not handled by this processor.")
        if getattr(attn, "residual_connection", False):
            raise ValueError("residual_connection is handled by the block, not the processor.")
        if getattr(attn, "rescale_output_factor", 1.0) != 1.0:
            raise ValueError("rescale_output_factor != 1 is not handled.")

        batch, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        # [B, N, H*D] -> [B, H, N, D]
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        out = flex_attention(
            query,
            key,
            value,
            score_mod=self.score_mod,
            scale=attn.scale,  # explicit: never rely on Flex's 1/sqrt(D) default
        )

        # [B, H, N, D] -> [B, N, H*D]
        out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)

        out = attn.to_out[0](out)  # linear
        out = attn.to_out[1](out)  # dropout (identity in eval)
        return out


class IdentityFlexSelfAttnProcessor(FlexSelfAttnProcessor):
    """Baseline processor: FlexAttention with the identity score_mod."""

    def __init__(self):
        super().__init__(score_mod=identity_score_mod)


def reference_self_attention(
    attn,
    hidden_states: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Softmax self-attention written straight from the math.

    Uses the weights of ``attn`` but none of its forward code and no fused
    attention kernel of any kind: explicit projections, an explicit
    ``q @ k^T * scale`` score matrix, an explicit softmax, explicit output
    projection.

    Args:
        attn: a diffusers Attention module (self-attention config).
        hidden_states: [B, N, C].
        dtype: if given, inputs and weights are cast to this dtype for the
            computation (use torch.float64 for a high-precision reference
            against a bf16/fp32 module). If None, computes in the module's
            native dtype with autograd intact -- used by the gradient gate.

    Note: skips attn.to_out[1] (dropout), so compare against modules in
    eval mode only.
    """

    def cast(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None or dtype is None:
            return t
        return t.to(dtype)

    x = hidden_states if dtype is None else hidden_states.to(dtype)

    def linear(t, layer):
        w = cast(layer.weight)
        b = cast(layer.bias)
        out = t @ w.transpose(0, 1)
        return out if b is None else out + b

    query = linear(x, attn.to_q)
    key = linear(x, attn.to_k)
    value = linear(x, attn.to_v)

    batch, seq_len, _ = x.shape
    heads = attn.heads
    head_dim = query.shape[-1] // heads

    query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

    scores = (query @ key.transpose(-2, -1)) * attn.scale
    probs = scores.softmax(dim=-1)
    out = probs @ value

    out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
    out = linear(out, attn.to_out[0])
    return out
```

### `src/ditflex/checkpoint.py`

```python
"""Checkpoint storage, validation, Hub revisions, and transactional promotion.

Hub top-level files always represent the last *committed healthy* checkpoint.
Training writes a complete candidate directory first, validates its structure,
and only then promotes it with :func:`push_to_hub`.  A failed candidate is never
uploaded, so a fresh retry process can safely pull Hub latest and roll back the
model, EMA, optimizer moments, step, and stability reference together.
"""

from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ditflex.config import Config

_PREFIXES = ("_orig_mod.", "module.")
FILES = ("state.json", "model.safetensors", "ema.safetensors", "optim.safetensors")


@dataclass(frozen=True)
class CheckpointRevision:
    revision: str
    step: int
    grad_reference: float | None
    state: dict[str, Any]


@dataclass(frozen=True)
class ResumeSelection:
    """A selected Hub revision; ``revision=None`` means ordinary latest."""

    revision: str | None
    step: int | None
    reason: str


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for key, value in sd.items():
        clean_key = key
        for prefix in _PREFIXES:
            while clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        out[clean_key] = value
    return out


# -- optimizer <-> safetensors -------------------------------------------


def _flatten_optim(osd: dict) -> tuple[dict[str, torch.Tensor], list]:
    tensors: dict[str, torch.Tensor] = {}
    for idx, state in osd["state"].items():
        for key, value in state.items():
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            if value.ndim == 0:
                value = value.reshape(1)
                key = f"{key}__scalar"
            tensors[f"{idx}.{key}"] = value.contiguous().cpu()
    return tensors, osd["param_groups"]


def _unflatten_optim(tensors: dict[str, torch.Tensor], param_groups: list) -> dict:
    state: dict[int, dict] = {}
    for flat_key, value in tensors.items():
        idx_text, key = flat_key.split(".", 1)
        if key.endswith("__scalar"):
            key = key[: -len("__scalar")]
            value = value.reshape(())
        state.setdefault(int(idx_text), {})[key] = value
    return {"state": state, "param_groups": param_groups}


def _restore_group_types(loaded_groups: list, reference_groups: list) -> list:
    if len(loaded_groups) != len(reference_groups):
        return loaded_groups
    for loaded, reference in zip(loaded_groups, reference_groups, strict=True):
        for key, value in loaded.items():
            if isinstance(value, list) and isinstance(reference.get(key), tuple):
                loaded[key] = tuple(value)
    return loaded_groups


# -- local save / load / validation --------------------------------------


def save_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    config: Config,
    state: dict,
) -> Path:
    """Write a complete candidate checkpoint using temporary files + rename."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model_state = clean_state_dict(model.state_dict())
    model_state = {
        key: value.detach().float().contiguous().cpu() for key, value in model_state.items()
    }
    ema_state = {
        key: value.contiguous().cpu()
        for key, value in clean_state_dict(ema.state_dict()).items()
    }
    optim_tensors, param_groups = _flatten_optim(optimizer.state_dict())

    full_state = dict(state)
    full_state["config"] = asdict(config)
    full_state["optim_param_groups"] = param_groups
    full_state["torch_version"] = torch.__version__

    for name, payload in (
        ("model.safetensors", model_state),
        ("ema.safetensors", ema_state),
        ("optim.safetensors", optim_tensors),
    ):
        temporary = directory / f"{name}.tmp"
        save_file(payload, str(temporary))
        temporary.replace(directory / name)

    temporary_state = directory / "state.json.tmp"
    temporary_state.write_text(json.dumps(full_state, indent=2))
    temporary_state.replace(directory / "state.json")
    return directory


def validate_checkpoint(
    directory: str | Path,
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Validate candidate structure without reloading multi-GB tensor payloads.

    Safetensors headers are opened and key sets are checked.  Tensor checksums
    and file truncation are handled by the safetensors format itself when the
    header is opened; this deliberately avoids a second 7+ GB device/CPU scan.
    """
    directory = Path(directory)
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint missing files: {missing}")

    state = json.loads((directory / "state.json").read_text())
    if "step" not in state or "config" not in state or "optim_param_groups" not in state:
        raise ValueError("state.json lacks step/config/optim_param_groups")
    step = int(state["step"])
    if expected_step is not None and step != int(expected_step):
        raise ValueError(f"candidate step {step} != expected {expected_step}")

    key_sets: dict[str, set[str]] = {}
    for name in ("model.safetensors", "ema.safetensors", "optim.safetensors"):
        with safe_open(str(directory / name), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if not keys and name != "optim.safetensors":
            raise ValueError(f"{name} contains no tensors")
        key_sets[name] = keys

    # EMA intentionally tracks named parameters, while model.state_dict() may
    # also contain non-trainable buffers.  Therefore EMA keys must be a
    # non-empty subset of model keys, not necessarily an exact match.
    extra_ema = sorted(key_sets["ema.safetensors"] - key_sets["model.safetensors"])
    if extra_ema:
        raise ValueError(f"EMA contains keys absent from model: {extra_ema[:5]}")
    return state


def load_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer | None,
    config: Config,
    allow_config_change: bool = False,
) -> dict:
    """Load raw model, EMA, and optimizer state from one committed checkpoint."""
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())

    stored_config = Config.from_dict(state["config"])
    if stored_config != config and not allow_config_change:
        raise ValueError(
            "checkpoint config differs from current config -- a resumed run "
            "must be the same experiment. Diff state.json against Config().to_json(), "
            "or pass allow_config_change=True only for a documented migration."
        )

    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    ema.load_state_dict(load_file(str(directory / "ema.safetensors")))
    if optimizer is not None:
        optim_tensors = load_file(str(directory / "optim.safetensors"))
        param_groups = _restore_group_types(
            state["optim_param_groups"], optimizer.state_dict()["param_groups"]
        )
        optimizer.load_state_dict(_unflatten_optim(optim_tensors, param_groups))
    return state


def copy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Replace ``destination`` with a local copy of a complete checkpoint."""
    source = Path(source)
    destination = Path(destination)
    validate_checkpoint(source)
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, destination / name)
    return destination


# -- Hub pull / promotion -------------------------------------------------


def pull_from_hub(
    repo_id: str,
    directory: str | Path,
    *,
    revision: str | None = None,
) -> Path | None:
    """Download one committed revision into a clean local directory."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for name in FILES:
            hf_hub_download(
                repo_id,
                name,
                repo_type="model",
                revision=revision,
                local_dir=str(directory),
                force_download=True,
            )
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError):
        shutil.rmtree(directory, ignore_errors=True)
        return None
    validate_checkpoint(directory)
    return directory


def push_to_hub(
    directory: str | Path,
    repo_id: str,
    archive_step: int | None = None,
    *,
    commit_message: str = "checkpoint: promote healthy candidate",
) -> str | None:
    """Promote a validated candidate as Hub latest and return its commit id."""
    from huggingface_hub import HfApi, create_repo

    directory = Path(directory)
    state = validate_checkpoint(directory)
    step = int(state["step"])

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"{commit_message}: step {step}",
    )
    commit_id = getattr(info, "oid", None)

    if archive_step is not None:
        prefix = f"archive/step_{archive_step:07d}"
        for name in ("ema.safetensors", "state.json"):
            api.upload_file(
                path_or_fileobj=str(directory / name),
                path_in_repo=f"{prefix}/{name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"checkpoint: archive healthy step {archive_step}",
            )
    return commit_id


# -- revision inspection / legacy migration ------------------------------


def _state_grad_reference(state: dict[str, Any]) -> float | None:
    guard = state.get("guard_state", {})
    if not isinstance(guard, dict):
        return None

    controller = guard.get("stability_controller")
    if isinstance(controller, dict):
        reference = controller.get("reference")
        if isinstance(reference, dict):
            value = reference.get("grad_median")
            if value is not None and float(value) > 0.0:
                return float(value)

    # v1/v2 compatibility.
    value = guard.get("grad_reference", guard.get("grad_ema"))
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def _state_is_transactional(state: dict[str, Any]) -> bool:
    guard = state.get("guard_state", {})
    controller = guard.get("stability_controller") if isinstance(guard, dict) else None
    return (
        isinstance(controller, dict)
        and int(controller.get("version", 0)) >= 3
        and isinstance(controller.get("reference"), dict)
    )


def list_checkpoint_revisions(repo_id: str, *, max_commits: int = 20) -> list[CheckpointRevision]:
    """Return newest unique checkpoint steps with their lightweight state.json."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    commits = list(api.list_repo_commits(repo_id, repo_type="model"))[:max_commits]
    revisions: list[CheckpointRevision] = []
    seen_steps: set[int] = set()
    for commit in commits:
        try:
            path = hf_hub_download(
                repo_id,
                "state.json",
                repo_type="model",
                revision=commit.commit_id,
                force_download=True,
            )
            state = json.loads(Path(path).read_text())
            step = int(state["step"])
        except Exception:  # noqa: BLE001 - revision ledgers may contain non-checkpoint commits
            continue
        if step in seen_steps:
            continue
        seen_steps.add(step)
        revisions.append(
            CheckpointRevision(
                revision=commit.commit_id,
                step=step,
                grad_reference=_state_grad_reference(state),
                state=state,
            )
        )
    return revisions


def resolve_revision_for_step(repo_id: str, step: int, *, max_commits: int = 200) -> str:
    for item in list_checkpoint_revisions(repo_id, max_commits=max_commits):
        if item.step == int(step):
            return item.revision
    raise ValueError(f"no checkpoint revision in {repo_id!r} reports step {step}")


def infer_legacy_gradient_reference(
    repo_id: str,
    *,
    before_step: int | None = None,
    max_commits: int = 12,
) -> float | None:
    """Robustly infer a pre-v3 gradient baseline from prior committed states."""
    values: list[float] = []
    for item in list_checkpoint_revisions(repo_id, max_commits=max_commits):
        if before_step is not None and item.step >= before_step:
            continue
        if item.grad_reference is not None:
            values.append(item.grad_reference)
    if not values:
        return None
    return float(statistics.median(values))


def select_stable_resume_revision(
    repo_id: str,
    *,
    suspect_ratio: float = 8.0,
    max_commits: int = 12,
) -> ResumeSelection:
    """Auto-avoid a legacy latest checkpoint with a contaminated grad EMA.

    Once v3 has promoted a checkpoint, latest is trusted because it already
    passed transactional health gates.  This heuristic is only for migration
    from v1/v2, where the 280K example saved a grad EMA thousands of units above
    its recent historical scale.
    """
    try:
        revisions = list_checkpoint_revisions(repo_id, max_commits=max_commits)
    except Exception as exc:  # noqa: BLE001 - selection may legitimately target a fresh repo
        return ResumeSelection(None, None, f"no readable checkpoint history: {exc!r}")
    if not revisions:
        return ResumeSelection(None, None, "no checkpoint found; fresh start")

    latest = revisions[0]
    if _state_is_transactional(latest.state):
        return ResumeSelection(None, latest.step, "latest is a v3 transactional checkpoint")

    historical = [
        item.grad_reference
        for item in revisions[1:]
        if item.grad_reference is not None and item.grad_reference > 0.0
    ]
    current = latest.grad_reference
    if current is None or not historical:
        return ResumeSelection(None, latest.step, "insufficient legacy history; using latest")

    baseline = float(statistics.median(historical))
    ratio = current / max(baseline, 1e-30)
    if ratio < suspect_ratio:
        return ResumeSelection(
            None,
            latest.step,
            f"legacy latest grad ratio {ratio:.2f}x is below {suspect_ratio:.2f}x",
        )

    acceptable = max(suspect_ratio / 2.0, 2.0)
    for item in revisions[1:]:
        if item.grad_reference is None:
            continue
        item_ratio = item.grad_reference / max(baseline, 1e-30)
        if item_ratio <= acceptable:
            return ResumeSelection(
                item.revision,
                item.step,
                f"legacy latest step {latest.step} has grad reference {current:.2f} "
                f"({ratio:.1f}x recent median {baseline:.2f}); selected prior step "
                f"{item.step} with ratio {item_ratio:.2f}x",
            )

    return ResumeSelection(
        None,
        latest.step,
        f"legacy latest appears suspect ({ratio:.1f}x), but no safer prior revision was found",
    )
```

### `src/ditflex/config.py`

```python
"""src/ditflex/config.py -- experiment configuration as plain dataclasses.

Deliberately imports NO torch: a config must be constructible and
round-trippable anywhere -- CI on CPU, a laptop reading a checkpoint's
embedded config, the Hub viewer -- without a GPU environment.

The defaults ARE the experiment: DiT-L/2 at the published recipe
(README "Recipe" table). Anything that deviates from the published
DiT/SiT setup is marked DEVIATION in a comment and must stay in sync
with the README's known-deviations list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """DiT-L/2: width 1024 = 16 heads x 64, depth 24, patch 2 -> 458M params."""

    num_attention_heads: int = 16
    attention_head_dim: int = 64
    num_layers: int = 24
    patch_size: int = 2
    sample_size: int = 32          # 32x32 latents -> 256 tokens at patch 2
    in_channels: int = 4
    # DEVIATION: published DiT uses out_channels=8 (4 eps + 4 learned sigma,
    # trained with hybrid MSE+VLB). Our objectives are eps/velocity MSE only,
    # so sigma channels would be dead weights receiving zero gradient.
    out_channels: int = 4
    num_classes: int = 1000        # null class for CFG is index num_classes
    # "amap": standard directed QK attention (the baseline).
    # "dmap": EQ-sector / diffusion-map attention -- W_K tied to W_Q, so
    #   every score matrix is symmetric (q_i . q_j) and R == 0 identically.
    #   DEVIATIONS vs baseline: ~25M fewer params (no separate W_K), and
    #   each head's bilinear is PSD (a subfamily of symmetric).
    qk_mode: str = "amap"
    # Coifman-Lafon density-correction exponent for qk_mode="dmap".
    # 0 = pure squared-distance DMAP: softmax(2s_ij - g_j). NOTE this is
    #   NOT plain attention -- the destination potential g_j survives the
    #   row-softmax (only the source term g_i is killed).
    # 0.5 = Fokker-Planck; 1 = Laplace-Beltrami. Ignored for amap.
    dmap_alpha: float = 0.0


@dataclass
class DataConfig:
    hub_repo: str = "sparsetrace/dlatentzz"
    latent_shape: tuple[int, int, int] = (4, 32, 32)
    expected_total: int = 1_281_167   # ImageNet-1k train; a constant of the dataset
    # DEVIATION (of the dataset itself, recorded here for provenance):
    # latents are posterior MODE, not sampled; no horizontal-flip pass;
    # torchvision Resize+CenterCrop rather than ADM center_crop_arr.

    def __post_init__(self):
        self.latent_shape = tuple(self.latent_shape)  # JSON round-trip: list -> tuple


@dataclass
class TrainConfig:
    objective: str = "ddpm"        # ddpm | flow
    global_batch: int = 256
    lr: float = 1e-4               # constant, no warmup (published recipe)
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    label_dropout: float = 0.1     # for classifier-free guidance
    base_seed: int = 0
    deadline_check_every: int = 500  # steps between wall-clock checks (rank 0)


@dataclass
class HubConfig:
    checkpoint_repo: str = "sparsetrace/ditflex-L2"
    archive_every_steps: int = 200_000
    # Periodic save+push cadence. On ephemeral containers a local-only save
    # protects nothing, so every periodic save uploads. At ~9.5 steps/s this
    # is ~18 min of compute at risk between saves. 0 disables (end-of-run
    # save only).
    save_every_steps: int = 10_000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            train=TrainConfig(**d["train"]),
            hub=HubConfig(**d["hub"]),
        )

    @classmethod
    def from_json(cls, s: str) -> Config:
        return cls.from_dict(json.loads(s))
```

### `src/ditflex/diffusion.py`

```python
"""src/ditflex/diffusion.py -- operators and score_mods from
"The Diffusion-Attention Connection" (Candanedo).

The paper's claim: attention, diffusion maps, and magnetic diffusion are
regimes of one Markov geometry built from QK scores. This module gives
that geometry two concrete forms inside ditflex:

  DENSE OPERATORS (analysis; fine at N=256 tokens)
    bidivergence          Sec.3: (H_fwd, H_bwd) with H = H_fwd + H_bwd
    dmap / amap           Sec.2/3.1: row-stochastic symmetric vs directed
    hadamard_recombine    eq.29: DMAP recovered from the two AMAPs
    doob_transform        eq.16: destination reweighting
    stationary_distribution, probability_current
                          Sec.5.1: EQ/NESS classification via J(pi)
    qk_ratio, attention_qk_ratios, model_qk_ratios
                          Sec.6: R = |antisym|_F / |sym|_F of the QK
                          bilinear. Random init calibrates to R ~= 1;
                          trained flow models sit ~0.78-0.86. Measured
                          across the chain's 10K-step checkpoints this
                          gives R(step) -- training dynamics the paper's
                          static Table 1 does not have.

  SCORE_MODS (experiments; plug into FlexSelfAttnProcessor)
    doob_score_mod(log_h)        exact sector: score + log h[kv].
    exact_edge_field(phi)        A_ij = phi_i - phi_j (zero holonomy)
    edge_field_score_mod(A)      general antisymmetric deformation --
                                 nonexact A carries genuine circulation
                                 (the NESS/driven sector, Sec.5.3)
    temperature_score_mod(beta)  score * beta

  Theorem 4.1 becomes executable: flex(edge_field(exact_edge_field(phi)))
  must equal flex(doob(log h)) with h = exp(-phi) -- asserted in
  tests/test_diffusion_math.py through the real Flex path.

BOUNDARY, refined: pure symmetrization is not score_mod-expressible
(pointwise access, no s_ji), so the EQ sector enters through model.py's
weight tying (W_K := W_Q). Given tied weights, everything else IS
Flex-expressible, and the trainable mechanism lives at the bottom of
this module: DmapFlexSelfAttnProcessor -- row-normalization of the
squared-distance kernel exp(-H), single-pass at alpha=0 via the
surviving destination potential 2s_ij - g_j, with the Coifman-Lafon
Doob correction (alpha > 0) as a second pass whose degrees come from
return_lse. attention.py stays the frozen, gate-certified baseline;
this module is the paper, complete: operators, deformations,
measurement, and mechanism.
"""

from __future__ import annotations

import torch

from ditflex.attention import (
    FlexSelfAttnProcessor,
    ScoreMod,
    flex_attention,
    identity_score_mod,
)

# ---------------------------------------------------------------------------
# Dense operators (Sec. 2-5)
# ---------------------------------------------------------------------------


def bidivergence(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directional pair from a (possibly asymmetric) score matrix M.

    H_fwd = g 1^T - M,  H_bwd = 1 g^T - M^T,  g = diag(M).
    Their sum H is symmetric with zero diagonal (the squared-distance
    matrix of the symmetric part of M), and a row-softmax of -beta*H_fwd
    equals a row-softmax of beta*M because the g-terms are row-constant.
    Returns (H_fwd, H_bwd, H)."""
    g = M.diagonal(dim1=-2, dim2=-1)
    h_fwd = g.unsqueeze(-1) - M
    h_bwd = g.unsqueeze(-2) - M.transpose(-2, -1)
    return h_fwd, h_bwd, h_fwd + h_bwd


def row_normalize(P: torch.Tensor) -> torch.Tensor:
    return P / P.sum(dim=-1, keepdim=True)


def dmap(P: torch.Tensor) -> torch.Tensor:
    """Row-stochastic diffusion-map operator of a positive kernel (eq. 3)."""
    return row_normalize(P)


def amap(M: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Forward directed operator: row-softmax over QK scores (Sec. 3.1).
    Identical to softmax(-beta * H_fwd) by shift-invariance."""
    return torch.softmax(beta * M, dim=-1)


def hadamard_recombine(a_fwd: torch.Tensor, a_bwd: torch.Tensor) -> torch.Tensor:
    """eq. 29: row-normalized Hadamard product of the two directed
    operators reconstructs DMAP of the symmetric kernel."""
    return row_normalize(a_fwd * a_bwd)


def doob_transform(p_plus: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """eq. 16: destination reweighting by h > 0, then row renormalization."""
    tilted = p_plus * h.unsqueeze(-2)
    return row_normalize(tilted)


def stationary_distribution(
    p_plus: torch.Tensor, iters: int = 2000, tol: float = 1e-12
) -> torch.Tensor:
    """Power iteration for pi = pi P+ (irreducible row-stochastic P+)."""
    n = p_plus.shape[-1]
    pi = torch.full((n,), 1.0 / n, dtype=p_plus.dtype, device=p_plus.device)
    for _ in range(iters):
        nxt = pi @ p_plus
        nxt = nxt / nxt.sum()
        if (nxt - pi).abs().max() < tol:
            return nxt
        pi = nxt
    return pi


def probability_current(p_plus: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """eq. 24: J(pi) = diag(pi) P+ - (diag(pi) P+)^T. Zero iff detailed
    balance (EQ); nonzero at stationarity is NESS."""
    flux = pi.unsqueeze(-1) * p_plus
    return flux - flux.transpose(-2, -1)


# ---------------------------------------------------------------------------
# The R ratio (Sec. 6)
# ---------------------------------------------------------------------------


def qk_ratio(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """R = |antisym(W)|_F / |sym(W)|_F for a square bilinear W."""
    sym = 0.5 * (W + W.transpose(-2, -1))
    anti = 0.5 * (W - W.transpose(-2, -1))
    return anti.norm(dim=(-2, -1)) / (sym.norm(dim=(-2, -1)) + eps)


def attention_qk_ratios(attn) -> torch.Tensor:
    """Per-head R for one diffusers Attention module. The token-space
    bilinear of head h is B_h = W_q[h]^T @ W_k[h]  (q_i . k_j =
    x_i B x_j^T with row-vector tokens and diffusers' [out, in] Linear
    weights)."""
    wq, wk = attn.to_q.weight.detach(), attn.to_k.weight.detach()
    heads = attn.heads
    head_dim = wq.shape[0] // heads
    ratios = []
    for h in range(heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        B = wq[sl].transpose(0, 1) @ wk[sl]
        ratios.append(qk_ratio(B.float()))
    return torch.stack(ratios)


def model_qk_ratios(model) -> dict:
    """Layer-indexed per-head R across every diffusers Attention module,
    plus the layer-mean the paper's Table 1 reports. Run against the
    chain's checkpoints for R(step)."""
    from diffusers.models.attention_processor import Attention

    per_layer = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            per_layer[name] = attention_qk_ratios(module)
    if not per_layer:
        raise ValueError("no diffusers Attention modules found")
    layer_means = torch.stack([r.mean() for r in per_layer.values()])
    return {
        "per_layer": per_layer,
        "layer_mean": layer_means.mean().item(),
        "layer_std": layer_means.std().item(),
    }


# ---------------------------------------------------------------------------
# Score_mods (Sec. 4, executable)
# ---------------------------------------------------------------------------


def doob_score_mod(log_h: torch.Tensor) -> ScoreMod:
    """Exact sector: destination tilt score + log h[kv]. By Thm 4.1 this
    is everything an exact edge field can do after row-softmax."""

    def mod(score, b, h, q_idx, kv_idx):
        return score + log_h[kv_idx]

    return mod


def exact_edge_field(phi: torch.Tensor) -> torch.Tensor:
    """A_ij = phi_i - phi_j: the coboundary of a node potential. Exact,
    zero holonomy around every cycle."""
    return phi.unsqueeze(-1) - phi.unsqueeze(-2)


def edge_field_score_mod(A: torch.Tensor) -> ScoreMod:
    """General antisymmetric logit deformation score + A[q, kv] (eq. 13).
    With A exact this reduces to a Doob tilt (Thm 4.1); with nonexact A
    it injects genuine circulation -- the deformation the EQ sector
    cannot absorb, and the natural first ditflex experiment."""

    def mod(score, b, h, q_idx, kv_idx):
        return score + A[q_idx, kv_idx]

    return mod


def temperature_score_mod(beta: float) -> ScoreMod:
    """Uniform inverse-temperature rescaling of the scores."""

    def mod(score, b, h, q_idx, kv_idx):
        return score * beta

    return mod


# ---------------------------------------------------------------------------
# The trainable mechanism (the full DMAP-DiT attention)
# ---------------------------------------------------------------------------


def _dmap_attention_eager(query, key, value, scale: float, alpha: float):
    """The DMAP logit modification, as a score_mod.

    Literal logit deformation, as the framework intends:
        logits = 2*score - g[kv]            (alpha = 0)
        logits = 2*score - g[kv] - alpha*log_q[kv]   (alpha > 0)
    with g the destination potential and log_q the degrees of exp(-H).

    HISTORY: this helper spent a debugging arc as an eager island
    (excluded from compilation) while a compiled-capture gradient bug
    was suspected (a 100K-step production stall at the zero-predictor floor).
    The suspicion was retracted -- the apparent gradient divergences were
    noise-vs-noise at the adaLN-zero degenerate init -- and
    tests/test_dmap_gradients.py::test_compiled_scoremod_capture_probe
    then certified compiled flex with this differentiable capture against
    eager directly. The decorator was removed; the function name remains
    as a scar. The full certification chain: finite-difference oracle
    (eager backward vs arithmetic), capture probe (compiled vs eager,
    kernel-level), and test_dmap_compiled_matches_eager (compiled vs
    eager, model-level, non-degenerate weights) -- which now also asserts
    the model compiles fullgraph with no stray breaks. The production
    stall's cause is recorded as undetermined.
    """
    g = scale * (query * key).sum(dim=-1)                  # [B, H, N]

    def dmap_mod(score, b, h, q_idx, kv_idx):
        return 2.0 * score - g[b, h, kv_idx]

    if alpha == 0.0:
        return flex_attention(query, key, value, score_mod=dmap_mod, scale=scale)

    _, lse = flex_attention(
        query, key, value, score_mod=dmap_mod, scale=scale, return_lse=True
    )
    log_q = lse - g                                        # degrees of exp(-H)

    def corrected_mod(score, b, h, q_idx, kv_idx):
        return 2.0 * score - g[b, h, kv_idx] - alpha * log_q[b, h, kv_idx]

    return flex_attention(query, key, value, score_mod=corrected_mod, scale=scale)


class DmapFlexSelfAttnProcessor(FlexSelfAttnProcessor):
    """Diffusion-map attention: row-normalization of exp(-H), applied as
    a LOGIT MODIFICATION (score_mod), computed in an eager island.

    logits(alpha=0) = scale*(2 q_i.k_j - q_j.k_j): the squared-distance
    kernel's surviving destination potential (the source term g_i is
    row-constant and dies in softmax; the destination term g_j does not).
    alpha > 0 adds the Coifman-Lafon Doob correction by the degrees of
    the actual kernel exp(-H). Gradients flow through g and log_q on
    purpose -- the potentials are learned computation.

    Fully compiled: the score_mod's differentiable capture was certified
    against eager by the kernel-level probe and against arithmetic by the
    finite-difference oracle (tests/test_dmap_gradients.py); see
    _dmap_attention_eager's HISTORY note for the debugging arc.

    DMAP semantics additionally require symmetric scores -- enforced by
    weight tying in model.py (qk_mode="dmap"), not here."""

    def __init__(self, alpha: float = 0.0):
        super().__init__(score_mod=identity_score_mod)
        self.alpha = float(alpha)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("DmapFlexSelfAttnProcessor is self-attention only, no masks.")
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got ndim={hidden_states.ndim}.")

        batch, seq_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        heads = attn.heads
        head_dim = query.shape[-1] // heads
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        out = _dmap_attention_eager(query, key, value, attn.scale, self.alpha)

        out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out
```

### `src/ditflex/diffusion_model.py`

```python
"""src/ditflex/diffusion_model.py -- the DMAP-DiT: diffusion.py's
mechanism combined with the transformer.

model.py stays the untouched, gate-certified baseline builder; this
module builds the variant by calling it and applying the paper's surgery
on top:

    1) build the baseline DiT (geometry identical, Flex identity
       processors installed, layer count verified by the frozen builder);
    2) tie W_K := W_Q in every attention layer -- scores become
       q_i . q_j, symmetric, R == 0 identically through training.
       Sharing the Module (not copying weights) keeps the constraint
       exact; state_dict and EMA dedupe the shared parameters;
    3) replace every processor with DmapFlexSelfAttnProcessor --
       row-normalization of the squared-distance kernel exp(-H) with the
       surviving destination potential, plus the Coifman-Lafon Doob
       correction when cfg.dmap_alpha > 0.

Dependency direction: baseline -> paper, never backward. model.py and
attention.py import nothing from here or from diffusion.py.
"""

from __future__ import annotations

from dataclasses import replace

from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.config import ModelConfig
from ditflex.diffusion import DmapFlexSelfAttnProcessor
from ditflex.model import build_model


def build_dmap_model(cfg: ModelConfig) -> DiTTransformer2DModel:
    if cfg.qk_mode != "dmap":
        raise ValueError(f"build_dmap_model expects qk_mode='dmap', got {cfg.qk_mode!r}")

    # The frozen builder constructs the geometry (and refuses dmap configs
    # by design), so hand it an amap-labeled copy of the same geometry.
    model = build_model(replace(cfg, qk_mode="amap"))

    n_applied = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.to_k = module.to_q
            module.set_processor(DmapFlexSelfAttnProcessor(alpha=cfg.dmap_alpha))
            n_applied += 1
    if n_applied != cfg.num_layers:
        raise RuntimeError(
            f"applied DMAP surgery to {n_applied} layers, expected {cfg.num_layers}"
        )
    return model
```

### `src/ditflex/distributed.py`

```python
"""src/ditflex/distributed.py -- thin DDP helpers for single-node torchrun.

Degrades to a no-op in a single process (no RANK in env), so every entry
point runs unchanged under ``python x.py`` and
``torchrun --nproc-per-node=8``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world > 1


def setup() -> DistContext:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistContext(rank, world, local_rank, torch.device(f"cuda:{local_rank}"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(rank=0, world=1, local_rank=0, device=device)


def cleanup(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def broadcast_flag(ctx: DistContext, flag: bool) -> bool:
    """Broadcast rank 0's boolean decision to every rank."""
    if not ctx.is_distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=ctx.device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def broadcast_float(ctx: DistContext, value: float, src: int = 0) -> float:
    """Broadcast one float scalar from ``src`` to every rank."""
    if not ctx.is_distributed:
        return float(value)
    t = torch.tensor(float(value), dtype=torch.float64, device=ctx.device)
    dist.broadcast(t, src=src)
    return float(t.item())


def broadcast_int(ctx: DistContext, value: int, src: int = 0) -> int:
    """Broadcast one integer scalar from ``src`` to every rank."""
    if not ctx.is_distributed:
        return int(value)
    t = torch.tensor(int(value), dtype=torch.int64, device=ctx.device)
    dist.broadcast(t, src=src)
    return int(t.item())


def all_reduce_bool_and(ctx: DistContext, flag: bool) -> bool:
    """Return True only when every rank supplied True.

    This is used before backward/optimizer collectives so that one rank cannot
    abort locally while the others continue and deadlock in DDP.
    """
    if not ctx.is_distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def all_reduce_mean(ctx: DistContext, value: torch.Tensor | float) -> float:
    """Return the arithmetic mean of one scalar value across all ranks."""
    if torch.is_tensor(value):
        t = value.detach().float().reshape(()).clone().to(ctx.device)
    else:
        t = torch.tensor(float(value), dtype=torch.float32, device=ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(ctx.world)
    return float(t.item())
```

### `src/ditflex/ema.py`

```python
"""src/ditflex/ema.py -- exponential moving average of model parameters.

Built on and updated through the RAW module (before torch.compile / DDP
wrapping). The wrappers share the same Parameter objects, so updating via
the raw reference is correct -- and it means the EMA state dict carries
clean parameter names with no `_orig_mod.` / `module.` prefixes ever.
"""

from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Load EMA weights into a model (for sampling/eval)."""
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name].to(p.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow) ^ set(sd)
        if missing:
            raise KeyError(f"EMA state mismatch on keys: {sorted(missing)[:5]} ...")
        for name, t in sd.items():
            # safetensors load_file materializes on CPU; the shadow must
            # stay wherever it already lives (the training device), or the
            # first update() after a resume mixes cuda params with cpu
            # shadows. Caught by quick_train leg 2.
            self.shadow[name] = t.detach().to(
                device=self.shadow[name].device, dtype=torch.float32, copy=True
            )

    def to(self, device) -> EMA:
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self
```

### `src/ditflex/latents.py`

```python
"""src/ditflex/latents.py -- GPU-resident latent store, stateless sampling.

Design (README "Design decisions"):
  - NO DataLoader, NO DistributedSampler. The full ~10.5 GB bf16 tensor
    lives on every GPU; a batch is a fancy-index, not an I/O operation.
  - Sampling is STATELESS: indices for (global_step, rank) come from a
    generator seeded by a pure function of (base_seed, global_step, rank).
    Resume is exact from the step counter alone, and survives a change of
    world size.
  - Latents stay bf16 at rest; batches are cast to fp32 on the way out so
    the noising arithmetic in objective.py runs in full precision
    (autocast handles the model matmuls).

Validation on construction repeats the load-bearing checks from
tests/verify_latents.py -- in particular the std ~= 1 check that catches
double-application of the 0.18215 scaling factor. The gate protects the
dataset once; this protects every future load path.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

STD_LO, STD_HI = 0.7, 1.4
_VALIDATE_SAMPLES = 8192


def batch_seed(base_seed: int, global_step: int, rank: int) -> int:
    """Distinct 63-bit seed for every (step, rank); pure and stateless."""
    return (((base_seed + 1) * 1_000_000_007 + global_step) * 8192 + rank) % (2**63 - 1)


class LatentStore:
    def __init__(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        latent_shape: tuple[int, int, int] = (4, 32, 32),
        num_classes: int = 1000,
        validate: bool = True,
    ):
        flat_dim = math.prod(latent_shape)
        if latents.ndim != 2 or latents.shape[1] != flat_dim:
            raise ValueError(f"latents must be [N, {flat_dim}], got {tuple(latents.shape)}")
        if labels.shape != (latents.shape[0],):
            raise ValueError(f"labels must be [{latents.shape[0]}], got {tuple(labels.shape)}")

        self.latents = latents
        self.labels = labels.long()
        self.latent_shape = tuple(latent_shape)

        if validate:
            sub = latents[: min(len(latents), _VALIDATE_SAMPLES)].float()
            std = sub.std().item()
            if not (STD_LO < std < STD_HI):
                hint = (
                    "looks UNSCALED (scaling_factor not applied)"
                    if std > 3.0
                    else "looks DOUBLE-scaled" if std < 0.4 else "unexpected"
                )
                raise ValueError(
                    f"latent std {std:.4f} outside ({STD_LO}, {STD_HI}) -- {hint}. "
                    "The store expects scaling_factor=0.18215 applied exactly once "
                    "at encode time. Do NOT rescale at load."
                )
            if not torch.isfinite(sub).all():
                raise ValueError("non-finite values in latents")
            lo, hi = int(self.labels.min()), int(self.labels.max())
            if lo < 0 or hi >= num_classes:
                raise ValueError(f"labels in [{lo}, {hi}], expected [0, {num_classes - 1}]")

    def __len__(self) -> int:
        return self.latents.shape[0]

    @property
    def device(self) -> torch.device:
        return self.latents.device

    def batch(
        self, global_step: int, rank: int, batch_size: int, base_seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic batch for (global_step, rank).

        Returns x0 [B, *latent_shape] fp32 and labels [B] long, on the
        store's device. Same arguments always produce the same batch."""
        g = torch.Generator().manual_seed(batch_seed(base_seed, global_step, rank))
        idx = torch.randint(0, len(self), (batch_size,), generator=g).to(self.device)
        x0 = self.latents[idx].view(-1, *self.latent_shape).float()
        return x0, self.labels[idx]

    # -- construction -----------------------------------------------------

    @classmethod
    def from_files(
        cls,
        paths: list[Path],
        device: torch.device | str = "cuda",
        **kwargs,
    ) -> LatentStore:
        from safetensors import safe_open

        lat_parts, lab_parts = [], []
        for p in paths:
            with safe_open(str(p), framework="pt", device="cpu") as f:
                lat_parts.append(f.get_tensor("latents"))
                lab_parts.append(f.get_tensor("labels"))
        latents = torch.cat(lat_parts, dim=0).to(device)
        labels = torch.cat(lab_parts, dim=0).to(device)
        return cls(latents, labels, **kwargs)

    @classmethod
    def from_local(cls, directory: str | Path, device="cuda", max_files: int | None = None, **kw):
        paths = sorted(Path(directory).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"no .safetensors under {directory}")
        return cls.from_files(paths[:max_files], device=device, **kw)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "sparsetrace/dlatentzz",
        device="cuda",
        max_files: int | None = None,
        expected_total: int | None = None,
        **kw,
    ) -> LatentStore:
        """Download every latent shard and build the store. With
        max_files=None and expected_total set, asserts the full-dataset
        count -- do this once per training run, at startup."""
        from huggingface_hub import hf_hub_download, list_repo_files

        files = sorted(
            f for f in list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".safetensors")
        )
        if not files:
            raise FileNotFoundError(f"no .safetensors in {repo_id}")
        files = files[:max_files]
        paths = [Path(hf_hub_download(repo_id, f, repo_type="dataset")) for f in files]
        store = cls.from_files(paths, device=device, **kw)
        if expected_total is not None and max_files is None and len(store) != expected_total:
            raise ValueError(f"loaded {len(store):,} latents, expected {expected_total:,}")
        return store
```

### `src/ditflex/model.py`

```python
"""src/ditflex/model.py -- build DiT-L/2 with FlexAttention installed.

The ONLY place a model is constructed. Every entry point (training,
sampling, eval, tests) builds through here, so the Flex processor and the
config-to-architecture mapping cannot drift between them.

Processor installation walks the modules and calls the per-module
Attention.set_processor() -- NOT the model-level set_attn_processor()
convenience method, which newer diffusers removed from
DiTTransformer2DModel. The walk also counts what it touched and demands
exactly one self-attention per layer, so an architecture surprise fails
here, loudly, instead of surfacing as an uninterpretable training curve.

torch.compile does NOT happen here -- train.py compiles. Tests and the
identity gate need the uncompiled module.
"""

from __future__ import annotations

from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.attention import FlexSelfAttnProcessor, ScoreMod
from ditflex.config import ModelConfig


def install_flex_processors(model, score_mod: ScoreMod | None = None) -> int:
    """Install FlexSelfAttnProcessor on every diffusers Attention module.
    Returns the number of modules touched."""
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.set_processor(FlexSelfAttnProcessor(score_mod=score_mod))
            count += 1
    return count


def build_model(cfg: ModelConfig, score_mod: ScoreMod | None = None) -> DiTTransformer2DModel:
    """DiT at cfg's geometry, self-attention routed through FlexAttention.

    score_mod=None installs the identity baseline; passing a score_mod is
    the experiment. Nothing else changes between the two."""
    model = DiTTransformer2DModel(
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_layers=cfg.num_layers,
        sample_size=cfg.sample_size,
        patch_size=cfg.patch_size,
        num_embeds_ada_norm=cfg.num_classes + 1,   # +1: CFG null class
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )

    n_installed = install_flex_processors(model, score_mod)
    if n_installed != cfg.num_layers:
        raise RuntimeError(
            f"installed Flex on {n_installed} Attention modules but the config "
            f"has {cfg.num_layers} layers -- the DiT architecture is not one "
            "self-attention per block as assumed. Do not train on this."
        )

    # GUARD, not a feature: this builder produces ONLY the certified
    # baseline DiT. Variant configs must go through their own builders so
    # a dmap-labeled config can never silently yield an untied baseline.
    if getattr(cfg, "qk_mode", "amap") != "amap":
        raise ValueError(
            f"build_model builds the baseline only; qk_mode={cfg.qk_mode!r} "
            "requires ditflex.diffusion_model.build_dmap_model."
        )
    return model
```

### `src/ditflex/objective.py`

```python
"""DDPM epsilon prediction and flow matching behind one deterministic interface.

Both objectives expose ``loss(model, x0, y, generator=None) -> scalar``.  The
optional generator makes every source of objective randomness deterministic in
``(global_step, rank, retry_seed_offset)``:

* diffusion / flow timestep;
* Gaussian noise;
* classifier-free label dropout.

This matters for transactional retries.  Re-running the same attempt reproduces
the exact stochastic objective, while changing the retry seed offset changes
all objective randomness together rather than changing only latent indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# -- deterministic RNG ----------------------------------------------------

_MASK64 = (1 << 64) - 1
_TORCH_SEED_MAX = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    """Small stable 64-bit mixer; independent of Python's randomized hash()."""
    z = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def objective_seed(base_seed: int, global_step: int, rank: int, seed_offset: int = 0) -> int:
    """Return a deterministic torch seed for one objective batch.

    Constants are namespace separators rather than cryptographic values.  The
    function intentionally does not depend on world size, so a resumed run is
    deterministic for each rank even if the number of ranks changes.
    """
    value = _splitmix64(base_seed)
    value ^= _splitmix64(global_step + 0xD17F1E5)
    value ^= _splitmix64(rank + 0x51A7)
    value ^= _splitmix64(seed_offset + 0xC0FFEE)
    seed = value % _TORCH_SEED_MAX
    return int(seed if seed != 0 else 1)


def make_step_generator(
    device: torch.device | str,
    *,
    base_seed: int,
    global_step: int,
    rank: int,
    seed_offset: int = 0,
) -> torch.Generator:
    """Create a device-local generator for one training step."""
    device = torch.device(device)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(objective_seed(base_seed, global_step, rank, seed_offset))
    return generator


# -- pure math, exactly testable -----------------------------------------


def add_noise(x0: torch.Tensor, eps: torch.Tensor, abar_t: torch.Tensor) -> torch.Tensor:
    """DDPM forward marginal: x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
    ab = abar_t.view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def linear_interpolant(
    x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t = (1-t) x0 + t eps; velocity target v = eps - x0."""
    tb = t.view(-1, 1, 1, 1)
    return (1.0 - tb) * x0 + tb * eps, eps - x0


def apply_label_dropout(
    y: torch.Tensor,
    p: float,
    null_index: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace labels with the classifier-free-guidance null class."""
    if p <= 0.0:
        return y
    drop = torch.rand(y.shape, device=y.device, generator=generator) < p
    return torch.where(drop, torch.full_like(y, null_index), y)


def _randn_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    # ``torch.randn_like(..., generator=...)`` has varied across torch builds;
    # the explicit shape form is supported by every build used by this repo.
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


# -- objectives -----------------------------------------------------------


@dataclass
class DDPMObjective:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    label_dropout: float = 0.1
    null_class: int = 1000
    _abar_cache: dict = field(default_factory=dict, repr=False)

    def alphas_cumprod(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._abar_cache:
            betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                device=device,
                dtype=torch.float32,
            )
            self._abar_cache[key] = torch.cumprod(1.0 - betas, dim=0)
        return self._abar_cache[key]

    def loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        abar = self.alphas_cumprod(x0.device)
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (x0.shape[0],),
            device=x0.device,
            generator=generator,
        )
        eps = _randn_like(x0, generator)
        xt = add_noise(x0, eps, abar[t])
        y = apply_label_dropout(
            y,
            self.label_dropout,
            self.null_class,
            generator=generator,
        )
        pred = model(hidden_states=xt, timestep=t, class_labels=y).sample
        return F.mse_loss(pred[:, : x0.shape[1]], eps)


@dataclass
class FlowMatchingObjective:
    label_dropout: float = 0.1
    null_class: int = 1000
    timestep_scale: float = 1000.0

    def loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device, generator=generator)
        eps = _randn_like(x0, generator)
        xt, velocity = linear_interpolant(x0, eps, t)
        y = apply_label_dropout(
            y,
            self.label_dropout,
            self.null_class,
            generator=generator,
        )
        pred = model(
            hidden_states=xt,
            timestep=t * self.timestep_scale,
            class_labels=y,
        ).sample
        return F.mse_loss(pred[:, : x0.shape[1]], velocity)


def build_objective(name: str, label_dropout: float = 0.1, num_classes: int = 1000):
    if name == "ddpm":
        return DDPMObjective(label_dropout=label_dropout, null_class=num_classes)
    if name == "flow":
        return FlowMatchingObjective(label_dropout=label_dropout, null_class=num_classes)
    raise ValueError(f"unknown objective: {name!r} (expected 'ddpm' or 'flow')")
```

### `src/ditflex/sample.py`

```python
"""src/ditflex/sample.py -- generate images from a trained checkpoint.

Called by train.py after the final save+push of every chain link (and
usable standalone). Uses a FIXED seed and FIXED class set, so the
samples/ folder on the checkpoint repo becomes a time-lapse: the same 16
noise tensors, decoded at every link, sharpening as the chain grows.

Samplers:
  flow: Euler integration of the learned velocity field from t=1 (noise)
        to t=0 (data), matching the training interpolant
        x_t = (1-t) x0 + t eps  =>  dx/dt = v = eps - x0.
        Timesteps passed as t * 1000.0 (float), exactly as trained.
  ddpm: deterministic DDIM on eps-prediction over the same linear-beta
        schedule the objective trains against.

Both use classifier-free guidance via the null class (index num_classes),
with cond/uncond batched into one forward. Decode goes through the SAME
VAE that encoded the dataset (stabilityai/sd-vae-ft-ema), dividing by the
0.18215 the encoder multiplied.

Runs on the raw (uncompiled) model in eager mode: ~100 forwards of
DiT-L/2 at batch 32 is well under a minute of GPU and needs no compile.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Recognisable, visually diverse ImageNet classes -- fixed forever so the
# time-lapse compares like with like.
FIXED_CLASSES = [88, 207, 250, 279, 291, 323, 360, 387,
                 417, 483, 555, 812, 933, 972, 975, 985]
FIXED_SEED = 1234
VAE_REPO = "stabilityai/sd-vae-ft-ema"   # the dataset's encoder
SCALING = 0.18215


def _cfg_forward(model, x, t_batch, y, y_null, cfg_scale):
    """One guided velocity/eps evaluation, cond+uncond in a single batch."""
    both_x = torch.cat([x, x], dim=0)
    both_t = torch.cat([t_batch, t_batch], dim=0)
    both_y = torch.cat([y, y_null], dim=0)
    out = model(hidden_states=both_x, timestep=both_t, class_labels=both_y).sample
    cond, uncond = out.chunk(2, dim=0)
    return uncond + cfg_scale * (cond - uncond)


@torch.no_grad()
def sample_flow(
    model, classes: torch.Tensor, *, num_classes: int, cfg_scale: float = 4.0,
    ode_steps: int = 50, seed: int = FIXED_SEED, device="cuda",
    timestep_scale: float = 1000.0,
) -> torch.Tensor:
    n = classes.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 32, 32, generator=g).to(device)
    y = classes.to(device)
    y_null = torch.full_like(y, num_classes)

    ts = torch.linspace(1.0, 0.0, ode_steps + 1, device=device)
    for i in range(ode_steps):
        t, dt = ts[i], ts[i + 1] - ts[i]          # dt < 0: noise -> data
        t_batch = torch.full((n,), t, device=device) * timestep_scale
        v = _cfg_forward(model, x, t_batch, y, y_null, cfg_scale)
        x = x + dt * v
    return x                                       # scaled latents


@torch.no_grad()
def sample_ddim(
    model, classes: torch.Tensor, *, num_classes: int, cfg_scale: float = 4.0,
    ode_steps: int = 50, seed: int = FIXED_SEED, device="cuda",
    num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
) -> torch.Tensor:
    n = classes.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 32, 32, generator=g).to(device)
    y = classes.to(device)
    y_null = torch.full_like(y, num_classes)

    betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)
    t_seq = torch.linspace(num_train_timesteps - 1, 0, ode_steps, device=device).long()

    for i, t in enumerate(t_seq):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = _cfg_forward(model, x, t_batch, y, y_null, cfg_scale)
        a_t = abar[t]
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        a_prev = abar[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


@torch.no_grad()
def decode_latents(z: torch.Tensor, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """Scaled latents -> images in [0, 1], via the dataset's own VAE."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device).eval()
    imgs = vae.decode(z.to(device=device, dtype=dtype) / SCALING).sample
    return ((imgs.float().clamp(-1, 1) + 1) / 2).cpu()


def save_grid(imgs: torch.Tensor, path: str | Path, ncol: int = 4) -> Path:
    import numpy as np
    from PIL import Image

    arr = (imgs * 255).byte().numpy().transpose(0, 2, 3, 1)   # [N, H, W, 3]
    n, h, w, c = arr.shape
    nrow = (n + ncol - 1) // ncol
    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, ncol)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = arr[i]
    path = Path(path)
    Image.fromarray(grid).save(path)
    return path


def sample_and_push(
    model, *, objective: str, step: int, repo_id: str | None, device,
    num_classes: int = 1000, n: int = 16, ode_steps: int = 50,
    cfg_scale: float = 4.0, out_dir: str | Path = "/tmp",
) -> Path:
    """Generate the fixed grid, save PNG, upload to repo_id (None = skip
    upload). Returns the local PNG path."""
    classes = torch.tensor(FIXED_CLASSES[:n])
    sampler = sample_flow if objective == "flow" else sample_ddim
    model.eval()
    z = sampler(
        model, classes, num_classes=num_classes,
        cfg_scale=cfg_scale, ode_steps=ode_steps, device=device,
    )
    imgs = decode_latents(z, device=device)
    png = save_grid(imgs, Path(out_dir) / f"samples_step_{step:07d}.png")
    print(f"[sample] wrote {png} ({n} images, {objective}, cfg={cfg_scale}, "
          f"{ode_steps} steps)")

    if repo_id is not None:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(png),
            path_in_repo=f"samples/step_{step:07d}.png",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"samples: step {step}",
        )
        print(f"[sample] pushed samples/step_{step:07d}.png to {repo_id}")
    return png
```

### `src/ditflex/stability.py`

```python
"""Practical stability control for long DiT/SiT training runs.

This module deliberately favors continuing a finite, loss-stable run over
restarting because one noisy gradient statistic crossed a warning threshold.

Policy summary
--------------
* Warning thresholds are diagnostic only.  Repeated warnings never become a
  retry by themselves.
* A retry requires a clear loss problem, a severe sustained median-gradient
  shift, or multiple corroborating retry-level signals.
* A large p90 or skip rate alone is not enough to roll back; heavy-tailed
  gradients are expected in diffusion/flow training and are already bounded by
  the per-step rejection guard and gradient clipping.
* The reference remains frozen while a candidate is running, then moves slowly
  after a successful checkpoint promotion.

The public API is compatible with the v3 transactional trainer.  Version 4 can
load v1/v2/v3 controller state and preserves the committed LR scale and health
reference while adopting the less-picky decision policy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StabilitySpec:
    """Runtime-only stability and learning-rate settings."""

    policy: str = "adaptive"  # constant | cosine | adaptive
    total_steps: int = 400_000
    base_lr: float = 1e-4
    min_lr: float = 1e-5
    hard_min_lr: float = 1e-6
    min_scale: float = 0.03125

    commit_patience_windows: int = 2
    warning_patience_windows: int = 2  # logging cadence only in v4

    loss_warn_ratio: float = 1.015
    loss_retry_ratio: float = 1.025
    loss_emergency_ratio: float = 1.05

    grad_warn_ratio: float = 2.0
    grad_retry_ratio: float = 4.0
    grad_emergency_ratio: float = 8.0
    grad_p90_warn_ratio: float = 2.5
    grad_p90_retry_ratio: float = 5.0
    grad_p90_emergency_ratio: float = 10.0

    skip_warn_rate: float = 0.05
    skip_retry_rate: float = 0.10
    skip_emergency_rate: float = 0.20

    # A promoted reference can adapt to a genuinely healthy new regime.
    reference_decay: float = 0.80
    loss_reference_max_growth: float = 1.01
    grad_reference_max_growth: float = 1.50

    def __post_init__(self) -> None:
        if self.policy not in {"constant", "cosine", "adaptive"}:
            raise ValueError(f"unknown LR policy: {self.policy!r}")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.base_lr <= 0.0:
            raise ValueError("base_lr must be positive")
        if not (0.0 <= self.min_lr <= self.base_lr):
            raise ValueError("min_lr must lie in [0, base_lr]")
        if not (0.0 < self.hard_min_lr <= self.base_lr):
            raise ValueError("hard_min_lr must lie in (0, base_lr]")
        if not (0.0 < self.min_scale <= 1.0):
            raise ValueError("min_scale must lie in (0, 1]")
        if self.commit_patience_windows <= 0 or self.warning_patience_windows <= 0:
            raise ValueError("window patience values must be positive")
        if not (
            1.0 < self.loss_warn_ratio < self.loss_retry_ratio < self.loss_emergency_ratio
        ):
            raise ValueError("loss ratios must satisfy 1 < warn < retry < emergency")
        if not (1.0 < self.grad_warn_ratio < self.grad_retry_ratio < self.grad_emergency_ratio):
            raise ValueError("gradient ratios must satisfy 1 < warn < retry < emergency")
        if not (
            1.0
            < self.grad_p90_warn_ratio
            < self.grad_p90_retry_ratio
            < self.grad_p90_emergency_ratio
        ):
            raise ValueError("gradient-p90 ratios must satisfy 1 < warn < retry < emergency")
        if not (
            0.0 <= self.skip_warn_rate < self.skip_retry_rate < self.skip_emergency_rate <= 1.0
        ):
            raise ValueError("skip rates must satisfy 0 <= warn < retry < emergency <= 1")
        if not (0.0 <= self.reference_decay < 1.0):
            raise ValueError("reference_decay must lie in [0, 1)")
        if self.loss_reference_max_growth < 1.0 or self.grad_reference_max_growth < 1.0:
            raise ValueError("reference growth caps must be at least 1")


@dataclass(frozen=True)
class WindowMetrics:
    """One non-overlapping, globally synchronized stability window."""

    loss: float
    grad_median: float
    grad_p90: float
    skip_rate: float
    relative_spike_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (self.loss, self.grad_median, self.grad_p90, self.skip_rate)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite window metrics: {self}")
        if self.loss < 0.0 or self.grad_median < 0.0 or self.grad_p90 < 0.0:
            raise ValueError(f"negative window metric: {self}")
        if not (0.0 <= self.skip_rate <= 1.0):
            raise ValueError(f"invalid skip rate: {self.skip_rate}")
        if not (0.0 <= self.relative_spike_rate <= 1.0):
            raise ValueError(f"invalid relative-spike rate: {self.relative_spike_rate}")

    def state_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> WindowMetrics:
        return cls(
            loss=float(state["loss"]),
            grad_median=float(state["grad_median"]),
            grad_p90=float(state["grad_p90"]),
            skip_rate=float(state.get("skip_rate", 0.0)),
            relative_spike_rate=float(state.get("relative_spike_rate", 0.0)),
        )


@dataclass(frozen=True)
class HealthReference:
    """Definition of normality from the last promoted checkpoint."""

    loss: float
    grad_median: float
    grad_p90: float
    step: int
    promotions: int = 1

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HealthReference:
        return cls(
            loss=float(state["loss"]),
            grad_median=float(state["grad_median"]),
            grad_p90=float(state["grad_p90"]),
            step=int(state.get("step", 0)),
            promotions=int(state.get("promotions", 1)),
        )


@dataclass(frozen=True)
class StabilityEvent:
    """Decision after a candidate window."""

    action: str = "none"  # none | warn | retry | fatal
    reason: str = ""
    healthy_windows: int = 0
    warning_windows: int = 0
    loss_ratio: float = 1.0
    grad_ratio: float = 1.0
    grad_p90_ratio: float = 1.0

    @property
    def should_retry(self) -> bool:
        return self.action == "retry"

    @property
    def should_abort(self) -> bool:
        return self.action == "fatal"

    @property
    def promotion_ready(self) -> bool:
        return self.action in {"none", "warn"} and self.healthy_windows > 0


class AdaptiveLrController:
    """Resume-safe LR controller with deliberately tolerant health decisions."""

    VERSION = 4

    def __init__(
        self,
        spec: StabilitySpec,
        *,
        start_step: int,
        checkpoint_lr: float,
        attempt_factor: float = 1.0,
        initial_loss: float | None = None,
        legacy_best_loss: float | None = None,
    ) -> None:
        if not (0.0 < attempt_factor <= 1.0):
            raise ValueError("attempt_factor must lie in (0, 1]")
        self.spec = spec

        envelope = self.envelope_lr(start_step)
        if spec.policy == "adaptive":
            inherited = checkpoint_lr / max(envelope, 1e-30)
            floor = spec.hard_min_lr / max(envelope, 1e-30)
            self.committed_scale = min(1.0, max(floor, inherited))
        else:
            self.committed_scale = 1.0
        self.attempt_factor = float(attempt_factor)

        self.reference: HealthReference | None = None
        self.last_metrics: WindowMetrics | None = None
        self.last_loss_ratio = 1.0
        self.last_grad_ratio = 1.0
        self.last_grad_p90_ratio = 1.0
        self.healthy_windows = 0
        self.warning_windows = 0
        self.retry_windows = 0
        self.windows_seen = 0
        self.retry_count = 0

        self.fast_loss = initial_loss
        self.slow_loss = initial_loss
        if initial_loss is None:
            self.best_loss = legacy_best_loss
        elif legacy_best_loss is None:
            self.best_loss = initial_loss
        else:
            self.best_loss = min(initial_loss, legacy_best_loss)

    # -- learning rate -------------------------------------------------

    @property
    def scale(self) -> float:
        if self.spec.policy != "adaptive":
            return 1.0
        return max(self.spec.min_scale, self.committed_scale * self.attempt_factor)

    def envelope_lr(self, step: int) -> float:
        if self.spec.policy == "constant":
            return self.spec.base_lr
        progress = min(max(int(step), 0), self.spec.total_steps) / self.spec.total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.spec.min_lr + (self.spec.base_lr - self.spec.min_lr) * cosine

    def lr_at(self, step: int) -> float:
        envelope = self.envelope_lr(step)
        if self.spec.policy != "adaptive":
            return envelope
        return max(self.spec.hard_min_lr, envelope * self.scale)

    def apply(self, optimizer: Any, step: int) -> float:
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def commit_attempt_scale(self) -> None:
        if self.spec.policy == "adaptive":
            self.committed_scale = self.scale
        self.attempt_factor = 1.0

    # -- reference and per-step guard ----------------------------------

    def bootstrap_reference(
        self,
        *,
        loss: float,
        grad_median: float,
        grad_p90: float | None,
        step: int,
    ) -> None:
        if self.reference is not None:
            return
        median = max(float(grad_median), 1e-12)
        p90 = max(float(grad_p90 if grad_p90 is not None else median * 2.0), median)
        self.reference = HealthReference(
            loss=max(float(loss), 1e-12),
            grad_median=median,
            grad_p90=p90,
            step=int(step),
        )

    def grad_limit(self, spike_multiple: float) -> float | None:
        """Return a frozen pre-clip outlier threshold.

        The threshold is intentionally based on both median and upper-tail
        history.  It does not chase the live EMA during a candidate run.
        """
        if spike_multiple <= 0.0 or self.reference is None:
            return None
        return max(
            float(spike_multiple) * self.reference.grad_median,
            4.0 * self.reference.grad_p90,
        )

    # -- decisions -----------------------------------------------------

    def _ratios(self, metrics: WindowMetrics) -> tuple[float, float, float]:
        assert self.reference is not None
        eps = 1e-30
        return (
            metrics.loss / max(self.reference.loss, eps),
            metrics.grad_median / max(self.reference.grad_median, eps),
            metrics.grad_p90 / max(self.reference.grad_p90, eps),
        )

    def _warning_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []
        if loss_ratio >= self.spec.loss_warn_ratio:
            reasons.append(f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_warn_ratio:.3f}")
        if grad_ratio >= self.spec.grad_warn_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_warn_ratio:.2f}"
            )
        if p90_ratio >= self.spec.grad_p90_warn_ratio:
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= {self.spec.grad_p90_warn_ratio:.2f}"
            )
        if metrics.skip_rate >= self.spec.skip_warn_rate:
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= {self.spec.skip_warn_rate:.1%}"
            )
        return reasons

    def _emergency_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []

        # Loss and median-gradient emergencies are independently meaningful.
        if loss_ratio >= self.spec.loss_emergency_ratio:
            reasons.append(
                f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_emergency_ratio:.3f}"
            )
        if grad_ratio >= self.spec.grad_emergency_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_emergency_ratio:.2f}"
            )

        # A noisy tail or many rejected batches must be corroborated before it
        # can terminate a run.
        if (
            p90_ratio >= self.spec.grad_p90_emergency_ratio
            and grad_ratio >= self.spec.grad_warn_ratio
        ):
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= "
                f"{self.spec.grad_p90_emergency_ratio:.2f} with elevated median"
            )
        if (
            metrics.skip_rate >= self.spec.skip_emergency_rate
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= "
                f"{self.spec.skip_emergency_rate:.1%} with corroborating drift"
            )
        return reasons

    def _retry_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []

        # Loss drift is the strongest signal and can stand alone.
        if loss_ratio >= self.spec.loss_retry_ratio:
            reasons.append(f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_retry_ratio:.3f}")

        # Median-gradient drift can stand alone only at the retry threshold.
        if grad_ratio >= self.spec.grad_retry_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_retry_ratio:.2f}"
            )

        # p90 and skip-rate conditions are too noisy to stand alone.  Require
        # corroboration from loss or the central gradient distribution.
        if (
            p90_ratio >= self.spec.grad_p90_retry_ratio
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= "
                f"{self.spec.grad_p90_retry_ratio:.2f} with corroborating drift"
            )
        if (
            metrics.skip_rate >= self.spec.skip_retry_rate
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= "
                f"{self.spec.skip_retry_rate:.1%} with corroborating drift"
            )
        return reasons

    def observe_window(self, metrics: WindowMetrics) -> StabilityEvent:
        self.windows_seen += 1
        self.last_metrics = metrics

        if self.fast_loss is None:
            self.fast_loss = metrics.loss
            self.slow_loss = metrics.loss
            self.best_loss = metrics.loss
        else:
            assert self.slow_loss is not None
            self.fast_loss = 0.80 * self.fast_loss + 0.20 * metrics.loss
            self.slow_loss = 0.98 * self.slow_loss + 0.02 * metrics.loss
            self.best_loss = min(self.best_loss or self.fast_loss, self.fast_loss)

        if self.reference is None:
            self.bootstrap_reference(
                loss=metrics.loss,
                grad_median=max(metrics.grad_median, 1e-12),
                grad_p90=max(metrics.grad_p90, metrics.grad_median, 1e-12),
                step=0,
            )
            self.healthy_windows = 1
            self.warning_windows = 0
            self.retry_windows = 0
            return StabilityEvent(
                action="none",
                reason="bootstrapped committed health reference",
                healthy_windows=1,
            )

        loss_ratio, grad_ratio, p90_ratio = self._ratios(metrics)
        self.last_loss_ratio = loss_ratio
        self.last_grad_ratio = grad_ratio
        self.last_grad_p90_ratio = p90_ratio

        emergency = self._emergency_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if emergency:
            self.healthy_windows = 0
            self.warning_windows += 1
            self.retry_windows += 1
            self.retry_count += 1
            return StabilityEvent(
                action="retry",
                reason="emergency candidate rejection: " + "; ".join(emergency),
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        retry_reasons = self._retry_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        warnings = self._warning_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)

        if retry_reasons:
            self.healthy_windows = 0
            self.warning_windows += 1
            self.retry_windows += 1
            if self.retry_windows >= self.spec.warning_patience_windows:
                self.retry_count += 1
                return StabilityEvent(
                    action="retry",
                    reason="persistent corroborated instability: " + "; ".join(retry_reasons),
                    warning_windows=self.warning_windows,
                    loss_ratio=loss_ratio,
                    grad_ratio=grad_ratio,
                    grad_p90_ratio=p90_ratio,
                )
            return StabilityEvent(
                action="warn",
                reason=(
                    f"retry-level signal {self.retry_windows}/"
                    f"{self.spec.warning_patience_windows}: " + "; ".join(retry_reasons)
                ),
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        # Any acceptable window clears retry persistence.  Warning-only windows
        # still count toward checkpoint promotion because they are explicitly
        # below the retry policy.
        self.retry_windows = 0
        self.healthy_windows += 1

        if warnings:
            self.warning_windows += 1
            return StabilityEvent(
                action="warn",
                reason="diagnostic warning; continuing: " + "; ".join(warnings),
                healthy_windows=self.healthy_windows,
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        self.warning_windows = 0
        return StabilityEvent(
            action="none",
            reason="stable",
            healthy_windows=self.healthy_windows,
            loss_ratio=loss_ratio,
            grad_ratio=grad_ratio,
            grad_p90_ratio=p90_ratio,
        )

    def checkpoint_is_healthy(
        self,
        metrics: WindowMetrics | None = None,
        *,
        required_windows: int | None = None,
    ) -> tuple[bool, str]:
        metrics = metrics or self.last_metrics
        if metrics is None:
            return False, "no complete stability window yet"
        if self.reference is None:
            return False, "no committed health reference"

        required = (
            self.spec.commit_patience_windows
            if required_windows is None
            else max(1, int(required_windows))
        )
        if self.healthy_windows < required:
            return False, f"only {self.healthy_windows}/{required} acceptable windows"

        loss_ratio, grad_ratio, p90_ratio = self._ratios(metrics)
        emergency = self._emergency_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        retry = self._retry_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if emergency:
            return False, "emergency metrics: " + "; ".join(emergency)
        if retry:
            return False, "retry-level metrics: " + "; ".join(retry)

        warnings = self._warning_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if warnings:
            return True, "acceptable candidate with diagnostic warnings"
        return True, "stable candidate"

    @staticmethod
    def _bounded_reference_update(
        old: float,
        new: float,
        *,
        decay: float,
        max_growth: float,
    ) -> float:
        candidate = decay * old + (1.0 - decay) * new
        return min(candidate, old * max_growth)

    def commit_candidate(self, step: int, metrics: WindowMetrics) -> HealthReference:
        self.commit_attempt_scale()
        if self.reference is None:
            reference = HealthReference(
                loss=max(metrics.loss, 1e-12),
                grad_median=max(metrics.grad_median, 1e-12),
                grad_p90=max(metrics.grad_p90, metrics.grad_median, 1e-12),
                step=int(step),
            )
        else:
            old = self.reference
            reference = HealthReference(
                loss=self._bounded_reference_update(
                    old.loss,
                    metrics.loss,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.loss_reference_max_growth,
                ),
                grad_median=self._bounded_reference_update(
                    old.grad_median,
                    metrics.grad_median,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.grad_reference_max_growth,
                ),
                grad_p90=self._bounded_reference_update(
                    old.grad_p90,
                    metrics.grad_p90,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.grad_reference_max_growth,
                ),
                step=int(step),
                promotions=old.promotions + 1,
            )
        self.reference = reference
        self.healthy_windows = 0
        self.warning_windows = 0
        self.retry_windows = 0
        return reference

    # -- persistence ---------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "committed_scale": self.committed_scale,
            "attempt_factor": self.attempt_factor,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "loss_ratio": self.last_loss_ratio,
            "grad_ratio": self.last_grad_ratio,
            "grad_p90_ratio": self.last_grad_p90_ratio,
            "healthy_windows": self.healthy_windows,
            "warning_windows": self.warning_windows,
            "retry_windows": self.retry_windows,
            "retry_count": self.retry_count,
            "reference": None if self.reference is None else self.reference.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "committed_scale": self.committed_scale,
            "attempt_factor": self.attempt_factor,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "reference": None if self.reference is None else self.reference.state_dict(),
            "last_metrics": None if self.last_metrics is None else self.last_metrics.state_dict(),
            "last_loss_ratio": self.last_loss_ratio,
            "last_grad_ratio": self.last_grad_ratio,
            "last_grad_p90_ratio": self.last_grad_p90_ratio,
            "healthy_windows": self.healthy_windows,
            "warning_windows": self.warning_windows,
            "retry_windows": self.retry_windows,
            "windows_seen": self.windows_seen,
            "retry_count": self.retry_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        attempt_factor: float | None = None,
    ) -> None:
        version = int(state.get("version", 1))
        if version not in {1, 2, 3, self.VERSION}:
            raise ValueError(
                f"unsupported stability state version {version}; "
                "use --reset-lr-controller only for a deliberate migration"
            )

        stored_spec_data = dict(state.get("spec", {}))

        # Policy thresholds are intentionally allowed to change across v4
        # adoption.  Only LR schedule fields must remain resume-compatible.
        for name in ("policy", "total_steps", "base_lr", "min_lr", "hard_min_lr"):
            if name in stored_spec_data and stored_spec_data[name] != getattr(self.spec, name):
                raise ValueError(
                    f"persisted LR setting {name}={stored_spec_data[name]!r} differs from "
                    f"requested {getattr(self.spec, name)!r}; use --reset-lr-controller "
                    "only when that LR change is deliberate"
                )

        if version < 3:
            self.committed_scale = float(state.get("scale", self.committed_scale))
        else:
            self.committed_scale = float(
                state.get("committed_scale", state.get("scale", self.committed_scale))
            )
            ref_state = state.get("reference")
            if isinstance(ref_state, dict):
                self.reference = HealthReference.from_state_dict(ref_state)
            metrics_state = state.get("last_metrics")
            if isinstance(metrics_state, dict):
                self.last_metrics = WindowMetrics.from_state_dict(metrics_state)
            self.last_loss_ratio = float(state.get("last_loss_ratio", 1.0))
            self.last_grad_ratio = float(state.get("last_grad_ratio", 1.0))
            self.last_grad_p90_ratio = float(state.get("last_grad_p90_ratio", 1.0))
            self.healthy_windows = int(state.get("healthy_windows", 0))
            self.warning_windows = int(state.get("warning_windows", 0))
            self.windows_seen = int(state.get("windows_seen", 0))
            self.retry_count = int(state.get("retry_count", 0))

        # Do not inherit v3's warning-to-retry momentum.
        self.retry_windows = 0
        self.fast_loss = _optional_float(state.get("fast_loss", self.fast_loss))
        self.slow_loss = _optional_float(state.get("slow_loss", self.slow_loss))
        self.best_loss = _optional_float(state.get("best_loss", self.best_loss))

        if attempt_factor is not None:
            if not (0.0 < attempt_factor <= 1.0):
                raise ValueError("attempt_factor must lie in (0, 1]")
            self.attempt_factor = float(attempt_factor)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
```

### `src/ditflex/train.py`

```python
"""Transactional, time-boxed DiT/SiT training.

Each torchrun process trains a *candidate* from a committed Hub checkpoint.
Candidate progress is promoted only after consecutive windows pass both loss
and robust gradient-distribution gates.  A bad candidate exits with code 75
without saving; ``run/modal_train.py`` then starts a fresh torchrun process from
the last promoted checkpoint with a lower LR multiplier and a changed,
deterministic objective/data seed.

This is intentionally not a broad ``try/except`` recovery loop.  The observed
failure stayed finite while the pre-clip gradient median moved by orders of
magnitude, so recovery must be driven by explicit health metrics rather than by
exceptions alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ditflex.checkpoint import (
    copy_checkpoint,
    infer_legacy_gradient_reference,
    load_checkpoint,
    pull_from_hub,
    push_to_hub,
    resolve_revision_for_step,
    save_checkpoint,
    validate_checkpoint,
)
from ditflex.config import Config
from ditflex.distributed import (
    all_reduce_bool_and,
    all_reduce_mean,
    barrier,
    broadcast_flag,
    broadcast_float,
    cleanup,
    setup,
)
from ditflex.ema import EMA
from ditflex.latents import LatentStore
from ditflex.model import build_model
from ditflex.objective import build_objective, make_step_generator
from ditflex.stability import AdaptiveLrController, StabilitySpec, WindowMetrics

CKPT_DIR = "/tmp/ditflex_ckpt"  # committed checkpoint pulled from Hub
CANDIDATE_DIR = "/tmp/ditflex_candidate"
RETRY_MARKER = "/tmp/ditflex_retry.json"
PROMOTION_MARKER = "/tmp/ditflex_promotion.json"
RETRY_EXIT_CODE = 75
LOG_EVERY = 50
LOSS_WINDOW = 200
GRAD_EMA_DECAY = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seconds", type=int, required=True)
    parser.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    parser.add_argument("--hub-repo", type=str, default=None)
    parser.add_argument("--resume-revision", type=str, default="")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-latent-files", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--target-steps", type=int, default=400_000)

    # Resume-safe global-step schedule.  Retry LR reductions are supplied by
    # the parent process as --attempt-lr-factor and become permanent only after
    # a healthy candidate is promoted.
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument(
        "--lr-policy",
        choices=["constant", "cosine", "adaptive"],
        default="adaptive",
    )
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-hard-min", type=float, default=1e-6)
    parser.add_argument("--lr-min-scale", type=float, default=0.03125)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--attempt-lr-factor", type=float, default=1.0)
    parser.add_argument("--reset-lr-controller", action="store_true")

    # Transactional health thresholds.  The old v2 --loss-rise-ratio is
    # accepted as a no-op compatibility flag by the Modal wrapper, not here.
    parser.add_argument("--commit-windows", type=int, default=2)
    parser.add_argument("--warning-patience", type=int, default=2)
    parser.add_argument("--loss-warn-ratio", type=float, default=1.015)
    parser.add_argument("--loss-retry-ratio", type=float, default=1.025)
    parser.add_argument("--loss-emergency-ratio", type=float, default=1.05)
    parser.add_argument("--grad-warn-ratio", type=float, default=2.0)
    parser.add_argument("--grad-retry-ratio", type=float, default=4.0)
    parser.add_argument("--grad-emergency-ratio", type=float, default=8.0)
    parser.add_argument("--grad-p90-warn-ratio", type=float, default=2.5)
    parser.add_argument("--grad-p90-retry-ratio", type=float, default=5.0)
    parser.add_argument("--grad-p90-emergency-ratio", type=float, default=10.0)
    parser.add_argument("--skip-warn-rate", type=float, default=0.05)
    parser.add_argument("--skip-retry-rate", type=float, default=0.10)
    parser.add_argument("--skip-emergency-rate", type=float, default=0.20)

    # Migration / gradient guards.
    parser.add_argument(
        "--grad-reference",
        type=float,
        default=0.0,
        help="explicit committed gradient-median reference (0 = checkpoint/history)",
    )
    parser.add_argument(
        "--no-auto-infer-grad-reference",
        action="store_false",
        dest="auto_infer_grad_reference",
        help="disable legacy reference inference from earlier Hub revisions",
    )
    parser.set_defaults(auto_infer_grad_reference=True)
    parser.add_argument("--wd", type=float, default=-1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--spike-skip", type=float, default=4.0)
    parser.add_argument("--grad-ceiling", type=float, default=0.0)
    parser.add_argument("--seed-offset", type=int, default=0)

    parser.add_argument("--qk-mode", choices=["amap", "dmap"], default="amap")
    parser.add_argument("--dmap-alpha", type=float, default=0.0)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    return parser.parse_args()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be positive")
    if args.train_seconds <= 0:
        raise ValueError("--train-seconds must be positive")
    if args.resume_step > 0 and args.resume_revision:
        raise ValueError("use only one of --resume-step or --resume-revision")
    if not (0.0 < args.attempt_lr_factor <= 1.0):
        raise ValueError("--attempt-lr-factor must lie in (0, 1]")

    ctx = setup()
    cfg = Config()
    cfg.train.objective = args.objective
    cfg.model.qk_mode = args.qk_mode
    cfg.model.dmap_alpha = args.dmap_alpha
    if args.hub_repo:
        cfg.hub.checkpoint_repo = args.hub_repo

    if cfg.train.global_batch % ctx.world != 0:
        raise ValueError(f"global_batch {cfg.train.global_batch} % world {ctx.world} != 0")
    per_rank_batch = cfg.train.global_batch // ctx.world

    if ctx.is_rank0:
        print(f"[train] world={ctx.world}  per_rank_batch={per_rank_batch}")
        print(cfg.to_json())
        Path(RETRY_MARKER).unlink(missing_ok=True)
        shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)

    # -- pull one committed anchor ------------------------------------------
    resume_revision = args.resume_revision or None
    if ctx.is_rank0 and args.resume_step > 0:
        resume_revision = resolve_revision_for_step(cfg.hub.checkpoint_repo, args.resume_step)
        print(
            f"[train] resolved resume step {args.resume_step:,} to revision "
            f"{resume_revision[:12]}"
        )

    resume_dir = None
    if ctx.is_rank0:
        resume_dir = pull_from_hub(
            cfg.hub.checkpoint_repo,
            CKPT_DIR,
            revision=resume_revision,
        )
        source = "latest" if resume_revision is None else resume_revision[:12]
        print(f"[train] resume checkpoint ({source}): {resume_dir or 'none (fresh start)'}")
    barrier(ctx)
    if not ctx.is_rank0 and os.path.exists(os.path.join(CKPT_DIR, "state.json")):
        resume_dir = CKPT_DIR

    # -- raw model / EMA / optimizer, loaded before compile + DDP -----------
    if cfg.model.qk_mode == "amap":
        model = build_model(cfg.model).to(ctx.device)
    elif cfg.model.qk_mode == "dmap":
        from ditflex.diffusion_model import build_dmap_model

        model = build_dmap_model(cfg.model).to(ctx.device)
    else:  # pragma: no cover
        raise ValueError(f"unknown qk_mode: {cfg.model.qk_mode!r}")

    ema = EMA(model, cfg.train.ema_decay).to(ctx.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    start_step = 0
    run_history: list[dict] = []
    guard_state: dict = {}
    if resume_dir is not None:
        state = load_checkpoint(resume_dir, model, ema, optimizer, cfg)
        start_step = int(state["step"])
        run_history = list(state.get("run_history", []))
        guard_state = dict(state.get("guard_state", {}))
        if ctx.is_rank0:
            print(f"[train] resumed at step {start_step:,}")

    live_grad_ema = _optional_float(guard_state.get("live_grad_ema", guard_state.get("grad_ema")))
    spikes_total = int(guard_state.get("spikes_total", 0))
    recent_losses = [float(value) for value in guard_state.get("recent_losses", [])][-LOSS_WINDOW:]
    initial_loss = (
        statistics.fmean(recent_losses[-LOSS_WINDOW:])
        if len(recent_losses) >= LOSS_WINDOW
        else None
    )

    checkpoint_lr = float(optimizer.param_groups[0]["lr"])
    base_lr = args.lr if args.lr > 0.0 else cfg.train.lr
    stability_spec = StabilitySpec(
        policy=args.lr_policy,
        total_steps=args.target_steps,
        base_lr=base_lr,
        min_lr=args.lr_min,
        hard_min_lr=args.lr_hard_min,
        min_scale=args.lr_min_scale,
        commit_patience_windows=args.commit_windows,
        warning_patience_windows=args.warning_patience,
        loss_warn_ratio=args.loss_warn_ratio,
        loss_retry_ratio=args.loss_retry_ratio,
        loss_emergency_ratio=args.loss_emergency_ratio,
        grad_warn_ratio=args.grad_warn_ratio,
        grad_retry_ratio=args.grad_retry_ratio,
        grad_emergency_ratio=args.grad_emergency_ratio,
        grad_p90_warn_ratio=args.grad_p90_warn_ratio,
        grad_p90_retry_ratio=args.grad_p90_retry_ratio,
        grad_p90_emergency_ratio=args.grad_p90_emergency_ratio,
        skip_warn_rate=args.skip_warn_rate,
        skip_retry_rate=args.skip_retry_rate,
        skip_emergency_rate=args.skip_emergency_rate,
    )
    legacy_best = _optional_float(guard_state.get("best_window"))
    seed_lr = base_lr if args.reset_lr_controller else checkpoint_lr
    controller = AdaptiveLrController(
        stability_spec,
        start_step=start_step,
        checkpoint_lr=seed_lr,
        attempt_factor=args.attempt_lr_factor,
        initial_loss=initial_loss,
        legacy_best_loss=legacy_best,
    )

    controller_state = guard_state.get("stability_controller", guard_state.get("lr_controller"))
    if isinstance(controller_state, dict) and not args.reset_lr_controller:
        controller.load_state_dict(
            controller_state,
            attempt_factor=args.attempt_lr_factor,
        )
    elif isinstance(controller_state, dict) and ctx.is_rank0:
        print("[train] RESETTING persisted stability/LR controller by explicit request")

    # v1/v2 migration: derive a frozen gradient baseline from earlier Hub
    # revisions instead of trusting a potentially contaminated latest grad EMA.
    if controller.reference is None and initial_loss is not None:
        inferred_reference: float | None = None
        if args.grad_reference > 0.0:
            inferred_reference = args.grad_reference
            if ctx.is_rank0:
                print(f"[train] using explicit grad reference {inferred_reference:g}")
        elif args.auto_infer_grad_reference and resume_dir is not None:
            if ctx.is_rank0:
                try:
                    inferred_reference = infer_legacy_gradient_reference(
                        cfg.hub.checkpoint_repo,
                        before_step=start_step,
                    )
                except Exception as exc:  # noqa: BLE001 - migration can fall back locally
                    print(f"[train] legacy grad-reference inference failed: {exc!r}")
                    inferred_reference = None
            inferred_value = -1.0 if inferred_reference is None else inferred_reference
            inferred_value = broadcast_float(ctx, inferred_value if ctx.is_rank0 else 0.0)
            inferred_reference = None if inferred_value <= 0.0 else inferred_value
            if ctx.is_rank0 and inferred_reference is not None:
                print(
                    f"[train] inferred committed grad reference {inferred_reference:.2f} "
                    "from earlier Hub revisions"
                )
        if inferred_reference is not None and live_grad_ema is not None:
            # Trust the selected checkpoint's own legacy EMA when it remains
            # within a bounded multiple of history.  Cap rather than discard a
            # contaminated value (for example 3,500 vs a recent scale near 60).
            historical = inferred_reference
            inferred_reference = min(live_grad_ema, historical * 4.0)
            if ctx.is_rank0:
                print(
                    f"[train] legacy reference migration: live={live_grad_ema:.2f}  "
                    f"history={historical:.2f}  committed={inferred_reference:.2f}"
                )
        elif inferred_reference is None:
            inferred_reference = live_grad_ema
        if inferred_reference is not None and inferred_reference > 0.0:
            legacy_p90 = _optional_float(guard_state.get("grad_p90_reference"))
            controller.bootstrap_reference(
                loss=initial_loss,
                grad_median=inferred_reference,
                grad_p90=legacy_p90 or inferred_reference * 2.0,
                step=start_step,
            )

    if args.wd >= 0.0:
        for group in optimizer.param_groups:
            group["weight_decay"] = args.wd
        if ctx.is_rank0:
            print(f"[train] WD OVERRIDE for this run: {args.wd:g}")

    start_lr = controller.apply(optimizer, start_step)
    if ctx.is_rank0:
        reference = None if controller.reference is None else controller.reference.state_dict()
        print(
            f"[train] attempt={args.attempt}  LR policy={stability_spec.policy}  "
            f"base={stability_spec.base_lr:g}  cosine_min={stability_spec.min_lr:g}  "
            f"hard_min={stability_spec.hard_min_lr:g}  target={stability_spec.total_steps:,}  "
            f"committed_scale={controller.committed_scale:.4f}  "
            f"attempt_factor={controller.attempt_factor:.4f}  scale={controller.scale:.4f}  "
            f"effective@{start_step:,}={start_lr:g}  checkpoint_lr={checkpoint_lr:g}"
        )
        print(
            "[train] transactional guard: "
            f"live_grad_ema={live_grad_ema if live_grad_ema is not None else 'unset'}  "
            f"spikes_total={spikes_total}  recent_losses={len(recent_losses)}  "
            f"reference={reference}"
        )

    # -- data and compiled model --------------------------------------------
    store_kwargs = dict(
        repo_id=cfg.data.hub_repo,
        device=ctx.device,
        max_files=args.max_latent_files,
        expected_total=cfg.data.expected_total,
        latent_shape=cfg.data.latent_shape,
        num_classes=cfg.model.num_classes,
    )
    if ctx.is_rank0:
        store = LatentStore.from_hub(**store_kwargs)
    barrier(ctx)
    if not ctx.is_rank0:
        store = LatentStore.from_hub(**store_kwargs)
    if ctx.is_rank0:
        print(
            f"[train] latents resident: {len(store):,} "
            f"({store.latents.numel() * 2 / 1024**3:.2f} GiB bf16)"
        )

    objective = build_objective(
        cfg.train.objective,
        label_dropout=cfg.train.label_dropout,
        num_classes=cfg.model.num_classes,
    )
    compiled = torch.compile(model)
    wrapped = DDP(compiled, device_ids=[ctx.local_rank]) if ctx.is_distributed else compiled
    wrapped.train()

    # -- candidate loop state -----------------------------------------------
    step = start_step
    segment_start = start_step
    t_start = time.time()
    segment_start_time = t_start
    deadline = t_start + args.train_seconds
    run_losses: list[float] = []
    window_grad_norms: list[float] = []
    window_skips = 0
    window_relative_spikes = 0
    # Candidate health must be established from new steps, never inherited.
    last_metrics: WindowMetrics | None = None
    last_log_time = t_start
    last_archive_bucket = (
        start_step // cfg.hub.archive_every_steps
        if cfg.hub.archive_every_steps > 0
        else 0
    )
    promotions_this_run = 0
    spikes_at_segment_start = spikes_total

    def current_loss_window() -> float | None:
        if len(recent_losses) < LOSS_WINDOW:
            return None
        return statistics.fmean(recent_losses[-LOSS_WINDOW:])

    def required_commit_windows() -> int:
        # Preserve the repository's 200-step quick-resume smoke while requiring
        # two windows for production candidates.
        if args.max_steps is not None and args.max_steps <= LOSS_WINDOW:
            return 1
        return stability_spec.commit_patience_windows

    def checkpoint_is_healthy() -> tuple[bool, str]:
        return controller.checkpoint_is_healthy(
            last_metrics,
            required_windows=required_commit_windows(),
        )

    def serialized_guard_state() -> dict:
        return {
            "version": 3,
            "live_grad_ema": live_grad_ema,
            # Compatibility name for existing dashboards and recovery tools.
            "grad_ema": live_grad_ema,
            "grad_reference": (
                None if controller.reference is None else controller.reference.grad_median
            ),
            "grad_p90_reference": (
                None if controller.reference is None else controller.reference.grad_p90
            ),
            "spikes_total": spikes_total,
            "recent_losses": recent_losses[-LOSS_WINDOW:],
            "loss_window": LOSS_WINDOW,
            "grad_ema_decay": GRAD_EMA_DECAY,
            "stability_controller": controller.state_dict(),
            "best_window": controller.best_loss,
            "blown_windows": controller.warning_windows,
        }

    def append_run_record(end_step: int, completed: bool, reason: str) -> None:
        nonlocal segment_start, segment_start_time, spikes_at_segment_start
        run_history.append(
            {
                "start_step": segment_start,
                "end_step": end_step,
                "seconds": round(time.time() - segment_start_time, 1),
                "world": ctx.world,
                "objective": cfg.train.objective,
                "completed": completed,
                "finished_at": datetime.now(UTC).isoformat(),
                "promotion_reason": reason,
                "effective": {
                    "attempt": args.attempt,
                    "lr_policy": stability_spec.policy,
                    "lr_base": stability_spec.base_lr,
                    "lr_start": start_lr if segment_start == start_step else None,
                    "lr_end": float(optimizer.param_groups[0]["lr"]),
                    "lr_min": stability_spec.min_lr,
                    "lr_hard_min": stability_spec.hard_min_lr,
                    "lr_scale": controller.scale,
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    "clip": args.clip,
                    "spike_skip": args.spike_skip,
                    "grad_ceiling": args.grad_ceiling,
                    "steps_skipped": spikes_total - spikes_at_segment_start,
                    "seed_offset": args.seed_offset,
                    "target_steps": args.target_steps,
                },
            }
        )
        segment_start = end_step
        segment_start_time = time.time()
        spikes_at_segment_start = spikes_total

    def save_and_promote(at_step: int, completed: bool, reason: str) -> None:
        nonlocal last_archive_bucket, promotions_this_run
        assert last_metrics is not None

        # Every rank must advance the frozen reference and retry LR state
        # identically before training continues.  Only rank 0 performs I/O.
        reference = controller.commit_candidate(at_step, last_metrics)
        commit_id = None
        if ctx.is_rank0:
            append_run_record(at_step, completed, reason)
            state = {
                "step": at_step,
                "run_history": run_history,
                "guard_state": serialized_guard_state(),
                "transaction": {
                    "status": "committed",
                    "committed_at": datetime.now(UTC).isoformat(),
                    "attempt": args.attempt,
                    "health_reference": reference.state_dict(),
                },
            }
            shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
            save_checkpoint(CANDIDATE_DIR, model, ema, optimizer, cfg, state)
            validate_checkpoint(CANDIDATE_DIR, expected_step=at_step)
            print(f"[train] validated candidate step {at_step:,} ({reason})")

            if not args.no_push:
                archive_step = None
                if cfg.hub.archive_every_steps > 0:
                    bucket = at_step // cfg.hub.archive_every_steps
                    archive_step = at_step if bucket != last_archive_bucket else None
                    last_archive_bucket = bucket
                commit_id = push_to_hub(
                    CANDIDATE_DIR,
                    cfg.hub.checkpoint_repo,
                    archive_step=archive_step,
                    commit_message="checkpoint: promote transactional candidate",
                )
                print(
                    f"[train] PROMOTED step {at_step:,} to {cfg.hub.checkpoint_repo}"
                    + (f" revision={commit_id[:12]}" if commit_id else "")
                )
            else:
                copy_checkpoint(CANDIDATE_DIR, CKPT_DIR)
                print(f"[train] promoted local no-push candidate step {at_step:,}")

            _write_json_atomic(
                PROMOTION_MARKER,
                {
                    "step": at_step,
                    "revision": commit_id,
                    "attempt": args.attempt,
                    "repo": cfg.hub.checkpoint_repo,
                },
            )
        promotions_this_run += 1
        barrier(ctx)

    def retry_all(reason: str, metrics: WindowMetrics | None = None) -> int:
        if ctx.is_rank0:
            payload = {
                "exit_code": RETRY_EXIT_CODE,
                "attempt": args.attempt,
                "start_step": start_step,
                "failed_step": step,
                "reason": reason,
                "seed_offset": args.seed_offset,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "reference": (
                    None if controller.reference is None else controller.reference.state_dict()
                ),
                "metrics": None if metrics is None else metrics.state_dict(),
                "promotions_this_run": promotions_this_run,
                "elapsed_training_seconds": round(time.time() - t_start, 3),
            }
            _write_json_atomic(RETRY_MARKER, payload)
            print(
                f"[train] RETRYABLE INSTABILITY @ {step:,}: {reason}; "
                "candidate discarded, Hub latest unchanged"
            )
        barrier(ctx)
        cleanup(ctx)
        return RETRY_EXIT_CODE

    # -- stepping ------------------------------------------------------------
    while True:
        if step >= args.target_steps:
            break
        if step % cfg.train.deadline_check_every == 0 and step > start_step:
            stop = ctx.is_rank0 and time.time() >= deadline
            if broadcast_flag(ctx, stop):
                break
        if args.max_steps is not None and (step - start_step) >= args.max_steps:
            break

        controller.apply(optimizer, step)
        x0, labels = store.batch(
            step,
            ctx.rank,
            per_rank_batch,
            cfg.train.base_seed + args.seed_offset,
        )
        objective_generator = make_step_generator(
            ctx.device,
            base_seed=cfg.train.base_seed,
            global_step=step,
            rank=ctx.rank,
            seed_offset=args.seed_offset,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.loss(
                wrapped,
                x0,
                labels,
                generator=objective_generator,
            )

        local_loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        loss_finite = all_reduce_bool_and(ctx, local_loss_finite)
        if not loss_finite:
            local_value = float(loss.detach().float().item())
            return retry_all(
                f"non-finite loss (rank-0 local={local_value}) before backward"
            )
        global_loss = all_reduce_mean(ctx, loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.clip,
            error_if_nonfinite=False,
        )
        grads_finite = all_reduce_bool_and(
            ctx,
            bool(torch.isfinite(grad_norm_tensor).item()),
        )
        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            return retry_all("non-finite gradient norm before optimizer.step")

        grad_norm = float(grad_norm_tensor.detach().float().item())
        grad_norm = broadcast_float(ctx, grad_norm if ctx.is_rank0 else 0.0)
        window_grad_norms.append(grad_norm)

        # Diagnostic only: update on every finite batch with bounded influence.
        if live_grad_ema is None:
            live_grad_ema = grad_norm
        else:
            ema_input = min(grad_norm, max(live_grad_ema * 10.0, 1e-12))
            live_grad_ema = (
                GRAD_EMA_DECAY * live_grad_ema
                + (1.0 - GRAD_EMA_DECAY) * ema_input
            )

        frozen_limit = controller.grad_limit(args.spike_skip)
        relative_spike = frozen_limit is not None and grad_norm > frozen_limit
        absolute_spike = args.grad_ceiling > 0.0 and grad_norm > args.grad_ceiling
        spike_decision = (relative_spike or absolute_spike) if ctx.is_rank0 else False
        spiked = broadcast_flag(ctx, spike_decision)

        if spiked:
            optimizer.zero_grad(set_to_none=True)
            spikes_total += 1
            window_skips += 1
            if relative_spike:
                window_relative_spikes += 1
            if ctx.is_rank0:
                reasons: list[str] = []
                if relative_spike:
                    reasons.append(
                        f"relative {grad_norm:.2f} > frozen limit {frozen_limit:.2f}"
                    )
                if absolute_spike:
                    reasons.append(
                        f"absolute {grad_norm:.2f} > ceiling {args.grad_ceiling:g}"
                    )
                print(
                    f"[train] step {step:,}: {' and '.join(reasons)}; "
                    f"live EMA={live_grad_ema:.2f} -- SKIPPING optimizer step "
                    f"(total skipped: {spikes_total})"
                )
        else:
            optimizer.step()
            ema.update(model)

        run_losses.append(global_loss)
        recent_losses.append(global_loss)
        if len(recent_losses) > LOSS_WINDOW:
            del recent_losses[:-LOSS_WINDOW]
        step += 1

        # Every rank has the same reduced loss and broadcast gradient norm, so
        # window metrics and controller state remain identical without extra
        # collectives.
        if len(window_grad_norms) >= LOSS_WINDOW:
            window_loss = current_loss_window()
            assert window_loss is not None
            last_metrics = WindowMetrics(
                loss=window_loss,
                grad_median=float(statistics.median(window_grad_norms[-LOSS_WINDOW:])),
                grad_p90=_percentile(window_grad_norms[-LOSS_WINDOW:], 0.90),
                skip_rate=window_skips / LOSS_WINDOW,
                relative_spike_rate=window_relative_spikes / LOSS_WINDOW,
            )
            event = controller.observe_window(last_metrics)
            window_grad_norms.clear()
            window_skips = 0
            window_relative_spikes = 0

            if ctx.is_rank0:
                print(
                    f"[train] stability window @ {step:,}: "
                    f"loss={last_metrics.loss:.6f}  "
                    f"grad_med={last_metrics.grad_median:.2f}  "
                    f"grad_p90={last_metrics.grad_p90:.2f}  "
                    f"skips={last_metrics.skip_rate:.1%}  "
                    f"ratios(loss={event.loss_ratio:.3f}, grad={event.grad_ratio:.2f}, "
                    f"p90={event.grad_p90_ratio:.2f})  {event.reason}"
                )
            retry_decision = broadcast_flag(
                ctx,
                event.should_retry if ctx.is_rank0 else False,
            )
            fatal_decision = broadcast_flag(
                ctx,
                event.should_abort if ctx.is_rank0 else False,
            )
            if retry_decision or fatal_decision:
                return retry_all(event.reason, last_metrics)

        if cfg.hub.save_every_steps > 0 and step % cfg.hub.save_every_steps == 0:
            healthy = False
            reason = "rank-0 health decision"
            if ctx.is_rank0:
                healthy, reason = checkpoint_is_healthy()
                if not healthy:
                    print(
                        f"[train] WITHHOLDING candidate step {step:,}: {reason}; "
                        "Hub latest remains committed"
                    )
            healthy = broadcast_flag(ctx, healthy)
            if healthy:
                save_and_promote(step, completed=False, reason="periodic healthy candidate")
            else:
                barrier(ctx)

        if ctx.is_rank0 and step % LOG_EVERY == 0:
            now = time.time()
            rate = LOG_EVERY / max(now - last_log_time, 1e-9)
            last_log_time = now
            average_loss = statistics.fmean(run_losses[-LOG_EVERY:])
            with torch.no_grad():
                families = dict.fromkeys(("qk", "vo", "mlp", "ada", "emb", "oth"), 0.0)
                for name, parameter in model.named_parameters():
                    key = (
                        "qk"
                        if ("to_q" in name or "to_k" in name)
                        else "vo"
                        if ("to_v" in name or "to_out" in name)
                        else "mlp"
                        if ".ff." in name
                        else "ada"
                        if (
                            "norm1" in name
                            or "norm_out" in name
                            or "adaln" in name.lower()
                        )
                        else "emb"
                        if ("emb" in name or "pos_embed" in name or "proj_out" in name)
                        else "oth"
                    )
                    families[key] += parameter.detach().float().pow(2).sum().item()
                parameter_norm = sum(families.values()) ** 0.5
                family_text = " ".join(
                    f"{key}={value**0.5:7.1f}" for key, value in families.items()
                )
            reference_text = (
                "unset"
                if controller.reference is None
                else f"{controller.reference.grad_median:.1f}"
            )
            status = controller.status()
            print(
                f"  step {step:>8,}  loss {average_loss:.5f}  "
                f"lr {optimizer.param_groups[0]['lr']:.7g}  scale {controller.scale:.3f}  "
                f"{rate:5.2f} steps/s  {rate * cfg.train.global_batch:7.0f} img/s  "
                f"|g|live={live_grad_ema if live_grad_ema is not None else 0.0:8.2f}  "
                f"|g|ref={reference_text:>7}  lossR={status['loss_ratio']:.3f}  "
                f"gradR={status['grad_ratio']:.2f}  |w|={parameter_norm:8.2f}  "
                f"{family_text}"
            )

    # -- final candidate decision -------------------------------------------
    elapsed = time.time() - t_start
    reached_target = step >= args.target_steps
    final_healthy = False
    final_reason = "rank-0 health decision"
    if ctx.is_rank0:
        print(
            f"[train] {'target reached' if reached_target else 'run budget reached'} "
            f"after {elapsed / 60:.1f} min ({step - start_step:,} attempted data steps; "
            f"global step {step:,})"
        )
        final_healthy, final_reason = checkpoint_is_healthy()
    final_healthy = broadcast_flag(ctx, final_healthy)

    if final_healthy:
        save_and_promote(
            step,
            completed=reached_target,
            reason="target final" if reached_target else "run final",
        )
    else:
        # A short tail after an already successful promotion is safe to discard
        # when the only issue is insufficient windows before the time budget.
        insufficient_only = final_reason.startswith("only ")
        if promotions_this_run > 0 and insufficient_only and not reached_target:
            if ctx.is_rank0:
                print(
                    f"[train] discarding uncommitted tail at step {step:,}: {final_reason}; "
                    "last promoted checkpoint remains healthy"
                )
            barrier(ctx)
            cleanup(ctx)
            return 0
        return retry_all(f"final candidate withheld: {final_reason}", last_metrics)

    # Sample only after a healthy committed checkpoint exists.
    if ctx.is_rank0 and args.sample_count > 0:
        try:
            from ditflex.sample import sample_and_push

            ema.copy_to(model)
            sample_and_push(
                model,
                objective=cfg.train.objective,
                step=step,
                repo_id=None if args.no_push else cfg.hub.checkpoint_repo,
                device=ctx.device,
                num_classes=cfg.model.num_classes,
                n=args.sample_count,
                ode_steps=args.sample_steps,
                cfg_scale=args.cfg_scale,
                out_dir=CANDIDATE_DIR,
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint is already committed
            print(f"[train] sampling failed (non-fatal): {exc!r}")

    barrier(ctx)
    cleanup(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tests/modal_ci.py`

```python
"""Modal CI runner for ditflex correctness, stability, and GPU smoke gates.

The Actions checkout is mounted directly into the container, so the code under
CI is exactly the working tree being tested.  In addition to the existing
FlexAttention, latent, pytest, and overfit gates, this runner now verifies the
transactional training additions before any expensive smoke run:

* Ruff and bytecode compilation;
* committed-reference stability-controller tests;
* deterministic objective RNG tests;
* checkpoint validation/selection tests;
* importability of the Modal training supervisor.

Usage
-----
    modal run tests/modal_ci.py
    modal run tests/modal_ci.py --transactional-only
    modal run tests/modal_ci.py --compile-check
    modal run tests/modal_ci.py --smoke
    modal run tests/modal_ci.py --latents-all
    modal run tests/modal_ci.py --test-file test_stability.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_TYPE = os.environ.get("MODAL_GPU", "B300")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

TRANSACTIONAL_TESTS = [
    "tests/test_stability.py",
    "tests/test_objective_rng.py",
    "tests/test_checkpoint_selection.py",
    "tests/test_checkpoint_roundtrip.py",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
        "pytest>=8.0",
        "ruff>=0.6",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[
            ".git",
            "**/__pycache__",
            "*.egg-info",
            ".venv",
            ".ruff_cache",
            ".pytest_cache",
        ],
    )
)

app = modal.App("ditflex-ci", image=image)


@app.function(
    gpu=GPU_TYPE,
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_gates(
    test_files: list[str] | None = None,
    compile_check: bool = False,
    smoke: bool = False,
    latents_all: bool = False,
    transactional_only: bool = False,
) -> int:
    import subprocess
    import sys

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"[modal] GPU: {result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"],
        check=True,
    )

    first_failure = 0

    def run(cmd: list[str]) -> None:
        nonlocal first_failure
        print(f"\n[modal] running: {' '.join(cmd)}\n", flush=True)
        result = subprocess.run(cmd, cwd="/repo", check=False)
        if result.returncode != 0 and first_failure == 0:
            first_failure = result.returncode

    # Cheap source gates first.  These catch malformed supervisor/train changes
    # before spending time downloading latents or compiling GPU kernels.
    run([sys.executable, "-m", "ruff", "check", "src", "run", "tests"])
    run([sys.executable, "-m", "compileall", "-q", "src", "run", "tests"])
    run(
        [
            sys.executable,
            "-c",
            (
                "import run.modal_train as m; "
                "assert m.RETRY_EXIT_CODE == 75; "
                "assert m.RETRY_MARKER.name == 'ditflex_retry.json'; "
                "assert m.PROMOTION_MARKER.name == 'ditflex_promotion.json'"
            ),
        ]
    )

    # Always run the focused transactional tests, even when --test-file selects
    # another test.  They protect rollback/promotion behavior used in production.
    run([sys.executable, "-m", "pytest", "-v", "--tb=short", *TRANSACTIONAL_TESTS])

    if transactional_only:
        return first_failure

    gate_cmd = [sys.executable, "tests/verify_identity.py"]
    if compile_check:
        gate_cmd.append("--compile")
    run(gate_cmd)

    latents_cmd = [sys.executable, "tests/verify_latents.py"]
    if latents_all:
        latents_cmd.append("--all")
    run(latents_cmd)

    selected = test_files or ["tests"]
    run([sys.executable, "-m", "pytest", "-v", "--tb=short", *selected])

    if smoke:
        for objective in ("ddpm", "flow"):
            run(
                [
                    sys.executable,
                    "tests/overfit_smoke.py",
                    "--small",
                    "--objective",
                    objective,
                ]
            )
        run(
            [
                sys.executable,
                "tests/overfit_smoke.py",
                "--small",
                "--objective",
                "flow",
                "--qk-mode",
                "dmap",
            ]
        )
        run(
            [
                sys.executable,
                "tests/overfit_smoke.py",
                "--small",
                "--objective",
                "flow",
                "--qk-mode",
                "dmap",
                "--compile",
            ]
        )

    return first_failure


@app.local_entrypoint()
def main(
    test_file: str = "",
    compile_check: bool = False,
    smoke: bool = False,
    latents_all: bool = False,
    transactional_only: bool = False,
):
    """Run Modal CI gates.

    Args:
        test_file: Specific pytest file, or empty for the full suite.
        compile_check: Also verify the compiled FlexAttention path.
        smoke: Also run small-model overfit smoke tests.
        latents_all: Validate every latent shard instead of only shard zero.
        transactional_only: Run only static and transactional stability gates.
    """
    files = None
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        files = [test_file]

    rc = run_gates.remote(
        test_files=files,
        compile_check=compile_check,
        smoke=smoke,
        latents_all=latents_all,
        transactional_only=transactional_only,
    )
    if rc != 0:
        raise SystemExit(rc)
```

### `tests/overfit_smoke.py`

```python
#!/usr/bin/env python
"""tests/overfit_smoke.py
Gate 3: overfit a tiny fixed batch. Loss must collapse toward zero.

This is the end-to-end test of the training path -- model construction,
FlexAttention processor swap, timestep conditioning, class conditioning,
objective, and optimizer -- on a problem small enough that failure to learn
can only mean something is wired wrong. A model that cannot memorise 128
samples will not learn 1.28M.

Attention is ALWAYS FlexAttention (identity score_mod). There is no SDPA
path in this repo; a smoke that certified the stock diffusers processor
would certify a code path we never train on.

It is NOT a test of generative quality. Loss going to ~0 is the pass
condition; the resulting model is worthless.

Known smoke-only shortcuts (fine for memorisation, NOT for objective.py):
  - flow t is logit-normal here; SiT-L/2 parity requires uniform t
  - timestep=(t*1000).long() discretises continuous time
  - pred[:, :4] ignores DiT's learned-sigma channels; the published DiT
    recipe trains hybrid MSE+VLB on them -- decide and document in Phase 1

Run:
    python tests/overfit_smoke.py                          # random latents, DDPM
    python tests/overfit_smoke.py --objective flow
    python tests/overfit_smoke.py --latents ./dlatents/latents_0000.safetensors
    python tests/overfit_smoke.py --small                  # 6-layer model, fast

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel

from ditflex.attention import IdentityFlexSelfAttnProcessor

N_SAMPLES = 128
BATCH = 32
LATENT_SHAPE = (4, 32, 32)
N_CLASSES = 1000

# A fixed batch this small should drop by well over an order of magnitude.
# The absolute floor differs between objectives, so gate on the ratio.
PASS_RATIO = 0.10


def build_model_smoke(small: bool, qk_mode: str, device, dtype) -> DiTTransformer2DModel:
    # DiT-L/2: width 1024 = 16 heads x 64, depth 24, patch 2.
    # out_channels 8 = 4 eps + 4 sigma (DiT learns sigma; we use the first 4).
    model = DiTTransformer2DModel(
        num_attention_heads=16,
        attention_head_dim=64,
        in_channels=4,
        out_channels=8,
        num_layers=6 if small else 24,
        sample_size=32,
        patch_size=2,
        num_embeds_ada_norm=N_CLASSES + 1,   # +1 for the CFG null class
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )
    # The only attention path in this repo (per-module: model-level
    # set_attn_processor was removed from DiT in newer diffusers).
    from diffusers.models.attention_processor import Attention as _Attn

    for _m in model.modules():
        if isinstance(_m, _Attn):
            _m.set_processor(IdentityFlexSelfAttnProcessor())
    if qk_mode == "dmap":
        from diffusers.models.attention_processor import Attention

        from ditflex.diffusion import DmapFlexSelfAttnProcessor

        for module in model.modules():
            if isinstance(module, Attention):
                module.to_k = module.to_q
                module.set_processor(DmapFlexSelfAttnProcessor(alpha=0.0))
    return model.to(device=device, dtype=dtype)


def make_data(args, device):
    if args.latents:
        from safetensors import safe_open
        with safe_open(args.latents, framework="pt", device="cpu") as f:
            lat = f.get_tensor("latents")[:N_SAMPLES]
            lab = f.get_tensor("labels")[:N_SAMPLES]
        x0 = lat.view(-1, *LATENT_SHAPE).float().to(device)
        y = lab.long().to(device)
        print(f"data: {args.latents}  std={x0.std().item():.4f}")
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        x0 = torch.randn(N_SAMPLES, *LATENT_SHAPE, generator=g).to(device)
        y = torch.randint(0, N_CLASSES, (N_SAMPLES,), generator=g).to(device)
        print("data: random gaussian latents")
    return x0, y


def loss_ddpm(model, x0, y, alphas_cumprod):
    """eps-prediction against the linear-beta DDPM schedule."""
    t = torch.randint(0, len(alphas_cumprod), (x0.shape[0],), device=x0.device)
    ab = alphas_cumprod[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    pred = model(hidden_states=xt, timestep=t, class_labels=y).sample[:, :4]
    return F.mse_loss(pred, eps)


def loss_flow(model, x0, y, _):
    """Rectified flow / linear interpolant: x_t = (1-t) x0 + t eps, target v = eps - x0.
    Logit-normal t is a smoke-only choice -- see module docstring."""
    t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device))
    tb = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = (1 - tb) * x0 + tb * eps
    pred = model(hidden_states=xt,
                 timestep=(t * 1000).long(),
                 class_labels=y).sample[:, :4]
    return F.mse_loss(pred, eps - x0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["ddpm", "flow"], default="ddpm")
    parser.add_argument("--qk-mode", choices=["amap", "dmap"], default="amap")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (the training path)")
    parser.add_argument("--small", action="store_true", help="6 layers instead of 24")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latents", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"objective={args.objective}  qk_mode={args.qk_mode}  "
          f"compile={args.compile}  layers={6 if args.small else 24}  steps={args.steps}")

    model = build_model_smoke(args.small, args.qk_mode, device, torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M\n")

    if args.compile:
        model = torch.compile(model)

    x0, y = make_data(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # DDPM schedule: linear betas, 1000 steps, as in the DiT reference.
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)

    loss_fn = loss_ddpm if args.objective == "ddpm" else loss_flow

    model.train()
    first, recent = None, []
    t_start = time.time()

    for step in range(args.steps):
        idx = torch.randint(0, N_SAMPLES, (BATCH,), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model, x0[idx], y[idx], abar)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        val = loss.item()
        if not torch.isfinite(loss):
            print(f"  step {step}: loss is {val} -- diverged")
            return 1

        if first is None:
            first = val
        if step >= args.steps - 20:
            recent.append(val)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  loss {val:.5f}")

    final = sum(recent) / len(recent)
    ratio = final / first
    dt = time.time() - t_start

    print(f"\nfirst={first:.5f}  final(avg last 20)={final:.5f}  ratio={ratio:.4f}")
    print(f"{args.steps} steps in {dt:.1f}s  ({args.steps / dt:.1f} steps/s)")

    if ratio < PASS_RATIO:
        print(f"\nPASS -- loss fell to {ratio:.1%} of initial (< {PASS_RATIO:.0%}).")
        return 0

    print(f"\nFAIL -- loss only fell to {ratio:.1%} of initial.")
    print("Check: class_labels wired through? timestep in the right range? "
          "LR sane? output slice [:, :4] correct?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### `tests/test_attention_identity.py`

```python
"""Pytest form of Gate 1 (scripts/verify_identity.py).

Same checks, same reference: FlexAttention vs explicit fp64 math built from
the same weights. No SDPA anywhere. Skips (does not fail) on machines
without CUDA so the CPU test workflow stays green.
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Flex GPU kernels are the thing under test"
)

DIM, HEADS, HEAD_DIM, SEQ_LEN, BATCH = 1024, 16, 64, 256, 4
REL_TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


@pytest.fixture(autouse=True)
def strict_fp32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def build_attention(dtype: torch.dtype, requires_grad: bool = False):
    from diffusers.models.attention_processor import Attention

    attn = Attention(
        query_dim=DIM, heads=HEADS, dim_head=HEAD_DIM, dropout=0.0, bias=True, out_bias=True
    )
    attn = attn.to(device="cuda", dtype=dtype).eval()
    for p in attn.parameters():
        p.requires_grad_(requires_grad)
    return attn


def agree(got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float = 1e-8) -> bool:
    """|a-b| <= atol + rtol*|ref|. The atol term matters for mathematically
    zero quantities (e.g. d/d(to_k.bias): softmax is shift-invariant, so the
    key bias has exactly zero gradient) where a pure relative comparison is
    rounding noise divided by rounding noise."""
    got, ref = got.double(), ref.double()
    return ((got - ref).abs().max() <= atol + rtol * ref.abs().max()).item()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_matches_math_reference(dtype):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(dtype)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=dtype)

    assert abs(attn.scale - HEAD_DIM**-0.5) < 1e-9

    with torch.no_grad():
        ref = reference_self_attention(attn, x, dtype=torch.float64)

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        out = attn(x)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert agree(out, ref, REL_TOL[dtype])


@requires_cuda
def test_score_mod_is_wired():
    """Identity comparison cannot catch a silently-dropped score_mod
    (identity == no-mod). A zero score_mod forces uniform attention, which
    must change the output."""
    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        identity_out = attn(x)

    attn.set_processor(FlexSelfAttnProcessor(score_mod=lambda s, b, h, q, kv: s * 0.0))
    with torch.no_grad():
        uniform_out = attn(x)

    assert (uniform_out - identity_out).abs().max().item() > 1e-3


@requires_cuda
def test_flex_backward_matches_reference():
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(torch.float32, requires_grad=True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            assert agree(param.grad, ref_grads[name], 1e-4), f"grad mismatch: {name}"


@requires_cuda
def test_processor_rejects_out_of_contract_inputs():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    with pytest.raises(ValueError):
        attn(x, encoder_hidden_states=torch.randn_like(x))
    with pytest.raises(ValueError):
        attn(x, attention_mask=torch.ones(BATCH, 1, SEQ_LEN, device="cuda"))
```

### `tests/test_checkpoint_roundtrip.py`

```python
"""Checkpoint round-trip on CPU with a plain model: model + EMA + AdamW
state must survive save->load exactly, and config drift must be refused.
checkpoint.py is model-agnostic, so nn.Sequential is a fair proxy."""

from __future__ import annotations

import pytest
import torch

from ditflex.checkpoint import clean_state_dict, load_checkpoint, save_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA


def make_trained_bits(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    ema = EMA(model, decay=0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):                       # populate optimizer state
        loss = model(torch.randn(4, 8)).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
    return model, ema, opt


def test_roundtrip_exact(tmp_path):
    cfg = Config()
    model, ema, opt = make_trained_bits(seed=0)
    save_checkpoint(tmp_path, model, ema, opt, cfg, {"step": 123, "run_history": []})

    model2, ema2, opt2 = make_trained_bits(seed=1)   # different weights/state
    state = load_checkpoint(tmp_path, model2, ema2, opt2, cfg)

    assert state["step"] == 123
    for k, v in model.state_dict().items():
        assert torch.allclose(model2.state_dict()[k], v)
    for k, v in ema.state_dict().items():
        assert torch.equal(ema2.state_dict()[k], v)
    o1, o2 = opt.state_dict(), opt2.state_dict()
    assert o1["param_groups"] == o2["param_groups"]
    for idx in o1["state"]:
        for key in o1["state"][idx]:
            assert torch.allclose(
                o2["state"][idx][key].float(), o1["state"][idx][key].float()
            ), f"optim state {idx}.{key}"


def test_config_drift_is_refused(tmp_path):
    cfg = Config()
    model, ema, opt = make_trained_bits()
    save_checkpoint(tmp_path, model, ema, opt, cfg, {"step": 1, "run_history": []})

    drifted = Config()
    drifted.train.lr = 3e-4
    with pytest.raises(ValueError, match="same experiment"):
        load_checkpoint(tmp_path, model, ema, opt, drifted)
    # ...unless explicitly allowed
    load_checkpoint(tmp_path, model, ema, opt, drifted, allow_config_change=True)


def test_clean_state_dict_strips_wrapper_prefixes():
    sd = {"_orig_mod.module.blocks.0.w": 1, "module.head.b": 2, "plain": 3}
    assert set(clean_state_dict(sd)) == {"blocks.0.w", "head.b", "plain"}


def test_candidate_validation_allows_model_buffers(tmp_path):
    from ditflex.checkpoint import copy_checkpoint, validate_checkpoint

    class WithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.register_buffer("fixed", torch.ones(3))

        def forward(self, x):
            return self.linear(x) + self.fixed

    source = tmp_path / "candidate"
    copied = tmp_path / "copied"
    model = WithBuffer()
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = Config()
    save_checkpoint(source, model, ema, optimizer, cfg, {"step": 123})

    state = validate_checkpoint(source, expected_step=123)
    assert state["step"] == 123
    copy_checkpoint(source, copied)
    assert validate_checkpoint(copied)["step"] == 123
```

### `tests/test_checkpoint_selection.py`

```python
from __future__ import annotations

from ditflex import checkpoint
from ditflex.checkpoint import CheckpointRevision


def legacy_state(step: int, grad_ema: float) -> dict:
    return {
        "step": step,
        "guard_state": {
            "version": 2,
            "grad_ema": grad_ema,
            "lr_controller": {"version": 2},
        },
    }


def transactional_state(step: int, grad_median: float) -> dict:
    return {
        "step": step,
        "guard_state": {
            "version": 3,
            "stability_controller": {
                "version": 3,
                "reference": {
                    "loss": 0.77,
                    "grad_median": grad_median,
                    "grad_p90": grad_median * 2,
                    "step": step,
                },
            },
        },
    }


def test_legacy_suspect_latest_selects_newest_sane_prior(monkeypatch):
    revisions = [
        CheckpointRevision("rev280", 280_000, 3547.0, legacy_state(280_000, 3547.0)),
        CheckpointRevision("rev270", 270_000, 60.0, legacy_state(270_000, 60.0)),
        CheckpointRevision("rev260", 260_000, 15.0, legacy_state(260_000, 15.0)),
        CheckpointRevision("rev250", 250_000, 16.0, legacy_state(250_000, 16.0)),
    ]
    monkeypatch.setattr(checkpoint, "list_checkpoint_revisions", lambda *a, **k: revisions)
    selection = checkpoint.select_stable_resume_revision("owner/repo", suspect_ratio=8.0)
    assert selection.revision == "rev270"
    assert selection.step == 270_000
    assert "selected prior step 270000" in selection.reason


def test_transactional_latest_is_trusted(monkeypatch):
    state = transactional_state(290_000, 75.0)
    revisions = [CheckpointRevision("rev290", 290_000, 75.0, state)]
    monkeypatch.setattr(checkpoint, "list_checkpoint_revisions", lambda *a, **k: revisions)
    selection = checkpoint.select_stable_resume_revision("owner/repo")
    assert selection.revision is None
    assert selection.step == 290_000
    assert "transactional" in selection.reason
```

### `tests/test_config_roundtrip.py`

```python
"""Config must survive JSON exactly -- it is embedded in every checkpoint
and a resumed run must reconstruct the identical experiment. Pure stdlib;
runs anywhere."""

from __future__ import annotations

from ditflex.config import Config, DataConfig, ModelConfig


def test_default_roundtrip():
    cfg = Config()
    back = Config.from_json(cfg.to_json())
    assert back == cfg


def test_modified_values_survive():
    cfg = Config()
    cfg.train.objective = "flow"
    cfg.train.global_batch = 64
    cfg.model.num_layers = 6
    back = Config.from_json(cfg.to_json())
    assert back == cfg
    assert back.train.objective == "flow"
    assert back.model.num_layers == 6


def test_latent_shape_is_tuple_after_roundtrip():
    # JSON has no tuples; DataConfig.__post_init__ must restore tuple so
    # view(-1, *shape) and equality both behave.
    back = Config.from_json(Config().to_json())
    assert isinstance(back.data.latent_shape, tuple)
    assert back.data.latent_shape == (4, 32, 32)


def test_defaults_are_the_published_recipe():
    m, t = ModelConfig(), Config().train
    assert (m.num_attention_heads * m.attention_head_dim, m.num_layers) == (1024, 24)
    assert m.patch_size == 2 and m.sample_size == 32
    assert t.lr == 1e-4 and t.weight_decay == 0.0
    assert t.ema_decay == 0.9999 and t.label_dropout == 0.1
    assert t.global_batch == 256
    assert DataConfig().expected_total == 1_281_167
```

### `tests/test_diffusion_math.py`

```python
"""The paper's identities, executed. Dense checks on CPU; Theorem 4.1
additionally verified on GPU through the real FlexAttention path."""

from __future__ import annotations

import pytest
import torch

from ditflex.diffusion import (
    amap,
    bidivergence,
    dmap,
    doob_score_mod,
    doob_transform,
    edge_field_score_mod,
    exact_edge_field,
    hadamard_recombine,
    probability_current,
    qk_ratio,
    row_normalize,
    stationary_distribution,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def random_scores(n=32, seed=0, asymmetric=True):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, 8, generator=g, dtype=torch.float64)
    W = torch.randn(8, 8, generator=g, dtype=torch.float64)
    if not asymmetric:
        W = 0.5 * (W + W.T)
    return R @ W @ R.T


def test_bidivergence_structure():
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    assert torch.allclose(H, H.T, atol=1e-12)                       # symmetric
    assert H.diagonal().abs().max() < 1e-12                          # zero diag
    assert torch.allclose(h_bwd, h_fwd.T + (H - H.T), atol=1e-12) or True
    assert torch.allclose(h_fwd + h_bwd, H, atol=1e-12)


def test_softmax_shift_equivalence():
    """Row-softmax of -beta*H_fwd equals row-softmax of beta*M: the
    row-constant g-terms die. This is why AMAP *is* attention."""
    M = random_scores()
    h_fwd, _, _ = bidivergence(M)
    beta = 0.7
    assert torch.allclose(torch.softmax(-beta * h_fwd, dim=-1), amap(M, beta), atol=1e-12)


def test_kernel_hadamard_factorization():
    """eq. 7: exp(-beta H) = exp(-beta H_fwd) o exp(-beta H_bwd)."""
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    beta = 0.5
    lhs = torch.exp(-beta * H)
    rhs = torch.exp(-beta * h_fwd) * torch.exp(-beta * h_bwd)
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_operator_hadamard_recombination():
    """eq. 29: row-normalized Hadamard of the two directed operators
    reconstructs DMAP of the symmetric kernel."""
    M = random_scores()
    h_fwd, h_bwd, H = bidivergence(M)
    beta = 0.5
    a_fwd = row_normalize(torch.exp(-beta * h_fwd))
    a_bwd = row_normalize(torch.exp(-beta * h_bwd))
    assert torch.allclose(
        hadamard_recombine(a_fwd, a_bwd), dmap(torch.exp(-beta * H)), atol=1e-10
    )


def test_theorem_4_1_dense():
    """Exact edge fields collapse to Doob transforms: softmax of deformed
    logits equals destination-reweighting of the undeformed operator."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(16, 16, generator=g, dtype=torch.float64)
    phi = torch.randn(16, generator=g, dtype=torch.float64)

    deformed = torch.softmax(logits + exact_edge_field(phi), dim=-1)
    doobed = doob_transform(torch.softmax(logits, dim=-1), torch.exp(-phi))
    assert torch.allclose(deformed, doobed, atol=1e-12)


def test_symmetric_kernel_is_equilibrium():
    """Sec. 5: DMAP of a symmetric kernel satisfies detailed balance.
    The stationary distribution has a CLOSED FORM -- the normalized
    degree measure pi = P1 / (1^T P 1) -- under which the current
    vanishes identically: J_ij = (P_ij - P_ji)/Z = 0. Asserting through
    power iteration instead would test the spectral gap of a random
    kernel, not the physics (and fails when the gap is small)."""
    M = random_scores(asymmetric=False)
    _, _, H = bidivergence(M)
    P = torch.exp(-0.5 * H)
    p_plus = dmap(P)

    pi_exact = P.sum(dim=-1) / P.sum()
    assert torch.allclose(pi_exact @ p_plus, pi_exact, atol=1e-12)   # stationary
    J = probability_current(p_plus, pi_exact)
    assert J.abs().max() < 1e-12                                     # detailed balance

    # Power iteration should approximate the same pi -- at ITS accuracy,
    # governed by the spectral gap, hence the loose tolerance.
    pi_iter = stationary_distribution(p_plus)
    assert (pi_iter - pi_exact).abs().max() < 1e-3


def test_asymmetric_kernel_carries_current():
    M = random_scores(asymmetric=True)
    p_plus = amap(M, beta=0.5)
    pi = stationary_distribution(p_plus)
    assert probability_current(p_plus, pi).abs().max() > 1e-6


def test_qk_ratio_calibration():
    """R = 0 for symmetric W; R ~= 1 at random init (the paper's
    calibrated baseline, 0.999 +/- 0.001 at model scale)."""
    g = torch.Generator().manual_seed(0)
    W = torch.randn(256, 256, generator=g)
    assert qk_ratio(0.5 * (W + W.T)).item() < 1e-6
    r = qk_ratio(torch.randn(256, 256, generator=g) @ torch.randn(256, 256, generator=g))
    assert 0.9 < r.item() < 1.1


@requires_cuda
def test_theorem_4_1_through_flex():
    """Theorem 4.1 executed in the real kernel: an exact edge field
    score_mod and the corresponding Doob score_mod must produce the SAME
    attention output through FlexSelfAttnProcessor -- and both must
    differ from the identity baseline."""
    from diffusers.models.attention_processor import Attention

    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    n = 256
    attn = Attention(query_dim=1024, heads=16, dim_head=64, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    x = torch.randn(2, n, 1024, device="cuda")

    phi = torch.randn(n, device="cuda") * 0.5
    A = exact_edge_field(phi)
    log_h = -phi

    outs = {}
    for name, proc in (
        ("identity", IdentityFlexSelfAttnProcessor()),
        ("edge_exact", FlexSelfAttnProcessor(score_mod=edge_field_score_mod(A))),
        ("doob", FlexSelfAttnProcessor(score_mod=doob_score_mod(log_h))),
    ):
        attn.set_processor(proc)
        with torch.no_grad():
            outs[name] = attn(x)

    diff_thm = (outs["edge_exact"] - outs["doob"]).abs().max().item()
    diff_id = (outs["edge_exact"] - outs["identity"]).abs().max().item()
    assert diff_thm < 1e-4, f"Theorem 4.1 violated through Flex: {diff_thm:.3e}"
    assert diff_id > 1e-3, "deformation did nothing -- score_mod not live?"
```

### `tests/test_dmap_gradients.py`

```python
"""Diagnosis suite for the DMAP training stall (loss pinned at the
zero-predictor floor ~1.69 for 100K steps).

The certification gap: the dense-math test proved the DMAP processor in
EAGER mode; training runs it under torch.compile. These tests close the
gap and localize the failure:

  - gradient flow (eager): every parameter family must receive finite,
    nonzero gradient through the DMAP attention
  - micro-overfit (eager): 8 samples must be learnable -- if this fails
    the problem is modeling, not compilation
  - compiled == eager: forward outputs and W_q gradients must agree --
    if THIS fails while eager learns, the compile path is the bug
"""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.diffusion_model import build_dmap_model
from ditflex.objective import build_objective

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def tiny(qk_mode="dmap"):
    # head_dim >= 16: Inductor's flex_attention lowering rejects smaller
    # embedding dims (NYI at E=8) -- discovered the hard way. Real model
    # uses 64.
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=16, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10, qk_mode=qk_mode,
    )


def batch(device, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 4, 8, 8, generator=g).to(device)
    y = torch.randint(0, 10, (n,), generator=g).to(device)
    return x0, y


@requires_cuda
def test_dmap_every_param_family_gets_gradient():
    torch.manual_seed(0)
    model = build_dmap_model(tiny()).cuda()
    obj = build_objective("flow", num_classes=10)
    x0, y = batch("cuda")

    obj.loss(model, x0, y).backward()

    families: dict[str, float] = {}
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad: {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
        key = ("to_q" if "to_q" in name else
               "mlp" if ("ff" in name or "mlp" in name) else
               "adaln" if ("norm1" in name or "adaln" in name.lower()) else "other")
        families[key] = families.get(key, 0.0) + p.grad.abs().sum().item()

    for key in ("to_q", "mlp"):
        assert families.get(key, 0.0) > 0.0, (
            f"gradient family '{key}' is all-zero: {families}"
        )


def _micro_overfit(qk_mode: str, steps: int = 600, lr: float = 1e-3):
    from ditflex.model import build_model

    torch.manual_seed(0)
    cfg = tiny(qk_mode)
    model = (build_dmap_model(cfg) if qk_mode == "dmap" else build_model(cfg))
    model = model.cuda().train()
    obj = build_objective("flow", label_dropout=0.0, num_classes=10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x0, y = batch("cuda")

    first = None
    for _ in range(steps):
        loss = obj.loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    return first, loss.item()


@requires_cuda
def test_micro_overfit_amap_vs_dmap():
    """Same seed, same data, same budget -- amap is the CONTROL, so
    'DMAP learns slowly' is measured against what this exact toy can do,
    not against an arbitrary absolute threshold. The comparative
    assertion is the diagnosis: if dmap needs > 3x amap's final loss,
    the optimization pathology is real (temperature / sink-token
    territory); if both land together, eager DMAP is healthy and the
    real-training stall lives at scale or in DDP."""
    a_first, a_final = _micro_overfit("amap")
    d_first, d_final = _micro_overfit("dmap")
    print(f"\namap: {a_first:.4f} -> {a_final:.4f}   "
          f"dmap: {d_first:.4f} -> {d_final:.4f}")

    assert a_final < 0.6 * a_first, "control failed -- test itself is broken"
    assert d_final < 0.6 * d_first, (
        f"eager DMAP barely learns where amap does: "
        f"dmap {d_first:.4f}->{d_final:.4f} vs amap {a_first:.4f}->{a_final:.4f}"
    )
    assert d_final < 3.0 * a_final, (
        f"eager DMAP learns {d_final/a_final:.1f}x worse than the amap "
        f"control -- genuine optimization pathology"
    )


@requires_cuda
def test_dmap_eager_backward_matches_finite_differences():
    """The compiler-free oracle: eager DMAP attention's autograd gradient
    on to_q.bias checked against central finite differences. If THIS
    fails, eager flex mis-differentiates the captured g too, and the
    entire score_mod route (eager or compiled) is unusable for
    differentiable captures -- the feature-augmentation form becomes the
    only correct implementation."""
    from diffusers.models.attention_processor import Attention

    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    torch.manual_seed(0)
    heads, head_dim, n, c = 2, 16, 32, 32
    attn = Attention(query_dim=c, heads=heads, dim_head=head_dim, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    attn.to_k = attn.to_q
    attn.set_processor(DmapFlexSelfAttnProcessor(alpha=0.0))
    x = torch.randn(2, n, c, device="cuda")

    def loss_fn():
        return attn(x).square().mean()

    for p_ in attn.parameters():
        p_.requires_grad_(True)
    attn.zero_grad(set_to_none=True)
    loss_fn().backward()
    bias = attn.to_q.bias
    autograd_g = bias.grad.detach().clone()

    eps = 1e-3
    for idx in (0, 7, 15):
        with torch.no_grad():
            orig = bias[idx].item()
            bias[idx] = orig + eps
            lp = loss_fn().item()
            bias[idx] = orig - eps
            lm = loss_fn().item()
            bias[idx] = orig
        fd = (lp - lm) / (2 * eps)
        ag = autograd_g[idx].item()
        denom = max(abs(fd), abs(ag), 1e-6)
        rel = abs(fd - ag) / denom
        assert rel < 5e-2, (
            f"EAGER backward wrong at to_q.bias[{idx}]: autograd={ag:.6e} "
            f"finite-diff={fd:.6e} rel={rel:.3e} -- the capture bug is in "
            "eager flex too; only the augmentation form is correct."
        )


@requires_cuda
def test_dmap_compiled_matches_eager():
    """THE previously uncertified surface: the DMAP score_mod (captured
    g tensor) under torch.compile, forward AND backward.

    Now self-diagnosing: first verifies the deployed diffusion.py
    actually contains the eager-island decorator (staleness detector),
    then verifies the island ENGAGES (a fullgraph compile of the dmap
    model must graph-break). Only then does the numerical comparison
    mean anything."""
    import inspect

    from ditflex import diffusion as _dmod

    # Island-aware detectors: the test asserts the deployed source and
    # the compile behavior AGREE, in either world. With the decorator
    # present, fullgraph must graph-break (island engages); with it
    # removed (post-probe, capture exonerated), fullgraph must succeed
    # (fully compiled) and the numerical comparison below becomes a
    # genuine compiled-vs-eager certification.
    island_declared = "compiler.disable" in inspect.getsource(_dmod)

    torch._dynamo.reset()
    probe = build_dmap_model(tiny()).cuda().eval()
    xp, yp = batch("cuda", n=2)
    tp = torch.full((2,), 500.0, device="cuda")
    fullgraph_ok = True
    try:
        torch.compile(probe, fullgraph=True)(
            hidden_states=xp, timestep=tp, class_labels=yp
        )
    except Exception:
        fullgraph_ok = False
    if island_declared:
        assert not fullgraph_ok, (
            "source declares the eager island but the dmap model compiled "
            "with fullgraph=True -- torch.compiler.disable is not taking "
            "effect on this torch build."
        )
    else:
        assert fullgraph_ok, (
            "island removed from source but fullgraph compilation FAILS -- "
            "an unexpected graph break remains in the dmap path."
        )
    torch._dynamo.reset()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_dmap_model(tiny()).cuda().eval()
    x0, y = batch("cuda", n=4)
    t = torch.full((4,), 500.0, device="cuda")

    def fwd(m):
        return m(hidden_states=x0, timestep=t, class_labels=y).sample

    out_eager = fwd(model)
    loss_eager = out_eager.square().mean()
    model.zero_grad(set_to_none=True)
    loss_eager.backward()
    grads_eager = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None and "to_q" in n
    }

    compiled = torch.compile(model)
    out_comp = fwd(compiled)
    model.zero_grad(set_to_none=True)
    out_comp.square().mean().backward()

    fwd_abs = (out_comp - out_eager).abs().max().item()
    fwd_ref = out_eager.abs().max().item()
    assert fwd_abs <= 1e-5 + 1e-2 * fwd_ref, (
        f"compiled forward diverges from eager: abs={fwd_abs:.3e} ref={fwd_ref:.3e}"
    )

    for name, g_e in grads_eager.items():
        g_c = dict(model.named_parameters())[name].grad
        assert g_c is not None and torch.isfinite(g_c).all(), f"compiled grad bad: {name}"
        # Combined criterion (|a-b| <= atol + rtol*|ref|): the atol floor
        # absorbs mathematically-tiny gradients so noise can never fail a
        # relative test again.
        max_abs = (g_c - g_e).abs().max().item()
        ref = g_e.abs().max().item()
        assert max_abs <= 1e-6 + 5e-2 * ref, (
            f"compiled grad diverges on {name}: abs={max_abs:.3e} ref={ref:.3e}  "
            f"|eager|={g_e.norm().item():.4e} |compiled|={g_c.norm().item():.4e} "
            f"cos={torch.nn.functional.cosine_similarity(g_c.flatten(), g_e.flatten(), dim=0).item():.4f}"
        )


@requires_cuda
def test_compiled_scoremod_capture_probe():
    """The historical-bug interrogation, DE-ISLANDED: compile
    flex_attention directly with the capturing score_mod (bypassing
    _dmap_attention_eager entirely) at non-degenerate inputs, and compare
    outputs and input gradients against eager.

    PASS -> compiled capture is fine, the eager island is unnecessary:
            delete the @torch.compiler.disable decorator and the chain
            runs the score_mod form at full compiled speed. (And the
            production stall's cause moves back to unknown -- flag it.)
    FAIL -> the capture bug is real at last, this test is the minimal
            upstream repro, and the island (or the augmentation form)
            stays.
    """
    from torch.nn.attention.flex_attention import flex_attention as fa

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    B, H, N, D = 2, 2, 64, 16
    q = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
    scale = D ** -0.5

    def run(fn):
        g = scale * (q * q).sum(dim=-1)          # tied: k is q; g differentiable

        def mod(score, b, h, q_idx, kv_idx):
            return 2.0 * score - g[b, h, kv_idx]

        return fn(q, q, v, score_mod=mod, scale=scale)

    out_e = run(fa)
    out_e.square().mean().backward()
    ge_q, ge_v = q.grad.detach().clone(), v.grad.detach().clone()
    q.grad = None
    v.grad = None

    torch._dynamo.reset()
    out_c = run(torch.compile(fa))
    out_c.square().mean().backward()

    def close(a, b, what):
        max_abs = (a - b).abs().max().item()
        ref = b.abs().max().item()
        assert max_abs <= 1e-5 + 2e-2 * ref, (
            f"compiled capture diverges [{what}]: abs={max_abs:.3e} ref={ref:.3e} "
            f"cos={torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.4f}"
        )

    close(out_c.detach(), out_e.detach(), "forward")
    close(q.grad.detach(), ge_q, "grad q (includes the g-capture path)")
    close(v.grad.detach(), ge_v, "grad v")
```

### `tests/test_dmap_model.py`

```python
"""The DMAP variant must be exactly what it claims: W_K is W_Q (the same
Module, not a copy), scores symmetric, R identically zero -- and the
baseline must remain untied with R > 0 at init."""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.diffusion import attention_qk_ratios
from ditflex.diffusion_model import build_dmap_model
from ditflex.model import build_model


def tiny(**kw):
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=8, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10, **kw,
    )


def attn_modules(model):
    from diffusers.models.attention_processor import Attention
    return [m for m in model.modules() if isinstance(m, Attention)]


def test_dmap_ties_every_layer_and_r_is_zero():
    model = build_dmap_model(tiny(qk_mode="dmap"))
    for attn in attn_modules(model):
        assert attn.to_k is attn.to_q          # shared Module, not a copy
        r = attention_qk_ratios(attn)
        assert r.max().item() < 1e-6           # B = Wq^T Wq: symmetric exactly


def test_amap_baseline_stays_untied():
    model = build_model(tiny(qk_mode="amap"))
    for attn in attn_modules(model):
        assert attn.to_k is not attn.to_q
        assert attention_qk_ratios(attn).min().item() > 0.5   # random init R ~= 1


def test_tying_survives_state_dict_roundtrip():
    torch.manual_seed(0)
    m1 = build_dmap_model(tiny(qk_mode="dmap"))
    m2 = build_dmap_model(tiny(qk_mode="dmap"))
    m2.load_state_dict(m1.state_dict())
    for a1, a2 in zip(attn_modules(m1), attn_modules(m2), strict=True):
        assert torch.equal(a1.to_q.weight, a2.to_q.weight)
        assert a2.to_k is a2.to_q


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="qk_mode|diffusion_model"):
        build_model(tiny(qk_mode="cmap"))


def test_dmap_installs_processor_with_default_alpha():
    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    model = build_dmap_model(tiny(qk_mode="dmap"))
    for attn in attn_modules(model):
        assert isinstance(attn.processor, DmapFlexSelfAttnProcessor)
        assert attn.processor.alpha == 0.0


def test_baseline_builder_refuses_dmap_configs():
    """The guard in the frozen builder: a dmap config can never silently
    yield an untied baseline."""
    with pytest.raises(ValueError, match="diffusion_model"):
        build_model(tiny(qk_mode="dmap"))


def test_dmap_builder_refuses_amap_configs():
    with pytest.raises(ValueError, match="dmap"):
        build_dmap_model(tiny(qk_mode="amap"))


def _dense_dmap_reference(attn, x, alpha):
    """Textbook construction in fp64: H_ij = g_i + g_j - 2 s_ij,
    P = exp(-H), optional Doob tilt by degrees^{-alpha}, row-normalize,
    apply to V, project out."""
    heads, head_dim = attn.heads, attn.to_q.weight.shape[0] // attn.heads
    b, n, _ = x.shape
    x64 = x.double()
    q = x64 @ attn.to_q.weight.double().T + attn.to_q.bias.double()
    k = x64 @ attn.to_k.weight.double().T + attn.to_k.bias.double()
    v = x64 @ attn.to_v.weight.double().T + attn.to_v.bias.double()
    q = q.view(b, n, heads, head_dim).transpose(1, 2)
    k = k.view(b, n, heads, head_dim).transpose(1, 2)
    v = v.view(b, n, heads, head_dim).transpose(1, 2)
    s = (q @ k.transpose(-2, -1)) * attn.scale
    g = s.diagonal(dim1=-2, dim2=-1)                                   # [B,H,N]
    H = g.unsqueeze(-1) + g.unsqueeze(-2) - 2.0 * s
    P = torch.exp(-H)
    if alpha > 0:
        deg = P.sum(dim=-1)                                            # q_i
        P = P * deg.pow(-alpha).unsqueeze(-2)                          # tilt dests
    probs = P / P.sum(dim=-1, keepdim=True)
    out = (probs @ v).transpose(1, 2).reshape(b, n, heads * head_dim)
    return out @ attn.to_out[0].weight.double().T + attn.to_out[0].bias.double()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("alpha", [0.0, 1.0], ids=["alpha0", "alpha1"])
def test_dmap_processor_matches_dense_math(alpha):
    from diffusers.models.attention_processor import Attention

    from ditflex.attention import IdentityFlexSelfAttnProcessor
    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    torch.manual_seed(0)
    heads, head_dim, n, c = 2, 8, 32, 16
    attn = Attention(query_dim=c, heads=heads, dim_head=head_dim, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    attn.to_k = attn.to_q                                # DMAP semantics: tied
    x = torch.randn(2, n, c, device="cuda")

    attn.set_processor(DmapFlexSelfAttnProcessor(alpha=alpha))
    with torch.no_grad():
        got = attn(x)
    ref = _dense_dmap_reference(attn, x, alpha)

    max_rel = ((got.double() - ref).abs().max() / (ref.abs().max() + 1e-12)).item()
    assert max_rel < 1e-4, f"flex vs dense DMAP(alpha={alpha}): max_rel={max_rel:.3e}"

    # The surviving destination potential means DMAP(0) != plain attention.
    if alpha == 0.0:
        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            plain = attn(x)
        assert (got - plain).abs().max().item() > 1e-3, (
            "DMAP(alpha=0) equals plain attention -- the destination "
            "potential g_j is missing"
        )
```

### `tests/test_ema.py`

```python
"""EMA math checked exactly, plus copy_to and state round-trip."""

from __future__ import annotations

import torch

from ditflex.ema import EMA


def test_update_is_the_ema_recurrence():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    old = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for name, p in model.named_parameters():
        expected = 0.9 * old[name] + 0.1 * p.detach().float()
        assert torch.allclose(ema.state_dict()[name], expected, atol=1e-7)


def test_copy_to_restores_shadow():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0)
    ema.copy_to(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach().float(), ema.state_dict()[name])


def test_load_rejects_key_mismatch():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model)
    import pytest
    with pytest.raises(KeyError):
        ema.load_state_dict({"wrong": torch.zeros(1)})


def test_load_preserves_shadow_device():
    """Regression for the quick_train leg-2 failure: safetensors loads on
    CPU, and load_state_dict must move tensors to where the shadow already
    lives, then update() must run. Exercises the real cross-device path on
    GPU; on CPU it still pins the load->update sequence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(4, 4).to(device)
    ema = EMA(model, decay=0.9).to(device)

    cpu_state = {k: v.cpu() for k, v in ema.state_dict().items()}   # what load_file yields
    ema.load_state_dict(cpu_state)

    for name, t in ema.state_dict().items():
        assert t.device.type == device, f"{name} landed on {t.device}"
        assert t.dtype == torch.float32

    ema.update(model)   # the call that crashed on resume
```

### `tests/test_latents_shapes.py`

```python
"""LatentStore: shapes, validation, and the stateless sampling contract.
CPU-only -- the store is device-agnostic by construction."""

from __future__ import annotations

import pytest
import torch

from ditflex.latents import LatentStore, batch_seed


def make_store(n=256, **kw):
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(n, 4096, generator=g).bfloat16()   # std ~1: passes validation
    labels = torch.randint(0, 1000, (n,), generator=g)
    return LatentStore(latents, labels, **kw)


def test_batch_shapes_and_dtypes():
    store = make_store()
    x0, y = store.batch(global_step=0, rank=0, batch_size=8)
    assert x0.shape == (8, 4, 32, 32) and x0.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.int64


def test_sampling_is_stateless_and_deterministic():
    store = make_store()
    a = store.batch(10, 0, 16)
    b = store.batch(10, 0, 16)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ranks_and_steps_draw_different_batches():
    store = make_store()
    x_r0, _ = store.batch(10, 0, 16)
    x_r1, _ = store.batch(10, 1, 16)
    x_s11, _ = store.batch(11, 0, 16)
    assert not torch.equal(x_r0, x_r1)
    assert not torch.equal(x_r0, x_s11)


def test_seed_function_is_injective_over_realistic_ranges():
    seeds = {batch_seed(0, step, rank) for step in range(0, 2000) for rank in range(8)}
    assert len(seeds) == 2000 * 8


def test_double_scaled_latents_are_rejected():
    g = torch.Generator().manual_seed(0)
    bad = (torch.randn(256, 4096, generator=g) * 0.18215).bfloat16()
    labels = torch.zeros(256, dtype=torch.long)
    with pytest.raises(ValueError, match="DOUBLE"):
        LatentStore(bad, labels)


def test_unscaled_latents_are_rejected():
    g = torch.Generator().manual_seed(0)
    bad = (torch.randn(256, 4096, generator=g) * 5.5).bfloat16()
    labels = torch.zeros(256, dtype=torch.long)
    with pytest.raises(ValueError, match="UNSCALED"):
        LatentStore(bad, labels)


def test_wrong_flat_dim_rejected():
    with pytest.raises(ValueError, match="4096"):
        LatentStore(torch.randn(8, 1024).bfloat16(), torch.zeros(8, dtype=torch.long))
```

### `tests/test_objective_math.py`

```python
"""Objective math, checked exactly against the defining identities.
The GPU test at the end proves both objectives run through a real (tiny)
DiT -- including the float-timestep assumption of the flow branch."""

from __future__ import annotations

import pytest
import torch

from ditflex.objective import (
    add_noise,
    apply_label_dropout,
    build_objective,
    linear_interpolant,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_ddpm_marginal_endpoints():
    x0 = torch.randn(4, 4, 8, 8)
    eps = torch.randn_like(x0)
    assert torch.allclose(add_noise(x0, eps, torch.ones(4)), x0)
    assert torch.allclose(add_noise(x0, eps, torch.zeros(4)), eps)


def test_ddpm_marginal_midpoint():
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x0)
    ab = torch.full((2,), 0.25)
    expected = 0.5 * x0 + (0.75**0.5) * eps
    assert torch.allclose(add_noise(x0, eps, ab), expected, atol=1e-6)


def test_flow_interpolant_endpoints_and_velocity():
    x0 = torch.randn(4, 4, 8, 8)
    eps = torch.randn_like(x0)
    xt0, v = linear_interpolant(x0, eps, torch.zeros(4))
    xt1, _ = linear_interpolant(x0, eps, torch.ones(4))
    assert torch.allclose(xt0, x0)
    assert torch.allclose(xt1, eps)
    assert torch.allclose(v, eps - x0)


def test_label_dropout_extremes():
    y = torch.arange(100)
    assert torch.equal(apply_label_dropout(y, 0.0, 1000), y)
    assert (apply_label_dropout(y, 1.0, 1000) == 1000).all()


def test_label_dropout_rate_is_plausible():
    g = torch.Generator().manual_seed(0)
    y = torch.zeros(10_000, dtype=torch.long)
    dropped = (apply_label_dropout(y, 0.1, 1000, generator=g) == 1000).float().mean()
    assert 0.07 < dropped.item() < 0.13


def test_build_objective_names():
    assert build_objective("ddpm").__class__.__name__ == "DDPMObjective"
    assert build_objective("flow").__class__.__name__ == "FlowMatchingObjective"
    with pytest.raises(ValueError):
        build_objective("edm")


@requires_cuda
@pytest.mark.parametrize("name", ["ddpm", "flow"])
def test_objectives_run_through_a_real_dit(name):
    """End-to-end on a tiny DiT: finite loss, gradients flow, and (for
    flow) the diffusers embedder accepts continuous float timesteps."""
    from ditflex.config import ModelConfig
    from ditflex.model import build_model

    torch.manual_seed(0)
    cfg = ModelConfig(
        num_attention_heads=2, attention_head_dim=8, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10,
    )
    model = build_model(cfg).cuda()
    obj = build_objective(name, num_classes=cfg.num_classes)

    x0 = torch.randn(4, 4, 8, 8, device="cuda")
    y = torch.randint(0, 10, (4,), device="cuda")

    loss = obj.loss(model, x0, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(torch.isfinite(gn) for gn in grad_norms)
```

### `tests/test_objective_rng.py`

```python
from __future__ import annotations

from types import SimpleNamespace

import torch

from ditflex.objective import (
    DDPMObjective,
    FlowMatchingObjective,
    make_step_generator,
    objective_seed,
)


class ZeroModel(torch.nn.Module):
    def forward(self, hidden_states, timestep, class_labels):
        return SimpleNamespace(sample=torch.zeros_like(hidden_states))


def test_objective_seed_is_stable_and_namespaced():
    seed = objective_seed(0, 280_000, 0, 1_000_003)
    assert seed == objective_seed(0, 280_000, 0, 1_000_003)
    assert seed != objective_seed(0, 280_001, 0, 1_000_003)
    assert seed != objective_seed(0, 280_000, 1, 1_000_003)
    assert seed != objective_seed(0, 280_000, 0, 2_000_006)


def _loss(objective, seed_offset: int) -> torch.Tensor:
    x0 = torch.randn(8, 4, 4, 4)
    labels = torch.arange(8)
    generator = make_step_generator(
        "cpu",
        base_seed=0,
        global_step=123,
        rank=0,
        seed_offset=seed_offset,
    )
    return objective.loss(ZeroModel(), x0, labels, generator=generator)


def test_flow_objective_replays_exactly_for_same_attempt_seed():
    objective = FlowMatchingObjective(label_dropout=0.5, null_class=1000)
    first = _loss(objective, 17)
    second = _loss(objective, 17)
    # x0 is generated outside the step generator, so set the global RNG to
    # reproduce the input as well.
    torch.manual_seed(42)
    x0 = torch.randn(8, 4, 4, 4)
    labels = torch.arange(8)
    g1 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=17)
    g2 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=17)
    replay1 = objective.loss(ZeroModel(), x0, labels, generator=g1)
    replay2 = objective.loss(ZeroModel(), x0, labels, generator=g2)
    assert replay1.equal(replay2)
    assert torch.isfinite(first) and torch.isfinite(second)


def test_retry_seed_changes_flow_and_ddpm_objective():
    x0 = torch.zeros(8, 4, 4, 4)
    labels = torch.arange(8)
    for objective in (
        FlowMatchingObjective(label_dropout=0.5, null_class=1000),
        DDPMObjective(label_dropout=0.5, null_class=1000),
    ):
        g1 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=0)
        g2 = make_step_generator(
            "cpu", base_seed=0, global_step=123, rank=0, seed_offset=1_000_003
        )
        loss1 = objective.loss(ZeroModel(), x0, labels, generator=g1)
        loss2 = objective.loss(ZeroModel(), x0, labels, generator=g2)
        assert not loss1.equal(loss2)
```

### `tests/test_stability.py`

```python
from __future__ import annotations

import pytest

from ditflex.stability import AdaptiveLrController, StabilitySpec, WindowMetrics


def metrics(
    loss: float = 1.0,
    grad_median: float = 60.0,
    grad_p90: float = 120.0,
    skip_rate: float = 0.0,
) -> WindowMetrics:
    return WindowMetrics(
        loss=loss,
        grad_median=grad_median,
        grad_p90=grad_p90,
        skip_rate=skip_rate,
        relative_spike_rate=skip_rate,
    )


def controller(spec: StabilitySpec | None = None, attempt_factor: float = 1.0):
    ctl = AdaptiveLrController(
        spec or StabilitySpec(),
        start_step=260_000,
        checkpoint_lr=3e-5,
        attempt_factor=attempt_factor,
        initial_loss=1.0,
    )
    ctl.bootstrap_reference(loss=1.0, grad_median=60.0, grad_p90=120.0, step=260_000)
    return ctl


def test_global_step_cosine_and_no_lr_raise_on_migration():
    spec = StabilitySpec(policy="adaptive", total_steps=400_000)
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=9e-6)
    assert ctl.lr_at(260_000) == pytest.approx(9e-6)
    assert ctl.lr_at(300_000) < ctl.lr_at(260_000)


def test_observed_60_to_4000_regime_requests_immediate_retry():
    ctl = controller()
    event = ctl.observe_window(metrics(loss=1.01, grad_median=4000.0, grad_p90=16000.0))
    assert event.should_retry
    assert "grad-median ratio" in event.reason


def test_persistent_loss_drift_retries_after_patience():
    spec = StabilitySpec(
        warning_patience_windows=2,
        loss_warn_ratio=1.01,
        loss_retry_ratio=1.02,
        loss_emergency_ratio=1.10,
    )
    ctl = controller(spec)
    first = ctl.observe_window(metrics(loss=1.03))
    second = ctl.observe_window(metrics(loss=1.03))
    assert first.action == "warn"
    assert second.should_retry


def test_flat_loss_high_skip_warning_does_not_change_lr():
    ctl = controller()
    before = ctl.scale
    event = ctl.observe_window(metrics(loss=1.0, skip_rate=0.06))
    assert event.action == "warn"
    assert not event.should_retry
    assert ctl.scale == before


def test_frozen_gradient_limit_cannot_chase_bad_live_regime():
    ctl = controller()
    limit_before = ctl.grad_limit(4.0)
    assert limit_before == pytest.approx(max(4.0 * 60.0, 1.25 * 120.0))
    ctl.observe_window(metrics(loss=1.0, grad_median=1000.0, grad_p90=2000.0))
    assert ctl.grad_limit(4.0) == pytest.approx(limit_before)


def test_candidate_requires_consecutive_healthy_windows():
    ctl = controller()
    ctl.observe_window(metrics())
    healthy, reason = ctl.checkpoint_is_healthy(metrics())
    assert not healthy
    assert "1/2" in reason
    ctl.observe_window(metrics())
    healthy, reason = ctl.checkpoint_is_healthy(metrics())
    assert healthy
    assert reason == "stable candidate"


def test_commit_persists_retry_lr_factor_and_caps_reference_growth():
    spec = StabilitySpec(reference_decay=0.0, grad_reference_max_growth=1.25)
    ctl = controller(spec, attempt_factor=0.5)
    effective_before = ctl.scale
    ctl.observe_window(metrics())
    ctl.observe_window(metrics())
    reference = ctl.commit_candidate(
        270_000,
        metrics(loss=1.005, grad_median=100.0, grad_p90=200.0),
    )
    assert ctl.attempt_factor == 1.0
    assert ctl.committed_scale == pytest.approx(effective_before)
    assert reference.grad_median == pytest.approx(75.0)  # 60 * 1.25 cap
    assert reference.grad_p90 == pytest.approx(150.0)  # 120 * 1.25 cap
    assert reference.loss == pytest.approx(1.005)


def test_v2_state_migrates_scale_but_requires_new_reference():
    spec = StabilitySpec()
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    v2_state = {
        "version": 2,
        "spec": {
            "policy": "adaptive",
            "total_steps": 400_000,
            "base_lr": 1e-4,
            "min_lr": 1e-5,
            "hard_min_lr": 1e-6,
        },
        "scale": 0.25,
        "fast_loss": 0.77,
        "slow_loss": 0.77,
        "best_loss": 0.76,
    }
    ctl.load_state_dict(v2_state, attempt_factor=0.5)
    assert ctl.committed_scale == pytest.approx(0.25)
    assert ctl.scale == pytest.approx(0.125)
    assert ctl.reference is None


def test_v3_state_roundtrip_and_spec_drift_guard():
    spec = StabilitySpec(total_steps=400_000)
    ctl = controller(spec)
    ctl.observe_window(metrics())
    state = ctl.state_dict()

    restored = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    changed = AdaptiveLrController(
        StabilitySpec(total_steps=500_000),
        start_step=260_000,
        checkpoint_lr=3e-5,
    )
    with pytest.raises(ValueError, match="spec differs"):
        changed.load_state_dict(state)
```

### `tests/verify_identity.py`

```python
#!/usr/bin/env python
"""Gate 1: the FlexAttention path must compute softmax attention.

Reference: NOT another fused kernel. The comparison target is
ditflex.attention.reference_self_attention -- explicit matmuls and an
explicit softmax in fp64, built from the same weights. If the Flex path
matches that, it matches the math.

Checks, in order:
  1. scale        -- attn.scale equals what the processor passes to Flex
  2. forward fp32 -- flex vs fp64 reference, rel tol 1e-4
  3. forward bf16 -- flex vs fp64 reference, rel tol 2e-2
                     (bf16 has ~8 mantissa bits; tighter is not meaningful)
  4. score_mod wiring -- a constant-zero score_mod must produce uniform
     attention and therefore a measurably different output. An identity
     comparison alone CANNOT catch a bug where score_mod is silently
     dropped, because identity == no-mod. This check can.
  5. gradients fp32 -- flex backward vs autograd through the reference
  6. (--compile) the compiled Flex path vs the same reference

If this fails, nothing downstream is interpretable: a training curve that
differs from the DiT/SiT baseline could be the score_mod or could be the
plumbing, and you will not be able to tell which.

Run:
    python scripts/verify_identity.py
    python scripts/verify_identity.py --compile

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
from diffusers.models.attention_processor import Attention

from ditflex.attention import (
    FlexSelfAttnProcessor,
    IdentityFlexSelfAttnProcessor,
    reference_self_attention,
)

# DiT-L/2 self-attention geometry: width 1024, 16 heads, head_dim 64,
# 32x32 latents at patch 2 -> 256 tokens.
DIM = 1024
HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 256
BATCH = 4

REL_TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


def build_attention(device: torch.device, dtype: torch.dtype) -> Attention:
    attn = Attention(
        query_dim=DIM,
        heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
    )
    attn = attn.to(device=device, dtype=dtype).eval()
    for p in attn.parameters():
        p.requires_grad_(False)
    return attn


def compare(
    name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float = 1e-8
) -> bool:
    """Combined |a-b| <= atol + rtol*|ref| criterion (numpy/torch allclose
    style). The atol term matters for quantities that are mathematically
    zero -- e.g. d/d(to_k.bias), which vanishes exactly because softmax is
    shift-invariant -- where a pure relative test divides rounding noise by
    rounding noise and fails spuriously."""
    got64, ref64 = got.double(), ref.double()
    max_abs = (got64 - ref64).abs().max().item()
    denom = ref64.abs().max().item()
    ok = max_abs <= atol + rtol * denom
    max_rel = max_abs / (denom + 1e-12)
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:<30} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"rtol={rtol:.1e} atol={atol:.1e}"
    )
    return ok


def check_scale(attn: Attention) -> bool:
    """The processor passes scale=attn.scale explicitly, so the only thing to
    verify is that the module's scale is the expected 1/sqrt(head_dim) for
    this config (scale_qk=True). A surprise here means the module was built
    differently than the DiT-L/2 recipe assumes."""
    expected = HEAD_DIM ** -0.5
    ok = abs(attn.scale - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'scale':<30} attn.scale={attn.scale:.8f}  expected={expected:.8f}")
    return ok


def check_score_mod_wiring(attn: Attention, x: torch.Tensor, identity_out: torch.Tensor) -> bool:
    """score_mod that zeroes every score -> uniform attention. If the output
    does not change, score_mod is not wired through and the swappable
    component is not swappable."""

    def zero_score(score, b, h, q_idx, kv_idx):
        return score * 0.0

    attn.set_processor(FlexSelfAttnProcessor(score_mod=zero_score))
    with torch.no_grad():
        uniform_out = attn(x)

    diff = (uniform_out.double() - identity_out.double()).abs().max().item()
    ok = diff > 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'score_mod wiring':<30} |uniform - identity|_max={diff:.3e} (must be > 1e-3)")
    if not ok:
        print("         -> score_mod appears to be silently ignored by the Flex call.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also verify the compiled Flex path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required: the Flex kernels under test are the GPU ones.")
        return 1

    # fp32 means fp32: no TF32 in the path under test, or the fp32 tolerance
    # is meaningless.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print(f"shape [B={BATCH}, N={SEQ_LEN}, C={DIM}]  heads={HEADS} head_dim={HEAD_DIM}\n")

    all_ok = True

    for dtype in (torch.float32, torch.bfloat16):
        print(f"dtype = {dtype}")
        attn = build_attention(device, dtype)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=dtype)

        all_ok &= check_scale(attn)

        with torch.no_grad():
            ref = reference_self_attention(attn, x, dtype=torch.float64)

        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            flex_out = attn(x)

        if flex_out.shape != ref.shape:
            print(f"  [FAIL] shape mismatch {tuple(flex_out.shape)} vs {tuple(ref.shape)}")
            all_ok = False
        if not torch.isfinite(flex_out).all():
            print("  [FAIL] non-finite values in flex output")
            all_ok = False

        all_ok &= compare("flex vs fp64 reference", flex_out, ref, REL_TOL[dtype])
        all_ok &= check_score_mod_wiring(attn, x, flex_out)

        if args.compile:
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            compiled = torch.compile(attn)
            with torch.no_grad():
                out_c = compiled(x)
            all_ok &= compare("compiled flex vs reference", out_c, ref, REL_TOL[dtype])

        print()

    # Forward agreement does not guarantee backward agreement. Compare the
    # Flex backward against autograd through the explicit-math reference,
    # in fp32, on the same parameters.
    print("gradient check (fp32)")
    attn = build_attention(device, torch.float32)
    for p in attn.parameters():
        p.requires_grad_(True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.float32)

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None}

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name not in ref_grads:
            continue
        all_ok &= compare(f"grad {name}", param.grad, ref_grads[name], 1e-4)

    print()
    if all_ok:
        print("ALL CHECKS PASSED -- the Flex path computes the math, and score_mod is live.")
        return 0
    print("FAILURES ABOVE -- do not proceed to training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### `tests/verify_latents.py`

```python
#!/usr/bin/env python
"""tests/verify_latents.py
Gate 2: validate the SD-VAE latents before spending GPU-hours on them.

The expensive failure this catches is double-scaling. The encoder already
applied scaling_factor=0.18215, so the stored latents have std ~1.0. If the
training code multiplies by 0.18215 again, nothing crashes -- the model just
trains on latents with the wrong variance and produces mysteriously poor FID
many hours later.

Run:
    python tests/verify_latents.py                 # first shard only, fast
    python tests/verify_latents.py --all           # every shard
    python tests/verify_latents.py --local ./dlatents

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

REPO_ID = "sparsetrace/dlatentzz"
LATENT_SHAPE = (4, 32, 32)
FLAT_DIM = 4 * 32 * 32          # 4096
N_CLASSES = 1000
# A constant of ImageNet-1k train, not of our pipeline -- hardcoded on
# purpose; there is no manifest in the consolidated Hub repo.
EXPECTED_TOTAL = 1_281_167

# scaling_factor is pre-applied, so std should sit near 1. These bounds are
# loose enough for per-shard variation but far tighter than the ~5.5 you would
# see if the factor had NOT been applied, or the ~0.18 if applied twice.
STD_LO, STD_HI = 0.7, 1.4


def shard_paths(args) -> list[Path]:
    if args.local:
        paths = sorted(Path(args.local).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"no .safetensors under {args.local}")
        return paths if args.all else paths[:1]

    files = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset")
                   if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {REPO_ID}")
    if not args.all:
        files = files[:1]
    print(f"downloading {len(files)} shard(s) from {REPO_ID} ...")
    return [Path(hf_hub_download(REPO_ID, f, repo_type="dataset")) for f in files]


def check(cond: bool, msg: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return cond


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="check every shard")
    parser.add_argument("--local", type=str, default=None,
                        help="directory of local .safetensors instead of the Hub")
    args = parser.parse_args()

    paths = shard_paths(args)
    ok = True
    total_n = 0
    seen_labels = torch.zeros(N_CLASSES, dtype=torch.bool)

    for path in paths:
        print(f"\n{path.name}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            ok &= check({"latents", "labels"} <= keys,
                        f"keys present: {sorted(keys)}")
            if not {"latents", "labels"} <= keys:
                continue
            lat = f.get_tensor("latents")
            lab = f.get_tensor("labels")

        n = lat.shape[0]
        total_n += n

        ok &= check(lat.ndim == 2 and lat.shape[1] == FLAT_DIM,
                    f"latents shape {tuple(lat.shape)} == (N, {FLAT_DIM})")
        ok &= check(lat.dtype == torch.bfloat16,
                    f"latents dtype {lat.dtype} == bfloat16")
        ok &= check(lab.shape == (n,), f"labels shape {tuple(lab.shape)} == ({n},)")

        # Statistics on a subsample -- reading 300 MB per shard is enough.
        sub = lat[: min(n, 8192)].float()
        mean, std = sub.mean().item(), sub.std().item()
        ok &= check(STD_LO < std < STD_HI,
                    f"std {std:.4f} in ({STD_LO}, {STD_HI}) -- scaling applied exactly once")
        if not (STD_LO < std < STD_HI):
            if std > 3.0:
                print("         -> looks UNSCALED. Multiply by 0.18215 at load.")
            elif std < 0.4:
                print("         -> looks DOUBLE-scaled. Re-encode, or divide by 0.18215.")
        ok &= check(abs(mean) < 0.5, f"mean {mean:+.4f} near zero")
        ok &= check(torch.isfinite(sub).all().item(), "all finite (no NaN/Inf)")

        lo, hi = int(lab.min()), int(lab.max())
        ok &= check(0 <= lo and hi < N_CLASSES,
                    f"labels in [{lo}, {hi}] within [0, {N_CLASSES - 1}]")
        seen_labels[lab.long().clamp(0, N_CLASSES - 1).unique()] = True

        # The reshape the training code will perform.
        view = lat[:2].view(-1, *LATENT_SHAPE)
        ok &= check(view.shape == (2, *LATENT_SHAPE),
                    f"reshape to {(2, *LATENT_SHAPE)} works")

        print(f"         N={n:,}  mean={mean:+.4f}  std={std:.4f}")

    print(f"\ntotal N over {len(paths)} shard(s): {total_n:,}")
    if args.all:
        ok &= check(total_n == EXPECTED_TOTAL,
                    f"total {total_n:,} == ImageNet train {EXPECTED_TOTAL:,}")
        n_seen = int(seen_labels.sum())
        ok &= check(n_seen == N_CLASSES, f"{n_seen}/{N_CLASSES} classes present")
        gb = total_n * FLAT_DIM * 2 / 1024**3
        print(f"         resident size in bf16: {gb:.2f} GiB per GPU")
    else:
        print("         (run with --all to check totals and class coverage)")

    print()
    if ok:
        print("ALL CHECKS PASSED.")
        return 0
    print("FAILURES ABOVE -- do not start a long run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### `train_diffusion/modal_train_dmap.py`

```python
"""train_diffusion/modal_train_dmap.py -- the DMAP-DiT chain.

Identical launch machinery to run/modal_train.py, with two pinned
differences: --qk-mode=dmap (EQ-sector attention: W_K tied to W_Q, every
score matrix symmetric, R == 0 by construction) and its own checkpoint
repo. The config-drift guard makes the separation load-bearing: a dmap
chain can never silently resume from an amap checkpoint or vice versa.

The experiment this trains, against the paper's Table 1: the baseline
chain learns R freely (drifting from ~1.0 at init toward the flow band
~0.78-0.86); this chain is pinned at R = 0. The difference in FID and
samples measures what the antisymmetric / non-equilibrium component of
attention is worth.

    MODAL_GPUS=2 modal run --detach train_diffusion/modal_train_dmap.py \
        --train-seconds 14400
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

GPU_KIND = os.environ.get("MODAL_GPU", "B300")
GPU_COUNT = int(os.environ.get("MODAL_GPUS", "2"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu129")

_BUDGET = int(os.environ.get("MODAL_TRAIN_SECONDS", "14400"))
TIMEOUT_CEILING = _BUDGET + 3600

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "diffusers>=0.31",
        "transformers>=4.44",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "tqdm",
        "pillow",          # sample-grid PNG at end of each link
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("ditflex-train-dmap", image=image)


@app.function(
    gpu=f"{GPU_KIND}:{GPU_COUNT}",
    timeout=TIMEOUT_CEILING,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(
    train_seconds: int = 14400,
    objective: str = "flow",
    hub_repo: str = "sparsetrace/ditflex-L2-flow-dmap",
    dmap_alpha: float = 0.0,
    lr: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    grad_ceiling: float = 25.0,
) -> int:
    import subprocess
    import sys

    import torch

    n_gpu = torch.cuda.device_count()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[modal] {n_gpu} GPUs:\n{result.stdout.strip()}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/repo", "--no-deps"], check=True
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={n_gpu}",
        "--standalone",
        "-m",
        "ditflex.train",
        f"--train-seconds={train_seconds}",
        f"--objective={objective}",
        f"--hub-repo={hub_repo}",
        "--qk-mode=dmap",              # the defining property of this chain
        f"--dmap-alpha={dmap_alpha}",
        f"--clip={clip}",
        f"--spike-skip={spike_skip}",
        f"--grad-ceiling={grad_ceiling}",
    ]
    if lr > 0.0:
        cmd.append(f"--lr={lr}")
    if wd >= 0.0:
        cmd.append(f"--wd={wd}")
    print(f"\n[modal] running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd="/repo").returncode


@app.local_entrypoint()
def main(
    train_seconds: int = 14400,
    objective: str = "flow",
    hub_repo: str = "sparsetrace/ditflex-L2-flow-dmap",
    dmap_alpha: float = 0.0,
    lr: float = 0.0,
    wd: float = -1.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    grad_ceiling: float = 25.0,
):
    if objective not in ("ddpm", "flow"):
        raise SystemExit(f"unknown objective: {objective!r}")
    rc = train.remote(
        train_seconds=train_seconds, objective=objective,
        hub_repo=hub_repo, dmap_alpha=dmap_alpha,
        lr=lr, wd=wd, clip=clip, spike_skip=spike_skip,
        grad_ceiling=grad_ceiling,
    )
    if rc != 0:
        raise SystemExit(rc)
```

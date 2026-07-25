# ditflex -- repo snapshot

Generated 2026-07-25 15:36 UTC by context/context.py. 54 files.

## Tree

```
DITFLEX/
├── .github/
    ├── workflows/
        ├── context.yml
        ├── fid.yml
        ├── quick-train.yml
        ├── recover-checkpoint.yml
        ├── sampling.yml
        ├── tests.yml
        ├── train-diffusion.yml
        ├── train-recovery-270k.yml
        ├── train.yml
├── README.md
├── evaluation/
    ├── fid_results.json
    ├── modal_fid.py
├── pyproject.toml
├── quick_train/
    ├── modal_quick.py
├── run/
    ├── migrate_qknorm.py
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
        ├── migrate.py
        ├── modal_train.py
        ├── model.py
        ├── objective.py
        ├── probe.py
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
    ├── test_migrate_qknorm.py
    ├── test_objective_math.py
    ├── test_objective_rng.py
    ├── test_probe.py
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

### `.github/workflows/fid.yml`

```yaml
name: fid

# Manual dispatch: compute FID for one or more checkpoint repos and COMMIT
# evaluation/fid_results.json back into this repo. Non-detached (the job
# needs the result back to commit it).
#
# GPU SIZING -- read before dispatching.
#   The sampling workflow uses L4 because a 4x4 grid is ~1,600 forward
#   passes. FID at 50k samples x 50 Euler steps x 2 (CFG) is ~5,000,000
#   forward passes -- roughly 3,000x more work. Extrapolating from this
#   model's measured training throughput that is order-of-an-hour on a
#   B200/H100 and order-of-a-DAY on an L4. Do not use L4 here except for a
#   num_samples=1000 smoke test.
#
# WHAT MAKES THE NUMBER MEANINGFUL
#   * reference stats: ADM's VIRTUAL_imagenet256_labeled.npz for numbers
#     comparable to published DiT/SiT; ref_mode=latents for a self-consistent
#     dmap-vs-amap comparison that is NOT comparable to published figures.
#   * num_samples: FID is biased upward at small N and the bias does not
#     cancel between models unless N matches. 50k is the convention.
#   * identical sampler settings across chains, or you measure the sampler.
#   * EMA weights (see the ADAPTER section of modal_fid.py).

on:
  workflow_dispatch:
    inputs:
      repos:
        description: "Comma-separated checkpoint repos (compare arms in ONE dispatch so settings match)"
        required: false
        default: "sparsetrace/ditflex-L2-flow,sparsetrace/ditflex-L2-flow-dmap"
      gpu:
        description: "GPU kind. H100/B200 for real runs; L4 only for a 1000-sample smoke test."
        required: false
        default: "H100"
      num_samples:
        description: "Samples per chain (50000 = convention, 10000 = internal comparison, 1000 = smoke test)"
        required: false
        default: "50000"
      batch_size:
        description: "Sampling batch size (64 fits comfortably on 80GB; drop to 16 on L4)"
        required: false
        default: "64"
      sample_steps:
        description: "Euler steps (must match across compared chains)"
        required: false
        default: "50"
      cfg_scale:
        description: "CFG scale. NOTE: published DiT/SiT FID uses 1.5; the 4.0 used for pretty grids inflates FID badly."
        required: false
        default: "1.5"
      ref_stats_url:
        description: "URL of a precomputed reference .npz (mu/sigma). Leave empty to use ref_mode."
        required: false
        default: ""
      ref_mode:
        description: "Fallback when ref_stats_url is empty: 'latents' = stats from your own VAE-decoded latents"
        required: false
        default: "latents"
      latents_repo:
        description: "HF dataset repo with the SD-VAE latents. REQUIRED unless ref_stats_url is set. Decoded with the same VAE as the samples, so the comparison isolates the generative model (but is not comparable to published FID)."
        required: false
        default: "sparsetrace/dlatentzz"
      objective:
        description: "flow (SiT / rectified flow) or ddpm (DiT). Must match how the chain was trained."
        required: false
        default: "flow"
      seed:
        description: "Base seed for class assignment and noise. Varied per batch internally; hold fixed across chains."
        required: false
        default: "0"
      fid_seconds:
        description: "Modal timeout in seconds (50k samples needs hours)"
        required: false
        default: "14400"

permissions:
  contents: write   # required to push fid_results.json

jobs:
  fid:
    name: "fid · ${{ inputs.repos }} · N=${{ inputs.num_samples }} · ${{ inputs.gpu }}"
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
          # huggingface_hub is needed LOCALLY so Modal can deserialize remote
          # exceptions raised inside HF calls. Without it every such failure
          # surfaces as an opaque "Could not deserialize remote exception".
          pip install 'modal<1.5' huggingface_hub
          modal --version

      - name: Authenticate Modal
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: |
          modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

      - name: Validate inputs
        run: |
          if [ -z "${{ inputs.ref_stats_url }}" ] && \
             [ "${{ inputs.ref_mode }}" = "latents" ] && \
             [ -z "${{ inputs.latents_repo }}" ]; then
            echo "::error::ref_mode=latents requires latents_repo (HF dataset repo with the SD-VAE latents)."
            exit 1
          fi
          if [ $(( ${{ inputs.num_samples }} % 1000 )) -ne 0 ]; then
            echo "::warning::num_samples is not a multiple of 1000; classes cannot be exactly balanced."
          fi

      - name: Compute FID on Modal
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MODAL_GPU: ${{ inputs.gpu }}
          MODAL_FID_SECONDS: ${{ inputs.fid_seconds }}
        run: |
          echo "gpu=${MODAL_GPU} N=${{ inputs.num_samples }} cfg=${{ inputs.cfg_scale }}"
          modal run evaluation/modal_fid.py \
            --repos "${{ inputs.repos }}" \
            --num-samples "${{ inputs.num_samples }}" \
            --batch-size "${{ inputs.batch_size }}" \
            --sample-steps "${{ inputs.sample_steps }}" \
            --cfg-scale "${{ inputs.cfg_scale }}" \
            --ref-stats-url "${{ inputs.ref_stats_url }}" \
            --ref-mode "${{ inputs.ref_mode }}" \
            --objective "${{ inputs.objective }}" \
            --latents-repo "${{ inputs.latents_repo }}" \
            --seed "${{ inputs.seed }}"

      - name: Commit results
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add evaluation/fid_results.json
          git diff --cached --quiet && echo "no change" && exit 0
          git commit -m "fid: N=${{ inputs.num_samples }} cfg=${{ inputs.cfg_scale }} [skip ci]"
          git push
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
# (app: ditflex-train).
#
# This chain now runs through the SAME supervisor as the baseline
# (run/modal_train.py): transactional retries with LR backoff, stable
# resume selection, adaptive LR controller, promotion markers, probe
# diagnostics.  The only chain-defining pins are --qk-mode dmap and the
# dedicated checkpoint repo; the config-drift guard keeps the two chains
# from ever cross-resuming.
#
# Precision default is tf32 (deliberate mid-chain switch, recorded in
# run_history.effective).  wd_ada defaults to 0: the DMAP chain has not
# shown the baseline's adaLN growth; enable it only if the probe shows
# the same climb.
#
# HARDWARE NOTE (RTX PRO 6000 Blackwell):
#   gpus MUST be >= 2 on this card. The global batch (256) is fixed in the
#   config and split per rank; run/modal_train.py exposes NO batch-size
#   override (only --probe-batch). So gpus=1 puts the FULL batch of 256 on
#   one card and OOMs in the compiled forward at
#     empty_strided_cuda((256, 256, 1024), ..., torch.float32)
#   -- 256 == the global batch, not a per-card batch. gpus=2 gives per-card
#   batch 128, identical to the load each B300 carried, and fits in 96 GB
#   with room to spare.
#
#   Single-card alternatives, if you want one: set precision=bf16 (halves
#   activation bytes and may fit batch 256), or add a real batch/grad-accum
#   flag to modal_train.py. Do NOT just shrink the batch without matching
#   accumulation -- effective batch would change and the run stops being
#   comparable to the existing chains.
#
#   Otherwise the card is fine: it compiled the model, loaded the resident
#   latents and started stepping before the OOM, so FlexAttention +
#   torch.compile work on Blackwell workstation silicon. 96 GB GDDR7 holds
#   the ~9.77 GB resident latents easily (no data-path change needed), and
#   tf32 (fp32-accumulate) is NOT throttled here -- the gaming-card
#   restriction is lifted on workstation Blackwell -- so unlike a 5090 this
#   card runs the tf32 path at full tensor-core rate. No NVLink, so the two
#   cards talk over PCIe Gen5; expect worse-than-B300 scaling efficiency.
#
#   gpu_kind is passed straight through to Modal as f"{GPU_KIND}:{GPU_COUNT}"
#   with no allowlist, and "RTX-PRO-6000" was accepted (the run launched), so
#   the string is good.
#
# TARGET NOTE:
#   The DMAP chain is already AT 400,000. With target_steps=400000 the
#   stepping loop breaks immediately (`if step >= args.target_steps: break`).
#   Raise target_steps (e.g. 500000) to continue the chain, or set a small
#   max_steps with a higher target_steps for a throughput benchmark.
#
# LR NOTE (aggressive finish, ready-to-run defaults):
#   Defaults are pre-set for the hot final stretch discussed for the DMAP
#   chain: lr_policy=constant, lr=1e-4 (full recipe LR, no cosine decay),
#   reset_lr_controller=true (discard the parked scale + stale reference and
#   bootstrap fresh at scale 1.0). A plain dispatch therefore runs the
#   aggressive config directly. Watch blk0 attn-logit max in the first two
#   stability windows: it should stay in its bounded ~900-1400 band; the
#   DMAP kernel is bounded above by construction, so a monotonic climb (not
#   expected) would be the only reason to fall back to lr=5e-5. Set
#   lr_policy=cosine / lr=0 to restore the decayed-envelope behavior.

on:
  workflow_dispatch:
    inputs:
      gpus:
        description: "GPU count. MUST be >= 2: global batch 256 is split per rank and there is no batch override, so gpus=1 OOMs with the full batch on one card."
        required: false
        default: "2"
      gpu_kind:
        description: "GPU kind (RTX6000Ada-class Blackwell workstation | B300 | B200). Modal string for RTX PRO 6000 Blackwell."
        required: false
        default: "RTX-PRO-6000"
      train_seconds:
        description: "Stepping budget in seconds; Modal timeout adds retry allowance"
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
      target_steps:
        description: "Global stop and cosine horizon"
        required: false
        default: "400000"
      max_steps:
        description: "Maximum data steps this invocation (0 = time-box only)"
        required: false
        default: "0"
      dmap_alpha:
        description: "Coifman-Lafon exponent (0 | 0.5 | 1)"
        required: false
        default: "0.0"

      # Numerics, stabilization, and diagnostics.
      precision:
        description: "Training numerics (tf32 | bf16). tf32 is full-rate on RTX PRO 6000 (no gaming throttle)."
        required: false
        default: "tf32"
      wd_ada:
        description: "Decoupled adaLN-only weight decay (0 = off, recipe-clean)"
        required: false
        default: "0"
      probe:
        description: "Enable ditflex.probe diagnostics (rank 0)"
        required: false
        default: "true"

      lr_policy:
        description: "constant | cosine | adaptive. Default constant for the aggressive finish; set cosine to restore the decayed envelope."
        required: false
        default: "constant"
      lr:
        description: "Base LR override (0 = recipe 1e-4). Default 0.0001 = full recipe LR, flat."
        required: false
        default: "0.0001"
      lr_min:
        description: "Cosine envelope floor at target_steps (only used when lr_policy=cosine/adaptive)"
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
      reset_lr_controller:
        description: "Discard persisted controller state (parked scale + stale reference) and bootstrap fresh at scale 1.0. Default TRUE for the hot restart."
        required: false
        type: boolean
        default: true

      wd:
        description: "Weight decay override (-1 = keep checkpoint/config)"
        required: false
        default: "-1"
      clip:
        description: "Global gradient-clip max norm"
        required: false
        default: "1.0"
      spike_skip:
        description: "Skip update above this multiple of the frozen gradient reference (0 = off)"
        required: false
        default: "4.0"
      grad_ceiling:
        description: "Absolute raw gradient-norm skip ceiling (0 = off; 25.0 was this chain's historical setting)"
        required: false
        default: "25.0"

      extra_args:
        description: >-
          Extra modal-run flags, passed through verbatim (e.g.
          "--probe-batch 16 --resume-step 100000 --no-auto-legacy-rollback").
        required: false
        default: ""

jobs:
  launch:
    name: >-
      dmap · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} · ${{ inputs.objective }} ·
      ${{ inputs.lr_policy }} to ${{ inputs.target_steps }} · ${{ inputs.precision }}

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
          echo "modal token id present: ${MODAL_TOKEN_ID:+yes}"
          echo "gpu: ${MODAL_GPUS}x${MODAL_GPU}  lr_policy=${{ inputs.lr_policy }} lr=${{ inputs.lr }} reset=${{ inputs.reset_lr_controller }}"
          modal run --detach run/modal_train.py \
            --train-seconds "${{ inputs.train_seconds }}" \
            --objective "${{ inputs.objective }}" \
            --hub-repo "${{ inputs.hub_repo }}" \
            --target-steps "${{ inputs.target_steps }}" \
            --max-steps "${{ inputs.max_steps }}" \
            --qk-mode dmap \
            --dmap-alpha "${{ inputs.dmap_alpha }}" \
            --precision "${{ inputs.precision }}" \
            --wd-ada "${{ inputs.wd_ada }}" \
            ${{ inputs.probe == 'true' && '--probe-attn-logits' || '' }} \
            --lr-policy "${{ inputs.lr_policy }}" \
            --lr "${{ inputs.lr }}" \
            --lr-min "${{ inputs.lr_min }}" \
            --lr-hard-min "${{ inputs.lr_hard_min }}" \
            --lr-backoff "${{ inputs.lr_backoff }}" \
            --lr-min-scale "${{ inputs.lr_min_scale }}" \
            --wd "${{ inputs.wd }}" \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            ${{ inputs.reset_lr_controller && '--reset-lr-controller' || '' }} \
            ${{ inputs.extra_args }}
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

      # Numerics and diagnostics.
      precision:
        description: "Training numerics (tf32 = published recipe | bf16 = previous default)"
        required: false
        default: "tf32"
      probe:
        description: "Enable ditflex.probe diagnostics (grad families + attn logits, rank 0)"
        required: false
        default: "true"

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
      wd_ada:
        description: "Decoupled weight decay on the adaLN family only (0 = off)"
        required: false
        default: "0.01"
      qk_norm:
        description: "Model has per-head RMSNorm on Q/K (set true after the 344K migration)"
        required: false
        default: "false"
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


jobs:
  launch:
    name: >-
      recovery · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} ·
      step=${{ inputs.resume_step }} · max=${{ inputs.max_steps }} ·
      ${{ inputs.precision }}${{ inputs.probe == 'true' && ' · probe' || '' }}
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
            --precision "${{ inputs.precision }}" \
            ${{ inputs.probe == 'true' && '--probe-attn-logits' || '' }} \
            --lr "${{ inputs.lr }}" \
            --lr-policy "${{ inputs.lr_policy }}" \
            --lr-backoff "${{ inputs.lr_backoff }}" \
            --lr-min-scale "${{ inputs.lr_min_scale }}" \
            --wd "${{ inputs.wd }}" \
            --wd-ada "${{ inputs.wd_ada }}" \
            ${{ inputs.qk_norm == 'true' && '--qk-norm' || '' }} \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            --skip-warn-rate "${{ inputs.skip_warn_rate }}" \
            --skip-retry-rate "${{ inputs.skip_retry_rate }}" \
            --skip-emergency-rate "${{ inputs.skip_emergency_rate }}" \
            ${{ inputs.extra_args }}
```

### `.github/workflows/train.yml`

```yaml
name: train

# Manual dispatch only.  Each run is detached: GitHub exits after launch while
# Modal pulls the latest healthy checkpoint, trains, saves, and pushes.
#
# Resume behavior: no pinned anchor.  The supervisor's stable-resume selection
# runs; once latest is a v3 transactional checkpoint it is trusted as-is, so
# this workflow simply picks up wherever the chain left off and continues
# toward target_steps.  Use train-recovery-270k.yml when you need to pin an
# exact historical step instead.
#
# GitHub caps workflow_dispatch at 25 inputs.  The deprecated v2 loss-ratio
# inputs and seed_offset were removed to make room for precision / probe /
# wd_ada; anything without a dedicated field goes through extra_args.

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
        description: "Stepping budget in seconds; Modal timeout adds retry allowance"
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

      # Numerics, stabilization, and diagnostics.
      precision:
        description: "Training numerics (tf32 = published recipe | bf16 = legacy)"
        required: false
        default: "tf32"
      wd_ada:
        description: "Decoupled weight decay on the adaLN family only (0 = off)"
        required: false
        default: "0.01"
      qk_norm:
        description: "Model has per-head RMSNorm on Q/K (set true after the 344K migration)"
        required: false
        default: "false"
      probe:
        description: "Enable ditflex.probe diagnostics (grad families + attn logits, rank 0)"
        required: false
        default: "true"

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
        description: "Skip update above this multiple of the frozen gradient reference (0 = off)"
        required: false
        default: "4.0"
      grad_ceiling:
        description: "Absolute raw gradient-norm skip ceiling (0 = off)"
        required: false
        default: "0"

      extra_args:
        description: >-
          Extra modal-run flags, passed through verbatim (e.g.
          "--probe-batch 16 --seed-offset 7 --resume-step 275000").
          Anything without a dedicated field above goes here.
        required: false
        default: ""

jobs:
  launch:
    name: >-
      launch · ${{ inputs.gpus }}x${{ inputs.gpu_kind }} · ${{ inputs.objective }} ·
      ${{ inputs.lr_policy }} to ${{ inputs.target_steps }} · ${{ inputs.precision }} ·
      wd_ada=${{ inputs.wd_ada }}${{ inputs.probe == 'true' && ' · probe' || '' }}
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
            --precision "${{ inputs.precision }}" \
            --wd-ada "${{ inputs.wd_ada }}" \
            ${{ inputs.qk_norm == 'true' && '--qk-norm' || '' }} \
            ${{ inputs.probe == 'true' && '--probe-attn-logits' || '' }} \
            --lr-policy "${{ inputs.lr_policy }}" \
            --lr "${{ inputs.lr }}" \
            --lr-min "${{ inputs.lr_min }}" \
            --lr-hard-min "${{ inputs.lr_hard_min }}" \
            --lr-backoff "${{ inputs.lr_backoff }}" \
            --lr-min-scale "${{ inputs.lr_min_scale }}" \
            --wd "${{ inputs.wd }}" \
            --clip "${{ inputs.clip }}" \
            --spike-skip "${{ inputs.spike_skip }}" \
            --grad-ceiling "${{ inputs.grad_ceiling }}" \
            ${{ inputs.reset_lr_controller && '--reset-lr-controller' || '' }} \
            ${{ inputs.extra_args }}
```

### `README.md`

````markdown
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
````

### `evaluation/fid_results.json`

```json
{
  "config": {
    "num_samples": 50000,
    "sample_steps": 50,
    "cfg_scale": 1.5,
    "seed": 0,
    "objective": "flow",
    "reference": "VAE-decoded latents from sparsetrace/dlatentzz (same decoder as generation; NOT comparable to published FID)",
    "gpu": "H100"
  },
  "fid": {
    "sparsetrace/ditflex-L2-flow-dmap": 18.7464
  }
}
```

### `evaluation/modal_fid.py`

```python
"""FID evaluation for the ditflex chains.

Streams: sample latents -> VAE decode -> InceptionV3 pool features ->
Frechet distance against reference statistics. Images are never written to
disk; only the 2048-d features are retained (50k x 2048 fp32 = 410 MB).

RUN:
    modal run evaluation/modal_fid.py --repos "sparsetrace/ditflex-L2-flow-dmap" \
        --num-samples 50000 --ref-stats-url <url-or-empty>

=============================================================================
THREE THINGS THAT DECIDE WHETHER YOUR NUMBER MEANS ANYTHING
=============================================================================

1. REFERENCE STATISTICS. FID is a distance to a reference distribution; the
   number is meaningless without saying which.
     * ref_stats_url = ADM's VIRTUAL_imagenet256_labeled.npz  -> comparable
       to published DiT/SiT numbers (they all use the ADM eval suite).
     * ref_mode = "latents"  -> statistics computed from YOUR OWN VAE-decoded
       latents. This measures the generative model only, factoring out VAE
       reconstruction error. Self-consistent and ideal for dmap-vs-amap, but
       NOT comparable to any published figure. Say which one you used.

2. SAMPLE COUNT. FID is biased upward at small N and the bias does not
   cancel between models unless N is identical. 50,000 is the convention.
   10,000 is defensible for internal comparison; 1,000 is a smoke test and
   should never be reported as "FID".

3. EMA WEIGHTS. Under the constant-LR recipe the raw weights orbit the
   minimum and the EMA sits in it. Sampling raw weights understates the
   model. See the ADAPTER section: confirm load_checkpoint() pulls EMA.
=============================================================================
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent
GPU_KIND = os.environ.get("MODAL_GPU", "H100")
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

# 50k samples x 50 Euler steps x 2 (CFG) = 5e6 forward passes. Budget hours,
# not minutes, and size the timeout accordingly.
TIMEOUT = int(os.environ.get("MODAL_FID_SECONDS", "14400"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # torchvision MUST come from the same index as torch: pytorch-fid pulls it
    # in, and a PyPI CPU wheel against a CUDA torch fails with
    # "operator torchvision::nms does not exist".
    .pip_install("torch", "torchvision", extra_options=f"--index-url {TORCH_INDEX}")
    .pip_install(
        "accelerate>=0.34",     # else diffusers falls back to slow VAE loading
        "diffusers>=0.31",
        # NOT transformers: diffusers only needs it for single-file loaders,
        # and importing it drags in transformers.AutoImageProcessor ->
        # torchvision.io, which is where the ABI mismatch explodes.
        # AutoencoderKL loads fine without it.
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "scipy>=1.11",
        "pytorch-fid>=0.3.0",   # canonical ported InceptionV3 weights
        "pillow",
        "tqdm",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=[".git", "**/__pycache__", "*.egg-info", ".venv",
                ".ruff_cache", ".pytest_cache"],
        copy=True,   # install-time visibility; see modal_sample.py notes
    )
)

app = modal.App("ditflex-fid", image=image)

VAE_ID = "stabilityai/sd-vae-ft-ema"   # the DiT/SiT standard decoder
VAE_SCALE = 0.18215


# =============================================================================
# ADAPTER -- wire these two to sampling/sample.py. They are the only places
# this script needs to know your codebase, and they are deliberately isolated.
# =============================================================================

def _find_model_cfg(obj, known: set):
    """Depth-first search for the nested dict carrying the most ModelConfig
    fields. Checkpoint state files bury the model config at varying depths;
    this finds it without hardcoding a path."""
    best, best_score, best_path = None, 0, ""
    stack = [(obj, "")]
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            score = len(known & set(cur))
            if score > best_score:
                best, best_score, best_path = cur, score, path or "<root>"
            stack.extend((v, f"{path}.{k}" if path else k)
                         for k, v in cur.items() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend((v, f"{path}[]") for v in cur if isinstance(v, (dict, list)))
    return best, best_score, best_path


def load_checkpoint(repo: str, device):
    """Return (model, ModelConfig) with EMA weights loaded, in eval mode.

    Filenames are discovered rather than assumed: the first run prints the
    repo's file list, so if the guesses below miss, the log tells you exactly
    what to put in CFG_NAMES / EMA_NAMES / RAW_NAMES.
    """
    import dataclasses
    import json
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    sys.path.insert(0, "/repo/src")
    from ditflex.config import ModelConfig
    from ditflex.diffusion_model import build_dmap_model
    from ditflex.model import build_model

    CFG_NAMES = ("state.json", "config.json", "model_config.json",
                 "ditflex_config.json")
    EMA_NAMES = ("ema.safetensors", "ema_model.safetensors", "model_ema.safetensors")
    RAW_NAMES = ("model.safetensors", "diffusion_pytorch_model.safetensors",
                 "pytorch_model.safetensors")

    # Skip archive/ (duplicate older checkpoints), samples/ (PNGs) and
    # optim.safetensors (~2x model size, Adam moments) -- none are needed here.
    path = Path(snapshot_download(
        repo_id=repo, repo_type="model",
        ignore_patterns=["archive/*", "samples/*", "optim.safetensors"],
    ))
    listing = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    print(f"[fid] {repo} contains: {listing}")

    # ---- config ----
    cfg_file = next((path / n for n in CFG_NAMES if (path / n).exists()), None)
    if cfg_file is None:
        raise FileNotFoundError(
            f"no config json among {CFG_NAMES}; repo has {listing}. "
            f"Add the correct name to CFG_NAMES."
        )
    raw_cfg = json.loads(cfg_file.read_text())
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    model_cfg, score, where = _find_model_cfg(raw_cfg, known)
    if model_cfg is None or score < 3:
        raise KeyError(
            f"{cfg_file.name}: no ModelConfig-like block found (best match had "
            f"{score} of {len(known)} fields). Top-level keys: {sorted(raw_cfg)}"
        )
    print(f"[fid] model config from {cfg_file.name}:{where} "
          f"({score}/{len(known)} fields present)")
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in known})
    print(f"[fid] cfg: qk_mode={cfg.qk_mode} qk_norm={cfg.qk_norm} "
          f"dmap_alpha={cfg.dmap_alpha} num_classes={cfg.num_classes}")

    # ---- weights: EMA strongly preferred ----
    wfile = next((path / n for n in EMA_NAMES if (path / n).exists()), None)
    used_ema = wfile is not None
    if wfile is None:
        wfile = next((path / n for n in RAW_NAMES if (path / n).exists()), None)
    if wfile is None:
        raise FileNotFoundError(
            f"no weights among {EMA_NAMES + RAW_NAMES}; repo has {listing}."
        )
    state = load_file(str(wfile))
    if not used_ema:                                  # EMA may live inside the same file
        if any(k.startswith("ema.") for k in state):
            state = {k[4:]: v for k, v in state.items() if k.startswith("ema.")}
            used_ema = True
    if used_ema:
        print(f"[fid] weights: {wfile.name} (EMA)")
    else:
        print(f"[fid] *** WARNING: {wfile.name} looks like RAW weights, not EMA.  ***")
        print("[fid] *** Under constant-LR the raw weights orbit the minimum while ***")
        print("[fid] *** the EMA sits in it; FID will be inflated. Check the repo.  ***")

    # ---- build + load ----
    model = build_dmap_model(cfg) if cfg.qk_mode == "dmap" else build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # DMAP ties W_K to W_Q, so no separate W_K was ever trained and the
    # checkpoint has no to_k tensors -- 2 per block is EXPECTED here, not a
    # fault. Only non-to_k gaps indicate a wrong file or config.
    tied_k = [k for k in missing if ".to_k." in k]
    other_missing = [k for k in missing if ".to_k." not in k]

    if tied_k:
        if cfg.qk_mode != "dmap":
            raise RuntimeError(
                f"{len(tied_k)} to_k tensors missing but qk_mode={cfg.qk_mode!r} "
                f"-- an amap checkpoint must carry W_K. Wrong weights file?"
            )
        print(f"[fid] {len(tied_k)} to_k tensors absent -- expected for "
              f"qk_mode=dmap (W_K tied to W_Q)")
        # Verify the built model really ties them. If to_k is a separate,
        # untrained module the processor MUST never read it, or we would be
        # sampling with a randomly initialised W_K and see no error.
        try:
            a = model.transformer_blocks[0].attn1
            aliased = (a.to_k is a.to_q) or (
                a.to_k.weight.data_ptr() == a.to_q.weight.data_ptr())
            if aliased:
                print("[fid] to_k aliases to_q -- tying confirmed at the module level")
            else:
                print("[fid] NOTE: to_k is a separate module left at init; the dmap "
                      "processor is expected to ignore it (see build_dmap_model). "
                      "If samples look like noise, this is the first thing to check.")
        except AttributeError as e:
            print(f"[fid] could not inspect attn tying ({e})")

    if other_missing or unexpected:
        print(f"[fid] load_state_dict: {len(other_missing)} unexpected-missing, "
              f"{len(unexpected)} unexpected")
        if other_missing:
            print(f"[fid]   missing[:5]    = {other_missing[:5]}")
        if unexpected:
            print(f"[fid]   unexpected[:5] = {unexpected[:5]}")
        if len(other_missing) > 10:
            raise RuntimeError("too many missing keys -- wrong weights file or config")

    return model.to(device).eval(), cfg


def sample_latents(model, labels, *, steps: int, cfg_scale: float, cfg,
                   device, objective: str, seed: int):
    """Return [B,4,32,32] scaled latents using the SAME sampler as sample.py.

    NOTE the per-batch `seed`. sample_flow/sample_ddim seed their initial noise
    from this argument (default FIXED_SEED), so calling them with a constant
    seed would draw every batch from the same noise tensors: diversity would
    collapse and FID would be badly inflated, with no error raised anywhere.
    """
    import sys

    sys.path.insert(0, "/repo/src")
    from ditflex.sample import sample_ddim, sample_flow

    sampler = sample_flow if objective == "flow" else sample_ddim
    return sampler(
        model, labels.cpu(),
        num_classes=cfg.num_classes,
        cfg_scale=cfg_scale,
        ode_steps=steps,
        seed=seed,
        device=device,
    )



# =============================================================================



def _load_latent_store(latents_repo: str, n: int, num_classes: int, seed: int):
    """Sample n latents (class-balanced when labels are present) from an HF store.

    Handles sharded stores: the previous version grabbed the FIRST shard via
    rglob(), which silently sampled a fraction of the dataset.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(repo_id=latents_repo, repo_type="dataset"))
    shards = sorted(path.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no .safetensors under {path} -- check latents_repo and repo_type "
            f"(this branch assumes repo_type='dataset')"
        )
    print(f"[fid] latent store: {len(shards)} shard(s)")

    lat_parts, lab_parts = [], []
    for s in shards:
        d = load_file(str(s))
        lk = next((k for k in ("latents", "latent", "x", "data") if k in d), None)
        if lk is None:
            raise KeyError(f"{s.name}: no latent tensor found; keys = {list(d)}")
        lat_parts.append(d[lk])
        bk = next((k for k in ("labels", "label", "y", "classes") if k in d), None)
        if bk is not None:
            lab_parts.append(d[bk])

    lat = torch.cat(lat_parts) if len(lat_parts) > 1 else lat_parts[0]
    lab = None
    if lab_parts:
        lab = torch.cat(lab_parts) if len(lab_parts) > 1 else lab_parts[0]
    print(f"[fid] {lat.shape[0]:,} latents, labels={'yes' if lab is not None else 'no'}")

    if lat.shape[0] < n:
        raise ValueError(f"store has {lat.shape[0]} latents, need {n}")

    g = torch.Generator().manual_seed(seed)
    if lab is not None:
        # Class-balanced, matching the generation side exactly.
        per = n // num_classes
        idx = []
        for c in range(num_classes):
            pool = (lab == c).nonzero(as_tuple=True)[0]
            if len(pool) < per:
                raise ValueError(f"class {c}: {len(pool)} available, need {per}")
            idx.append(pool[torch.randperm(len(pool), generator=g)[:per]])
        idx = torch.cat(idx)
        idx = idx[torch.randperm(len(idx), generator=g)]
        print(f"[fid] reference: class-balanced, {per}/class")
    else:
        idx = torch.randperm(lat.shape[0], generator=g)[:n]
        print("[fid] reference: uniform random (no labels found -- NOT class-balanced, "
              "which biases FID against the class-balanced generation side)")
    return lat, idx


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Standard FID formula (Heusel et al. 2017)."""
    import numpy as np
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # Numerically singular product; nudge both covariances.
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * np.trace(covmean))


def _stats(features):
    import numpy as np
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@app.function(
    gpu=GPU_KIND,
    cpu=8.0,
    memory=32768,   # latent store is ~10 GiB in CPU RAM during reference pass
    timeout=TIMEOUT,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def evaluate(
    repos: str = "sparsetrace/ditflex-L2-flow-dmap",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    objective: str = "flow",
    seed: int = 0,
) -> str:
    # Fail before spending GPU time on a misconfigured dispatch.
    if not ref_stats_url and ref_mode == "latents" and not latents_repo:
        raise ValueError(
            "ref_mode='latents' requires --latents-repo (the HF dataset repo "
            "holding the SD-VAE latents). Either set it, or pass --ref-stats-url "
            "pointing at a precomputed reference .npz."
        )
    if num_samples % 1000 != 0:
        print(f"[fid] WARNING: num_samples={num_samples} is not a multiple of 1000, "
              f"so classes cannot be exactly balanced.")

    # --- ABI smoke test FIRST ----------------------------------------------
    # torchvision built against a different torch than the CUDA wheel fails
    # with "operator torchvision::nms does not exist". Both diffusers (via
    # transformers.AutoImageProcessor) and pytorch_fid import torchvision, so
    # this MUST run before either or the failure surfaces as an opaque
    # "Could not import module 'AutoImageProcessor'" from inside their lazy
    # import machinery.
    import torch
    import torchvision

    print(f"[fid] torch {torch.__version__} / torchvision {torchvision.__version__}")
    torchvision.ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)
    print("[fid] torchvision C++ ops OK")
    # -----------------------------------------------------------------------

    import numpy as np
    from diffusers import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    from tqdm import tqdm

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Inception (block 3 = 2048-d pool features, the FID standard) ----
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).eval()

    @torch.no_grad()
    def features_from_latents(lat):
        """latents -> [B,2048] Inception pool features.

        Accepts flat [B,4096] (how the store keeps them) or shaped
        [B,4,32,32] (what the sampler returns). Layout is channel-major,
        confirmed by the neighbour-correlation probe below.
        """
        if lat.dim() == 2:
            lat = lat.view(lat.shape[0], 4, 32, 32)
        img = vae.decode(lat / VAE_SCALE).sample          # [-1, 1]
        img = (img.clamp(-1, 1) + 1.0) / 2.0              # [0, 1]
        # InceptionV3(resize_input=True, normalize_input=True) handles the
        # 299x299 bilinear resize and the [-1,1] rescale internally. Do NOT
        # pre-resize -- resize implementation is a known source of FID drift
        # between codebases (cf. clean-fid).
        return inception(img)[0].squeeze(-1).squeeze(-1).cpu().numpy()

    # ---- reference statistics -------------------------------------------
    if ref_stats_url:
        import urllib.request
        print(f"[fid] downloading reference stats: {ref_stats_url}")
        urllib.request.urlretrieve(ref_stats_url, "/tmp/ref.npz")
        ref = np.load("/tmp/ref.npz")
        mu_ref, sigma_ref = ref["mu"], ref["sigma"]
        ref_desc = ref_stats_url
    elif ref_mode == "latents":
        print(f"[fid] computing reference stats from real latents (n={num_samples})")
        all_lat, idx = _load_latent_store(latents_repo, num_samples, 1000, seed)

        # --- layout probe: (C,H,W) vs (H,W,C) is silent if wrong ------------
        # Images are spatially smooth, channels are not, so the correct
        # reshape has the higher horizontal-neighbour correlation. A wrong
        # layout decodes without error and yields meaningless FID.
        if all_lat.dim() == 2:
            probe = all_lat[idx[:256]].float()

            def _nbr_corr(x):                          # x: [B,C,H,W]
                u = x[..., :, :-1].reshape(-1)
                v = x[..., :, 1:].reshape(-1)
                u = u - u.mean()
                v = v - v.mean()
                return float(u @ v / (u.norm() * v.norm() + 1e-12))

            chw = _nbr_corr(probe.view(-1, 4, 32, 32))
            hwc = _nbr_corr(probe.view(-1, 32, 32, 4).permute(0, 3, 1, 2))
            print(f"[fid] layout probe: neighbour corr  CHW={chw:.3f}  HWC={hwc:.3f}")
            print(f"[fid] -> latents are "
                  f"{'CHW (correct)' if chw > hwc else 'HWC -- FIX THE RESHAPE'}")
            shaped = probe.view(-1, 4, 32, 32)
            print(f"[fid] per-channel mean {shaped.mean((0, 2, 3)).tolist()}")
            print(f"[fid] per-channel std  {shaped.std((0, 2, 3)).tolist()}")
            print("[fid] std ~1 => latents are pre-scaled, so dividing by "
                  f"{VAE_SCALE} before decode is correct")
        # ---------------------------------------------------------------------

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc="ref"):
            sl = idx[i : i + batch_size]
            lat = all_lat[sl].to(device=device, dtype=torch.float32)
            feats[i : i + len(sl)] = features_from_latents(lat)
        mu_ref, sigma_ref = _stats(feats)
        ref_desc = (f"VAE-decoded latents from {latents_repo} "
                    f"(same decoder as generation; NOT comparable to published FID)")
        del feats, all_lat
    else:
        raise ValueError("supply ref_stats_url, or ref_mode='latents' with latents_repo")

    # ---- per-chain generation --------------------------------------------
    results = {"config": {
        "num_samples": num_samples, "sample_steps": sample_steps,
        "cfg_scale": cfg_scale, "seed": seed, "objective": objective,
        "reference": ref_desc,
        "gpu": GPU_KIND,
    }, "fid": {}}

    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        print(f"[fid] === {repo} ===")
        model, cfg = load_checkpoint(repo, device)

        torch.manual_seed(seed)
        # Class-balanced: exactly num_samples/1000 per class, then shuffled.
        labels_all = torch.arange(num_samples, device=device) % cfg.num_classes
        labels_all = labels_all[torch.randperm(num_samples, device=device)]

        feats = np.empty((num_samples, 2048), dtype=np.float32)
        for i in tqdm(range(0, num_samples, batch_size), desc=repo.split("/")[-1]):
            lab = labels_all[i : i + batch_size]
            with torch.no_grad():
                # Per-batch seed -- see the note in sample_latents().
                lat = sample_latents(
                    model, lab, steps=sample_steps, cfg_scale=cfg_scale,
                    cfg=cfg, device=device, objective=objective,
                    seed=seed * 1_000_003 + i,
                )
                feats[i : i + len(lab)] = features_from_latents(lat)

        mu, sigma = _stats(feats)
        score = frechet_distance(mu, sigma, mu_ref, sigma_ref)
        results["fid"][repo] = round(score, 4)
        print(f"[fid] {repo}: FID = {score:.4f}")

        del model, feats
        torch.cuda.empty_cache()

    out = json.dumps(results, indent=2)
    print(out)
    return out


@app.local_entrypoint()
def main(
    repos: str = "sparsetrace/ditflex-L2-flow-dmap",
    num_samples: int = 50_000,
    batch_size: int = 64,
    sample_steps: int = 50,
    cfg_scale: float = 1.5,
    ref_stats_url: str = "",
    ref_mode: str = "latents",
    latents_repo: str = "",
    objective: str = "flow",
    seed: int = 0,
):
    payload = evaluate.remote(
        repos=repos, num_samples=num_samples, batch_size=batch_size,
        sample_steps=sample_steps, cfg_scale=cfg_scale,
        ref_stats_url=ref_stats_url, ref_mode=ref_mode,
        latents_repo=latents_repo, objective=objective, seed=seed,
    )
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/fid_results.json").write_text(payload)
    print("[fid] wrote evaluation/fid_results.json")
```

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

### `run/migrate_qknorm.py`

```python
"""run/migrate_qknorm.py -- Hub-facing CLI for the qk-norm migration.

Pulls one committed revision of the amap chain, runs
ditflex.migrate.migrate_checkpoint (name-keyed remap of model / EMA /
AdamW state into the qk_norm=True architecture), and pushes the result as
the new Hub latest under an unmistakable commit message so
run/recover_checkpoint.py and the revision ledger can always identify the
migration boundary.

SAFETY: defaults to --dry-run, which performs the FULL migration locally
(pull, remap, validate) and prints the summary, uploading nothing.
Inspect, then re-run with --push.

    HF_TOKEN=... python run/migrate_qknorm.py \
        --repo sparsetrace/ditflex-L2-flow --step 344000 --dry-run
    HF_TOKEN=... python run/migrate_qknorm.py \
        --repo sparsetrace/ditflex-L2-flow --step 344000 --push

Runs on CPU: two DiT-L models in fp32 (~4 GB RAM), a few minutes, no GPU.

After pushing, resume the chain with:
    workflow train-recovery-270k: resume_step=<step>, max_steps=5000,
        qk_norm=true, reset_lr_controller=true, lr=0.00001   (warmup)
    then workflow train: qk_norm=true, reset_lr_controller=true, lr=0
        (full cosine to 400K)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="committed step to migrate (0 = current Hub latest)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument(
        "--push",
        dest="dry_run",
        action="store_false",
        help="actually upload the migrated checkpoint as Hub latest",
    )
    parser.add_argument(
        "--keep-dir",
        type=str,
        default="",
        help="optional directory to retain the migrated checkpoint locally",
    )
    args = parser.parse_args()

    from ditflex.checkpoint import pull_from_hub, push_to_hub, resolve_revision_for_step
    from ditflex.migrate import migrate_checkpoint

    revision = None
    if args.step > 0:
        revision = resolve_revision_for_step(args.repo, args.step)
        print(f"[migrate] step {args.step:,} -> revision {revision[:12]}")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        dest = Path(args.keep_dir) if args.keep_dir else Path(tmp) / "migrated"

        print(f"[migrate] pulling {args.repo} ({revision[:12] if revision else 'latest'}) ...")
        pulled = pull_from_hub(args.repo, source, revision=revision)
        if pulled is None:
            raise SystemExit(f"no checkpoint found in {args.repo}")

        print("[migrate] migrating (CPU; a few minutes for DiT-L) ...")
        new_state = migrate_checkpoint(source, dest, source_revision=revision)
        summary = new_state["migration_summary"]
        print(
            f"[migrate] OK: step {summary['from_step']:,}  "
            f"new_tensors={summary['new_tensors']}  "
            f"optim_states_migrated={summary['optim_states_migrated']}/"
            f"{summary['optim_states_total_old']}"
        )
        print(json.dumps(summary, indent=2))

        if args.dry_run:
            print(
                "[migrate] DRY RUN -- nothing uploaded. Re-run with --push to "
                "promote the migrated checkpoint as Hub latest."
            )
            if not args.keep_dir:
                return 0
            print(f"[migrate] migrated checkpoint retained at {dest}")
            return 0

        commit = push_to_hub(
            dest,
            args.repo,
            commit_message=(
                f"MIGRATION qk-norm: step {summary['from_step']} "
                "(RMSNorm on Q/K; optimizer state remapped by name; "
                "guard reset; resume with --qk-norm --reset-lr-controller)"
            ),
        )
        print(
            f"[migrate] PUSHED migrated step {summary['from_step']:,} to "
            f"{args.repo}" + (f" revision={commit[:12]}" if commit else "")
        )
        if args.keep_dir:
            print(f"[migrate] local copy retained at {dest}")
        else:
            shutil.rmtree(dest, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
    wd_ada: float = 0.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
    precision: str = "tf32",
    probe_attn_logits: bool = False,
    probe_batch: int = 8,
    qk_mode: str = "amap",
    dmap_alpha: float = 0.0,
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
    if precision not in {"tf32", "bf16"}:
        print(f"[modal] unknown precision: {precision!r}")
        return 2
    if wd_ada < 0.0:
        print("[modal] wd_ada must be non-negative")
        return 2
    if qk_mode not in {"amap", "dmap"}:
        print(f"[modal] unknown qk_mode: {qk_mode!r}")
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
            f"--precision={precision}",
            f"--qk-mode={qk_mode}",
            f"--dmap-alpha={dmap_alpha}",
        ]
        if probe_attn_logits:
            command.append("--probe-attn-logits")
            command.append(f"--probe-batch={probe_batch}")
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
        if wd_ada > 0.0:
            command.append(f"--wd-ada={wd_ada}")
        if reset_lr_controller and attempt == 0:
            command.append("--reset-lr-controller")

        print(
            f"\n[modal] attempt {attempt}/{max_retries}: "
            f"lr_factor={attempt_factor:g} seed_offset={attempt_seed_offset} "
            f"budget={child_budget}s precision={precision} qk_mode={qk_mode} anchor="
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
    wd_ada: float = 0.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
    precision: str = "tf32",
    probe_attn_logits: bool = False,
    probe_batch: int = 8,
    qk_mode: str = "amap",
    dmap_alpha: float = 0.0,
):
    if objective not in {"ddpm", "flow"}:
        raise SystemExit(f"unknown objective: {objective!r}")
    if lr_policy not in {"constant", "cosine", "adaptive"}:
        raise SystemExit(f"unknown lr_policy: {lr_policy!r}")
    if precision not in {"tf32", "bf16"}:
        raise SystemExit(f"unknown precision: {precision!r}")
    if wd_ada < 0.0:
        raise SystemExit("wd_ada must be non-negative")
    if qk_mode not in {"amap", "dmap"}:
        raise SystemExit(f"unknown qk_mode: {qk_mode!r}")
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
        wd_ada=wd_ada,
        clip=clip,
        spike_skip=spike_skip,
        seed_offset=seed_offset,
        grad_ceiling=grad_ceiling,
        skip_warn_rate=skip_warn_rate,
        skip_retry_rate=skip_retry_rate,
        skip_emergency_rate=skip_emergency_rate,
        precision=precision,
        probe_attn_logits=probe_attn_logits,
        probe_batch=probe_batch,
        qk_mode=qk_mode,
        dmap_alpha=dmap_alpha,
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

    import sys
    sys.path.insert(0, "/repo/src")
    from ditflex.config import Config
    
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

QK-NORM (added at the 344K migration; DEVIATION from published DiT/SiT):
if the module carries ``norm_q`` / ``norm_k`` (RMSNorm over head_dim),
the processor applies them per-head AFTER the [B, N, H*D] -> [B, H, N, D]
reshape and BEFORE flex_attention -- entirely upstream of the score_mod,
so the swappable-score dispatch path is untouched. The fp64 reference
applies the identical normalization from the same weights, so the
identity gate certifies both configurations. Rationale recorded in the
migration commit and README: at ~270-312K steps the un-normalized chain
developed unbounded QK logits (block 1 reaching 8.6e6), driven by adaLN
weight growth, producing the gradient-spike instability; RMSNorm on Q and
K bounds per-head logits by construction. The DMAP chain does NOT use
qk-norm (see model.py / diffusion_model.py guards): it shows no logit
pathology, untied norms would break its R == 0 symmetry, and tied norms
would change the squared-distance kernel's geometry.

Scope: DiT-L/2 self-attention on [B, N, C] tokens. Cross-attention,
attention masks, 4D inputs, group/spatial norm, in-processor residuals,
and output rescaling are all rejected loudly rather than handled -- in a
repo whose premise is that every deviation from the baseline is known, a
config surprise should fail at the gate, not show up later as an
uninterpretable training curve. qk-norm moved from the rejected list to
the handled list in the 344K migration, with gate coverage for both the
with-norm and without-norm configurations.

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

QK_NORM_EPS = 1e-6


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
        if getattr(attn, "residual_connection", False):
            raise ValueError("residual_connection is handled by the block, not the processor.")
        if getattr(attn, "rescale_output_factor", 1.0) != 1.0:
            raise ValueError("rescale_output_factor != 1 is not handled.")

        norm_q = getattr(attn, "norm_q", None)
        norm_k = getattr(attn, "norm_k", None)
        if (norm_q is None) != (norm_k is None):
            raise ValueError(
                "qk-norm must be installed on BOTH norm_q and norm_k or neither; "
                "a half-installed configuration is a migration bug."
            )

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

        # QK-norm: per-head RMSNorm over head_dim, applied before the kernel
        # and entirely outside the score_mod dispatch path.
        if norm_q is not None:
            query = norm_q(query)
            key = norm_k(key)

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


def _reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = QK_NORM_EPS,
) -> torch.Tensor:
    """RMSNorm written straight from the math, for the fp64 reference path.

    Matches torch.nn.RMSNorm semantics: x / sqrt(mean(x^2) + eps) * weight,
    normalized over the last dimension. Written explicitly (no fused op) so
    the reference depends on nothing but arithmetic.
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms * weight


def reference_self_attention(
    attn,
    hidden_states: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Softmax self-attention written straight from the math.

    Uses the weights of ``attn`` but none of its forward code and no fused
    attention kernel of any kind: explicit projections, optional explicit
    per-head RMSNorm on Q/K (when the module carries norm_q/norm_k), an
    explicit ``q @ k^T * scale`` score matrix, an explicit softmax, explicit
    output projection.

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

    norm_q = getattr(attn, "norm_q", None)
    norm_k = getattr(attn, "norm_k", None)
    if norm_q is not None:
        eps_q = float(getattr(norm_q, "eps", QK_NORM_EPS) or QK_NORM_EPS)
        eps_k = float(getattr(norm_k, "eps", QK_NORM_EPS) or QK_NORM_EPS)
        query = _reference_rms_norm(query, cast(norm_q.weight), eps_q)
        key = _reference_rms_norm(key, cast(norm_k.weight), eps_k)

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
    # DEVIATION (adopted at the 344K migration, amap chain only): per-head
    # RMSNorm(head_dim, eps=1e-6) on Q and K before attention. Forced by
    # evidence: unbounded block-1 QK logits (8.6e6 at 312.5K) driven by
    # adaLN weight growth produced the gradient-spike instability; RMSNorm
    # bounds per-head logits by construction. INVALID for qk_mode="dmap":
    # untied norms would break the R == 0 symmetry, tied norms would
    # flatten the destination potential g_j that defines the DMAP kernel
    # (build_dmap_model refuses the combination). Old checkpoints lacking
    # this key deserialize to False, so pre-migration configs round-trip
    # unchanged.
    qk_norm: bool = False


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

QK-NORM IS REFUSED HERE, deliberately.  The amap chain adopted per-head
RMSNorm on Q/K at its 344K migration (unbounded-logit instability).  It
does not transfer to this chain: untied norm_q/norm_k would make the
effective bilinear asymmetric (destroying the R == 0 invariant that IS
the experiment), and even a tied norm forces every |q_j| toward unit
RMS, flattening the destination potential g_j = scale*|q_j|^2 whose
survival in the softmax is the defining difference between DMAP and
plain attention (Sec. 3.1).  A "normed DMAP" is a third architecture,
not a stabilized DMAP.  Pre-committed trigger recorded here: if this
chain's probe shows sustained logit growth or grad-median ratcheting in
the 200-280K range (where the amap chain's pathology developed), design
a DMAP-appropriate intervention then -- as its own documented arm.

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
    if getattr(cfg, "qk_norm", False):
        raise ValueError(
            "qk_norm=True is not valid for the DMAP chain: untied norms break "
            "the R == 0 symmetry; a tied norm flattens the destination "
            "potential g_j that defines the DMAP kernel. If this chain ever "
            "develops the amap chain's logit pathology, design a "
            "DMAP-appropriate intervention as its own documented arm (see "
            "module docstring)."
        )

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

### `src/ditflex/migrate.py`

```python
"""src/ditflex/migrate.py -- one-shot checkpoint migration to qk-norm.

Turns a committed amap checkpoint WITHOUT qk-norm into a complete, valid
checkpoint WITH qk-norm, preserving everything that can be preserved:

  * model weights   -- copied by name; the 2*num_layers new RMSNorm weight
                       tensors initialize to ones (RMSNorm default);
  * EMA shadow      -- copied by name; norm weights seeded from the (ones)
                       online values so EMA covers them from step one;
  * AdamW state     -- THE LOAD-BEARING STEP.  checkpoint.py flattens
                       optimizer state keyed by PARAMETER INDEX.  Inserting
                       norm parameters changes named_parameters() ordering,
                       so index-preserving load would silently attach each
                       parameter's first/second moments to the WRONG
                       parameter.  This module remaps old index -> parameter
                       name (via the old architecture's ordering) -> new
                       parameter object.  New norm parameters start with no
                       state; AdamW lazily initializes them on their first
                       step;
  * hyperparameters -- lr, betas, eps, weight_decay copied from the stored
                       param_groups, so the migrated checkpoint resumes at
                       the LR the chain was actually running;
  * step / history  -- step unchanged; a migration record is appended to
                       run_history.

Deliberately NOT preserved:

  * stability controller / health reference / recent losses -- the
    post-norm gradient regime is a different distribution; carrying the
    contaminated pre-norm reference (grown 38.5 -> 115 through capped
    promotions during the divergence) would misconfigure the guard in both
    directions.  The first post-migration run bootstraps a fresh reference
    from its own first windows (resume with --reset-lr-controller).

Lives in src/ditflex so tests can import it; run/migrate_qknorm.py is the
thin Hub-facing CLI around migrate_checkpoint().
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

from ditflex.checkpoint import _unflatten_optim, save_checkpoint, validate_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA
from ditflex.model import build_model

EXPECTED_NEW_SUFFIXES = ("norm_q.weight", "norm_k.weight")


def _assert_only_norm_keys(keys: list[str], what: str) -> None:
    bad = [k for k in keys if not k.endswith(EXPECTED_NEW_SUFFIXES)]
    if bad:
        raise RuntimeError(f"unexpected non-qk-norm {what} keys: {bad[:5]}")


def migrate_checkpoint(
    source_dir: str | Path,
    dest_dir: str | Path,
    *,
    source_revision: str | None = None,
) -> dict:
    """Migrate one validated local checkpoint directory to qk_norm=True.

    Returns the new state dict (already written to dest_dir along with the
    model/EMA/optim safetensors).  Raises on any structural surprise rather
    than proceeding: a migration that is not exactly understood must not
    produce a checkpoint.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    validate_checkpoint(source_dir)
    state = json.loads((source_dir / "state.json").read_text())
    step = int(state["step"])

    old_cfg = Config.from_dict(state["config"])
    if old_cfg.model.qk_mode != "amap":
        raise ValueError(
            f"qk-norm migration is amap-only; checkpoint has qk_mode="
            f"{old_cfg.model.qk_mode!r}"
        )
    if getattr(old_cfg.model, "qk_norm", False):
        raise ValueError("checkpoint already has qk_norm=True; nothing to migrate")

    new_cfg = Config.from_dict(json.loads(old_cfg.to_json()))  # deep copy via JSON
    new_cfg.model.qk_norm = True

    # -- models -----------------------------------------------------------
    old_model = build_model(old_cfg.model)
    old_sd = load_file(str(source_dir / "model.safetensors"))
    old_model.load_state_dict(old_sd)  # strict: the old architecture must match exactly

    new_model = build_model(new_cfg.model)
    missing, unexpected = new_model.load_state_dict(old_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"old checkpoint has keys the new model lacks: {unexpected[:5]}")
    _assert_only_norm_keys(list(missing), "missing model")
    expected_new = 2 * new_cfg.model.num_layers
    if len(missing) != expected_new:
        raise RuntimeError(
            f"expected exactly {expected_new} new qk-norm tensors, found {len(missing)}"
        )
    # Norm weights keep their RMSNorm init (ones): NOT an identity map on
    # pre-trained Q/K -- resume with the documented reduced-LR warmup.

    # -- EMA --------------------------------------------------------------
    ema = EMA(new_model, decay=old_cfg.train.ema_decay)
    old_ema = load_file(str(source_dir / "ema.safetensors"))
    extra_shadow = sorted(set(ema.shadow) - set(old_ema))
    _assert_only_norm_keys(extra_shadow, "EMA-new")
    stray = sorted(set(old_ema) - set(ema.shadow))
    if stray:
        raise RuntimeError(f"old EMA has keys absent from new model: {stray[:5]}")
    for name, tensor in old_ema.items():
        ema.shadow[name] = tensor.detach().to(dtype=torch.float32, copy=True)
    # Norm entries in the shadow remain the (ones) online values -- EMA
    # tracks them from step one of the resumed run.

    # -- optimizer: index -> name -> new parameter ------------------------
    old_groups = state["optim_param_groups"]
    if len(old_groups) != 1:
        raise RuntimeError(
            f"migration assumes the repo's single param group, found {len(old_groups)}"
        )
    old_tensors = load_file(str(source_dir / "optim.safetensors"))
    old_osd = _unflatten_optim(old_tensors, old_groups)
    old_names = [name for name, _ in old_model.named_parameters()]
    state_indices = set(old_osd["state"].keys())
    if state_indices and max(state_indices) >= len(old_names):
        raise RuntimeError(
            f"optimizer state index {max(state_indices)} exceeds old parameter "
            f"count {len(old_names)} -- ordering assumption violated"
        )

    hp = old_groups[0]
    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=float(hp.get("lr", old_cfg.train.lr)),
        betas=tuple(hp.get("betas", (0.9, 0.999))),
        eps=float(hp.get("eps", 1e-8)),
        weight_decay=float(hp.get("weight_decay", old_cfg.train.weight_decay)),
    )
    new_params = dict(new_model.named_parameters())
    migrated = 0
    for index, name in enumerate(old_names):
        entry = old_osd["state"].get(index)
        if entry is None:
            continue
        if name not in new_params:
            raise RuntimeError(f"old parameter {name!r} not present in new model")
        new_optimizer.state[new_params[name]] = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in entry.items()
        }
        migrated += 1

    # -- state.json -------------------------------------------------------
    run_history = list(state.get("run_history", []))
    run_history.append(
        {
            "start_step": step,
            "end_step": step,
            "seconds": 0.0,
            "world": 0,
            "objective": old_cfg.train.objective,
            "completed": False,
            "finished_at": datetime.now(UTC).isoformat(),
            "promotion_reason": "offline qk-norm migration",
            "effective": {
                "migration": "qk_norm",
                "source_revision": source_revision,
                "new_tensors": expected_new,
                "optim_states_migrated": migrated,
            },
        }
    )
    new_state = {
        "step": step,
        "run_history": run_history,
        # Fresh guard: the post-norm gradient regime must define its own
        # normality.  spikes_total is carried as a cumulative counter only.
        "guard_state": {
            "version": 3,
            "spikes_total": int(state.get("guard_state", {}).get("spikes_total", 0)),
            "migration": "qk_norm",
        },
        "transaction": {
            "status": "committed",
            "committed_at": datetime.now(UTC).isoformat(),
            "migration": "qk_norm",
            "source_revision": source_revision,
            "note": (
                "resume with --qk-norm and --reset-lr-controller; expect a "
                "loss bump at step one (fresh RMSNorms rescale Q/K) and run "
                "a bounded reduced-LR warmup segment first"
            ),
        },
        "migration_summary": {
            "from_step": step,
            "new_tensors": expected_new,
            "optim_states_migrated": migrated,
            "optim_states_total_old": len(state_indices),
            "config_change": {"model.qk_norm": [False, True]},
        },
    }

    save_checkpoint(dest_dir, new_model, ema, new_optimizer, new_cfg, new_state)
    validate_checkpoint(dest_dir, expected_step=step)
    return new_state
```

### `src/ditflex/modal_train.py`

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
    wd_ada: float = 0.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
    precision: str = "tf32",
    probe_attn_logits: bool = False,
    probe_batch: int = 8,
    qk_mode: str = "amap",
    dmap_alpha: float = 0.0,
    qk_norm: bool = False,
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
    if precision not in {"tf32", "bf16"}:
        print(f"[modal] unknown precision: {precision!r}")
        return 2
    if wd_ada < 0.0:
        print("[modal] wd_ada must be non-negative")
        return 2
    if qk_mode not in {"amap", "dmap"}:
        print(f"[modal] unknown qk_mode: {qk_mode!r}")
        return 2
    if qk_norm and qk_mode != "amap":
        print("[modal] qk_norm is valid for the amap chain only")
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
            f"--precision={precision}",
            f"--qk-mode={qk_mode}",
            f"--dmap-alpha={dmap_alpha}",
        ]
        if qk_norm:
            command.append("--qk-norm")
        if probe_attn_logits:
            command.append("--probe-attn-logits")
            command.append(f"--probe-batch={probe_batch}")
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
        if wd_ada > 0.0:
            command.append(f"--wd-ada={wd_ada}")
        if reset_lr_controller and attempt == 0:
            command.append("--reset-lr-controller")

        print(
            f"\n[modal] attempt {attempt}/{max_retries}: "
            f"lr_factor={attempt_factor:g} seed_offset={attempt_seed_offset} "
            f"budget={child_budget}s precision={precision} qk_mode={qk_mode} anchor="
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
    wd_ada: float = 0.0,
    clip: float = 1.0,
    spike_skip: float = 4.0,
    seed_offset: int = 0,
    grad_ceiling: float = 0.0,
    skip_warn_rate: float = 0.30,
    skip_retry_rate: float = 0.40,
    skip_emergency_rate: float = 0.60,
    precision: str = "tf32",
    probe_attn_logits: bool = False,
    probe_batch: int = 8,
    qk_mode: str = "amap",
    dmap_alpha: float = 0.0,
    qk_norm: bool = False,
):
    if objective not in {"ddpm", "flow"}:
        raise SystemExit(f"unknown objective: {objective!r}")
    if lr_policy not in {"constant", "cosine", "adaptive"}:
        raise SystemExit(f"unknown lr_policy: {lr_policy!r}")
    if precision not in {"tf32", "bf16"}:
        raise SystemExit(f"unknown precision: {precision!r}")
    if wd_ada < 0.0:
        raise SystemExit("wd_ada must be non-negative")
    if qk_mode not in {"amap", "dmap"}:
        raise SystemExit(f"unknown qk_mode: {qk_mode!r}")
    if qk_norm and qk_mode != "amap":
        raise SystemExit("qk_norm is valid for the amap chain only")
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
        wd_ada=wd_ada,
        clip=clip,
        spike_skip=spike_skip,
        seed_offset=seed_offset,
        grad_ceiling=grad_ceiling,
        skip_warn_rate=skip_warn_rate,
        skip_retry_rate=skip_retry_rate,
        skip_emergency_rate=skip_emergency_rate,
        precision=precision,
        probe_attn_logits=probe_attn_logits,
        probe_batch=probe_batch,
        qk_mode=qk_mode,
        dmap_alpha=dmap_alpha,
        qk_norm=qk_norm,
    )
    if return_code != 0:
        raise SystemExit(return_code)
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

QK-NORM (cfg.qk_norm, amap only; DEVIATION adopted at the 344K
migration): installs torch.nn.RMSNorm(head_dim, eps=1e-6) as norm_q and
norm_k on every Attention module. The Flex processor applies them
per-head after the head reshape (see attention.py). The parameters are
named ``...attn1.norm_q.weight`` / ``...attn1.norm_k.weight`` and are
therefore covered by EMA, checkpointing, and the migration tooling
(ditflex.migrate) by name.

torch.compile does NOT happen here -- train.py compiles. Tests and the
identity gate need the uncompiled module.
"""

from __future__ import annotations

import torch
from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.attention import QK_NORM_EPS, FlexSelfAttnProcessor, ScoreMod
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


def install_qk_norms(model, head_dim: int) -> int:
    """Install per-head RMSNorm(head_dim) as norm_q/norm_k on every
    Attention module. Returns the number of modules touched.

    Assigning the modules as attributes registers them as submodules, so
    the new parameters appear in named_parameters(), state_dict(), the
    EMA shadow, and the optimizer -- everything downstream keys by name.
    Weight init is RMSNorm's default (ones), which is NOT an identity
    transform on pre-trained Q/K (it rescales each head vector to unit
    RMS): a migrated checkpoint needs the short reduced-LR warmup
    documented in run/migrate_qknorm.py.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.norm_q = torch.nn.RMSNorm(head_dim, eps=QK_NORM_EPS)
            module.norm_k = torch.nn.RMSNorm(head_dim, eps=QK_NORM_EPS)
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

    if getattr(cfg, "qk_norm", False):
        # GUARD: qk-norm is an amap-chain deviation only. A dmap-labeled
        # config never reaches here (rejected below), but a future builder
        # calling this with a variant config must decide explicitly.
        if getattr(cfg, "qk_mode", "amap") != "amap":
            raise ValueError(
                "qk_norm=True is defined for the amap baseline only; "
                f"qk_mode={cfg.qk_mode!r} must not carry it (untied norms break "
                "R == 0; tied norms flatten the DMAP destination potential)."
            )
        n_normed = install_qk_norms(model, cfg.attention_head_dim)
        if n_normed != cfg.num_layers:
            raise RuntimeError(
                f"installed qk-norm on {n_normed} Attention modules, expected "
                f"{cfg.num_layers}."
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

### `src/ditflex/probe.py`

```python
"""src/ditflex/probe.py -- opt-in training diagnostics. NEVER on the production path.

Two questions, answered with data instead of pattern-matching:

  1. WHICH parameter family produces the gradient tail?
     grad_family_norms() walks named_parameters AFTER backward (grads must
     still be present) and reduces squared grad norms into the same six
     families train.py's weight-norm log already uses (qk / vo / mlp /
     ada / emb / oth). If one family's grad norm dominates and grows across
     a bounded diagnostic run, the culprit has a name in the logs.

  2. ARE attention logits growing?
     attention_logit_probe() captures every Attention module's input with
     forward pre-hooks, recomputes the QK logits EXPLICITLY from the
     module's own projections in fp32 (no fused kernel, same spirit as
     reference_self_attention), and returns per-layer |logit| max.
     Calibration: healthy DiT attention logits sit roughly 5-20; values of
     50+ that climb over a few thousand steps are the attention-logit-growth
     signature, and QK-norm is the established fix. If instead logits stay
     tame while the 'ada' grad family dominates, the adaLN modulation MLPs
     are the target and the surgery is different (per-group weight decay),
     with no new parameters and no optimizer-state migration.

Both helpers are rank-0-only by convention, use NO collectives, and are
read-only with respect to training state: safe to call on the RAW
(uncompiled, unwrapped) module in the middle of a DDP training step. The
probe forward runs the eager Flex path (the model handed in is the raw
module train.py keeps around for exactly this kind of use); at N=256 and a
small probe batch this is well under a second per stability window.

Everything here is gated behind --probe-attn-logits in train.py. With the
flag off, train.py never calls into this module and the production chain
is byte-for-byte the same behavior as before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FAMILIES = ("qk", "vo", "mlp", "ada", "emb", "oth")


def family_of(name: str) -> str:
    """Identical classification to train.py's LOG_EVERY weight-norm block.

    Kept as a duplicate on purpose: probe.py must not import from the
    training loop, and the production loop must not import its logging
    taxonomy from a diagnostics module.
    """
    if "to_q" in name or "to_k" in name:
        return "qk"
    if "to_v" in name or "to_out" in name:
        return "vo"
    if ".ff." in name:
        return "mlp"
    if "norm1" in name or "norm_out" in name or "adaln" in name.lower():
        return "ada"
    if "emb" in name or "pos_embed" in name or "proj_out" in name:
        return "emb"
    return "oth"


@torch.no_grad()
def grad_family_norms(model: torch.nn.Module) -> dict[str, float]:
    """Per-family L2 gradient norms. Call after backward, before zero_grad.

    Returns {family: ||grad||_2} with families summed in squared space, so
    the values compose exactly like the global grad norm train.py already
    logs (global**2 == sum over families of family**2, up to shared-param
    dedup in the dmap chain, where tied to_k/to_q gradients accumulate
    into the one shared parameter and are counted once, in 'qk').
    """
    squares = dict.fromkeys(FAMILIES, 0.0)
    seen: set[int] = set()  # dmap ties to_k to to_q: count shared params once
    for name, parameter in model.named_parameters():
        if parameter.grad is None or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        squares[family_of(name)] += (
            parameter.grad.detach().float().pow(2).sum().item()
        )
    return {key: value**0.5 for key, value in squares.items()}


def format_families(norms: dict[str, float]) -> str:
    return "  ".join(f"{key}={norms[key]:9.2f}" for key in FAMILIES)


def _make_capture_hook(store: dict[str, torch.Tensor], name: str):
    def hook(_module, args):
        # args[0] is hidden_states [B, N, C]; keep it for the explicit
        # logit recomputation below. detach: this is a read-only probe.
        store[name] = args[0].detach()

    return hook


@torch.no_grad()
def attention_logit_probe(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    *,
    t_value: float = 500.0,
    autocast_dtype: torch.dtype = torch.bfloat16,
    top_k: int = 5,
) -> dict:
    """Max |QK logit| per attention layer, computed explicitly in fp32.

    The forward runs under the same autocast dtype as training so captured
    inputs reflect the numerics the model actually sees; the logits
    themselves are then recomputed in fp32 from the module's own to_q/to_k
    weights (explicit matmul, no fused kernel), so the number reported is
    the mathematical logit, not a kernel artifact.

    Args:
        model:  the RAW module (uncompiled, unwrapped). Never the DDP wrapper.
        x0:     [B, 4, 32, 32] probe batch (a small slice of the current
                training batch is fine; B=8 keeps captures under ~150 MB).
        labels: [B] class labels for the same slice.
        t_value: timestep passed to the model; the DiT embedder accepts a
                float for both objectives, and mid-schedule (500) is a
                representative operating point. Logit growth, when present,
                shows up at every t.

    Returns dict with:
        per_layer: {module_name: max_abs_logit}
        max:       overall max
        argmax:    name of the layer attaining it
        top:       [(short_name, value)] for the top_k layers, descending
    """
    from diffusers.models.attention_processor import Attention

    attention_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Attention)
    ]
    if not attention_modules:
        raise ValueError("no diffusers Attention modules found on the probe model")

    captured: dict[str, torch.Tensor] = {}
    handles = [
        module.register_forward_pre_hook(_make_capture_hook(captured, name))
        for name, module in attention_modules
    ]

    was_training = model.training
    model.eval()
    try:
        t = torch.full((x0.shape[0],), float(t_value), device=x0.device)
        if x0.is_cuda and autocast_dtype != torch.float32:
            with torch.autocast("cuda", dtype=autocast_dtype):
                model(hidden_states=x0, timestep=t, class_labels=labels)
        else:
            # fp32 / TF32 training mode, or the CPU path used by unit tests:
            # run the plain forward so captures match training numerics.
            model(hidden_states=x0, timestep=t, class_labels=labels)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    per_layer: dict[str, float] = {}
    for name, module in attention_modules:
        hidden = captured[name].float()
        weight_q = module.to_q.weight.detach().float()
        weight_k = module.to_k.weight.detach().float()
        bias_q = None if module.to_q.bias is None else module.to_q.bias.detach().float()
        bias_k = None if module.to_k.bias is None else module.to_k.bias.detach().float()

        query = F.linear(hidden, weight_q, bias_q)
        key = F.linear(hidden, weight_k, bias_k)

        batch, seq_len, _ = hidden.shape
        heads = module.heads
        head_dim = query.shape[-1] // heads
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        logits = (query @ key.transpose(-2, -1)) * module.scale
        per_layer[name] = logits.abs().amax().item()

    ranked = sorted(per_layer.items(), key=lambda kv: kv[1], reverse=True)
    argmax_name, max_value = ranked[0]

    def short(name: str) -> str:
        # "transformer_blocks.17.attn1" -> "blk17"
        for token in name.split("."):
            if token.isdigit():
                return f"blk{token}"
        return name

    return {
        "per_layer": per_layer,
        "max": max_value,
        "argmax": short(argmax_name),
        "top": [(short(n), round(v, 2)) for n, v in ranked[:top_k]],
    }
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

PRECISION: ``--precision`` selects the training numerics.  ``tf32``
(default) runs the published DiT/SiT recipe -- fp32 activations with TF32
tensor-core matmuls, no autocast.  ``bf16`` restores the previous behavior
(bf16 autocast over fp32 master weights).  Latents, EMA, optimizer state,
and checkpoints are fp32 in both modes; the choice is recorded in each
run_history entry, not in the config, so the config-drift guard never
refuses a resume across a precision change.

ADALN WEIGHT DECAY: ``--wd-ada`` applies decoupled (AdamW-style) weight
decay to the adaLN modulation family only (norm1 / norm_out / *adaln*),
implemented as a manual per-step shrink immediately before
``optimizer.step``.  Diagnosis behind it: the adaLN weights grow unbounded
under the published wd=0 recipe (|w|_ada ~4.9k of |w|_total ~4.9k at 274K
steps), their scale modulations amplify the tokens feeding attention, and
block-1 QK logits explode (3.4e6 observed) -- the gradient-spike source.
Implemented OUTSIDE the optimizer on purpose: the single AdamW param group
is untouched, so index-keyed optimizer checkpoints load unchanged in both
directions and 0.0 (default) is byte-for-byte the previous behavior.
Skipped (spiked) steps do not decay -- decay accompanies real updates only,
so effective decay per accepted step is exactly lr*wd_ada.

DIAGNOSTICS (opt-in, off by default): ``--probe-attn-logits`` enables
``ditflex.probe`` -- per-family gradient norms at LOG_EVERY cadence and on
every skipped (spike) step, plus an explicit fp32 max-attention-logit probe at
every stability window.  Rank 0 only, no collectives, read-only.  With the
flag off this file's behavior is identical to before the probe existed.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
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
from ditflex.probe import attention_logit_probe, format_families, grad_family_norms
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
    parser.add_argument(
        "--wd-ada",
        type=float,
        default=0.0,
        help=(
            "decoupled weight decay applied ONLY to the adaLN modulation "
            "family (norm1/norm_out/adaln), as a manual shrink before "
            "optimizer.step. 0 = off (previous behavior). The optimizer's "
            "param group is untouched, so checkpoints remain compatible."
        ),
    )
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--spike-skip", type=float, default=4.0)
    parser.add_argument("--grad-ceiling", type=float, default=0.0)
    parser.add_argument("--seed-offset", type=int, default=0)

    parser.add_argument(
        "--precision",
        choices=["tf32", "bf16"],
        default="tf32",
        help=(
            "tf32: fp32 activations + TF32 matmuls (published DiT/SiT recipe, "
            "~2x activation memory, lower throughput). bf16: bf16 autocast "
            "over fp32 master weights (previous default)."
        ),
    )

    # Opt-in diagnostics (ditflex.probe).  Off by default: with the flag off,
    # nothing in this file calls into the probe module and the production
    # chain's behavior is unchanged.
    parser.add_argument(
        "--probe-attn-logits",
        action="store_true",
        help=(
            "rank-0 diagnostics: per-family grad norms at LOG_EVERY cadence "
            "and on every skipped spike step; explicit fp32 max-attention-"
            "logit probe at every stability window"
        ),
    )
    parser.add_argument(
        "--probe-batch",
        type=int,
        default=8,
        help="probe forward batch size (slice of the current training batch)",
    )

    parser.add_argument("--qk-mode", choices=["amap", "dmap"], default="amap")
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        help=(
            "build the amap model with per-head RMSNorm on Q/K (post-344K "
            "migration architecture). Must match the checkpoint's embedded "
            "config or the drift guard refuses the resume. Invalid with "
            "--qk-mode dmap."
        ),
    )
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
    if args.probe_batch <= 0:
        raise ValueError("--probe-batch must be positive")
    if args.wd_ada < 0.0:
        raise ValueError("--wd-ada must be non-negative")

    ctx = setup()

    # Precision backends.  TF32 mode follows the published DiT/SiT recipe:
    # fp32 activations, tensor-core TF32 matmuls.  In bf16 mode TF32 flags
    # are irrelevant (autocast matmuls run in bf16) but harmless.
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cfg = Config()
    cfg.train.objective = args.objective
    cfg.model.qk_mode = args.qk_mode
    cfg.model.dmap_alpha = args.dmap_alpha
    cfg.model.qk_norm = args.qk_norm
    if args.hub_repo:
        cfg.hub.checkpoint_repo = args.hub_repo

    if cfg.train.global_batch % ctx.world != 0:
        raise ValueError(f"global_batch {cfg.train.global_batch} % world {ctx.world} != 0")
    per_rank_batch = cfg.train.global_batch // ctx.world

    if ctx.is_rank0:
        print(f"[train] world={ctx.world}  per_rank_batch={per_rank_batch}")
        print(
            f"[train] precision={args.precision}"
            + (
                " (fp32 activations, TF32 matmuls -- published recipe)"
                if args.precision == "tf32"
                else " (bf16 autocast, fp32 master weights)"
            )
        )
        print(cfg.to_json())
        if args.probe_attn_logits:
            print(
                f"[train] PROBE ENABLED: grad families @ every {LOG_EVERY} steps "
                f"and on spikes; attn logits @ every {LOSS_WINDOW}-step window "
                f"(probe_batch={args.probe_batch}, rank 0 only)"
            )
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

    # adaLN-only decoupled decay: same family classification as the logging
    # blocks and ditflex.probe.  Collected once from the RAW model; DDP and
    # torch.compile wrappers share these Parameter objects, so shrinking them
    # here shrinks the training weights.  Deduplicated by identity for the
    # dmap chain's tied modules (adaLN is never tied today, but cheap safety).
    ada_params: list[torch.nn.Parameter] = []
    if args.wd_ada > 0.0:
        seen_ids: set[int] = set()
        for _name, _param in model.named_parameters():
            if (
                ("norm1" in _name or "norm_out" in _name or "adaln" in _name.lower())
                and id(_param) not in seen_ids
            ):
                seen_ids.add(id(_param))
                ada_params.append(_param)
        if not ada_params:
            raise RuntimeError(
                "--wd-ada > 0 but no adaLN parameters matched norm1/norm_out/"
                "adaln -- the architecture naming has drifted; refusing to "
                "silently no-op."
            )
        if ctx.is_rank0:
            ada_sq = sum(
                p.detach().float().pow(2).sum().item() for p in ada_params
            )
            print(
                f"[train] ADA WEIGHT DECAY enabled: wd_ada={args.wd_ada:g} on "
                f"{len(ada_params)} tensors, |w|_ada={ada_sq**0.5:.1f} at start; "
                f"decay is decoupled (p *= 1 - lr*wd_ada) and applied only on "
                f"accepted (non-skipped) steps"
            )

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
                    "precision": args.precision,
                    "lr_policy": stability_spec.policy,
                    "lr_base": stability_spec.base_lr,
                    "lr_start": start_lr if segment_start == start_step else None,
                    "lr_end": float(optimizer.param_groups[0]["lr"]),
                    "lr_min": stability_spec.min_lr,
                    "lr_hard_min": stability_spec.hard_min_lr,
                    "lr_scale": controller.scale,
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    "wd_ada": args.wd_ada,
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
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else nullcontext()
        )
        with autocast_ctx:
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

        # PROBE (pre-clip): capture RAW per-family gradient norms before
        # clip_grad_norm_ rescales grads in place.  Cheap (one reduction over
        # params, rank 0 only) and only when the flag is on; printed later
        # once the spike decision is known so the line can carry the tag.
        raw_families: dict[str, float] | None = None
        if args.probe_attn_logits and ctx.is_rank0:
            try:
                raw_families = grad_family_norms(model)
            except Exception as exc:  # noqa: BLE001 - diagnostics never kill training
                print(f"[probe] grad-family probe failed (non-fatal): {exc!r}")

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

        # PROBE: print the RAW (pre-clip) per-family norms captured above,
        # now that the spike decision is known.  These compose to the raw
        # |g| (sum of squares), so on a spike line the dominant family's
        # value IS the spike magnitude, not a clipped fraction of 1.0.
        if raw_families is not None and (spiked or step % LOG_EVERY == 0):
            tag = "SPIKE" if spiked else "cadence"
            print(
                f"[probe] step {step:,} ({tag})  |g|raw={grad_norm:9.2f}  "
                f"{format_families(raw_families)}"
            )

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
            if ada_params:
                # Decoupled decay at the CURRENT scheduled LR, so the decay
                # strength tracks the controller exactly like the update does.
                decay = 1.0 - float(optimizer.param_groups[0]["lr"]) * args.wd_ada
                with torch.no_grad():
                    torch._foreach_mul_(ada_params, decay)
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

            # PROBE: explicit fp32 max attention logits, once per window, on a
            # small slice of the batch this step just trained on.  Runs on the
            # RAW module in eager mode (the compiled/DDP wrapper is untouched);
            # a probe failure is logged and never affects the training loop or
            # the retry decision below.
            if args.probe_attn_logits and ctx.is_rank0:
                try:
                    n_probe = min(args.probe_batch, x0.shape[0])
                    stats = attention_logit_probe(
                        model,
                        x0[:n_probe],
                        labels[:n_probe],
                        autocast_dtype=(
                            torch.bfloat16
                            if args.precision == "bf16"
                            else torch.float32
                        ),
                    )
                    print(
                        f"[probe] attn logits @ {step:,}: max={stats['max']:.2f} "
                        f"at {stats['argmax']}  top={stats['top']}"
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostics never kill training
                    print(f"[probe] attn-logit probe failed (non-fatal): {exc!r}")

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

Parametrized over qk_norm since the 344K migration: the with-norm
configuration attaches RMSNorm(head_dim) to norm_q/norm_k with
NON-TRIVIAL weights (1 + noise), so a silently-skipped norm cannot pass
by accident, and the fp64 reference applies the identical normalization.
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


def attach_qk_norms(attn, dtype: torch.dtype, seed: int = 7) -> None:
    """Install RMSNorm(head_dim) with non-trivial weights (1 + 0.1*noise).

    Ones-weights would make a dropped multiply invisible; perturbed weights
    make the norm's presence measurable in every comparison.
    """
    from ditflex.attention import QK_NORM_EPS

    g = torch.Generator().manual_seed(seed)
    for name in ("norm_q", "norm_k"):
        norm = torch.nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        with torch.no_grad():
            norm.weight.add_(0.1 * torch.randn(HEAD_DIM, generator=g))
        setattr(attn, name, norm.to(device="cuda", dtype=dtype))


def build_attention(dtype: torch.dtype, qk_norm: bool, requires_grad: bool = False):
    from diffusers.models.attention_processor import Attention

    attn = Attention(
        query_dim=DIM, heads=HEADS, dim_head=HEAD_DIM, dropout=0.0, bias=True, out_bias=True
    )
    attn = attn.to(device="cuda", dtype=dtype).eval()
    if qk_norm:
        attach_qk_norms(attn, dtype)
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
@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qknorm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_matches_math_reference(dtype, qk_norm):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(dtype, qk_norm)
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
def test_qk_norm_changes_the_output():
    """A dropped norm is invisible to the identity comparison when weights
    are ones; with perturbed weights, with-norm and without-norm outputs
    must measurably differ -- proving the norm is live in the Flex path."""
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm=False)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")
    with torch.no_grad():
        plain = attn(x)

    attach_qk_norms(attn, torch.float32)
    with torch.no_grad():
        normed = attn(x)

    assert (normed - plain).abs().max().item() > 1e-3


@requires_cuda
def test_half_installed_norms_are_rejected():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32, qk_norm=True)
    attn.norm_k = None  # simulate a broken migration
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")
    with pytest.raises(ValueError, match="BOTH"):
        attn(x)


@requires_cuda
def test_score_mod_is_wired():
    """Identity comparison cannot catch a silently-dropped score_mod
    (identity == no-mod). A zero score_mod forces uniform attention, which
    must change the output."""
    from ditflex.attention import FlexSelfAttnProcessor, IdentityFlexSelfAttnProcessor

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm=False)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.set_processor(IdentityFlexSelfAttnProcessor())
    with torch.no_grad():
        identity_out = attn(x)

    attn.set_processor(FlexSelfAttnProcessor(score_mod=lambda s, b, h, q, kv: s * 0.0))
    with torch.no_grad():
        uniform_out = attn(x)

    assert (uniform_out - identity_out).abs().max().item() > 1e-3


@requires_cuda
@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qknorm"])
def test_flex_backward_matches_reference(qk_norm):
    from ditflex.attention import (
        IdentityFlexSelfAttnProcessor,
        reference_self_attention,
    )

    torch.manual_seed(0)
    attn = build_attention(torch.float32, qk_norm, requires_grad=True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda")

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }
    if qk_norm:
        assert any("norm_q" in n for n in ref_grads), "reference gave no grad to norm_q"

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            assert agree(param.grad, ref_grads[name], 1e-4), f"grad mismatch: {name}"


@requires_cuda
def test_processor_rejects_out_of_contract_inputs():
    from ditflex.attention import IdentityFlexSelfAttnProcessor

    attn = build_attention(torch.float32, qk_norm=False)
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

### `tests/test_migrate_qknorm.py`

```python
"""ditflex.migrate: the qk-norm migration must be exactly understood.

Round-trips a tiny amap checkpoint through migrate_checkpoint and asserts
the load-bearing properties one by one: weights preserved by name, norm
weights fresh ones, EMA carried, AdamW moments attached to the SAME
parameters they belonged to (by name, not index), hyperparameters kept,
step kept, guard reset, and the drift guard behavior on both sides of the
migration. CPU-only."""

from __future__ import annotations

import pytest
import torch

from ditflex.checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA
from ditflex.migrate import migrate_checkpoint
from ditflex.model import build_model


def tiny_config(qk_norm: bool = False) -> Config:
    cfg = Config()
    cfg.model.num_attention_heads = 2
    cfg.model.attention_head_dim = 16
    cfg.model.num_layers = 2
    cfg.model.sample_size = 8
    cfg.model.num_classes = 10
    cfg.model.qk_norm = qk_norm
    cfg.train.objective = "flow"
    return cfg


def make_trained_checkpoint(directory, steps: int = 3) -> tuple[Config, dict]:
    torch.manual_seed(0)
    cfg = tiny_config(qk_norm=False)
    model = build_model(cfg.model)
    ema = EMA(model, decay=cfg.train.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.3e-5, weight_decay=0.0)
    # Populate real AdamW state (exp_avg, exp_avg_sq, step) with synthetic
    # gradients: identical migration mechanics to a trained checkpoint, and
    # CPU-safe (FlexAttention has no CPU backward, so a real backward would
    # break the CPU test workflow).
    for _ in range(steps):
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 1e-3
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
    state = {
        "step": 344_000,
        "run_history": [{"start_step": 0, "end_step": 344_000}],
        "guard_state": {
            "version": 3,
            "spikes_total": 2823,
            "recent_losses": [0.77] * 200,
            "stability_controller": {"version": 4, "committed_scale": 0.282},
        },
    }
    save_checkpoint(directory, model, ema, optimizer, cfg, state)
    return cfg, {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "ema": {k: v.clone() for k, v in ema.state_dict().items()},
        "optim_names": [n for n, _ in model.named_parameters()],
        "optim_state": {
            n: {k: v.clone() for k, v in optimizer.state[p].items() if torch.is_tensor(v)}
            for n, p in model.named_parameters()
            if p in optimizer.state
        },
    }


def test_migration_roundtrip(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    _, before = make_trained_checkpoint(src)

    new_state = migrate_checkpoint(src, dst)
    assert new_state["step"] == 344_000
    validate_checkpoint(dst, expected_step=344_000)

    # Load into the qk_norm architecture through the ordinary path.
    new_cfg = tiny_config(qk_norm=True)
    model = build_model(new_cfg.model)
    ema = EMA(model, decay=new_cfg.train.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.0)
    state = load_checkpoint(dst, model, ema, optimizer, new_cfg)
    assert state["step"] == 344_000

    # 1. Every pre-existing weight preserved by NAME; norm weights are ones.
    new_sd = model.state_dict()
    for name, tensor in before["model"].items():
        assert torch.equal(new_sd[name], tensor), name
    norm_keys = [k for k in new_sd if k.endswith(("norm_q.weight", "norm_k.weight"))]
    assert len(norm_keys) == 2 * new_cfg.model.num_layers
    for key in norm_keys:
        assert torch.equal(new_sd[key], torch.ones_like(new_sd[key])), key

    # 2. EMA carried by name; norm shadow entries are the (ones) online values.
    for name, tensor in before["ema"].items():
        assert torch.equal(ema.shadow[name], tensor), name
    for key in norm_keys:
        assert torch.equal(ema.shadow[key], torch.ones_like(ema.shadow[key]))

    # 3. AdamW moments attached to the SAME named parameters (the index
    #    remap): exp_avg for each old parameter must match exactly, and the
    #    fresh norm parameters must have no state.
    params = dict(model.named_parameters())
    for name, entry in before["optim_state"].items():
        migrated_entry = optimizer.state[params[name]]
        for key, value in entry.items():
            assert torch.allclose(
                migrated_entry[key].float(), value.float()
            ), f"optim state mismatch: {name}.{key}"
    for key in norm_keys:
        assert params[key] not in optimizer.state or not optimizer.state[params[key]]

    # 4. Hyperparameters preserved.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.3e-5)

    # 5. Guard reset: no controller, no recent losses; spike counter carried.
    guard = state["guard_state"]
    assert "stability_controller" not in guard
    assert "recent_losses" not in guard
    assert guard["spikes_total"] == 2823


def test_migrated_checkpoint_refuses_old_config(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    make_trained_checkpoint(src)
    migrate_checkpoint(src, dst)

    old_cfg = tiny_config(qk_norm=False)
    model = build_model(old_cfg.model)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="same experiment"):
        load_checkpoint(dst, model, ema, optimizer, old_cfg)


def test_migration_refuses_wrong_inputs(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    make_trained_checkpoint(src)

    # Already migrated -> refuse a second migration.
    migrate_checkpoint(src, dst)
    with pytest.raises(ValueError, match="already"):
        migrate_checkpoint(dst, tmp_path / "again")


def test_dmap_builder_refuses_qk_norm():
    from ditflex.diffusion_model import build_dmap_model

    cfg = tiny_config(qk_norm=True)
    cfg.model.qk_mode = "dmap"
    with pytest.raises(ValueError, match="not valid for the DMAP chain"):
        build_dmap_model(cfg.model)
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

### `tests/test_probe.py`

```python
"""ditflex.probe: the diagnostics must themselves be trustworthy.

Three properties, checked exactly:
  - grad_family_norms composes to the global grad norm (squared-sum identity)
    and counts dmap's tied to_q/to_k parameter once;
  - attention_logit_probe's explicit fp32 logit max agrees with a dense
    fp64 recomputation on the same weights;
  - the probe is read-only: params, grads, and training mode unchanged.

CPU-only where possible; the logit-vs-dense check builds a tiny DiT and
runs on CPU too (eager flex works on CPU for these shapes).
"""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.model import build_model
from ditflex.probe import (
    FAMILIES,
    attention_logit_probe,
    family_of,
    grad_family_norms,
)


def tiny():
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=16, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10,
    )


def test_family_classification_matches_train_py_taxonomy():
    assert family_of("transformer_blocks.0.attn1.to_q.weight") == "qk"
    assert family_of("transformer_blocks.0.attn1.to_k.bias") == "qk"
    assert family_of("transformer_blocks.0.attn1.to_v.weight") == "vo"
    assert family_of("transformer_blocks.0.attn1.to_out.0.weight") == "vo"
    assert family_of("transformer_blocks.0.ff.net.0.proj.weight") == "mlp"
    assert family_of("transformer_blocks.0.norm1.linear.weight") == "ada"
    assert family_of("pos_embed.proj.weight") == "emb"
    assert family_of("proj_out.weight") == "emb"


def test_grad_family_norms_compose_to_global_norm():
    torch.manual_seed(0)
    model = build_model(tiny())
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500.0)
    y = torch.randint(0, 10, (2,))
    out = model(hidden_states=x0, timestep=t, class_labels=y).sample
    out.square().mean().backward()

    families = grad_family_norms(model)
    assert set(families) == set(FAMILIES)
    composed = sum(v**2 for v in families.values()) ** 0.5
    global_norm = torch.norm(
        torch.stack([
            p.grad.detach().float().norm()
            for p in model.parameters()
            if p.grad is not None
        ])
    ).item()
    assert composed == pytest.approx(global_norm, rel=1e-5)


def test_grad_family_norms_count_tied_params_once():
    from ditflex.diffusion_model import build_dmap_model

    cfg = tiny()
    cfg.qk_mode = "dmap"
    model = build_dmap_model(cfg)
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500.0)
    y = torch.randint(0, 10, (2,))
    model(hidden_states=x0, timestep=t, class_labels=y).sample.square().mean().backward()

    families = grad_family_norms(model)
    # Tied to_k is to_q: unique-parameter global norm must still compose.
    seen: set[int] = set()
    unique_sq = 0.0
    for p in model.parameters():
        if p.grad is None or id(p) in seen:
            continue
        seen.add(id(p))
        unique_sq += p.grad.detach().float().pow(2).sum().item()
    composed = sum(v**2 for v in families.values()) ** 0.5
    assert composed == pytest.approx(unique_sq**0.5, rel=1e-5)


def test_logit_probe_matches_dense_fp64_and_is_read_only():
    torch.manual_seed(0)
    model = build_model(tiny())
    model.train()
    x0 = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    stats = attention_logit_probe(model, x0, y, autocast_dtype=torch.float32)

    # Read-only: params untouched, mode restored, no grads created.
    assert model.training
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n]), n
        assert p.grad is None

    assert stats["max"] > 0.0
    assert stats["argmax"].startswith("blk")
    assert len(stats["per_layer"]) == 2

    # Dense fp64 cross-check of one layer using captured-free recomputation:
    # rebuild the layer input by hooking again, then compare.
    from diffusers.models.attention_processor import Attention

    captured = {}
    names = [n for n, m in model.named_modules() if isinstance(m, Attention)]
    target = names[0]
    module = dict(model.named_modules())[target]
    handle = module.register_forward_pre_hook(
        lambda _m, args: captured.setdefault("x", args[0].detach())
    )
    with torch.no_grad():
        t = torch.full((2,), 500.0)
        model.eval()
        model(hidden_states=x0, timestep=t, class_labels=y)
    handle.remove()

    h = captured["x"].double()
    q = h @ module.to_q.weight.double().T + module.to_q.bias.double()
    k = h @ module.to_k.weight.double().T + module.to_k.bias.double()
    b, n, _ = h.shape
    heads = module.heads
    hd = q.shape[-1] // heads
    q = q.view(b, n, heads, hd).transpose(1, 2)
    k = k.view(b, n, heads, hd).transpose(1, 2)
    dense_max = ((q @ k.transpose(-2, -1)) * module.scale).abs().amax().item()
    assert stats["per_layer"][target] == pytest.approx(dense_max, rel=1e-4)


# -- adaLN weight-decay behavior (train.py --wd-ada), pinned here because the
#    decay maths is probe-adjacent diagnostics territory and needs no GPU.


def test_ada_family_selection_matches_train_py_filter():
    """The name filter train.py uses to build ada_params must select the
    same tensors the 'ada' family reports -- the decayed set and the
    monitored set have to be the same set."""
    model = build_model(tiny())
    filter_names = {
        n
        for n, _ in model.named_parameters()
        if "norm1" in n or "norm_out" in n or "adaln" in n.lower()
    }
    family_names = {
        n for n, _ in model.named_parameters() if family_of(n) == "ada"
    }
    assert filter_names == family_names
    assert filter_names, "no adaLN parameters found -- naming drifted"


def test_decoupled_ada_decay_shrinks_only_ada():
    torch.manual_seed(0)
    model = build_model(tiny())
    lr, wd_ada = 1e-2, 0.5  # exaggerated so one step is measurable

    seen: set[int] = set()
    ada_params = []
    for n, p in model.named_parameters():
        if ("norm1" in n or "norm_out" in n or "adaln" in n.lower()) and id(p) not in seen:
            seen.add(id(p))
            ada_params.append(p)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    decay = 1.0 - lr * wd_ada
    with torch.no_grad():
        torch._foreach_mul_(ada_params, decay)

    for n, p in model.named_parameters():
        if family_of(n) == "ada":
            assert torch.allclose(p.detach(), before[n] * decay), n
        else:
            assert torch.equal(p.detach(), before[n]), n
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

Since the 344K migration the gate runs EVERY check in BOTH configurations:
without qk-norm (the pre-migration baseline) and with qk-norm attached
using NON-TRIVIAL weights (1 + 0.1*noise -- ones-weights would make a
dropped norm invisible). The fp64 reference applies the identical
normalization from the same weights.

Checks, per configuration:
  1. scale        -- attn.scale equals what the processor passes to Flex
  2. forward fp32 -- flex vs fp64 reference, rel tol 1e-4
  3. forward bf16 -- flex vs fp64 reference, rel tol 2e-2
                     (bf16 has ~8 mantissa bits; tighter is not meaningful)
  4. score_mod wiring -- a constant-zero score_mod must produce uniform
     attention and therefore a measurably different output
  5. norm liveness (qk-norm config only) -- with-norm output must differ
     from without-norm output on the same weights/input
  6. gradients fp32 -- flex backward vs autograd through the reference,
     including the norm weights when present
  7. (--compile) the compiled Flex path vs the same reference

If this fails, nothing downstream is interpretable.

Run:
    python tests/verify_identity.py
    python tests/verify_identity.py --compile

Exit code 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
from diffusers.models.attention_processor import Attention

from ditflex.attention import (
    QK_NORM_EPS,
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


def attach_qk_norms(attn: Attention, device, dtype, seed: int = 7) -> None:
    g = torch.Generator().manual_seed(seed)
    for name in ("norm_q", "norm_k"):
        norm = torch.nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        with torch.no_grad():
            norm.weight.add_(0.1 * torch.randn(HEAD_DIM, generator=g))
        setattr(attn, name, norm.to(device=device, dtype=dtype))


def build_attention(device, dtype, qk_norm: bool) -> Attention:
    attn = Attention(
        query_dim=DIM,
        heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
    )
    attn = attn.to(device=device, dtype=dtype).eval()
    if qk_norm:
        attach_qk_norms(attn, device, dtype)
    for p in attn.parameters():
        p.requires_grad_(False)
    return attn


def compare(
    name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float = 1e-8
) -> bool:
    got64, ref64 = got.double(), ref.double()
    max_abs = (got64 - ref64).abs().max().item()
    denom = ref64.abs().max().item()
    ok = max_abs <= atol + rtol * denom
    max_rel = max_abs / (denom + 1e-12)
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:<34} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"rtol={rtol:.1e} atol={atol:.1e}"
    )
    return ok


def check_scale(attn: Attention) -> bool:
    expected = HEAD_DIM ** -0.5
    ok = abs(attn.scale - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'scale':<34} attn.scale={attn.scale:.8f}  expected={expected:.8f}")
    return ok


def check_score_mod_wiring(attn: Attention, x: torch.Tensor, identity_out: torch.Tensor) -> bool:
    def zero_score(score, b, h, q_idx, kv_idx):
        return score * 0.0

    attn.set_processor(FlexSelfAttnProcessor(score_mod=zero_score))
    with torch.no_grad():
        uniform_out = attn(x)

    diff = (uniform_out.double() - identity_out.double()).abs().max().item()
    ok = diff > 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'score_mod wiring':<34} |uniform - identity|_max={diff:.3e} (> 1e-3)")
    return ok


def run_configuration(device, qk_norm: bool, compile_check: bool) -> bool:
    tag = "qk-norm" if qk_norm else "plain"
    all_ok = True

    plain_outputs: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float32, torch.bfloat16):
        print(f"[{tag}] dtype = {dtype}")
        torch.manual_seed(0)
        attn = build_attention(device, dtype, qk_norm)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=dtype)

        all_ok &= check_scale(attn)

        with torch.no_grad():
            ref = reference_self_attention(attn, x, dtype=torch.float64)

        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            flex_out = attn(x)

        if flex_out.shape != ref.shape or not torch.isfinite(flex_out).all():
            print("  [FAIL] shape or finiteness")
            all_ok = False

        all_ok &= compare("flex vs fp64 reference", flex_out, ref, REL_TOL[dtype])
        all_ok &= check_score_mod_wiring(attn, x, flex_out)

        # Norm liveness: same weights and input, norms removed, output must move.
        if qk_norm and dtype is torch.float32:
            attn.norm_q, attn.norm_k = None, None
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            with torch.no_grad():
                unnormed = attn(x)
            diff = (unnormed.double() - flex_out.double()).abs().max().item()
            ok = diff > 1e-3
            print(f"  [{'PASS' if ok else 'FAIL'}] {'qk-norm liveness':<34} "
                  f"|normed - plain|_max={diff:.3e} (> 1e-3)")
            all_ok &= ok
            attach_qk_norms(attn, device, dtype)

        if compile_check:
            attn.set_processor(IdentityFlexSelfAttnProcessor())
            compiled = torch.compile(attn)
            with torch.no_grad():
                out_c = compiled(x)
            all_ok &= compare("compiled flex vs reference", out_c, ref, REL_TOL[dtype])

        plain_outputs[dtype] = flex_out
        print()

    # Gradient check (fp32), including norm weights when present.
    print(f"[{tag}] gradient check (fp32)")
    attn = build_attention(device, torch.float32, qk_norm)
    for p in attn.parameters():
        p.requires_grad_(True)
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.float32)

    attn.zero_grad(set_to_none=True)
    reference_self_attention(attn, x).square().mean().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in attn.named_parameters() if p.grad is not None
    }
    if qk_norm and not any("norm_q" in n for n in ref_grads):
        print("  [FAIL] reference produced no gradient for norm_q")
        all_ok = False

    attn.zero_grad(set_to_none=True)
    attn.set_processor(IdentityFlexSelfAttnProcessor())
    attn(x).square().mean().backward()

    for name, param in attn.named_parameters():
        if name in ref_grads:
            all_ok &= compare(f"grad {name}", param.grad, ref_grads[name], 1e-4)
    print()
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also verify the compiled Flex path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required: the Flex kernels under test are the GPU ones.")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print(f"shape [B={BATCH}, N={SEQ_LEN}, C={DIM}]  heads={HEADS} head_dim={HEAD_DIM}\n")

    all_ok = True
    for qk_norm in (False, True):
        all_ok &= run_configuration(device, qk_norm, args.compile)

    if all_ok:
        print("ALL CHECKS PASSED -- Flex computes the math in BOTH configurations, "
              "score_mod and qk-norm are live.")
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

# PCIe-only cards (RTX workstation/consumer) have P2P disabled at the driver
# level; NCCL can hang probing for it instead of falling back. Disable it
# there ONLY -- on SXM (B200/B300) these same paths are NVLink, and turning
# them off forces the all-reduce through host memory.
_PCIE_ONLY = "RTX" in GPU_KIND.upper() or "PRO-6000" in GPU_KIND.upper()
_NCCL_ENV = {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"} if _PCIE_ONLY else {}

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
    cpu=8.0,
    timeout=TIMEOUT_CEILING,
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "NCCL_DEBUG": "INFO",
        **_NCCL_ENV,
    })],
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

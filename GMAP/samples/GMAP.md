# GMAP — Girsanov-MAP: frozen coupling-plane scans on SiT

**No training.** GMAP takes one set of SiT-XL/2 weights, replaces every
attention forward with a two-coupling operator, and measures how generation
responds as the couplings are dialed away from the trained operating point.
It is the vision-side twin of the frozen anti-attention scan we ran on
nanochat-d32 for the ICL project, where the same protocol produced a master
curve in the relative directed coupling with a critical threshold for
in-context copying.

## The idea

Every attention score matrix splits into a symmetric and an antisymmetric
part. In the operator-decomposition language of this project (AMAP / DMAP /
HMAP / FMAP), these are two different computational sectors:

- **symmetric sector** ("metric", electric-like): similarity / aggregation.
  For standard attention it is the indefinite form `1/2(qk^T + kq^T)`; for
  AMAP it is the PSD Gram `1/2 <m_i, m_j>`, `m = (q+k)/sqrt(2)`.
- **antisymmetric sector** ("flux", magnetic-like): directed transport,
  `1/2(qk^T - kq^T)`. Identical in both variants.

GMAP scales the two sectors independently:

    logits = ( c_sym * sym  +  c_flux * asym ) * scale

`(c_sym, c_flux) = (1, 1)` reproduces the base operator **exactly** (the
harness asserts this against the released forward, up to eager-vs-SDPA
kernel rounding, ~3e-4 under tf32). Because the couplings live in one shared
config object across all 28 attention blocks, a scan mutates two floats
between evaluations — no re-patching, no weight edits, state_dict untouched.

The name: annealing `c_flux -> 0` removes the directed (drift) component of
the kernel while keeping the symmetric medium — the discrete analogue of
inverting a Girsanov change of measure back to a reversible reference
process. SiT attention is bidirectional (no causal mask), so unlike the
language-model version this is a *literal* detailed-balance statement: at
`c_flux = 0` the kernel is symmetric and the row-normalised operator is
reversible. The "anti-attention" path of the ICL project,

    s  ->  s + (1 - xi) * s^T  =  (2 - xi) * sym + xi * asym,

is the diagonal ray of this plane: progressively multiplying in the reverse
kernel cools the metric while annealing away direction, landing at `xi = 0`
on the pure symmetrized kernel at doubled coupling.

## The scan

Four rays through the `(c_sym, c_flux)` plane, all passing through the
baseline `(1, 1)`:

| ray       | points        | meaning                                            |
|-----------|---------------|----------------------------------------------------|
| xi        | `(2 - t, t)`  | anti-attention diagonal (metric heats, flux anneals) |
| fluxcut   | `(1, t)`      | pinned metric, flux -> 0 — the frozen coexact dial |
| symheat   | `(c, 1)`      | pinned flux, metric coupling 1 -> 2                |
| fluxboost | `(1, t > 1)`  | flux over-driven past the trained coupling         |

At every point (`xi_sit.py`):

1. a **sample grid** with the project-wide fixed classes / noise seed
   (`amap_common.FIXED_CLASSES` / `FIXED_SEED`) — every grid in this repo is
   pixel-comparable to every other, including the AMAP/sample0 panels;
2. a **paired frozen flow-matching loss** — the same deterministic latent
   batches and pinned time/noise draws at every point, so the FM column
   differs only through the operator.

Two base arms select whose plane is being scanned:

- `--base standard` : released 7M SiT-XL/2, `variant="standard"` —
  `(1,1)` **is** the released model;
- `--base amap`     : an AMAP checkpoint from `--ckpt-repo`
  (`--ckpt-step latest|<int>|base`, `--weights ema|model`),
  `variant="amap"` — `(1,1)` is the AMAP operator on those weights.
  `--ckpt-step base` probes the *released* weights under the AMAP operator
  (the graft moment, no finetuning anywhere).

## Results so far (spacing 0.25, 19 points/arm, B200, tf32)

**Standard plane** (released 7M weights; eyeball labels True = realistic,
False = noise, LSD = structured-but-dreamy):

- `(1,1)` FM 0.729. **fluxcut**: realistic down to `c_flux = 0.5`
  (FM 0.746), noise at 0.25 and 0 — the antisymmetric sector is
  load-bearing for generation in the *vanilla* model.
- **fluxboost** to `c_flux = 2` costs only +0.021 FM, grids stay realistic:
  the loss landscape is *soft along the flux direction*.
- **symheat** to `c_sym = 2` costs +0.175 FM and produces the LSD regime
  (structure preserved, textures homogenized/dreamlike) — *stiff along the
  metric direction*, ~8x stiffer than flux at matched displacement.
- Failure modes are visually distinct at similar FM: flux loss -> noise
  (directed transport lost), metric heating -> LSD (over-aggregation /
  clustering made visible). The FM loss is blind to a distinction the
  samples display.

**AMAP plane, 40k-step finetuned checkpoint** (both EMA and raw `model`
weights — near-identical surfaces, every point within ~0.015, so the
profile is a property of the basin, not the averaging):

- `(1,1)` FM 0.737. The plateau is *narrower in both directions*:
  fluxcut@0.5 is already noise (FM 1.106–1.127), symheat is noise by 1.5.
  Anisotropy amplified: fluxboost to 2 costs +0.02, symheat to 2 costs
  +0.79.
- Reading (trajectory claim, robust): **40k steps under the PSD metric
  visibly migrate computational load onto the flux** — trained under an
  always-positive similarity kernel, the model offloads less discrimination
  onto the symmetric sector and leans harder on circulation. Exact parallel
  to the language-side AMAP profile.
- The symheat failure *type* also differs: standard -> LSD (clustering),
  AMAP -> noise (PSD diagonal `1/2||m_i||^2` amplifies self-attention lock:
  no mixing rather than over-mixing).

**Fairness caveat (important).** The standard plane is a 7M-step model; the
AMAP plane is 7M + 40k finetune steps — 0.6% of the budget under the new
operator. Robust across this asymmetry: the *sign* structure (flux
soft / metric stiff; load migration onto flux). Confounded by it: all
cross-plane *magnitudes* (plateau widths, the 8x-vs-38x anisotropy ratio,
the xi=0 endpoint comparison). See the pinned controls.

## What this is for

The ICL paper takes exactly one figure and two paragraphs from here, framed
as an out-of-domain check: the **standard-plane** panel (baseline grid,
fluxcut noise, symheat LSD, FM-vs-coupling curves) supporting "the sector
localization is not a language artifact" — vanilla SiT, no exotic operator,
no training asymmetry. Nothing about the AMAP plane's magnitudes, nothing
calibrated. Everything else below is companion-paper (DAC) material.

## Pinned follow-ups (deliberate scope boundary — decided, not drifted)

1. `--base amap --ckpt-step base`: released weights under the AMAP operator.
   Zero training asymmetry (identical weights on both planes) — rescues the
   `xi = 0` PSD-vs-indefinite symmetrization comparison and quantifies the
   graft-moment FM cost. **NOTE: still un-run** — the second AMAP-plane scan
   accidentally re-probed step_0040000 with `weights=model` (which is what
   produced the free EMA-vs-raw replication above).
2. Matched-budget control: 40k steps of *standard-attention* continued
   training from the same 7M checkpoint, then scan that plane — separates
   "40k more steps" from "40k AMAP steps" (the magnitudes' rescue).
3. AdaLN calibration rung: finetune the AdaLN conditioning MLPs only
   (SiT's native scalars-only family), everything else frozen, per scan
   point — upgrades frozen fragility to attributable necessity, exactly as
   per-head temperatures did for the nanochat master curve.
4. Fine fluxcut spacing (0.05 over [0.4, 0.8]) on both planes: locate the
   cliffs; sharp-vs-smooth is the modality contrast with the induction
   transition.
5. Coupling-plane portrait vs finetune time (scan at several checkpoints
   along a longer AMAP finetune): watch the fluxcut cliff migrate.

## Files

- `gmap_attention.py` — the operator + selftest (`python gmap_attention.py`):
  (1,1) corners exact for both variants, anti-attention identity, flux-only
  antisymmetric, xi=0 symmetric/reversible.
- `xi_sit.py` — the Modal scan harness (grids + paired FM column, rays and
  spacing configurable). Resolves `amap_common.py` from this folder or
  falls back to `../AMAP/amap_common.py` — no copy required.
- `GMAP.yml` — workflow (`.github/workflows/`): choose base arm, rays,
  spacing; grids land in `<push_repo>/samples/gmap/`, the scan table in
  `<push_repo>/probes/`, and grids are committed back to `GMAP/samples/`.

Grid naming: `gmap_<variant>_<ray>_t<t>.png`. The `(1,1)` grid appears once
per ray by construction — identical files are the free replicates (byte-equal
PNGs are the scan's internal consistency check).

# MAPtest

FID across the three SiT-XL/2-family models on one shared reference:

- **sit** — released SiT-XL/2 (7M), full dot-product attention (the upper bound)
- **amap** — SiT + AMAP (symmetric PSD core + antisymmetric flux)
- **dmap** — SiT + folded DMAP (symmetric distance kernel only, R≡0)

Expected ordering: **FID(SiT) ≤ FID(AMAP) ≤ FID(DMAP)** — the gap AMAP→DMAP is
the measured value of the flux term; SiT→AMAP is the cost of forcing the
symmetric sector PSD.

All three are loaded via the same helpers used to train them
(`amap_common.load_amap_checkpoint`, `dmap_common.load_dmap_checkpoint`) and
sampled with the same official SiT transport ODE + CFG, against one reference —
so the *differences* are trustworthy even where absolute latents-mode numbers
aren't publication figures.

## Run (all three in one job so they share a reference)

    modal run MAPtest/maptest_fid.py \
        --models "sit,amap:jcandane/AMAP,dmap:jcandane/DMAP" \
        --num-samples 50000 --cfg-scale 1.5

Model spec: `kind[:repo[:step[:weights]]]`, e.g. `amap:jcandane/AMAP:40000:ema`.
`step` defaults to `latest`, `weights` to `ema`. `sit` needs no repo.

## Reference

- default `ref_mode="latents"` → stats from VAE-decoded `sparsetrace/dlatentzz`.
  Self-consistent, factors out VAE error, ideal for SiT-vs-AMAP-vs-DMAP, but
  NOT comparable to any published SiT FID.
- `--ref-stats-url <ADM VIRTUAL_imagenet256_labeled.npz>` → publication-comparable
  (needed to check SiT against the paper's ~2.06; use cfg 1.5, 50k).

## Notes

- 3 models × 50k × 50 ODE steps × 2 (CFG) is hours — timeout defaults to 8h
  (`MAPTEST_SECONDS`), GPU H200 (`MAPTEST_GPU`). For a quick pass use
  `--num-samples 10000` (defensible for internal comparison, not a reported FID).
- To split across jobs, run one `--models` at a time; but note the reference is
  recomputed each job unless you pass a fixed `--ref-stats-url`.
- Writes `MAPtest/fid_results.json`.

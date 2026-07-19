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

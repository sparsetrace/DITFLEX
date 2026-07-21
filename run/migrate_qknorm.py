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

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

FILES = ["model.safetensors", "ema.safetensors", "optimizer.safetensors", "state.json"]


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

"""src/ditflex/train.py -- time-boxed DDP training loop.

Launched by run/modal_train.py via torchrun:

    python -m torch.distributed.run --nproc-per-node=8 --standalone \
        -m ditflex.train --train-seconds 7200 --objective ddpm

Long training is many short runs: pull latest checkpoint from the Hub
(if any) -> train until the wall-clock budget is spent (rank 0 checks the
deadline every cfg.train.deadline_check_every steps and broadcasts the
stop, so per-rank clock drift cannot desynchronise the exit) -> save ->
push -> exit. Resume is exact because data sampling is stateless in
(global_step, rank).

Order of operations (each is load-bearing):
    build raw model -> load checkpoint into it -> EMA on raw params
    -> torch.compile(model) -> DDP-wrap the compiled module
Compile-inner-then-DDP is the README's chosen order; the interaction is
version-sensitive, so if compile fails here, flipping to
torch.compile(DDP(model)) is the first thing to try.

Recipe fidelity: AdamW lr 1e-4 constant, no warmup, no weight decay,
EMA 0.9999, bf16 autocast with fp32 master weights, label dropout 0.1
(inside the objective), NO gradient clipping (published DiT does not
clip; the overfit smoke's clipping is smoke-only).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ditflex.checkpoint import (
    load_checkpoint,
    pull_from_hub,
    push_to_hub,
    save_checkpoint,
)
from ditflex.config import Config
from ditflex.distributed import barrier, broadcast_flag, cleanup, setup
from ditflex.ema import EMA
from ditflex.latents import LatentStore
from ditflex.model import build_model
from ditflex.objective import build_objective

CKPT_DIR = "/tmp/ditflex_ckpt"
LOG_EVERY = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seconds", type=int, required=True)
    p.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    p.add_argument("--hub-repo", type=str, default=None,
                   help="override cfg.hub.checkpoint_repo")
    p.add_argument("--no-push", action="store_true",
                   help="skip the Hub upload (local smokes)")
    p.add_argument("--max-latent-files", type=int, default=None,
                   help="load only the first N latent shards (smokes)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="hard step cap regardless of wall clock (smokes)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = setup()

    cfg = Config()
    cfg.train.objective = args.objective
    if args.hub_repo:
        cfg.hub.checkpoint_repo = args.hub_repo

    if cfg.train.global_batch % ctx.world != 0:
        raise ValueError(f"global_batch {cfg.train.global_batch} % world {ctx.world} != 0")
    per_rank_batch = cfg.train.global_batch // ctx.world

    if ctx.is_rank0:
        print(f"[train] world={ctx.world}  per_rank_batch={per_rank_batch}")
        print(cfg.to_json())

    # -- checkpoint pull (rank 0 downloads, shared FS on one node) -------
    resume_dir = None
    if ctx.is_rank0:
        resume_dir = pull_from_hub(cfg.hub.checkpoint_repo, CKPT_DIR)
        print(f"[train] resume checkpoint: {resume_dir or 'none (fresh start)'}")
    barrier(ctx)
    if not ctx.is_rank0 and os.path.exists(os.path.join(CKPT_DIR, "state.json")):
        resume_dir = CKPT_DIR

    # -- model / ema / optimizer, load BEFORE compile/wrap ---------------
    model = build_model(cfg.model).to(ctx.device)
    ema = EMA(model, cfg.train.ema_decay).to(ctx.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    start_step = 0
    run_history: list = []
    if resume_dir is not None:
        state = load_checkpoint(resume_dir, model, ema, optimizer, cfg)
        start_step = int(state["step"])
        run_history = state.get("run_history", [])
        if ctx.is_rank0:
            print(f"[train] resumed at step {start_step:,}")

    # -- latents: rank 0 warms the HF cache, then everyone loads ---------
    store_kw = dict(
        repo_id=cfg.data.hub_repo,
        device=ctx.device,
        max_files=args.max_latent_files,
        expected_total=cfg.data.expected_total,
        latent_shape=cfg.data.latent_shape,
        num_classes=cfg.model.num_classes,
    )
    if ctx.is_rank0:
        store = LatentStore.from_hub(**store_kw)
    barrier(ctx)
    if not ctx.is_rank0:
        store = LatentStore.from_hub(**store_kw)
    if ctx.is_rank0:
        print(f"[train] latents resident: {len(store):,} "
              f"({store.latents.numel() * 2 / 1024**3:.2f} GiB bf16)")

    objective = build_objective(
        cfg.train.objective,
        label_dropout=cfg.train.label_dropout,
        num_classes=cfg.model.num_classes,
    )

    # -- compile inner, then DDP-wrap (README order; version-sensitive) --
    compiled = torch.compile(model)
    wrapped = (
        DDP(compiled, device_ids=[ctx.local_rank]) if ctx.is_distributed else compiled
    )

    # -- the loop --------------------------------------------------------
    step = start_step
    t_start = time.time()
    deadline = t_start + args.train_seconds
    losses: list[float] = []
    last_archive_bucket = start_step // cfg.hub.archive_every_steps
    wrapped.train()

    def run_record(end_step: int, completed: bool) -> dict:
        return {
            "start_step": start_step,
            "end_step": end_step,
            "seconds": round(time.time() - t_start, 1),
            "world": ctx.world,
            "objective": cfg.train.objective,
            "completed": completed,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_and_push(at_step: int, completed: bool) -> None:
        """Rank-0 only. Periodic saves record the run-in-progress
        (completed=False) so a crash-resume still sees honest history."""
        nonlocal last_archive_bucket
        state = {
            "step": at_step,
            "run_history": run_history + [run_record(at_step, completed)],
        }
        save_checkpoint(CKPT_DIR, model, ema, optimizer, cfg, state)
        tag = "final" if completed else "periodic"
        print(f"[train] saved step {at_step:,} ({tag})")
        if not args.no_push:
            bucket = at_step // cfg.hub.archive_every_steps
            archive = at_step if bucket != last_archive_bucket else None
            last_archive_bucket = bucket
            push_to_hub(CKPT_DIR, cfg.hub.checkpoint_repo, archive_step=archive)
            print(f"[train] pushed to {cfg.hub.checkpoint_repo}")

    while True:
        if step % cfg.train.deadline_check_every == 0 and step > start_step:
            stop = ctx.is_rank0 and time.time() >= deadline
            if broadcast_flag(ctx, stop):
                break
        if args.max_steps is not None and (step - start_step) >= args.max_steps:
            break

        x0, y = store.batch(step, ctx.rank, per_rank_batch, cfg.train.base_seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.loss(wrapped, x0, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)          # raw module: clean names, shared params

        val = loss.item()
        if not torch.isfinite(loss):
            print(f"[train][rank {ctx.rank}] step {step}: loss={val} -- diverged, "
                  "aborting WITHOUT saving so the Hub 'latest' stays healthy.")
            cleanup(ctx)
            return 1

        losses.append(val)
        step += 1

        # Periodic checkpoint: every rank reaches the barrier, rank 0
        # saves+pushes behind it, so the DDP collectives never see a
        # half-absent rank during the upload.
        if cfg.hub.save_every_steps > 0 and step % cfg.hub.save_every_steps == 0:
            if ctx.is_rank0:
                save_and_push(step, completed=False)
            barrier(ctx)

        if ctx.is_rank0 and step % LOG_EVERY == 0:
            rate = LOG_EVERY / max(time.time() - getattr(main, "_t", t_start), 1e-9)
            main._t = time.time()
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            print(f"  step {step:>8,}  loss {avg:.5f}  {rate:5.2f} steps/s  "
                  f"{rate * cfg.train.global_batch:7.0f} img/s")

    # -- final save + push (rank 0), everyone waits ----------------------
    elapsed = time.time() - t_start
    if ctx.is_rank0:
        print(f"[train] run done after {elapsed/60:.1f} min "
              f"({step - start_step:,} steps this run)")
        save_and_push(step, completed=True)
    barrier(ctx)
    cleanup(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

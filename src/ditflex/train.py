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
    p.add_argument("--lr", type=float, default=0.0,
                   help="override the learning rate for THIS RUN (0 = keep the "
                        "config/checkpoint value). Applied to optimizer param "
                        "groups only, after checkpoint load -- cfg.train.lr is "
                        "NOT modified, so the config-drift guard still passes. "
                        "Added for the 250K instability: dropping lr stalls the "
                        "norm growth that clipping alone only contains.")
    p.add_argument("--wd", type=float, default=-1.0,
                   help="override AdamW weight decay for THIS RUN (-1 = keep "
                        "config value). Param-groups only, post-load; drift "
                        "guard unaffected. Added for the 250K instability: lr "
                        "cuts only DELAY the norm-growth blowup (1e-4: +850 "
                        "steps; 3e-5: +4.7K steps); decay reverses the growth "
                        "itself. Cumulative shrink ~ lr*wd*steps.")
    p.add_argument("--grad-ceiling", type=float, default=25.0,
                   help="ABSOLUTE skip ceiling: refuse any step whose raw grad "
                        "norm exceeds this, regardless of the EMA (0 = off). "
                        "The relative 4x-EMA guard RATCHETS under a drifting "
                        "distribution (observed: EMA 1.8 -> 600 in 4K steps as "
                        "near-threshold steps fed the average); the ceiling "
                        "cannot be chased.")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="offset added to the data base seed for this run "
                        "(runtime-only; drift-safe). Discriminator for the "
                        "247-256K instability cluster: four onsets across "
                        "three trajectories share deterministic batches -- a "
                        "shifted order tests data-region vs state-intrinsic.")
    p.add_argument("--clip", type=float, default=1.0,
                   help="gradient-clip max-norm for this run")
    p.add_argument("--spike-skip", type=float, default=4.0,
                   help="SKIP the optimizer step when the pre-clip grad norm "
                        "exceeds this factor times its running EMA (0 = off). "
                        "Clipping caps a bad step's SIZE but still takes it; "
                        "skipping refuses it. Added after the DMAP 46K cliff: "
                        "loss 0.80 -> floor in ~400 steps with |w| moving only "
                        "0.002%% -- a single-event explosion, not norm growth.")
    p.add_argument("--qk-mode", choices=["amap", "dmap"], default="amap",
                   help="amap = baseline directed attention; dmap = diffusion-map "
                        "attention (W_K tied to W_Q, R == 0, plus Coifman-Lafon "
                        "density correction)")
    p.add_argument("--dmap-alpha", type=float, default=0.0,
                   help="density-correction exponent for --qk-mode=dmap "
                        "(0 = none, 0.5 = Fokker-Planck, 1 = Laplace-Beltrami)")
    p.add_argument("--sample-count", type=int, default=16,
                   help="images to sample after the final save (0 disables)")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
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

    # -- checkpoint pull (rank 0 downloads, shared FS on one node) -------
    resume_dir = None
    if ctx.is_rank0:
        resume_dir = pull_from_hub(cfg.hub.checkpoint_repo, CKPT_DIR)
        print(f"[train] resume checkpoint: {resume_dir or 'none (fresh start)'}")
    barrier(ctx)
    if not ctx.is_rank0 and os.path.exists(os.path.join(CKPT_DIR, "state.json")):
        resume_dir = CKPT_DIR

    # -- model / ema / optimizer, load BEFORE compile/wrap ---------------
    if cfg.model.qk_mode == "amap":
        model = build_model(cfg.model).to(ctx.device)
    elif cfg.model.qk_mode == "dmap":
        from ditflex.diffusion_model import build_dmap_model

        model = build_dmap_model(cfg.model).to(ctx.device)
    else:
        raise ValueError(f"unknown qk_mode: {cfg.model.qk_mode!r}")
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

    if args.lr > 0.0:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if ctx.is_rank0:
            print(f"[train] LR OVERRIDE for this run: {args.lr:g} "
                  f"(config value {cfg.train.lr:g} unchanged; drift guard unaffected)")
    if args.wd >= 0.0:
        for group in optimizer.param_groups:
            group["weight_decay"] = args.wd
        if ctx.is_rank0:
            print(f"[train] WD OVERRIDE for this run: {args.wd:g} "
                  "(decoupled AdamW decay; drift guard unaffected)")

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
            # Effective hyperparameters ON THE RECORD: overrides are
            # runtime-only (drift-safe), so without this the checkpoint
            # would not know its own training history.
            "effective": {
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "clip": args.clip,
                "spike_skip": args.spike_skip,
                "steps_skipped": getattr(main, "_spikes", 0),
                "seed_offset": args.seed_offset,
                "grad_ceiling": args.grad_ceiling,
            },
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

        x0, y = store.batch(
            step, ctx.rank, per_rank_batch,
            cfg.train.base_seed + args.seed_offset,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.loss(wrapped, x0, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # DEVIATION (stability): DiT/SiT recipes neither clip nor skip.
        # Clipping added after the baseline's ~247.4K norm-growth blowup;
        # spike-skip added after the DMAP chain's 46K single-event cliff
        # (sharp explosion at flat |w| -- clipping caps a bad step's size
        # but still takes it in the bad direction; skipping refuses it).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.clip
        ).item()
        gema = getattr(main, "_grad_ema", None)
        spiked = (
            args.spike_skip > 0.0
            and gema is not None
            and grad_norm > args.spike_skip * gema
        ) or (args.grad_ceiling > 0.0 and grad_norm > args.grad_ceiling)
        if spiked:
            optimizer.zero_grad(set_to_none=True)   # refuse the step entirely
            main._spikes = getattr(main, "_spikes", 0) + 1
            if ctx.is_rank0:
                print(f"[train] step {step:,}: grad norm {grad_norm:.2f} > "
                      f"{args.spike_skip:g}x EMA {gema:.2f} -- SKIPPING step "
                      f"(total skipped: {main._spikes})")
        else:
            optimizer.step()
            ema.update(model)      # raw module: clean names, shared params
            # EMA of the grad norm updates only on accepted steps, so one
            # spike cannot poison the baseline it is judged against.
            main._grad_ema = (
                grad_norm if gema is None else 0.99 * gema + 0.01 * grad_norm
            )

        val = loss.item()
        if not torch.isfinite(loss):
            print(f"[train][rank {ctx.rank}] step {step}: loss={val} -- diverged, "
                  "aborting WITHOUT saving so the Hub 'latest' stays healthy.")
            cleanup(ctx)
            return 1

        losses.append(val)
        step += 1

        # Divergence guard (the 247K lesson): NaN is not the only failure
        # mode -- a gradient explosion can collapse the model to the
        # zero-predictor floor while every loss stays finite, and periodic
        # saves would then push poisoned checkpoints for hours. Track the
        # best 200-step mean; if the current mean exceeds 1.6x best for
        # two consecutive windows, abort WITHOUT saving.
        if step % 200 == 0 and len(losses) >= 200 and (step - start_step) > 400:
            window = sum(losses[-200:]) / 200
            best = getattr(main, "_best_window", None)
            if best is None or window < best:
                main._best_window = window
                main._blown_windows = 0
            elif window > 1.6 * best:
                main._blown_windows = getattr(main, "_blown_windows", 0) + 1
                if main._blown_windows >= 2:
                    print(f"[train][rank {ctx.rank}] step {step}: windowed loss "
                          f"{window:.4f} > 1.6x best {best:.4f} for 2 windows -- "
                          "DIVERGENCE, aborting WITHOUT saving.")
                    cleanup(ctx)
                    return 1
            else:
                main._blown_windows = 0

        # Periodic checkpoint: every rank reaches the barrier, rank 0
        # saves+pushes behind it, so the DDP collectives never see a
        # half-absent rank during the upload.
        if cfg.hub.save_every_steps > 0 and step % cfg.hub.save_every_steps == 0:
            # Push-health gate (the 250K lesson, stricter form): the Hub
            # should only ever receive checkpoints from a model near its
            # best. Noise is ~2%; +10% means something is off -- keep
            # training (it may recover) but WITHHOLD the push. The 1.6x
            # divergence guard above still aborts sustained blowups.
            best = getattr(main, "_best_window", None)
            healthy = True
            if best is not None and len(losses) >= 200:
                window = sum(losses[-200:]) / 200
                healthy = window <= 1.10 * best
                if not healthy and ctx.is_rank0:
                    print(f"[train] step {step:,}: windowed loss {window:.4f} > "
                          f"1.10x best {best:.4f} -- WITHHOLDING periodic push")
            if ctx.is_rank0 and healthy:
                save_and_push(step, completed=False)
            barrier(ctx)

        if ctx.is_rank0 and step % LOG_EVERY == 0:
            rate = LOG_EVERY / max(time.time() - getattr(main, "_t", t_start), 1e-9)
            main._t = time.time()
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            with torch.no_grad():
                # Per-family norms: the global scalar HID the disease (it
                # fell while the instability re-emerged), so decompose --
                # whichever family moves against the tide is the culprit.
                fams = dict.fromkeys(("qk", "vo", "mlp", "ada", "emb", "oth"), 0.0)
                for pname, p_ in model.named_parameters():
                    key = ("qk" if ("to_q" in pname or "to_k" in pname) else
                           "vo" if ("to_v" in pname or "to_out" in pname) else
                           "mlp" if ".ff." in pname else
                           "ada" if ("norm1" in pname or "norm_out" in pname
                                     or "adaln" in pname.lower()) else
                           "emb" if ("emb" in pname or "pos_embed" in pname
                                     or "proj_out" in pname) else "oth")
                    fams[key] += p_.detach().float().pow(2).sum().item()
                pnorm = sum(fams.values()) ** 0.5
                fstr = " ".join(f"{k}={v**0.5:7.1f}" for k, v in fams.items())
            print(f"  step {step:>8,}  loss {avg:.5f}  {rate:5.2f} steps/s  "
                  f"{rate * cfg.train.global_batch:7.0f} img/s  |w|={pnorm:8.2f}  {fstr}")

    # -- final save + push (rank 0), everyone waits ----------------------
    elapsed = time.time() - t_start
    if ctx.is_rank0:
        print(f"[train] run done after {elapsed/60:.1f} min "
              f"({step - start_step:,} steps this run)")
        save_and_push(step, completed=True)

    # -- fixed-seed sample grid: the time-lapse frame for this link ------
    # Strictly AFTER the final save+push, and non-fatal: sampling must
    # never be able to endanger a completed checkpoint. copy_to clobbers
    # the training weights, which is fine -- they are already on the Hub.
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
                out_dir=CKPT_DIR,
            )
        except Exception as e:  # noqa: BLE001 -- deliberately broad
            print(f"[train] sampling failed (non-fatal; checkpoint already pushed): {e!r}")
    barrier(ctx)
    cleanup(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

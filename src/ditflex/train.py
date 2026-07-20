"""src/ditflex/train.py -- time-boxed DDP training loop.

Launched by run/modal_train.py via torchrun:

    python -m torch.distributed.run --nproc-per-node=8 --standalone \
        -m ditflex.train --train-seconds 7200 --objective ddpm

Long training is many short runs: pull latest checkpoint from the Hub
(if any) -> train until the wall-clock budget is spent -> save -> push ->
exit. Data sampling is stateless in (global_step, rank), and the numerical
stability guards are checkpointed so they also survive run boundaries.

Order of operations:
    build raw model -> load checkpoint into it -> EMA on raw params
    -> torch.compile(model) -> DDP-wrap the compiled module

Stability deviations from the published DiT/SiT recipes:
    * global gradient clipping;
    * relative and absolute raw-gradient spike rejection;
    * non-finite loss/gradient rejection before optimizer.step();
    * finite-loss collapse detection;
    * periodic and final checkpoint health gates;
    * guard-state persistence across chained runs;
    * rank-synchronised safety decisions under DDP.
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
from ditflex.distributed import (
    all_reduce_bool_and,
    all_reduce_mean,
    barrier,
    broadcast_flag,
    broadcast_float,
    broadcast_int,
    cleanup,
    setup,
)
from ditflex.ema import EMA
from ditflex.latents import LatentStore
from ditflex.model import build_model
from ditflex.objective import build_objective

CKPT_DIR = "/tmp/ditflex_ckpt"
LOG_EVERY = 50
LOSS_WINDOW = 200
DIVERGENCE_MULTIPLIER = 1.60
CHECKPOINT_HEALTH_MULTIPLIER = 1.10
GRAD_EMA_DECAY = 0.99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seconds", type=int, required=True)
    p.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    p.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="override cfg.hub.checkpoint_repo",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="skip the Hub upload (local smokes)",
    )
    p.add_argument(
        "--max-latent-files",
        type=int,
        default=None,
        help="load only the first N latent shards (smokes)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="hard step cap regardless of wall clock (smokes)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help=(
            "override the learning rate for THIS RUN (0 = keep the "
            "config/checkpoint value). Applied to optimizer param groups only, "
            "after checkpoint load; cfg.train.lr is not modified."
        ),
    )
    p.add_argument(
        "--wd",
        type=float,
        default=-1.0,
        help=(
            "override AdamW weight decay for THIS RUN (-1 = keep config/checkpoint "
            "value). Applied to optimizer param groups only, after checkpoint load."
        ),
    )
    p.add_argument(
        "--grad-ceiling",
        type=float,
        default=25.0,
        help=(
            "absolute raw-gradient-norm skip ceiling; refuse any step whose norm "
            "exceeds this value, regardless of the running EMA (0 = off)"
        ),
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="runtime-only offset added to the data base seed",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="gradient-clip max norm for this run",
    )
    p.add_argument(
        "--spike-skip",
        type=float,
        default=4.0,
        help=(
            "skip the optimizer step when the pre-clip gradient norm exceeds this "
            "factor times its running EMA (0 = off)"
        ),
    )
    p.add_argument(
        "--qk-mode",
        choices=["amap", "dmap"],
        default="amap",
        help=(
            "amap = baseline directed attention; dmap = diffusion-map attention "
            "(W_K tied to W_Q, R == 0, plus density correction)"
        ),
    )
    p.add_argument(
        "--dmap-alpha",
        type=float,
        default=0.0,
        help="density-correction exponent for --qk-mode=dmap",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="images to sample after a healthy final save (0 disables)",
    )
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    return p.parse_args()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


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
    guard_state: dict = {}
    if resume_dir is not None:
        state = load_checkpoint(resume_dir, model, ema, optimizer, cfg)
        start_step = int(state["step"])
        run_history = state.get("run_history", [])
        guard_state = state.get("guard_state", {})
        if ctx.is_rank0:
            print(f"[train] resumed at step {start_step:,}")

    if args.lr > 0.0:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if ctx.is_rank0:
            print(
                f"[train] LR OVERRIDE for this run: {args.lr:g} "
                f"(config value {cfg.train.lr:g} unchanged; drift guard unaffected)"
            )
    if args.wd >= 0.0:
        for group in optimizer.param_groups:
            group["weight_decay"] = args.wd
        if ctx.is_rank0:
            print(
                f"[train] WD OVERRIDE for this run: {args.wd:g} "
                "(decoupled AdamW decay; drift guard unaffected)"
            )

    # Restore numerical guard history. Old checkpoints simply start with
    # empty guard state and become fully persistent at the next healthy save.
    grad_ema = _optional_float(guard_state.get("grad_ema"))
    best_window = _optional_float(guard_state.get("best_window"))
    blown_windows = int(guard_state.get("blown_windows", 0))
    spikes_total = int(guard_state.get("spikes_total", 0))
    recent_losses = [float(x) for x in guard_state.get("recent_losses", [])]
    recent_losses = recent_losses[-LOSS_WINDOW:]
    spikes_at_start = spikes_total

    if ctx.is_rank0:
        print(
            "[train] guard state: "
            f"grad_ema={grad_ema if grad_ema is not None else 'unset'}  "
            f"best_window={best_window if best_window is not None else 'unset'}  "
            f"blown_windows={blown_windows}  spikes_total={spikes_total}  "
            f"recent_losses={len(recent_losses)}"
        )

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
        print(
            f"[train] latents resident: {len(store):,} "
            f"({store.latents.numel() * 2 / 1024**3:.2f} GiB bf16)"
        )

    objective = build_objective(
        cfg.train.objective,
        label_dropout=cfg.train.label_dropout,
        num_classes=cfg.model.num_classes,
    )

    # -- compile inner, then DDP-wrap (README order; version-sensitive) --
    compiled = torch.compile(model)
    wrapped = DDP(compiled, device_ids=[ctx.local_rank]) if ctx.is_distributed else compiled

    # -- the loop --------------------------------------------------------
    step = start_step
    t_start = time.time()
    deadline = t_start + args.train_seconds
    run_losses: list[float] = []
    last_archive_bucket = start_step // cfg.hub.archive_every_steps
    last_log_time = t_start
    wrapped.train()

    def current_window() -> float | None:
        if len(recent_losses) < LOSS_WINDOW:
            return None
        return sum(recent_losses[-LOSS_WINDOW:]) / LOSS_WINDOW

    def checkpoint_is_healthy() -> tuple[bool, float | None]:
        window = current_window()
        if best_window is None or window is None:
            return True, window
        return window <= CHECKPOINT_HEALTH_MULTIPLIER * best_window, window

    def serialized_guard_state() -> dict:
        return {
            "version": 1,
            "grad_ema": grad_ema,
            "best_window": best_window,
            "blown_windows": blown_windows,
            "spikes_total": spikes_total,
            "recent_losses": recent_losses[-LOSS_WINDOW:],
            "loss_window": LOSS_WINDOW,
            "grad_ema_decay": GRAD_EMA_DECAY,
            "divergence_multiplier": DIVERGENCE_MULTIPLIER,
            "checkpoint_health_multiplier": CHECKPOINT_HEALTH_MULTIPLIER,
        }

    def run_record(end_step: int, completed: bool) -> dict:
        return {
            "start_step": start_step,
            "end_step": end_step,
            "seconds": round(time.time() - t_start, 1),
            "world": ctx.world,
            "objective": cfg.train.objective,
            "completed": completed,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "effective": {
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "clip": args.clip,
                "spike_skip": args.spike_skip,
                "steps_skipped_this_run": spikes_total - spikes_at_start,
                "steps_skipped_total": spikes_total,
                "seed_offset": args.seed_offset,
                "grad_ceiling": args.grad_ceiling,
            },
        }

    def save_and_push(at_step: int, completed: bool) -> None:
        """Save a checkpoint from rank 0 after the caller's health decision."""
        nonlocal last_archive_bucket
        state = {
            "step": at_step,
            "run_history": run_history + [run_record(at_step, completed)],
            "guard_state": serialized_guard_state(),
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

    def abort_all(message: str, code: int = 1) -> int:
        if ctx.is_rank0:
            print(message)
        barrier(ctx)
        cleanup(ctx)
        return code

    while True:
        if step % cfg.train.deadline_check_every == 0 and step > start_step:
            stop = ctx.is_rank0 and time.time() >= deadline
            if broadcast_flag(ctx, stop):
                break
        if args.max_steps is not None and (step - start_step) >= args.max_steps:
            break

        x0, y = store.batch(
            step,
            ctx.rank,
            per_rank_batch,
            cfg.train.base_seed + args.seed_offset,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.loss(wrapped, x0, y)

        # Every rank must agree before any rank enters backward. Otherwise a
        # rank-local NaN exit can strand its peers inside a DDP collective.
        local_loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        loss_finite = all_reduce_bool_and(ctx, local_loss_finite)
        if not loss_finite:
            local_value = float(loss.detach().float().item())
            return abort_all(
                f"[train] step {step:,}: non-finite loss detected "
                f"(rank-0 local loss={local_value}) -- aborting WITHOUT saving."
            )

        # Use the global batch's mean loss for every guard and log. This keeps
        # guard history identical on all ranks and makes rank-0 decisions valid.
        global_loss = all_reduce_mean(ctx, loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # clip_grad_norm_ returns the raw pre-clip global norm. Do not use
        # error_if_nonfinite=True here: explicit rank-synchronised handling is
        # safer than allowing one process to raise before its peers.
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.clip,
            error_if_nonfinite=False,
        )
        grad_norm = float(grad_norm_tensor.detach().float().item())
        grads_finite = all_reduce_bool_and(ctx, bool(torch.isfinite(grad_norm_tensor).item()))
        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            return abort_all(
                f"[train] step {step:,}: non-finite gradient norm detected "
                "-- aborting BEFORE optimizer.step and WITHOUT saving."
            )

        # DDP should already make gradients identical. Broadcasting the measured
        # norm keeps the persisted EMA bit-for-bit aligned even if kernels report
        # tiny rank-local rounding differences.
        grad_norm = broadcast_float(ctx, grad_norm if ctx.is_rank0 else 0.0)

        # Rank 0 owns the decision; everybody receives the same answer.
        relative_spike = (
            args.spike_skip > 0.0
            and grad_ema is not None
            and grad_norm > args.spike_skip * grad_ema
        )
        absolute_spike = args.grad_ceiling > 0.0 and grad_norm > args.grad_ceiling
        spike_decision = relative_spike or absolute_spike if ctx.is_rank0 else False
        spiked = broadcast_flag(ctx, spike_decision)

        if spiked:
            optimizer.zero_grad(set_to_none=True)
            spikes_total += 1
            if ctx.is_rank0:
                reasons: list[str] = []
                if relative_spike:
                    reasons.append(
                        f"relative {grad_norm:.2f} > {args.spike_skip:g}x EMA {grad_ema:.2f}"
                    )
                if absolute_spike:
                    reasons.append(f"absolute {grad_norm:.2f} > ceiling {args.grad_ceiling:g}")
                reason = " and ".join(reasons) if reasons else "rank-0 spike decision"
                ema_text = "unset" if grad_ema is None else f"{grad_ema:.2f}"
                print(
                    f"[train] step {step:,}: {reason}; grad EMA={ema_text} "
                    f"-- SKIPPING optimizer step (total skipped: {spikes_total})"
                )
        else:
            optimizer.step()
            ema.update(model)
            grad_ema = (
                grad_norm
                if grad_ema is None
                else GRAD_EMA_DECAY * grad_ema + (1.0 - GRAD_EMA_DECAY) * grad_norm
            )

        run_losses.append(global_loss)
        recent_losses.append(global_loss)
        if len(recent_losses) > LOSS_WINDOW:
            del recent_losses[:-LOSS_WINDOW]
        step += 1

        # Evaluate one non-overlapping global window every LOSS_WINDOW data
        # steps. The state is persistent, so a warning window at the end of one
        # Modal invocation remains active at the beginning of the next.
        if step % LOSS_WINDOW == 0 and len(recent_losses) >= LOSS_WINDOW:
            window = current_window()
            assert window is not None
            diverged = False
            if ctx.is_rank0:
                if best_window is None or window < best_window:
                    best_window = window
                    blown_windows = 0
                elif window > DIVERGENCE_MULTIPLIER * best_window:
                    blown_windows += 1
                    diverged = blown_windows >= 2
                else:
                    blown_windows = 0
            # Rank 0 has updated the state. Broadcast the small state scalars so
            # every rank serialises identical guards and evaluates identically.
            best_value = -1.0 if best_window is None else best_window
            best_value = broadcast_float(ctx, best_value if ctx.is_rank0 else 0.0)
            best_window = None if best_value < 0.0 else best_value
            blown_windows = broadcast_int(
                ctx, blown_windows if ctx.is_rank0 else 0
            )
            diverged = broadcast_flag(ctx, diverged if ctx.is_rank0 else False)
            if diverged:
                return abort_all(
                    f"[train] step {step:,}: windowed loss {window:.4f} > "
                    f"{DIVERGENCE_MULTIPLIER:.2f}x best {best_window:.4f} for "
                    "2 consecutive windows -- DIVERGENCE, aborting WITHOUT saving."
                )

        # Periodic checkpoint: rank 0 decides health, broadcasts it, then every
        # rank reaches the same barrier regardless of whether the push occurs.
        if cfg.hub.save_every_steps > 0 and step % cfg.hub.save_every_steps == 0:
            healthy = False
            window = current_window()
            if ctx.is_rank0:
                healthy, window = checkpoint_is_healthy()
                if not healthy:
                    assert window is not None and best_window is not None
                    print(
                        f"[train] step {step:,}: windowed loss {window:.4f} > "
                        f"{CHECKPOINT_HEALTH_MULTIPLIER:.2f}x best "
                        f"{best_window:.4f} -- WITHHOLDING periodic checkpoint"
                    )
            healthy = broadcast_flag(ctx, healthy)
            if ctx.is_rank0 and healthy:
                save_and_push(step, completed=False)
            barrier(ctx)

        if ctx.is_rank0 and step % LOG_EVERY == 0:
            now = time.time()
            rate = LOG_EVERY / max(now - last_log_time, 1e-9)
            last_log_time = now
            avg = sum(run_losses[-LOG_EVERY:]) / len(run_losses[-LOG_EVERY:])
            with torch.no_grad():
                fams = dict.fromkeys(("qk", "vo", "mlp", "ada", "emb", "oth"), 0.0)
                for pname, parameter in model.named_parameters():
                    key = (
                        "qk"
                        if ("to_q" in pname or "to_k" in pname)
                        else "vo"
                        if ("to_v" in pname or "to_out" in pname)
                        else "mlp"
                        if ".ff." in pname
                        else "ada"
                        if (
                            "norm1" in pname
                            or "norm_out" in pname
                            or "adaln" in pname.lower()
                        )
                        else "emb"
                        if ("emb" in pname or "pos_embed" in pname or "proj_out" in pname)
                        else "oth"
                    )
                    fams[key] += parameter.detach().float().pow(2).sum().item()
                parameter_norm = sum(fams.values()) ** 0.5
                family_text = " ".join(f"{key}={value**0.5:7.1f}" for key, value in fams.items())
            grad_ema_text = "unset" if grad_ema is None else f"{grad_ema:.2f}"
            print(
                f"  step {step:>8,}  loss {avg:.5f}  {rate:5.2f} steps/s  "
                f"{rate * cfg.train.global_batch:7.0f} img/s  "
                f"|g|ema={grad_ema_text:>7}  |w|={parameter_norm:8.2f}  {family_text}"
            )

    # -- final health gate, save and push -------------------------------
    elapsed = time.time() - t_start
    final_healthy = False
    final_window = current_window()
    if ctx.is_rank0:
        print(
            f"[train] run done after {elapsed / 60:.1f} min "
            f"({step - start_step:,} steps this run)"
        )
        final_healthy, final_window = checkpoint_is_healthy()
        if not final_healthy:
            assert final_window is not None and best_window is not None
            print(
                f"[train] FINAL CHECKPOINT WITHHELD: windowed loss "
                f"{final_window:.4f} > {CHECKPOINT_HEALTH_MULTIPLIER:.2f}x "
                f"best {best_window:.4f}. Hub latest remains unchanged."
            )
    final_healthy = broadcast_flag(ctx, final_healthy)

    if not final_healthy:
        barrier(ctx)
        cleanup(ctx)
        return 2

    if ctx.is_rank0:
        save_and_push(step, completed=True)
    barrier(ctx)

    # -- fixed-seed sample grid: only after a healthy final save ---------
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
        except Exception as exc:  # noqa: BLE001 -- deliberately non-fatal
            print(f"[train] sampling failed (non-fatal; checkpoint already pushed): {exc!r}")

    barrier(ctx)
    cleanup(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

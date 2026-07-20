"""src/ditflex/train.py -- time-boxed, resume-safe DiT/SiT training.

Launched by ``run/modal_train.py`` via torchrun.  Long training is a chain of
short Modal jobs: pull the latest healthy checkpoint, train until the wall
clock or global target is reached, save, push, and exit.

The stability policy has three layers:

* non-finite and gradient-spike rejection before ``optimizer.step``;
* a global-step cosine LR envelope with persistent adaptive backoffs;
* checkpoint withholding and a final abort that leave Hub ``latest`` healthy.

The LR policy is runtime-only and lives in ``guard_state``.  It can therefore be
introduced at step 260K without changing ``Config`` and without bypassing the
checkpoint experiment-drift guard.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime

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
    cleanup,
    setup,
)
from ditflex.ema import EMA
from ditflex.latents import LatentStore
from ditflex.model import build_model
from ditflex.objective import build_objective
from ditflex.stability import AdaptiveLrController, StabilitySpec

CKPT_DIR = "/tmp/ditflex_ckpt"
LOG_EVERY = 50
LOSS_WINDOW = 200
GRAD_EMA_DECAY = 0.99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seconds", type=int, required=True)
    p.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    p.add_argument("--hub-repo", type=str, default=None, help="override checkpoint repo")
    p.add_argument("--no-push", action="store_true", help="skip Hub upload")
    p.add_argument("--max-latent-files", type=int, default=None, help="smoke-test shard cap")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="maximum data steps in this invocation; wall clock may stop first",
    )
    p.add_argument(
        "--target-steps",
        type=int,
        default=400_000,
        help="global stop and cosine horizon; defaults to the DiT/SiT 400K target",
    )

    # LR policy.  --lr remains backward-compatible as the recipe/base LR
    # override.  Under adaptive mode, leave it at 0 to use base=1e-4; at step
    # 260K the cosine envelope is then about 3.46e-5 before adaptive backoffs.
    p.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help="base LR override (0 = Config value 1e-4), before cosine/backoff",
    )
    p.add_argument(
        "--lr-policy",
        choices=["constant", "cosine", "adaptive"],
        default="adaptive",
    )
    p.add_argument("--lr-min", type=float, default=1e-5, help="cosine envelope floor")
    p.add_argument(
        "--lr-hard-min",
        type=float,
        default=1e-6,
        help="absolute floor after adaptive multiplier",
    )
    p.add_argument("--lr-backoff", type=float, default=0.5)
    p.add_argument("--lr-min-scale", type=float, default=0.125)
    p.add_argument("--loss-rise-ratio", type=float, default=1.08)
    p.add_argument("--loss-emergency-ratio", type=float, default=1.35)
    p.add_argument(
        "--reset-lr-controller",
        action="store_true",
        help="deliberately discard persisted adaptive state after changing policy settings",
    )

    p.add_argument(
        "--wd",
        type=float,
        default=-1.0,
        help="AdamW weight decay override for this run (-1 = checkpoint/config value)",
    )
    p.add_argument(
        "--grad-ceiling",
        type=float,
        default=25.0,
        help="absolute raw-gradient-norm skip ceiling (0 = off)",
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="runtime-only offset added to the stateless data seed",
    )
    p.add_argument("--clip", type=float, default=1.0, help="global gradient max norm")
    p.add_argument(
        "--spike-skip",
        type=float,
        default=4.0,
        help="skip update when raw grad norm exceeds this multiple of grad EMA (0 = off)",
    )
    p.add_argument("--qk-mode", choices=["amap", "dmap"], default="amap")
    p.add_argument("--dmap-alpha", type=float, default=0.0)
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    return p.parse_args()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def main() -> int:
    args = parse_args()
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be positive")

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

    # -- pull latest on rank 0; all ranks share the one-node filesystem -----
    resume_dir = None
    if ctx.is_rank0:
        resume_dir = pull_from_hub(cfg.hub.checkpoint_repo, CKPT_DIR)
        print(f"[train] resume checkpoint: {resume_dir or 'none (fresh start)'}")
    barrier(ctx)
    if not ctx.is_rank0 and os.path.exists(os.path.join(CKPT_DIR, "state.json")):
        resume_dir = CKPT_DIR

    # -- raw model / EMA / optimizer; load before compile and DDP ------------
    if cfg.model.qk_mode == "amap":
        model = build_model(cfg.model).to(ctx.device)
    elif cfg.model.qk_mode == "dmap":
        from ditflex.diffusion_model import build_dmap_model

        model = build_dmap_model(cfg.model).to(ctx.device)
    else:  # pragma: no cover - argparse and Config constrain this
        raise ValueError(f"unknown qk_mode: {cfg.model.qk_mode!r}")

    ema = EMA(model, cfg.train.ema_decay).to(ctx.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
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

    # Restore legacy guard fields before constructing the new controller so
    # its fast/slow EMAs begin from the recent checkpoint window, not the
    # all-time best (which may be much lower late in training).
    grad_ema = _optional_float(guard_state.get("grad_ema"))
    spikes_total = int(guard_state.get("spikes_total", 0))
    recent_losses = [float(x) for x in guard_state.get("recent_losses", [])][-LOSS_WINDOW:]
    window_spikes = int(guard_state.get("window_spikes", 0))
    spikes_at_start = spikes_total
    initial_loss = (
        sum(recent_losses[-LOSS_WINDOW:]) / LOSS_WINDOW
        if len(recent_losses) >= LOSS_WINDOW
        else None
    )

    checkpoint_lr = float(optimizer.param_groups[0]["lr"])
    base_lr = args.lr if args.lr > 0.0 else cfg.train.lr
    lr_spec = StabilitySpec(
        policy=args.lr_policy,
        total_steps=args.target_steps,
        base_lr=base_lr,
        min_lr=args.lr_min,
        hard_min_lr=args.lr_hard_min,
        backoff_factor=args.lr_backoff,
        min_scale=args.lr_min_scale,
        rise_ratio=args.loss_rise_ratio,
        emergency_ratio=args.loss_emergency_ratio,
    )
    legacy_best = _optional_float(guard_state.get("best_window"))
    lr_controller = AdaptiveLrController(
        lr_spec,
        start_step=start_step,
        checkpoint_lr=checkpoint_lr,
        initial_loss=initial_loss,
        legacy_best_loss=legacy_best,
    )
    controller_state = guard_state.get("lr_controller")
    if controller_state and not args.reset_lr_controller:
        lr_controller.load_state_dict(controller_state)
    elif controller_state and args.reset_lr_controller and ctx.is_rank0:
        print("[train] RESETTING persisted LR-controller state by explicit request")

    if args.wd >= 0.0:
        for group in optimizer.param_groups:
            group["weight_decay"] = args.wd
        if ctx.is_rank0:
            print(f"[train] WD OVERRIDE for this run: {args.wd:g}")

    start_lr = lr_controller.apply(optimizer, start_step)
    if ctx.is_rank0:
        inherited = "fresh" if resume_dir is None else f"checkpoint lr={checkpoint_lr:g}"
        print(
            f"[train] LR policy={lr_spec.policy}  base={lr_spec.base_lr:g}  "
            f"cosine_min={lr_spec.min_lr:g}  hard_min={lr_spec.hard_min_lr:g}  "
            f"target={lr_spec.total_steps:,}  scale={lr_controller.scale:.4f}  "
            f"effective@{start_step:,}={start_lr:g}  ({inherited})"
        )

    if ctx.is_rank0:
        status = lr_controller.status()
        print(
            "[train] guard state: "
            f"grad_ema={grad_ema if grad_ema is not None else 'unset'}  "
            f"spikes_total={spikes_total}  recent_losses={len(recent_losses)}  "
            f"window_spikes={window_spikes}  controller={status}"
        )

    # -- latents: rank 0 warms cache, then every rank creates its GPU store --
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

    # Compile inner, then DDP-wrap.  EMA continues to reference raw parameters.
    compiled = torch.compile(model)
    wrapped = DDP(compiled, device_ids=[ctx.local_rank]) if ctx.is_distributed else compiled

    # -- loop state ----------------------------------------------------------
    step = start_step
    t_start = time.time()
    deadline = t_start + args.train_seconds
    run_losses: list[float] = []
    last_archive_bucket = (
        start_step // cfg.hub.archive_every_steps if cfg.hub.archive_every_steps > 0 else 0
    )
    last_log_time = t_start
    wrapped.train()

    def current_window() -> float | None:
        if len(recent_losses) < LOSS_WINDOW:
            return None
        return sum(recent_losses[-LOSS_WINDOW:]) / LOSS_WINDOW

    def checkpoint_is_healthy() -> tuple[bool, str, float | None]:
        window = current_window()
        healthy, reason = lr_controller.checkpoint_is_healthy(window)
        return healthy, reason, window

    def serialized_guard_state() -> dict:
        return {
            "version": 2,
            "grad_ema": grad_ema,
            "spikes_total": spikes_total,
            "recent_losses": recent_losses[-LOSS_WINDOW:],
            "window_spikes": window_spikes,
            "loss_window": LOSS_WINDOW,
            "grad_ema_decay": GRAD_EMA_DECAY,
            "lr_controller": lr_controller.state_dict(),
            # Compatibility breadcrumbs for older analysis/recovery scripts.
            "best_window": lr_controller.best_loss,
            "blown_windows": lr_controller.bad_windows,
        }

    def run_record(end_step: int, completed: bool) -> dict:
        return {
            "start_step": start_step,
            "end_step": end_step,
            "seconds": round(time.time() - t_start, 1),
            "world": ctx.world,
            "objective": cfg.train.objective,
            "completed": completed,
            "finished_at": datetime.now(UTC).isoformat(),
            "effective": {
                "lr_policy": lr_spec.policy,
                "lr_base": lr_spec.base_lr,
                "lr_start": start_lr,
                "lr_end": float(optimizer.param_groups[0]["lr"]),
                "lr_min": lr_spec.min_lr,
                "lr_hard_min": lr_spec.hard_min_lr,
                "lr_scale": lr_controller.scale,
                "lr_backoffs_total": lr_controller.backoff_count,
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "clip": args.clip,
                "spike_skip": args.spike_skip,
                "grad_ceiling": args.grad_ceiling,
                "steps_skipped_this_run": spikes_total - spikes_at_start,
                "steps_skipped_total": spikes_total,
                "seed_offset": args.seed_offset,
                "target_steps": args.target_steps,
            },
        }

    def save_and_push(at_step: int, completed: bool, reason: str) -> None:
        """Save from rank 0 after the caller has passed the health gate."""
        nonlocal last_archive_bucket
        state = {
            "step": at_step,
            "run_history": run_history + [run_record(at_step, completed)],
            "guard_state": serialized_guard_state(),
        }
        save_checkpoint(CKPT_DIR, model, ema, optimizer, cfg, state)
        print(f"[train] saved step {at_step:,} ({reason})")
        if args.no_push:
            return

        archive = None
        if cfg.hub.archive_every_steps > 0:
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

        lr_controller.apply(optimizer, step)
        x0, y = store.batch(
            step,
            ctx.rank,
            per_rank_batch,
            cfg.train.base_seed + args.seed_offset,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.loss(wrapped, x0, y)

        # Agree before backward so a rank-local NaN cannot strand peers in DDP.
        local_loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        loss_finite = all_reduce_bool_and(ctx, local_loss_finite)
        if not loss_finite:
            local_value = float(loss.detach().float().item())
            return abort_all(
                f"[train] step {step:,}: non-finite loss detected "
                f"(rank-0 local loss={local_value}) -- aborting WITHOUT saving"
            )

        global_loss = all_reduce_mean(ctx, loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Return value is the raw pre-clip norm.  Explicit synchronized handling
        # is safer than error_if_nonfinite=True under DDP.
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.clip, error_if_nonfinite=False
        )
        grads_finite = all_reduce_bool_and(
            ctx, bool(torch.isfinite(grad_norm_tensor).item())
        )
        if not grads_finite:
            optimizer.zero_grad(set_to_none=True)
            return abort_all(
                f"[train] step {step:,}: non-finite gradient norm -- "
                "aborting BEFORE optimizer.step and WITHOUT saving"
            )

        grad_norm = float(grad_norm_tensor.detach().float().item())
        grad_norm = broadcast_float(ctx, grad_norm if ctx.is_rank0 else 0.0)
        relative_spike = (
            args.spike_skip > 0.0
            and grad_ema is not None
            and grad_norm > args.spike_skip * grad_ema
        )
        absolute_spike = args.grad_ceiling > 0.0 and grad_norm > args.grad_ceiling
        spike_decision = (relative_spike or absolute_spike) if ctx.is_rank0 else False
        spiked = broadcast_flag(ctx, spike_decision)

        if spiked:
            optimizer.zero_grad(set_to_none=True)
            spikes_total += 1
            window_spikes += 1
            if ctx.is_rank0:
                reasons: list[str] = []
                if relative_spike:
                    reasons.append(
                        f"relative {grad_norm:.2f} > {args.spike_skip:g}x EMA {grad_ema:.2f}"
                    )
                if absolute_spike:
                    reasons.append(f"absolute {grad_norm:.2f} > ceiling {args.grad_ceiling:g}")
                ema_text = "unset" if grad_ema is None else f"{grad_ema:.2f}"
                print(
                    f"[train] step {step:,}: {' and '.join(reasons)}; "
                    f"grad EMA={ema_text} -- SKIPPING optimizer step "
                    f"(total skipped: {spikes_total})"
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

        recovery_saved = False
        if step % LOSS_WINDOW == 0 and len(recent_losses) >= LOSS_WINDOW:
            window = current_window()
            assert window is not None
            spike_rate = window_spikes / LOSS_WINDOW
            event = lr_controller.observe_window(window, spike_rate)
            window_spikes = 0

            # Rank 0 owns external actions even though every rank updates the
            # deterministic controller from the same synchronized inputs.
            abort_decision = broadcast_flag(
                ctx, event.should_abort if ctx.is_rank0 else False
            )
            backoff_decision = broadcast_flag(
                ctx, event.backed_off if ctx.is_rank0 else False
            )
            recovery_decision = broadcast_flag(
                ctx, event.request_checkpoint if ctx.is_rank0 else False
            )

            if ctx.is_rank0 and event.reason:
                print(
                    f"[train] stability window @ {step:,}: loss={window:.6f}  "
                    f"spikes={spike_rate:.1%}  {event.reason}"
                )
            if backoff_decision:
                new_lr = lr_controller.apply(optimizer, step)
                if ctx.is_rank0:
                    print(
                        f"[train] ADAPTIVE LR BACKOFF @ {step:,}: "
                        f"scale {event.old_scale:.4f} -> {event.new_scale:.4f}; "
                        f"effective lr now {new_lr:.7g}"
                    )
            if abort_decision:
                return abort_all(
                    f"[train] step {step:,}: {event.reason} -- DIVERGENCE, "
                    "aborting WITHOUT saving; Hub latest remains healthy",
                    code=2,
                )

            if recovery_decision:
                healthy, health_reason, _ = checkpoint_is_healthy()
                healthy = broadcast_flag(ctx, healthy if ctx.is_rank0 else False)
                if ctx.is_rank0 and healthy:
                    save_and_push(step, completed=False, reason="post-backoff recovery")
                elif ctx.is_rank0:
                    print(
                        f"[train] recovery checkpoint request withheld: {health_reason}"
                    )
                recovery_saved = healthy
                barrier(ctx)

        # Periodic checkpoint.  A post-backoff recovery save at the same step
        # already performed the expensive upload, so avoid a duplicate.
        if (
            not recovery_saved
            and cfg.hub.save_every_steps > 0
            and step % cfg.hub.save_every_steps == 0
        ):
            healthy = False
            reason = "rank-0 health decision"
            if ctx.is_rank0:
                healthy, reason, window = checkpoint_is_healthy()
                if not healthy:
                    print(
                        f"[train] step {step:,}: WITHHOLDING periodic checkpoint: "
                        f"{reason}; current_window={window}"
                    )
            healthy = broadcast_flag(ctx, healthy)
            if ctx.is_rank0 and healthy:
                save_and_push(step, completed=False, reason="periodic")
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
                family_text = " ".join(
                    f"{key}={value**0.5:7.1f}" for key, value in fams.items()
                )
            grad_text = "unset" if grad_ema is None else f"{grad_ema:.2f}"
            status = lr_controller.status()
            print(
                f"  step {step:>8,}  loss {avg:.5f}  lr {optimizer.param_groups[0]['lr']:.7g}  "
                f"scale {lr_controller.scale:.3f}  {rate:5.2f} steps/s  "
                f"{rate * cfg.train.global_batch:7.0f} img/s  "
                f"|g|ema={grad_text:>7}  trend={status['trend_ratio']:.3f}  "
                f"|w|={parameter_norm:8.2f}  {family_text}"
            )

    # -- final health gate / save -------------------------------------------
    elapsed = time.time() - t_start
    final_healthy = False
    final_reason = "rank-0 health decision"
    final_window = current_window()
    if ctx.is_rank0:
        reached = "target reached" if step >= args.target_steps else "run budget reached"
        print(
            f"[train] {reached} after {elapsed / 60:.1f} min "
            f"({step - start_step:,} data steps this run; global step {step:,})"
        )
        final_healthy, final_reason, final_window = checkpoint_is_healthy()
        if not final_healthy:
            print(
                f"[train] FINAL CHECKPOINT WITHHELD: {final_reason}; "
                f"current_window={final_window}. Hub latest remains unchanged."
            )
    final_healthy = broadcast_flag(ctx, final_healthy)

    if not final_healthy:
        barrier(ctx)
        cleanup(ctx)
        return 2

    if ctx.is_rank0:
        completed = step >= args.target_steps
        save_and_push(
            step,
            completed=completed,
            reason="target final" if completed else "run final",
        )
    barrier(ctx)

    # Fixed-seed sample grid only after a healthy final checkpoint exists.
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
        except Exception as exc:  # noqa: BLE001 - checkpoint is already safe
            print(f"[train] sampling failed (non-fatal): {exc!r}")

    barrier(ctx)
    cleanup(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

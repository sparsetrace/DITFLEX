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

        # PROBE: per-family gradient norms.  Gradients are still present here
        # for BOTH branches below (the skip branch zeroes them a few lines
        # down), so this sees exactly the gradient the spike guard judged.
        # Rank 0 only, read-only, no collectives.  The spike-step prints are
        # the money data: they name the family responsible for each rejected
        # gradient rather than only the steady-state tail.
        if args.probe_attn_logits and ctx.is_rank0 and (
            spiked or step % LOG_EVERY == 0
        ):
            try:
                families = grad_family_norms(model)
                tag = "SPIKE" if spiked else "cadence"
                print(
                    f"[probe] step {step:,} ({tag})  |g|={grad_norm:9.2f}  "
                    f"{format_families(families)}"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics never kill training
                print(f"[probe] grad-family probe failed (non-fatal): {exc!r}")

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

"""Resume-safe learning-rate and loss-stability control.

The published DiT and SiT recipes use AdamW with a constant 1e-4 learning
rate.  This module is deliberately a runtime policy rather than part of
``Config`` so an existing checkpoint can adopt it without tripping the
experiment config-drift guard.

The controller combines two independent mechanisms:

1. A closed-form cosine envelope indexed by the *global* training step.  It
   therefore resumes exactly without relying on scheduler call counts.
2. A monotonic adaptive multiplier.  Sustained loss rises or a high rate of
   rejected gradient spikes reduce the multiplier.  It never increases again
   during the run, which avoids LR oscillation near the end of training.

All state is JSON-serializable and is intended to live under
``state.json["guard_state"]["lr_controller"]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StabilitySpec:
    """Immutable policy settings persisted with the controller state."""

    policy: str = "adaptive"  # constant | cosine | adaptive
    total_steps: int = 400_000
    base_lr: float = 1e-4
    min_lr: float = 1e-5
    hard_min_lr: float = 1e-6

    # Adaptive multiplier settings.
    backoff_factor: float = 0.5
    min_scale: float = 0.125
    patience_windows: int = 2
    cooldown_windows: int = 5
    recovery_windows: int = 2
    emergency_patience: int = 2

    # Loss windows are smoothed before decisions.  ``rise_ratio`` is measured
    # against the slow EMA; ``best_emergency_ratio`` preserves the old 1.6x
    # all-time-best escape hatch as a final catastrophic-loss check.
    fast_decay: float = 0.80
    slow_decay: float = 0.98
    rise_ratio: float = 1.08
    emergency_ratio: float = 1.35
    best_emergency_ratio: float = 1.60

    # Fraction of data steps in a loss window whose optimizer update was
    # rejected by the gradient-spike guard.
    spike_rate_threshold: float = 0.02
    spike_rate_emergency: float = 0.10

    def __post_init__(self) -> None:
        if self.policy not in {"constant", "cosine", "adaptive"}:
            raise ValueError(f"unknown LR policy: {self.policy!r}")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not (self.base_lr > 0.0):
            raise ValueError("base_lr must be positive")
        if not (0.0 <= self.min_lr <= self.base_lr):
            raise ValueError("min_lr must be between 0 and base_lr")
        if not (0.0 < self.hard_min_lr <= max(self.min_lr, self.base_lr)):
            raise ValueError("hard_min_lr must be positive and no larger than the LR range")
        if not (0.0 < self.backoff_factor < 1.0):
            raise ValueError("backoff_factor must be in (0, 1)")
        if not (0.0 < self.min_scale <= 1.0):
            raise ValueError("min_scale must be in (0, 1]")
        if self.patience_windows <= 0 or self.emergency_patience <= 0:
            raise ValueError("patience values must be positive")
        if self.cooldown_windows < 0 or self.recovery_windows <= 0:
            raise ValueError("cooldown must be non-negative and recovery_windows positive")
        if not (0.0 <= self.fast_decay < 1.0 and 0.0 <= self.slow_decay < 1.0):
            raise ValueError("EMA decays must be in [0, 1)")
        if not (1.0 < self.rise_ratio < self.emergency_ratio):
            raise ValueError("expected 1 < rise_ratio < emergency_ratio")
        if self.best_emergency_ratio <= 1.0:
            raise ValueError("best_emergency_ratio must exceed 1")
        if not (0.0 <= self.spike_rate_threshold < self.spike_rate_emergency <= 1.0):
            raise ValueError("invalid spike-rate thresholds")


@dataclass(frozen=True)
class StabilityEvent:
    """Decision produced after one non-overlapping loss window."""

    action: str = "none"  # none | backoff | abort
    reason: str = ""
    old_scale: float = 1.0
    new_scale: float = 1.0
    request_checkpoint: bool = False

    @property
    def backed_off(self) -> bool:
        return self.action == "backoff"

    @property
    def should_abort(self) -> bool:
        return self.action == "abort"


class AdaptiveLrController:
    """Closed-form LR schedule plus a persistent loss-trend controller."""

    VERSION = 1

    def __init__(
        self,
        spec: StabilitySpec,
        *,
        start_step: int,
        checkpoint_lr: float,
        initial_loss: float | None = None,
        legacy_best_loss: float | None = None,
    ) -> None:
        self.spec = spec

        envelope = self.envelope_lr(start_step)
        if spec.policy == "adaptive":
            # Never raise LR when adopting the controller mid-run.  A checkpoint
            # may already contain a manual rescue LR below the cosine envelope.
            inherited = checkpoint_lr / max(envelope, 1e-30)
            hard_floor_scale = spec.hard_min_lr / max(envelope, 1e-30)
            self.scale = min(1.0, max(hard_floor_scale, inherited))
        else:
            self.scale = 1.0

        self.fast_loss: float | None = initial_loss
        self.slow_loss: float | None = initial_loss
        if initial_loss is None:
            self.best_loss = legacy_best_loss
        elif legacy_best_loss is None:
            self.best_loss = initial_loss
        else:
            self.best_loss = min(initial_loss, legacy_best_loss)
        self.last_window_loss: float | None = None
        self.last_trend_ratio = 1.0
        self.last_best_ratio = 1.0
        self.last_spike_rate = 0.0

        self.bad_windows = 0
        self.emergency_windows = 0
        self.cooldown_remaining = 0
        self.backoff_count = 0
        self.windows_seen = 0

        # A backoff is not checkpointed immediately because the weights that
        # triggered it may be unhealthy.  After N healthy windows, the caller
        # is asked to make an out-of-cadence recovery checkpoint.
        self.pending_recovery_checkpoint = False
        self.good_windows_after_backoff = 0

    # -- schedule -----------------------------------------------------

    def envelope_lr(self, step: int) -> float:
        """Return the deterministic LR envelope at a global data step."""
        if self.spec.policy == "constant":
            return self.spec.base_lr
        progress = min(max(int(step), 0), self.spec.total_steps) / self.spec.total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.spec.min_lr + (self.spec.base_lr - self.spec.min_lr) * cosine

    def lr_at(self, step: int) -> float:
        envelope = self.envelope_lr(step)
        if self.spec.policy != "adaptive":
            return envelope
        return max(self.spec.hard_min_lr, envelope * self.scale)

    def apply(self, optimizer: Any, step: int) -> float:
        """Set all optimizer param groups to the LR for ``step``."""
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- observations -------------------------------------------------

    def _can_backoff(self) -> bool:
        return self.spec.policy == "adaptive" and self.scale > self.spec.min_scale * (1.0 + 1e-12)

    def _backoff(self, reason: str) -> StabilityEvent:
        old_scale = self.scale
        self.scale = max(self.spec.min_scale, self.scale * self.spec.backoff_factor)
        self.backoff_count += 1
        self.bad_windows = 0
        self.emergency_windows = 0
        self.cooldown_remaining = self.spec.cooldown_windows
        self.pending_recovery_checkpoint = True
        self.good_windows_after_backoff = 0
        return StabilityEvent(
            action="backoff",
            reason=reason,
            old_scale=old_scale,
            new_scale=self.scale,
        )

    def observe_window(self, loss: float, spike_rate: float) -> StabilityEvent:
        """Observe one aligned loss window and possibly lower the LR.

        ``loss`` must be the globally reduced mean for the window.  ``spike_rate``
        is the fraction of steps in that same window whose optimizer update was
        skipped.  Every DDP rank can call this independently because both inputs
        are already rank-synchronized by the training loop.
        """
        loss = float(loss)
        spike_rate = float(spike_rate)
        if not math.isfinite(loss) or loss < 0.0:
            return StabilityEvent(action="abort", reason=f"invalid window loss {loss!r}")
        if not math.isfinite(spike_rate) or not (0.0 <= spike_rate <= 1.0):
            return StabilityEvent(action="abort", reason=f"invalid spike rate {spike_rate!r}")

        self.windows_seen += 1
        self.last_window_loss = loss
        self.last_spike_rate = spike_rate
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if self.fast_loss is None:
            self.fast_loss = loss
            self.slow_loss = loss
            self.best_loss = loss
            self.last_trend_ratio = 1.0
            self.last_best_ratio = 1.0
            return StabilityEvent()

        assert self.slow_loss is not None and self.best_loss is not None
        self.fast_loss = (
            self.spec.fast_decay * self.fast_loss + (1.0 - self.spec.fast_decay) * loss
        )
        self.slow_loss = (
            self.spec.slow_decay * self.slow_loss + (1.0 - self.spec.slow_decay) * loss
        )
        self.best_loss = min(self.best_loss, self.fast_loss)

        eps = 1e-30
        self.last_trend_ratio = self.fast_loss / max(self.slow_loss, eps)
        self.last_best_ratio = self.fast_loss / max(self.best_loss, eps)

        loss_emergency = (
            self.last_trend_ratio >= self.spec.emergency_ratio
            or self.last_best_ratio >= self.spec.best_emergency_ratio
        )
        spike_emergency = spike_rate >= self.spec.spike_rate_emergency
        emergency = loss_emergency or spike_emergency

        # Require both a rising short/long trend and a modest distance from the
        # best smoothed loss, otherwise ordinary plateau noise can cause decay.
        loss_bad = self.last_trend_ratio >= self.spec.rise_ratio and self.last_best_ratio >= 1.03
        spike_bad = spike_rate >= self.spec.spike_rate_threshold
        bad = loss_bad or spike_bad

        if emergency:
            self.emergency_windows += 1
            self.bad_windows += 1
            reasons: list[str] = []
            if loss_emergency:
                reasons.append(
                    f"loss trend={self.last_trend_ratio:.3f}, best-ratio={self.last_best_ratio:.3f}"
                )
            if spike_emergency:
                reasons.append(f"spike-rate={spike_rate:.1%}")
            reason = "; ".join(reasons)
            if self._can_backoff():
                return self._backoff(f"emergency backoff: {reason}")
            if self.emergency_windows >= self.spec.emergency_patience:
                return StabilityEvent(
                    action="abort",
                    reason=(
                        f"emergency persisted at minimum adaptive scale for "
                        f"{self.emergency_windows} windows: {reason}"
                    ),
                    old_scale=self.scale,
                    new_scale=self.scale,
                )
            return StabilityEvent(reason=f"emergency warning at minimum scale: {reason}")

        self.emergency_windows = 0
        if bad:
            self.bad_windows += 1
            self.good_windows_after_backoff = 0
            reasons = []
            if loss_bad:
                reasons.append(
                    f"loss trend={self.last_trend_ratio:.3f}, best-ratio={self.last_best_ratio:.3f}"
                )
            if spike_bad:
                reasons.append(f"spike-rate={spike_rate:.1%}")
            reason = "; ".join(reasons)
            if (
                self.bad_windows >= self.spec.patience_windows
                and self.cooldown_remaining == 0
                and self._can_backoff()
            ):
                return self._backoff(f"sustained instability: {reason}")
            return StabilityEvent(reason=f"instability warning {self.bad_windows}: {reason}")

        self.bad_windows = 0
        request_checkpoint = False
        if self.pending_recovery_checkpoint:
            self.good_windows_after_backoff += 1
            if self.good_windows_after_backoff >= self.spec.recovery_windows:
                self.pending_recovery_checkpoint = False
                self.good_windows_after_backoff = 0
                request_checkpoint = True
        return StabilityEvent(request_checkpoint=request_checkpoint)

    # -- checkpoint health / state -----------------------------------

    def checkpoint_is_healthy(self, current_window: float | None = None) -> tuple[bool, str]:
        """Conservative health gate for periodic and final checkpoints."""
        if self.fast_loss is None or self.slow_loss is None or self.best_loss is None:
            return True, "controller has not accumulated a full window yet"

        trend = self.last_trend_ratio
        best_ratio = self.last_best_ratio
        if current_window is not None:
            current_window = float(current_window)
            trend = max(trend, current_window / max(self.slow_loss, 1e-30))
            best_ratio = max(best_ratio, current_window / max(self.best_loss, 1e-30))

        if trend >= self.spec.emergency_ratio:
            return False, f"loss trend ratio {trend:.3f} >= {self.spec.emergency_ratio:.3f}"
        if best_ratio >= self.spec.best_emergency_ratio:
            return False, (
                f"loss/best ratio {best_ratio:.3f} >= {self.spec.best_emergency_ratio:.3f}"
            )
        if self.bad_windows > 0:
            return False, f"{self.bad_windows} unresolved instability warning window(s)"
        return True, "stable"

    def status(self) -> dict[str, float | int | bool | None]:
        return {
            "scale": self.scale,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "trend_ratio": self.last_trend_ratio,
            "best_ratio": self.last_best_ratio,
            "spike_rate": self.last_spike_rate,
            "bad_windows": self.bad_windows,
            "emergency_windows": self.emergency_windows,
            "cooldown_remaining": self.cooldown_remaining,
            "backoff_count": self.backoff_count,
            "pending_recovery_checkpoint": self.pending_recovery_checkpoint,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "scale": self.scale,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "last_window_loss": self.last_window_loss,
            "last_trend_ratio": self.last_trend_ratio,
            "last_best_ratio": self.last_best_ratio,
            "last_spike_rate": self.last_spike_rate,
            "bad_windows": self.bad_windows,
            "emergency_windows": self.emergency_windows,
            "cooldown_remaining": self.cooldown_remaining,
            "backoff_count": self.backoff_count,
            "windows_seen": self.windows_seen,
            "pending_recovery_checkpoint": self.pending_recovery_checkpoint,
            "good_windows_after_backoff": self.good_windows_after_backoff,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError(
                f"unsupported LR-controller state version {state.get('version')!r}; "
                "pass --reset-lr-controller to deliberately start a new controller"
            )
        stored_spec = StabilitySpec(**state["spec"])
        if stored_spec != self.spec:
            raise ValueError(
                "LR-controller settings differ from the checkpoint. Keep the same runtime "
                "arguments, or pass --reset-lr-controller for a deliberate policy reset.\n"
                f"stored={asdict(stored_spec)}\ncurrent={asdict(self.spec)}"
            )

        def float_or_none(value: object) -> float | None:
            return None if value is None else float(value)
        self.scale = float(state["scale"])
        self.fast_loss = float_or_none(state.get("fast_loss"))
        self.slow_loss = float_or_none(state.get("slow_loss"))
        self.best_loss = float_or_none(state.get("best_loss"))
        self.last_window_loss = float_or_none(state.get("last_window_loss"))
        self.last_trend_ratio = float(state.get("last_trend_ratio", 1.0))
        self.last_best_ratio = float(state.get("last_best_ratio", 1.0))
        self.last_spike_rate = float(state.get("last_spike_rate", 0.0))
        self.bad_windows = int(state.get("bad_windows", 0))
        self.emergency_windows = int(state.get("emergency_windows", 0))
        self.cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self.backoff_count = int(state.get("backoff_count", 0))
        self.windows_seen = int(state.get("windows_seen", 0))
        self.pending_recovery_checkpoint = bool(
            state.get("pending_recovery_checkpoint", False)
        )
        self.good_windows_after_backoff = int(state.get("good_windows_after_backoff", 0))

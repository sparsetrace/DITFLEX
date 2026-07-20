"""Practical stability control for long DiT/SiT training runs.

This module deliberately favors continuing a finite, loss-stable run over
restarting because one noisy gradient statistic crossed a warning threshold.

Policy summary
--------------
* Warning thresholds are diagnostic only.  Repeated warnings never become a
  retry by themselves.
* A retry requires a clear loss problem, a severe sustained median-gradient
  shift, or multiple corroborating retry-level signals.
* A large p90 or skip rate alone is not enough to roll back; heavy-tailed
  gradients are expected in diffusion/flow training and are already bounded by
  the per-step rejection guard and gradient clipping.
* The reference remains frozen while a candidate is running, then moves slowly
  after a successful checkpoint promotion.

The public API is compatible with the v3 transactional trainer.  Version 4 can
load v1/v2/v3 controller state and preserves the committed LR scale and health
reference while adopting the less-picky decision policy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StabilitySpec:
    """Runtime-only stability and learning-rate settings."""

    policy: str = "adaptive"  # constant | cosine | adaptive
    total_steps: int = 400_000
    base_lr: float = 1e-4
    min_lr: float = 1e-5
    hard_min_lr: float = 1e-6
    min_scale: float = 0.03125

    commit_patience_windows: int = 2
    warning_patience_windows: int = 2  # logging cadence only in v4

    loss_warn_ratio: float = 1.015
    loss_retry_ratio: float = 1.025
    loss_emergency_ratio: float = 1.05

    grad_warn_ratio: float = 2.0
    grad_retry_ratio: float = 4.0
    grad_emergency_ratio: float = 8.0
    grad_p90_warn_ratio: float = 2.5
    grad_p90_retry_ratio: float = 5.0
    grad_p90_emergency_ratio: float = 10.0

    skip_warn_rate: float = 0.05
    skip_retry_rate: float = 0.10
    skip_emergency_rate: float = 0.20

    # A promoted reference can adapt to a genuinely healthy new regime.
    reference_decay: float = 0.80
    loss_reference_max_growth: float = 1.01
    grad_reference_max_growth: float = 1.50

    def __post_init__(self) -> None:
        if self.policy not in {"constant", "cosine", "adaptive"}:
            raise ValueError(f"unknown LR policy: {self.policy!r}")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.base_lr <= 0.0:
            raise ValueError("base_lr must be positive")
        if not (0.0 <= self.min_lr <= self.base_lr):
            raise ValueError("min_lr must lie in [0, base_lr]")
        if not (0.0 < self.hard_min_lr <= self.base_lr):
            raise ValueError("hard_min_lr must lie in (0, base_lr]")
        if not (0.0 < self.min_scale <= 1.0):
            raise ValueError("min_scale must lie in (0, 1]")
        if self.commit_patience_windows <= 0 or self.warning_patience_windows <= 0:
            raise ValueError("window patience values must be positive")
        if not (
            1.0 < self.loss_warn_ratio < self.loss_retry_ratio < self.loss_emergency_ratio
        ):
            raise ValueError("loss ratios must satisfy 1 < warn < retry < emergency")
        if not (1.0 < self.grad_warn_ratio < self.grad_retry_ratio < self.grad_emergency_ratio):
            raise ValueError("gradient ratios must satisfy 1 < warn < retry < emergency")
        if not (
            1.0
            < self.grad_p90_warn_ratio
            < self.grad_p90_retry_ratio
            < self.grad_p90_emergency_ratio
        ):
            raise ValueError("gradient-p90 ratios must satisfy 1 < warn < retry < emergency")
        if not (
            0.0 <= self.skip_warn_rate < self.skip_retry_rate < self.skip_emergency_rate <= 1.0
        ):
            raise ValueError("skip rates must satisfy 0 <= warn < retry < emergency <= 1")
        if not (0.0 <= self.reference_decay < 1.0):
            raise ValueError("reference_decay must lie in [0, 1)")
        if self.loss_reference_max_growth < 1.0 or self.grad_reference_max_growth < 1.0:
            raise ValueError("reference growth caps must be at least 1")


@dataclass(frozen=True)
class WindowMetrics:
    """One non-overlapping, globally synchronized stability window."""

    loss: float
    grad_median: float
    grad_p90: float
    skip_rate: float
    relative_spike_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (self.loss, self.grad_median, self.grad_p90, self.skip_rate)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite window metrics: {self}")
        if self.loss < 0.0 or self.grad_median < 0.0 or self.grad_p90 < 0.0:
            raise ValueError(f"negative window metric: {self}")
        if not (0.0 <= self.skip_rate <= 1.0):
            raise ValueError(f"invalid skip rate: {self.skip_rate}")
        if not (0.0 <= self.relative_spike_rate <= 1.0):
            raise ValueError(f"invalid relative-spike rate: {self.relative_spike_rate}")

    def state_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> WindowMetrics:
        return cls(
            loss=float(state["loss"]),
            grad_median=float(state["grad_median"]),
            grad_p90=float(state["grad_p90"]),
            skip_rate=float(state.get("skip_rate", 0.0)),
            relative_spike_rate=float(state.get("relative_spike_rate", 0.0)),
        )


@dataclass(frozen=True)
class HealthReference:
    """Definition of normality from the last promoted checkpoint."""

    loss: float
    grad_median: float
    grad_p90: float
    step: int
    promotions: int = 1

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HealthReference:
        return cls(
            loss=float(state["loss"]),
            grad_median=float(state["grad_median"]),
            grad_p90=float(state["grad_p90"]),
            step=int(state.get("step", 0)),
            promotions=int(state.get("promotions", 1)),
        )


@dataclass(frozen=True)
class StabilityEvent:
    """Decision after a candidate window."""

    action: str = "none"  # none | warn | retry | fatal
    reason: str = ""
    healthy_windows: int = 0
    warning_windows: int = 0
    loss_ratio: float = 1.0
    grad_ratio: float = 1.0
    grad_p90_ratio: float = 1.0

    @property
    def should_retry(self) -> bool:
        return self.action == "retry"

    @property
    def should_abort(self) -> bool:
        return self.action == "fatal"

    @property
    def promotion_ready(self) -> bool:
        return self.action in {"none", "warn"} and self.healthy_windows > 0


class AdaptiveLrController:
    """Resume-safe LR controller with deliberately tolerant health decisions."""

    VERSION = 4

    def __init__(
        self,
        spec: StabilitySpec,
        *,
        start_step: int,
        checkpoint_lr: float,
        attempt_factor: float = 1.0,
        initial_loss: float | None = None,
        legacy_best_loss: float | None = None,
    ) -> None:
        if not (0.0 < attempt_factor <= 1.0):
            raise ValueError("attempt_factor must lie in (0, 1]")
        self.spec = spec

        envelope = self.envelope_lr(start_step)
        if spec.policy == "adaptive":
            inherited = checkpoint_lr / max(envelope, 1e-30)
            floor = spec.hard_min_lr / max(envelope, 1e-30)
            self.committed_scale = min(1.0, max(floor, inherited))
        else:
            self.committed_scale = 1.0
        self.attempt_factor = float(attempt_factor)

        self.reference: HealthReference | None = None
        self.last_metrics: WindowMetrics | None = None
        self.last_loss_ratio = 1.0
        self.last_grad_ratio = 1.0
        self.last_grad_p90_ratio = 1.0
        self.healthy_windows = 0
        self.warning_windows = 0
        self.retry_windows = 0
        self.windows_seen = 0
        self.retry_count = 0

        self.fast_loss = initial_loss
        self.slow_loss = initial_loss
        if initial_loss is None:
            self.best_loss = legacy_best_loss
        elif legacy_best_loss is None:
            self.best_loss = initial_loss
        else:
            self.best_loss = min(initial_loss, legacy_best_loss)

    # -- learning rate -------------------------------------------------

    @property
    def scale(self) -> float:
        if self.spec.policy != "adaptive":
            return 1.0
        return max(self.spec.min_scale, self.committed_scale * self.attempt_factor)

    def envelope_lr(self, step: int) -> float:
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
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def commit_attempt_scale(self) -> None:
        if self.spec.policy == "adaptive":
            self.committed_scale = self.scale
        self.attempt_factor = 1.0

    # -- reference and per-step guard ----------------------------------

    def bootstrap_reference(
        self,
        *,
        loss: float,
        grad_median: float,
        grad_p90: float | None,
        step: int,
    ) -> None:
        if self.reference is not None:
            return
        median = max(float(grad_median), 1e-12)
        p90 = max(float(grad_p90 if grad_p90 is not None else median * 2.0), median)
        self.reference = HealthReference(
            loss=max(float(loss), 1e-12),
            grad_median=median,
            grad_p90=p90,
            step=int(step),
        )

    def grad_limit(self, spike_multiple: float) -> float | None:
        """Return a frozen pre-clip outlier threshold.

        The threshold is intentionally based on both median and upper-tail
        history.  It does not chase the live EMA during a candidate run.
        """
        if spike_multiple <= 0.0 or self.reference is None:
            return None
        return max(
            float(spike_multiple) * self.reference.grad_median,
            4.0 * self.reference.grad_p90,
        )

    # -- decisions -----------------------------------------------------

    def _ratios(self, metrics: WindowMetrics) -> tuple[float, float, float]:
        assert self.reference is not None
        eps = 1e-30
        return (
            metrics.loss / max(self.reference.loss, eps),
            metrics.grad_median / max(self.reference.grad_median, eps),
            metrics.grad_p90 / max(self.reference.grad_p90, eps),
        )

    def _warning_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []
        if loss_ratio >= self.spec.loss_warn_ratio:
            reasons.append(f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_warn_ratio:.3f}")
        if grad_ratio >= self.spec.grad_warn_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_warn_ratio:.2f}"
            )
        if p90_ratio >= self.spec.grad_p90_warn_ratio:
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= {self.spec.grad_p90_warn_ratio:.2f}"
            )
        if metrics.skip_rate >= self.spec.skip_warn_rate:
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= {self.spec.skip_warn_rate:.1%}"
            )
        return reasons

    def _emergency_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []

        # Loss and median-gradient emergencies are independently meaningful.
        if loss_ratio >= self.spec.loss_emergency_ratio:
            reasons.append(
                f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_emergency_ratio:.3f}"
            )
        if grad_ratio >= self.spec.grad_emergency_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_emergency_ratio:.2f}"
            )

        # A noisy tail or many rejected batches must be corroborated before it
        # can terminate a run.
        if (
            p90_ratio >= self.spec.grad_p90_emergency_ratio
            and grad_ratio >= self.spec.grad_warn_ratio
        ):
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= "
                f"{self.spec.grad_p90_emergency_ratio:.2f} with elevated median"
            )
        if (
            metrics.skip_rate >= self.spec.skip_emergency_rate
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= "
                f"{self.spec.skip_emergency_rate:.1%} with corroborating drift"
            )
        return reasons

    def _retry_reasons(
        self,
        metrics: WindowMetrics,
        loss_ratio: float,
        grad_ratio: float,
        p90_ratio: float,
    ) -> list[str]:
        reasons: list[str] = []

        # Loss drift is the strongest signal and can stand alone.
        if loss_ratio >= self.spec.loss_retry_ratio:
            reasons.append(f"loss ratio {loss_ratio:.3f} >= {self.spec.loss_retry_ratio:.3f}")

        # Median-gradient drift can stand alone only at the retry threshold.
        if grad_ratio >= self.spec.grad_retry_ratio:
            reasons.append(
                f"grad-median ratio {grad_ratio:.2f} >= {self.spec.grad_retry_ratio:.2f}"
            )

        # p90 and skip-rate conditions are too noisy to stand alone.  Require
        # corroboration from loss or the central gradient distribution.
        if (
            p90_ratio >= self.spec.grad_p90_retry_ratio
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"grad-p90 ratio {p90_ratio:.2f} >= "
                f"{self.spec.grad_p90_retry_ratio:.2f} with corroborating drift"
            )
        if (
            metrics.skip_rate >= self.spec.skip_retry_rate
            and (
                loss_ratio >= self.spec.loss_warn_ratio
                or grad_ratio >= self.spec.grad_warn_ratio
            )
        ):
            reasons.append(
                f"skip rate {metrics.skip_rate:.1%} >= "
                f"{self.spec.skip_retry_rate:.1%} with corroborating drift"
            )
        return reasons

    def observe_window(self, metrics: WindowMetrics) -> StabilityEvent:
        self.windows_seen += 1
        self.last_metrics = metrics

        if self.fast_loss is None:
            self.fast_loss = metrics.loss
            self.slow_loss = metrics.loss
            self.best_loss = metrics.loss
        else:
            assert self.slow_loss is not None
            self.fast_loss = 0.80 * self.fast_loss + 0.20 * metrics.loss
            self.slow_loss = 0.98 * self.slow_loss + 0.02 * metrics.loss
            self.best_loss = min(self.best_loss or self.fast_loss, self.fast_loss)

        if self.reference is None:
            self.bootstrap_reference(
                loss=metrics.loss,
                grad_median=max(metrics.grad_median, 1e-12),
                grad_p90=max(metrics.grad_p90, metrics.grad_median, 1e-12),
                step=0,
            )
            self.healthy_windows = 1
            self.warning_windows = 0
            self.retry_windows = 0
            return StabilityEvent(
                action="none",
                reason="bootstrapped committed health reference",
                healthy_windows=1,
            )

        loss_ratio, grad_ratio, p90_ratio = self._ratios(metrics)
        self.last_loss_ratio = loss_ratio
        self.last_grad_ratio = grad_ratio
        self.last_grad_p90_ratio = p90_ratio

        emergency = self._emergency_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if emergency:
            self.healthy_windows = 0
            self.warning_windows += 1
            self.retry_windows += 1
            self.retry_count += 1
            return StabilityEvent(
                action="retry",
                reason="emergency candidate rejection: " + "; ".join(emergency),
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        retry_reasons = self._retry_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        warnings = self._warning_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)

        if retry_reasons:
            self.healthy_windows = 0
            self.warning_windows += 1
            self.retry_windows += 1
            if self.retry_windows >= self.spec.warning_patience_windows:
                self.retry_count += 1
                return StabilityEvent(
                    action="retry",
                    reason="persistent corroborated instability: " + "; ".join(retry_reasons),
                    warning_windows=self.warning_windows,
                    loss_ratio=loss_ratio,
                    grad_ratio=grad_ratio,
                    grad_p90_ratio=p90_ratio,
                )
            return StabilityEvent(
                action="warn",
                reason=(
                    f"retry-level signal {self.retry_windows}/"
                    f"{self.spec.warning_patience_windows}: " + "; ".join(retry_reasons)
                ),
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        # Any acceptable window clears retry persistence.  Warning-only windows
        # still count toward checkpoint promotion because they are explicitly
        # below the retry policy.
        self.retry_windows = 0
        self.healthy_windows += 1

        if warnings:
            self.warning_windows += 1
            return StabilityEvent(
                action="warn",
                reason="diagnostic warning; continuing: " + "; ".join(warnings),
                healthy_windows=self.healthy_windows,
                warning_windows=self.warning_windows,
                loss_ratio=loss_ratio,
                grad_ratio=grad_ratio,
                grad_p90_ratio=p90_ratio,
            )

        self.warning_windows = 0
        return StabilityEvent(
            action="none",
            reason="stable",
            healthy_windows=self.healthy_windows,
            loss_ratio=loss_ratio,
            grad_ratio=grad_ratio,
            grad_p90_ratio=p90_ratio,
        )

    def checkpoint_is_healthy(
        self,
        metrics: WindowMetrics | None = None,
        *,
        required_windows: int | None = None,
    ) -> tuple[bool, str]:
        metrics = metrics or self.last_metrics
        if metrics is None:
            return False, "no complete stability window yet"
        if self.reference is None:
            return False, "no committed health reference"

        required = (
            self.spec.commit_patience_windows
            if required_windows is None
            else max(1, int(required_windows))
        )
        if self.healthy_windows < required:
            return False, f"only {self.healthy_windows}/{required} acceptable windows"

        loss_ratio, grad_ratio, p90_ratio = self._ratios(metrics)
        emergency = self._emergency_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        retry = self._retry_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if emergency:
            return False, "emergency metrics: " + "; ".join(emergency)
        if retry:
            return False, "retry-level metrics: " + "; ".join(retry)

        warnings = self._warning_reasons(metrics, loss_ratio, grad_ratio, p90_ratio)
        if warnings:
            return True, "acceptable candidate with diagnostic warnings"
        return True, "stable candidate"

    @staticmethod
    def _bounded_reference_update(
        old: float,
        new: float,
        *,
        decay: float,
        max_growth: float,
    ) -> float:
        candidate = decay * old + (1.0 - decay) * new
        return min(candidate, old * max_growth)

    def commit_candidate(self, step: int, metrics: WindowMetrics) -> HealthReference:
        self.commit_attempt_scale()
        if self.reference is None:
            reference = HealthReference(
                loss=max(metrics.loss, 1e-12),
                grad_median=max(metrics.grad_median, 1e-12),
                grad_p90=max(metrics.grad_p90, metrics.grad_median, 1e-12),
                step=int(step),
            )
        else:
            old = self.reference
            reference = HealthReference(
                loss=self._bounded_reference_update(
                    old.loss,
                    metrics.loss,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.loss_reference_max_growth,
                ),
                grad_median=self._bounded_reference_update(
                    old.grad_median,
                    metrics.grad_median,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.grad_reference_max_growth,
                ),
                grad_p90=self._bounded_reference_update(
                    old.grad_p90,
                    metrics.grad_p90,
                    decay=self.spec.reference_decay,
                    max_growth=self.spec.grad_reference_max_growth,
                ),
                step=int(step),
                promotions=old.promotions + 1,
            )
        self.reference = reference
        self.healthy_windows = 0
        self.warning_windows = 0
        self.retry_windows = 0
        return reference

    # -- persistence ---------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "committed_scale": self.committed_scale,
            "attempt_factor": self.attempt_factor,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "loss_ratio": self.last_loss_ratio,
            "grad_ratio": self.last_grad_ratio,
            "grad_p90_ratio": self.last_grad_p90_ratio,
            "healthy_windows": self.healthy_windows,
            "warning_windows": self.warning_windows,
            "retry_windows": self.retry_windows,
            "retry_count": self.retry_count,
            "reference": None if self.reference is None else self.reference.state_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "committed_scale": self.committed_scale,
            "attempt_factor": self.attempt_factor,
            "fast_loss": self.fast_loss,
            "slow_loss": self.slow_loss,
            "best_loss": self.best_loss,
            "reference": None if self.reference is None else self.reference.state_dict(),
            "last_metrics": None if self.last_metrics is None else self.last_metrics.state_dict(),
            "last_loss_ratio": self.last_loss_ratio,
            "last_grad_ratio": self.last_grad_ratio,
            "last_grad_p90_ratio": self.last_grad_p90_ratio,
            "healthy_windows": self.healthy_windows,
            "warning_windows": self.warning_windows,
            "retry_windows": self.retry_windows,
            "windows_seen": self.windows_seen,
            "retry_count": self.retry_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        attempt_factor: float | None = None,
    ) -> None:
        version = int(state.get("version", 1))
        if version not in {1, 2, 3, self.VERSION}:
            raise ValueError(
                f"unsupported stability state version {version}; "
                "use --reset-lr-controller only for a deliberate migration"
            )

        stored_spec_data = dict(state.get("spec", {}))

        # Policy thresholds are intentionally allowed to change across v4
        # adoption.  Only LR schedule fields must remain resume-compatible.
        for name in ("policy", "total_steps", "base_lr", "min_lr", "hard_min_lr"):
            if name in stored_spec_data and stored_spec_data[name] != getattr(self.spec, name):
                raise ValueError(
                    f"persisted LR setting {name}={stored_spec_data[name]!r} differs from "
                    f"requested {getattr(self.spec, name)!r}; use --reset-lr-controller "
                    "only when that LR change is deliberate"
                )

        if version < 3:
            self.committed_scale = float(state.get("scale", self.committed_scale))
        else:
            self.committed_scale = float(
                state.get("committed_scale", state.get("scale", self.committed_scale))
            )
            ref_state = state.get("reference")
            if isinstance(ref_state, dict):
                self.reference = HealthReference.from_state_dict(ref_state)
            metrics_state = state.get("last_metrics")
            if isinstance(metrics_state, dict):
                self.last_metrics = WindowMetrics.from_state_dict(metrics_state)
            self.last_loss_ratio = float(state.get("last_loss_ratio", 1.0))
            self.last_grad_ratio = float(state.get("last_grad_ratio", 1.0))
            self.last_grad_p90_ratio = float(state.get("last_grad_p90_ratio", 1.0))
            self.healthy_windows = int(state.get("healthy_windows", 0))
            self.warning_windows = int(state.get("warning_windows", 0))
            self.windows_seen = int(state.get("windows_seen", 0))
            self.retry_count = int(state.get("retry_count", 0))

        # Do not inherit v3's warning-to-retry momentum.
        self.retry_windows = 0
        self.fast_loss = _optional_float(state.get("fast_loss", self.fast_loss))
        self.slow_loss = _optional_float(state.get("slow_loss", self.slow_loss))
        self.best_loss = _optional_float(state.get("best_loss", self.best_loss))

        if attempt_factor is not None:
            if not (0.0 < attempt_factor <= 1.0):
                raise ValueError("attempt_factor must lie in (0, 1]")
            self.attempt_factor = float(attempt_factor)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)

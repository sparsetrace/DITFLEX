from __future__ import annotations

import pytest

from ditflex.stability import AdaptiveLrController, StabilitySpec, WindowMetrics


def metrics(
    loss: float = 1.0,
    grad_median: float = 60.0,
    grad_p90: float = 120.0,
    skip_rate: float = 0.0,
) -> WindowMetrics:
    return WindowMetrics(
        loss=loss,
        grad_median=grad_median,
        grad_p90=grad_p90,
        skip_rate=skip_rate,
        relative_spike_rate=skip_rate,
    )


def controller(spec: StabilitySpec | None = None, attempt_factor: float = 1.0):
    ctl = AdaptiveLrController(
        spec or StabilitySpec(),
        start_step=260_000,
        checkpoint_lr=3e-5,
        attempt_factor=attempt_factor,
        initial_loss=1.0,
    )
    ctl.bootstrap_reference(loss=1.0, grad_median=60.0, grad_p90=120.0, step=260_000)
    return ctl


def test_global_step_cosine_and_no_lr_raise_on_migration():
    spec = StabilitySpec(policy="adaptive", total_steps=400_000)
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=9e-6)
    assert ctl.lr_at(260_000) == pytest.approx(9e-6)
    assert ctl.lr_at(300_000) < ctl.lr_at(260_000)


def test_observed_60_to_4000_regime_requests_immediate_retry():
    ctl = controller()
    event = ctl.observe_window(metrics(loss=1.01, grad_median=4000.0, grad_p90=16000.0))
    assert event.should_retry
    assert "grad-median ratio" in event.reason


def test_persistent_loss_drift_retries_after_patience():
    spec = StabilitySpec(
        warning_patience_windows=2,
        loss_warn_ratio=1.01,
        loss_retry_ratio=1.02,
        loss_emergency_ratio=1.10,
    )
    ctl = controller(spec)
    first = ctl.observe_window(metrics(loss=1.03))
    second = ctl.observe_window(metrics(loss=1.03))
    assert first.action == "warn"
    assert second.should_retry


def test_flat_loss_high_skip_warning_does_not_change_lr():
    ctl = controller()
    before = ctl.scale
    event = ctl.observe_window(metrics(loss=1.0, skip_rate=0.06))
    assert event.action == "warn"
    assert not event.should_retry
    assert ctl.scale == before


def test_frozen_gradient_limit_cannot_chase_bad_live_regime():
    ctl = controller()
    limit_before = ctl.grad_limit(4.0)
    assert limit_before == pytest.approx(max(4.0 * 60.0, 1.25 * 120.0))
    ctl.observe_window(metrics(loss=1.0, grad_median=1000.0, grad_p90=2000.0))
    assert ctl.grad_limit(4.0) == pytest.approx(limit_before)


def test_candidate_requires_consecutive_healthy_windows():
    ctl = controller()
    ctl.observe_window(metrics())
    healthy, reason = ctl.checkpoint_is_healthy(metrics())
    assert not healthy
    assert "1/2" in reason
    ctl.observe_window(metrics())
    healthy, reason = ctl.checkpoint_is_healthy(metrics())
    assert healthy
    assert reason == "stable candidate"


def test_commit_persists_retry_lr_factor_and_caps_reference_growth():
    spec = StabilitySpec(reference_decay=0.0, grad_reference_max_growth=1.25)
    ctl = controller(spec, attempt_factor=0.5)
    effective_before = ctl.scale
    ctl.observe_window(metrics())
    ctl.observe_window(metrics())
    reference = ctl.commit_candidate(
        270_000,
        metrics(loss=1.005, grad_median=100.0, grad_p90=200.0),
    )
    assert ctl.attempt_factor == 1.0
    assert ctl.committed_scale == pytest.approx(effective_before)
    assert reference.grad_median == pytest.approx(75.0)  # 60 * 1.25 cap
    assert reference.grad_p90 == pytest.approx(150.0)  # 120 * 1.25 cap
    assert reference.loss == pytest.approx(1.005)


def test_v2_state_migrates_scale_but_requires_new_reference():
    spec = StabilitySpec()
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    v2_state = {
        "version": 2,
        "spec": {
            "policy": "adaptive",
            "total_steps": 400_000,
            "base_lr": 1e-4,
            "min_lr": 1e-5,
            "hard_min_lr": 1e-6,
        },
        "scale": 0.25,
        "fast_loss": 0.77,
        "slow_loss": 0.77,
        "best_loss": 0.76,
    }
    ctl.load_state_dict(v2_state, attempt_factor=0.5)
    assert ctl.committed_scale == pytest.approx(0.25)
    assert ctl.scale == pytest.approx(0.125)
    assert ctl.reference is None


def test_v3_state_roundtrip_and_spec_drift_guard():
    spec = StabilitySpec(total_steps=400_000)
    ctl = controller(spec)
    ctl.observe_window(metrics())
    state = ctl.state_dict()

    restored = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    changed = AdaptiveLrController(
        StabilitySpec(total_steps=500_000),
        start_step=260_000,
        checkpoint_lr=3e-5,
    )
    with pytest.raises(ValueError, match="spec differs"):
        changed.load_state_dict(state)

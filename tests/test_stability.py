"""Unit tests for the resume-safe LR/loss controller."""

from __future__ import annotations

import pytest

from ditflex.stability import AdaptiveLrController, StabilitySpec


class DummyOptimizer:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


def test_cosine_envelope_matches_global_step():
    spec = StabilitySpec(policy="cosine", total_steps=400_000, base_lr=1e-4, min_lr=1e-5)
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)

    assert ctl.lr_at(0) == pytest.approx(1e-4)
    assert ctl.lr_at(260_000) == pytest.approx(3.4570427512e-5)
    assert ctl.lr_at(400_000) == pytest.approx(1e-5)
    assert ctl.lr_at(500_000) == pytest.approx(1e-5)


def test_adopting_mid_run_never_raises_checkpoint_lr():
    spec = StabilitySpec(policy="adaptive")
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    opt = DummyOptimizer(lr=3e-5)

    assert ctl.apply(opt, 260_000) == pytest.approx(3e-5)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)


def test_sustained_loss_rise_causes_one_backoff():
    spec = StabilitySpec(policy="adaptive", patience_windows=2, cooldown_windows=0)
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)

    assert ctl.observe_window(1.0, 0.0).action == "none"
    assert ctl.observe_window(1.5, 0.0).action == "none"
    event = ctl.observe_window(1.5, 0.0)

    assert event.backed_off
    assert event.old_scale == pytest.approx(1.0)
    assert event.new_scale == pytest.approx(0.5)
    assert ctl.backoff_count == 1


def test_spike_storm_with_flat_loss_does_not_lower_lr():
    spec = StabilitySpec(policy="adaptive", patience_windows=1, cooldown_windows=0)
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)

    ctl.observe_window(1.0, 0.0)
    event = ctl.observe_window(1.0, 0.75)

    assert not event.backed_off
    assert not event.should_abort
    assert ctl.scale == pytest.approx(1.0)
    assert "LR unchanged" in event.reason


def test_spikes_can_corroborate_rising_loss():
    spec = StabilitySpec(
        policy="adaptive",
        patience_windows=1,
        cooldown_windows=0,
        fast_decay=0.0,
        slow_decay=0.9,
    )
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)

    ctl.observe_window(1.0, 0.0)
    event = ctl.observe_window(1.05, 0.20)

    assert event.backed_off


def test_emergency_at_minimum_scale_aborts_after_patience():
    spec = StabilitySpec(policy="adaptive", min_scale=1.0, emergency_patience=2)
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)

    ctl.observe_window(1.0, 0.0)
    first = ctl.observe_window(5.0, 0.0)
    second = ctl.observe_window(5.0, 0.0)

    assert not first.should_abort
    assert second.should_abort
    assert "minimum adaptive scale" in second.reason


def test_recovery_checkpoint_requested_only_after_good_windows():
    spec = StabilitySpec(
        policy="adaptive",
        patience_windows=2,
        cooldown_windows=0,
        recovery_windows=2,
    )
    ctl = AdaptiveLrController(spec, start_step=0, checkpoint_lr=1e-4)
    ctl.observe_window(1.0, 0.0)
    ctl.observe_window(1.5, 0.0)
    assert ctl.observe_window(1.5, 0.0).backed_off

    requested = False
    for _ in range(10):
        event = ctl.observe_window(0.8, 0.0)
        requested = requested or event.request_checkpoint
        if requested:
            break

    assert requested
    assert not ctl.pending_recovery_checkpoint


def test_v1_state_migration_clears_pending_spike_only_decisions():
    spec = StabilitySpec(policy="adaptive")
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    state = ctl.state_dict()
    state["version"] = 1
    state["bad_windows"] = 3
    state["emergency_windows"] = 1
    state["cooldown_remaining"] = 4
    state["pending_recovery_checkpoint"] = True

    restored = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=3e-5)
    restored.load_state_dict(state)

    assert restored.bad_windows == 0
    assert restored.emergency_windows == 0
    assert restored.cooldown_remaining == 0
    assert not restored.pending_recovery_checkpoint
    assert restored.scale == pytest.approx(ctl.scale)


def test_state_roundtrip_and_spec_drift_guard():
    spec = StabilitySpec(policy="adaptive", total_steps=400_000)
    ctl = AdaptiveLrController(spec, start_step=260_000, checkpoint_lr=1e-4)
    ctl.observe_window(1.0, 0.0)
    ctl.observe_window(1.5, 0.0)
    ctl.observe_window(1.5, 0.0)
    state = ctl.state_dict()

    restored = AdaptiveLrController(
        spec,
        start_step=260_000,
        checkpoint_lr=ctl.lr_at(260_000),
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    changed = AdaptiveLrController(
        StabilitySpec(policy="adaptive", total_steps=500_000),
        start_step=260_000,
        checkpoint_lr=1e-4,
    )
    with pytest.raises(ValueError, match="settings differ"):
        changed.load_state_dict(state)

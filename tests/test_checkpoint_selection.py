from __future__ import annotations

from ditflex import checkpoint
from ditflex.checkpoint import CheckpointRevision


def legacy_state(step: int, grad_ema: float) -> dict:
    return {
        "step": step,
        "guard_state": {
            "version": 2,
            "grad_ema": grad_ema,
            "lr_controller": {"version": 2},
        },
    }


def transactional_state(step: int, grad_median: float) -> dict:
    return {
        "step": step,
        "guard_state": {
            "version": 3,
            "stability_controller": {
                "version": 3,
                "reference": {
                    "loss": 0.77,
                    "grad_median": grad_median,
                    "grad_p90": grad_median * 2,
                    "step": step,
                },
            },
        },
    }


def test_legacy_suspect_latest_selects_newest_sane_prior(monkeypatch):
    revisions = [
        CheckpointRevision("rev280", 280_000, 3547.0, legacy_state(280_000, 3547.0)),
        CheckpointRevision("rev270", 270_000, 60.0, legacy_state(270_000, 60.0)),
        CheckpointRevision("rev260", 260_000, 15.0, legacy_state(260_000, 15.0)),
        CheckpointRevision("rev250", 250_000, 16.0, legacy_state(250_000, 16.0)),
    ]
    monkeypatch.setattr(checkpoint, "list_checkpoint_revisions", lambda *a, **k: revisions)
    selection = checkpoint.select_stable_resume_revision("owner/repo", suspect_ratio=8.0)
    assert selection.revision == "rev270"
    assert selection.step == 270_000
    assert "selected prior step 270000" in selection.reason


def test_transactional_latest_is_trusted(monkeypatch):
    state = transactional_state(290_000, 75.0)
    revisions = [CheckpointRevision("rev290", 290_000, 75.0, state)]
    monkeypatch.setattr(checkpoint, "list_checkpoint_revisions", lambda *a, **k: revisions)
    selection = checkpoint.select_stable_resume_revision("owner/repo")
    assert selection.revision is None
    assert selection.step == 290_000
    assert "transactional" in selection.reason

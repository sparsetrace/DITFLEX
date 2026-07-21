"""ditflex.migrate: the qk-norm migration must be exactly understood.

Round-trips a tiny amap checkpoint through migrate_checkpoint and asserts
the load-bearing properties one by one: weights preserved by name, norm
weights fresh ones, EMA carried, AdamW moments attached to the SAME
parameters they belonged to (by name, not index), hyperparameters kept,
step kept, guard reset, and the drift guard behavior on both sides of the
migration. CPU-only."""

from __future__ import annotations

import pytest
import torch

from ditflex.checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA
from ditflex.migrate import migrate_checkpoint
from ditflex.model import build_model


def tiny_config(qk_norm: bool = False) -> Config:
    cfg = Config()
    cfg.model.num_attention_heads = 2
    cfg.model.attention_head_dim = 16
    cfg.model.num_layers = 2
    cfg.model.sample_size = 8
    cfg.model.num_classes = 10
    cfg.model.qk_norm = qk_norm
    cfg.train.objective = "flow"
    return cfg


def make_trained_checkpoint(directory, steps: int = 3) -> tuple[Config, dict]:
    torch.manual_seed(0)
    cfg = tiny_config(qk_norm=False)
    model = build_model(cfg.model)
    ema = EMA(model, decay=cfg.train.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.3e-5, weight_decay=0.0)
    # Populate real AdamW state (exp_avg, exp_avg_sq, step) with synthetic
    # gradients: identical migration mechanics to a trained checkpoint, and
    # CPU-safe (FlexAttention has no CPU backward, so a real backward would
    # break the CPU test workflow).
    for _ in range(steps):
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 1e-3
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
    state = {
        "step": 344_000,
        "run_history": [{"start_step": 0, "end_step": 344_000}],
        "guard_state": {
            "version": 3,
            "spikes_total": 2823,
            "recent_losses": [0.77] * 200,
            "stability_controller": {"version": 4, "committed_scale": 0.282},
        },
    }
    save_checkpoint(directory, model, ema, optimizer, cfg, state)
    return cfg, {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "ema": {k: v.clone() for k, v in ema.state_dict().items()},
        "optim_names": [n for n, _ in model.named_parameters()],
        "optim_state": {
            n: {k: v.clone() for k, v in optimizer.state[p].items() if torch.is_tensor(v)}
            for n, p in model.named_parameters()
            if p in optimizer.state
        },
    }


def test_migration_roundtrip(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    _, before = make_trained_checkpoint(src)

    new_state = migrate_checkpoint(src, dst)
    assert new_state["step"] == 344_000
    validate_checkpoint(dst, expected_step=344_000)

    # Load into the qk_norm architecture through the ordinary path.
    new_cfg = tiny_config(qk_norm=True)
    model = build_model(new_cfg.model)
    ema = EMA(model, decay=new_cfg.train.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.0)
    state = load_checkpoint(dst, model, ema, optimizer, new_cfg)
    assert state["step"] == 344_000

    # 1. Every pre-existing weight preserved by NAME; norm weights are ones.
    new_sd = model.state_dict()
    for name, tensor in before["model"].items():
        assert torch.equal(new_sd[name], tensor), name
    norm_keys = [k for k in new_sd if k.endswith(("norm_q.weight", "norm_k.weight"))]
    assert len(norm_keys) == 2 * new_cfg.model.num_layers
    for key in norm_keys:
        assert torch.equal(new_sd[key], torch.ones_like(new_sd[key])), key

    # 2. EMA carried by name; norm shadow entries are the (ones) online values.
    for name, tensor in before["ema"].items():
        assert torch.equal(ema.shadow[name], tensor), name
    for key in norm_keys:
        assert torch.equal(ema.shadow[key], torch.ones_like(ema.shadow[key]))

    # 3. AdamW moments attached to the SAME named parameters (the index
    #    remap): exp_avg for each old parameter must match exactly, and the
    #    fresh norm parameters must have no state.
    params = dict(model.named_parameters())
    for name, entry in before["optim_state"].items():
        migrated_entry = optimizer.state[params[name]]
        for key, value in entry.items():
            assert torch.allclose(
                migrated_entry[key].float(), value.float()
            ), f"optim state mismatch: {name}.{key}"
    for key in norm_keys:
        assert params[key] not in optimizer.state or not optimizer.state[params[key]]

    # 4. Hyperparameters preserved.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.3e-5)

    # 5. Guard reset: no controller, no recent losses; spike counter carried.
    guard = state["guard_state"]
    assert "stability_controller" not in guard
    assert "recent_losses" not in guard
    assert guard["spikes_total"] == 2823


def test_migrated_checkpoint_refuses_old_config(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    make_trained_checkpoint(src)
    migrate_checkpoint(src, dst)

    old_cfg = tiny_config(qk_norm=False)
    model = build_model(old_cfg.model)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="same experiment"):
        load_checkpoint(dst, model, ema, optimizer, old_cfg)


def test_migration_refuses_wrong_inputs(tmp_path):
    src = tmp_path / "old"
    dst = tmp_path / "new"
    make_trained_checkpoint(src)

    # Already migrated -> refuse a second migration.
    migrate_checkpoint(src, dst)
    with pytest.raises(ValueError, match="already"):
        migrate_checkpoint(dst, tmp_path / "again")


def test_dmap_builder_refuses_qk_norm():
    from ditflex.diffusion_model import build_dmap_model

    cfg = tiny_config(qk_norm=True)
    cfg.model.qk_mode = "dmap"
    with pytest.raises(ValueError, match="not valid for the DMAP chain"):
        build_dmap_model(cfg.model)

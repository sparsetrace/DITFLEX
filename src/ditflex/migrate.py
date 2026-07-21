"""src/ditflex/migrate.py -- one-shot checkpoint migration to qk-norm.

Turns a committed amap checkpoint WITHOUT qk-norm into a complete, valid
checkpoint WITH qk-norm, preserving everything that can be preserved:

  * model weights   -- copied by name; the 2*num_layers new RMSNorm weight
                       tensors initialize to ones (RMSNorm default);
  * EMA shadow      -- copied by name; norm weights seeded from the (ones)
                       online values so EMA covers them from step one;
  * AdamW state     -- THE LOAD-BEARING STEP.  checkpoint.py flattens
                       optimizer state keyed by PARAMETER INDEX.  Inserting
                       norm parameters changes named_parameters() ordering,
                       so index-preserving load would silently attach each
                       parameter's first/second moments to the WRONG
                       parameter.  This module remaps old index -> parameter
                       name (via the old architecture's ordering) -> new
                       parameter object.  New norm parameters start with no
                       state; AdamW lazily initializes them on their first
                       step;
  * hyperparameters -- lr, betas, eps, weight_decay copied from the stored
                       param_groups, so the migrated checkpoint resumes at
                       the LR the chain was actually running;
  * step / history  -- step unchanged; a migration record is appended to
                       run_history.

Deliberately NOT preserved:

  * stability controller / health reference / recent losses -- the
    post-norm gradient regime is a different distribution; carrying the
    contaminated pre-norm reference (grown 38.5 -> 115 through capped
    promotions during the divergence) would misconfigure the guard in both
    directions.  The first post-migration run bootstraps a fresh reference
    from its own first windows (resume with --reset-lr-controller).

Lives in src/ditflex so tests can import it; run/migrate_qknorm.py is the
thin Hub-facing CLI around migrate_checkpoint().
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

from ditflex.checkpoint import _unflatten_optim, save_checkpoint, validate_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA
from ditflex.model import build_model

EXPECTED_NEW_SUFFIXES = ("norm_q.weight", "norm_k.weight")


def _assert_only_norm_keys(keys: list[str], what: str) -> None:
    bad = [k for k in keys if not k.endswith(EXPECTED_NEW_SUFFIXES)]
    if bad:
        raise RuntimeError(f"unexpected non-qk-norm {what} keys: {bad[:5]}")


def migrate_checkpoint(
    source_dir: str | Path,
    dest_dir: str | Path,
    *,
    source_revision: str | None = None,
) -> dict:
    """Migrate one validated local checkpoint directory to qk_norm=True.

    Returns the new state dict (already written to dest_dir along with the
    model/EMA/optim safetensors).  Raises on any structural surprise rather
    than proceeding: a migration that is not exactly understood must not
    produce a checkpoint.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    validate_checkpoint(source_dir)
    state = json.loads((source_dir / "state.json").read_text())
    step = int(state["step"])

    old_cfg = Config.from_dict(state["config"])
    if old_cfg.model.qk_mode != "amap":
        raise ValueError(
            f"qk-norm migration is amap-only; checkpoint has qk_mode="
            f"{old_cfg.model.qk_mode!r}"
        )
    if getattr(old_cfg.model, "qk_norm", False):
        raise ValueError("checkpoint already has qk_norm=True; nothing to migrate")

    new_cfg = Config.from_dict(json.loads(old_cfg.to_json()))  # deep copy via JSON
    new_cfg.model.qk_norm = True

    # -- models -----------------------------------------------------------
    old_model = build_model(old_cfg.model)
    old_sd = load_file(str(source_dir / "model.safetensors"))
    old_model.load_state_dict(old_sd)  # strict: the old architecture must match exactly

    new_model = build_model(new_cfg.model)
    missing, unexpected = new_model.load_state_dict(old_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"old checkpoint has keys the new model lacks: {unexpected[:5]}")
    _assert_only_norm_keys(list(missing), "missing model")
    expected_new = 2 * new_cfg.model.num_layers
    if len(missing) != expected_new:
        raise RuntimeError(
            f"expected exactly {expected_new} new qk-norm tensors, found {len(missing)}"
        )
    # Norm weights keep their RMSNorm init (ones): NOT an identity map on
    # pre-trained Q/K -- resume with the documented reduced-LR warmup.

    # -- EMA --------------------------------------------------------------
    ema = EMA(new_model, decay=old_cfg.train.ema_decay)
    old_ema = load_file(str(source_dir / "ema.safetensors"))
    extra_shadow = sorted(set(ema.shadow) - set(old_ema))
    _assert_only_norm_keys(extra_shadow, "EMA-new")
    stray = sorted(set(old_ema) - set(ema.shadow))
    if stray:
        raise RuntimeError(f"old EMA has keys absent from new model: {stray[:5]}")
    for name, tensor in old_ema.items():
        ema.shadow[name] = tensor.detach().to(dtype=torch.float32, copy=True)
    # Norm entries in the shadow remain the (ones) online values -- EMA
    # tracks them from step one of the resumed run.

    # -- optimizer: index -> name -> new parameter ------------------------
    old_groups = state["optim_param_groups"]
    if len(old_groups) != 1:
        raise RuntimeError(
            f"migration assumes the repo's single param group, found {len(old_groups)}"
        )
    old_tensors = load_file(str(source_dir / "optim.safetensors"))
    old_osd = _unflatten_optim(old_tensors, old_groups)
    old_names = [name for name, _ in old_model.named_parameters()]
    state_indices = set(old_osd["state"].keys())
    if state_indices and max(state_indices) >= len(old_names):
        raise RuntimeError(
            f"optimizer state index {max(state_indices)} exceeds old parameter "
            f"count {len(old_names)} -- ordering assumption violated"
        )

    hp = old_groups[0]
    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=float(hp.get("lr", old_cfg.train.lr)),
        betas=tuple(hp.get("betas", (0.9, 0.999))),
        eps=float(hp.get("eps", 1e-8)),
        weight_decay=float(hp.get("weight_decay", old_cfg.train.weight_decay)),
    )
    new_params = dict(new_model.named_parameters())
    migrated = 0
    for index, name in enumerate(old_names):
        entry = old_osd["state"].get(index)
        if entry is None:
            continue
        if name not in new_params:
            raise RuntimeError(f"old parameter {name!r} not present in new model")
        new_optimizer.state[new_params[name]] = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in entry.items()
        }
        migrated += 1

    # -- state.json -------------------------------------------------------
    run_history = list(state.get("run_history", []))
    run_history.append(
        {
            "start_step": step,
            "end_step": step,
            "seconds": 0.0,
            "world": 0,
            "objective": old_cfg.train.objective,
            "completed": False,
            "finished_at": datetime.now(UTC).isoformat(),
            "promotion_reason": "offline qk-norm migration",
            "effective": {
                "migration": "qk_norm",
                "source_revision": source_revision,
                "new_tensors": expected_new,
                "optim_states_migrated": migrated,
            },
        }
    )
    new_state = {
        "step": step,
        "run_history": run_history,
        # Fresh guard: the post-norm gradient regime must define its own
        # normality.  spikes_total is carried as a cumulative counter only.
        "guard_state": {
            "version": 3,
            "spikes_total": int(state.get("guard_state", {}).get("spikes_total", 0)),
            "migration": "qk_norm",
        },
        "transaction": {
            "status": "committed",
            "committed_at": datetime.now(UTC).isoformat(),
            "migration": "qk_norm",
            "source_revision": source_revision,
            "note": (
                "resume with --qk-norm and --reset-lr-controller; expect a "
                "loss bump at step one (fresh RMSNorms rescale Q/K) and run "
                "a bounded reduced-LR warmup segment first"
            ),
        },
        "migration_summary": {
            "from_step": step,
            "new_tensors": expected_new,
            "optim_states_migrated": migrated,
            "optim_states_total_old": len(state_indices),
            "config_change": {"model.qk_norm": [False, True]},
        },
    }

    save_checkpoint(dest_dir, new_model, ema, new_optimizer, new_cfg, new_state)
    validate_checkpoint(dest_dir, expected_step=step)
    return new_state

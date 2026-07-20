"""Checkpoint storage, validation, Hub revisions, and transactional promotion.

Hub top-level files always represent the last *committed healthy* checkpoint.
Training writes a complete candidate directory first, validates its structure,
and only then promotes it with :func:`push_to_hub`.  A failed candidate is never
uploaded, so a fresh retry process can safely pull Hub latest and roll back the
model, EMA, optimizer moments, step, and stability reference together.
"""

from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ditflex.config import Config

_PREFIXES = ("_orig_mod.", "module.")
FILES = ("state.json", "model.safetensors", "ema.safetensors", "optim.safetensors")


@dataclass(frozen=True)
class CheckpointRevision:
    revision: str
    step: int
    grad_reference: float | None
    state: dict[str, Any]


@dataclass(frozen=True)
class ResumeSelection:
    """A selected Hub revision; ``revision=None`` means ordinary latest."""

    revision: str | None
    step: int | None
    reason: str


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for key, value in sd.items():
        clean_key = key
        for prefix in _PREFIXES:
            while clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        out[clean_key] = value
    return out


# -- optimizer <-> safetensors -------------------------------------------


def _flatten_optim(osd: dict) -> tuple[dict[str, torch.Tensor], list]:
    tensors: dict[str, torch.Tensor] = {}
    for idx, state in osd["state"].items():
        for key, value in state.items():
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            if value.ndim == 0:
                value = value.reshape(1)
                key = f"{key}__scalar"
            tensors[f"{idx}.{key}"] = value.contiguous().cpu()
    return tensors, osd["param_groups"]


def _unflatten_optim(tensors: dict[str, torch.Tensor], param_groups: list) -> dict:
    state: dict[int, dict] = {}
    for flat_key, value in tensors.items():
        idx_text, key = flat_key.split(".", 1)
        if key.endswith("__scalar"):
            key = key[: -len("__scalar")]
            value = value.reshape(())
        state.setdefault(int(idx_text), {})[key] = value
    return {"state": state, "param_groups": param_groups}


def _restore_group_types(loaded_groups: list, reference_groups: list) -> list:
    if len(loaded_groups) != len(reference_groups):
        return loaded_groups
    for loaded, reference in zip(loaded_groups, reference_groups, strict=True):
        for key, value in loaded.items():
            if isinstance(value, list) and isinstance(reference.get(key), tuple):
                loaded[key] = tuple(value)
    return loaded_groups


# -- local save / load / validation --------------------------------------


def save_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    config: Config,
    state: dict,
) -> Path:
    """Write a complete candidate checkpoint using temporary files + rename."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model_state = clean_state_dict(model.state_dict())
    model_state = {
        key: value.detach().float().contiguous().cpu() for key, value in model_state.items()
    }
    ema_state = {
        key: value.contiguous().cpu()
        for key, value in clean_state_dict(ema.state_dict()).items()
    }
    optim_tensors, param_groups = _flatten_optim(optimizer.state_dict())

    full_state = dict(state)
    full_state["config"] = asdict(config)
    full_state["optim_param_groups"] = param_groups
    full_state["torch_version"] = torch.__version__

    for name, payload in (
        ("model.safetensors", model_state),
        ("ema.safetensors", ema_state),
        ("optim.safetensors", optim_tensors),
    ):
        temporary = directory / f"{name}.tmp"
        save_file(payload, str(temporary))
        temporary.replace(directory / name)

    temporary_state = directory / "state.json.tmp"
    temporary_state.write_text(json.dumps(full_state, indent=2))
    temporary_state.replace(directory / "state.json")
    return directory


def validate_checkpoint(
    directory: str | Path,
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Validate candidate structure without reloading multi-GB tensor payloads.

    Safetensors headers are opened and key sets are checked.  Tensor checksums
    and file truncation are handled by the safetensors format itself when the
    header is opened; this deliberately avoids a second 7+ GB device/CPU scan.
    """
    directory = Path(directory)
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint missing files: {missing}")

    state = json.loads((directory / "state.json").read_text())
    if "step" not in state or "config" not in state or "optim_param_groups" not in state:
        raise ValueError("state.json lacks step/config/optim_param_groups")
    step = int(state["step"])
    if expected_step is not None and step != int(expected_step):
        raise ValueError(f"candidate step {step} != expected {expected_step}")

    key_sets: dict[str, set[str]] = {}
    for name in ("model.safetensors", "ema.safetensors", "optim.safetensors"):
        with safe_open(str(directory / name), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if not keys and name != "optim.safetensors":
            raise ValueError(f"{name} contains no tensors")
        key_sets[name] = keys

    # EMA intentionally tracks named parameters, while model.state_dict() may
    # also contain non-trainable buffers.  Therefore EMA keys must be a
    # non-empty subset of model keys, not necessarily an exact match.
    extra_ema = sorted(key_sets["ema.safetensors"] - key_sets["model.safetensors"])
    if extra_ema:
        raise ValueError(f"EMA contains keys absent from model: {extra_ema[:5]}")
    return state


def load_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer | None,
    config: Config,
    allow_config_change: bool = False,
) -> dict:
    """Load raw model, EMA, and optimizer state from one committed checkpoint."""
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())

    stored_config = Config.from_dict(state["config"])
    if stored_config != config and not allow_config_change:
        raise ValueError(
            "checkpoint config differs from current config -- a resumed run "
            "must be the same experiment. Diff state.json against Config().to_json(), "
            "or pass allow_config_change=True only for a documented migration."
        )

    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    ema.load_state_dict(load_file(str(directory / "ema.safetensors")))
    if optimizer is not None:
        optim_tensors = load_file(str(directory / "optim.safetensors"))
        param_groups = _restore_group_types(
            state["optim_param_groups"], optimizer.state_dict()["param_groups"]
        )
        optimizer.load_state_dict(_unflatten_optim(optim_tensors, param_groups))
    return state


def copy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Replace ``destination`` with a local copy of a complete checkpoint."""
    source = Path(source)
    destination = Path(destination)
    validate_checkpoint(source)
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, destination / name)
    return destination


# -- Hub pull / promotion -------------------------------------------------


def pull_from_hub(
    repo_id: str,
    directory: str | Path,
    *,
    revision: str | None = None,
) -> Path | None:
    """Download one committed revision into a clean local directory."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for name in FILES:
            hf_hub_download(
                repo_id,
                name,
                repo_type="model",
                revision=revision,
                local_dir=str(directory),
                force_download=True,
            )
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError):
        shutil.rmtree(directory, ignore_errors=True)
        return None
    validate_checkpoint(directory)
    return directory


def push_to_hub(
    directory: str | Path,
    repo_id: str,
    archive_step: int | None = None,
    *,
    commit_message: str = "checkpoint: promote healthy candidate",
) -> str | None:
    """Promote a validated candidate as Hub latest and return its commit id."""
    from huggingface_hub import HfApi, create_repo

    directory = Path(directory)
    state = validate_checkpoint(directory)
    step = int(state["step"])

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"{commit_message}: step {step}",
    )
    commit_id = getattr(info, "oid", None)

    if archive_step is not None:
        prefix = f"archive/step_{archive_step:07d}"
        for name in ("ema.safetensors", "state.json"):
            api.upload_file(
                path_or_fileobj=str(directory / name),
                path_in_repo=f"{prefix}/{name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"checkpoint: archive healthy step {archive_step}",
            )
    return commit_id


# -- revision inspection / legacy migration ------------------------------


def _state_grad_reference(state: dict[str, Any]) -> float | None:
    guard = state.get("guard_state", {})
    if not isinstance(guard, dict):
        return None

    controller = guard.get("stability_controller")
    if isinstance(controller, dict):
        reference = controller.get("reference")
        if isinstance(reference, dict):
            value = reference.get("grad_median")
            if value is not None and float(value) > 0.0:
                return float(value)

    # v1/v2 compatibility.
    value = guard.get("grad_reference", guard.get("grad_ema"))
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def _state_is_transactional(state: dict[str, Any]) -> bool:
    guard = state.get("guard_state", {})
    controller = guard.get("stability_controller") if isinstance(guard, dict) else None
    return (
        isinstance(controller, dict)
        and int(controller.get("version", 0)) >= 3
        and isinstance(controller.get("reference"), dict)
    )


def list_checkpoint_revisions(repo_id: str, *, max_commits: int = 20) -> list[CheckpointRevision]:
    """Return newest unique checkpoint steps with their lightweight state.json."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    commits = list(api.list_repo_commits(repo_id, repo_type="model"))[:max_commits]
    revisions: list[CheckpointRevision] = []
    seen_steps: set[int] = set()
    for commit in commits:
        try:
            path = hf_hub_download(
                repo_id,
                "state.json",
                repo_type="model",
                revision=commit.commit_id,
                force_download=True,
            )
            state = json.loads(Path(path).read_text())
            step = int(state["step"])
        except Exception:  # noqa: BLE001 - revision ledgers may contain non-checkpoint commits
            continue
        if step in seen_steps:
            continue
        seen_steps.add(step)
        revisions.append(
            CheckpointRevision(
                revision=commit.commit_id,
                step=step,
                grad_reference=_state_grad_reference(state),
                state=state,
            )
        )
    return revisions


def resolve_revision_for_step(repo_id: str, step: int, *, max_commits: int = 200) -> str:
    for item in list_checkpoint_revisions(repo_id, max_commits=max_commits):
        if item.step == int(step):
            return item.revision
    raise ValueError(f"no checkpoint revision in {repo_id!r} reports step {step}")


def infer_legacy_gradient_reference(
    repo_id: str,
    *,
    before_step: int | None = None,
    max_commits: int = 12,
) -> float | None:
    """Robustly infer a pre-v3 gradient baseline from prior committed states."""
    values: list[float] = []
    for item in list_checkpoint_revisions(repo_id, max_commits=max_commits):
        if before_step is not None and item.step >= before_step:
            continue
        if item.grad_reference is not None:
            values.append(item.grad_reference)
    if not values:
        return None
    return float(statistics.median(values))


def select_stable_resume_revision(
    repo_id: str,
    *,
    suspect_ratio: float = 8.0,
    max_commits: int = 12,
) -> ResumeSelection:
    """Auto-avoid a legacy latest checkpoint with a contaminated grad EMA.

    Once v3 has promoted a checkpoint, latest is trusted because it already
    passed transactional health gates.  This heuristic is only for migration
    from v1/v2, where the 280K example saved a grad EMA thousands of units above
    its recent historical scale.
    """
    try:
        revisions = list_checkpoint_revisions(repo_id, max_commits=max_commits)
    except Exception as exc:  # noqa: BLE001 - selection may legitimately target a fresh repo
        return ResumeSelection(None, None, f"no readable checkpoint history: {exc!r}")
    if not revisions:
        return ResumeSelection(None, None, "no checkpoint found; fresh start")

    latest = revisions[0]
    if _state_is_transactional(latest.state):
        return ResumeSelection(None, latest.step, "latest is a v3 transactional checkpoint")

    historical = [
        item.grad_reference
        for item in revisions[1:]
        if item.grad_reference is not None and item.grad_reference > 0.0
    ]
    current = latest.grad_reference
    if current is None or not historical:
        return ResumeSelection(None, latest.step, "insufficient legacy history; using latest")

    baseline = float(statistics.median(historical))
    ratio = current / max(baseline, 1e-30)
    if ratio < suspect_ratio:
        return ResumeSelection(
            None,
            latest.step,
            f"legacy latest grad ratio {ratio:.2f}x is below {suspect_ratio:.2f}x",
        )

    acceptable = max(suspect_ratio / 2.0, 2.0)
    for item in revisions[1:]:
        if item.grad_reference is None:
            continue
        item_ratio = item.grad_reference / max(baseline, 1e-30)
        if item_ratio <= acceptable:
            return ResumeSelection(
                item.revision,
                item.step,
                f"legacy latest step {latest.step} has grad reference {current:.2f} "
                f"({ratio:.1f}x recent median {baseline:.2f}); selected prior step "
                f"{item.step} with ratio {item_ratio:.2f}x",
            )

    return ResumeSelection(
        None,
        latest.step,
        f"legacy latest appears suspect ({ratio:.1f}x), but no safer prior revision was found",
    )

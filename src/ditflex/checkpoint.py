"""src/ditflex/checkpoint.py -- save/load/resume, HF Hub push/pull.

Hub layout (README "Checkpointing"):
    state.json              step, config, run_history, environment
    model.safetensors       fp32 weights, clean names
    ema.safetensors         fp32 EMA shadow
    optim.safetensors       AdamW state, flattened to {param_idx}.{key}
    archive/step_XXXXXXX/   periodic EMA+state snapshots, kept forever

Everything tensor-shaped lives in safetensors (no pickle); AdamW's
param_groups (plain python) ride inside state.json. State dicts are
passed through clean_state_dict defensively -- EMA and the raw-module
save path never produce `_orig_mod.` / `module.` prefixes, but a
checkpoint that cannot load into a bare model for sampling is the
failure mode the README warns about, so we strip anyway.

Resume refuses config drift: state.json embeds the full Config, and
load_checkpoint hard-errors if it differs from the current one. A
resumed run must be the same experiment.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ditflex.config import Config

_PREFIXES = ("_orig_mod.", "module.")
FILES = ("state.json", "model.safetensors", "ema.safetensors", "optim.safetensors")


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for p in _PREFIXES:
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


# -- optimizer <-> safetensors -------------------------------------------


def _flatten_optim(osd: dict) -> tuple[dict[str, torch.Tensor], list]:
    tensors: dict[str, torch.Tensor] = {}
    for idx, st in osd["state"].items():
        for key, v in st.items():
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            if v.ndim == 0:                      # safetensors-safe scalars
                v = v.reshape(1)
                key = f"{key}__scalar"
            tensors[f"{idx}.{key}"] = v.contiguous().cpu()
    return tensors, osd["param_groups"]


def _unflatten_optim(tensors: dict[str, torch.Tensor], param_groups: list) -> dict:
    state: dict[int, dict] = {}
    for flat_key, v in tensors.items():
        idx_s, key = flat_key.split(".", 1)
        if key.endswith("__scalar"):
            key = key[: -len("__scalar")]
            v = v.reshape(())
        state.setdefault(int(idx_s), {})[key] = v
    return {"state": state, "param_groups": param_groups}


# -- save / load ----------------------------------------------------------


def save_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    config: Config,
    state: dict,
) -> Path:
    """Write the four checkpoint files atomically-ish (tmp then rename)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model_sd = clean_state_dict(model.state_dict())
    model_sd = {k: v.detach().float().contiguous().cpu() for k, v in model_sd.items()}
    ema_sd = {k: v.contiguous().cpu() for k, v in clean_state_dict(ema.state_dict()).items()}
    optim_tensors, param_groups = _flatten_optim(optimizer.state_dict())

    full_state = dict(state)
    full_state["config"] = asdict(config)
    full_state["optim_param_groups"] = param_groups
    full_state["torch_version"] = torch.__version__

    for name, payload in (
        ("model.safetensors", model_sd),
        ("ema.safetensors", ema_sd),
        ("optim.safetensors", optim_tensors),
    ):
        tmp = directory / (name + ".tmp")
        save_file(payload, str(tmp))
        tmp.replace(directory / name)

    tmp = directory / "state.json.tmp"
    tmp.write_text(json.dumps(full_state, indent=2))
    tmp.replace(directory / "state.json")
    return directory


def load_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer | None,
    config: Config,
    allow_config_change: bool = False,
) -> dict:
    """Load into the raw (unwrapped, uncompiled) model. Returns state."""
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())

    stored_cfg = Config.from_dict(state["config"])
    if stored_cfg != config and not allow_config_change:
        raise ValueError(
            "checkpoint config differs from current config -- a resumed run "
            "must be the same experiment. Diff the state.json against "
            "Config().to_json(), or pass allow_config_change=True if the "
            "change is deliberate and documented."
        )

    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    ema.load_state_dict(load_file(str(directory / "ema.safetensors")))
    if optimizer is not None:
        optim_tensors = load_file(str(directory / "optim.safetensors"))
        optimizer.load_state_dict(
            _unflatten_optim(optim_tensors, state["optim_param_groups"])
        )
    return state


# -- HF Hub ---------------------------------------------------------------


def push_to_hub(directory: str | Path, repo_id: str, archive_step: int | None = None) -> None:
    """Upload the checkpoint dir as the repo's 'latest'; optionally also
    snapshot EMA+state under archive/step_XXXXXXX/ (kept forever -- the
    top level is overwritten each run, but HF repos are git, so prior
    revisions stay recoverable regardless)."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message="checkpoint: latest",
    )
    if archive_step is not None:
        prefix = f"archive/step_{archive_step:07d}"
        for name in ("ema.safetensors", "state.json"):
            api.upload_file(
                path_or_fileobj=str(Path(directory) / name),
                path_in_repo=f"{prefix}/{name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"checkpoint: archive step {archive_step}",
            )


def pull_from_hub(repo_id: str, directory: str | Path) -> Path | None:
    """Download the latest checkpoint files. Returns the local dir, or
    None if the repo does not exist / has no checkpoint yet (fresh start)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for name in FILES:
            hf_hub_download(
                repo_id, name, repo_type="model",
                local_dir=str(directory),
            )
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    return directory

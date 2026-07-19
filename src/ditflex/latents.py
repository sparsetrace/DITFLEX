"""src/ditflex/latents.py -- GPU-resident latent store, stateless sampling.

Design (README "Design decisions"):
  - NO DataLoader, NO DistributedSampler. The full ~10.5 GB bf16 tensor
    lives on every GPU; a batch is a fancy-index, not an I/O operation.
  - Sampling is STATELESS: indices for (global_step, rank) come from a
    generator seeded by a pure function of (base_seed, global_step, rank).
    Resume is exact from the step counter alone, and survives a change of
    world size.
  - Latents stay bf16 at rest; batches are cast to fp32 on the way out so
    the noising arithmetic in objective.py runs in full precision
    (autocast handles the model matmuls).

Validation on construction repeats the load-bearing checks from
tests/verify_latents.py -- in particular the std ~= 1 check that catches
double-application of the 0.18215 scaling factor. The gate protects the
dataset once; this protects every future load path.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

STD_LO, STD_HI = 0.7, 1.4
_VALIDATE_SAMPLES = 8192


def batch_seed(base_seed: int, global_step: int, rank: int) -> int:
    """Distinct 63-bit seed for every (step, rank); pure and stateless."""
    return (((base_seed + 1) * 1_000_000_007 + global_step) * 8192 + rank) % (2**63 - 1)


class LatentStore:
    def __init__(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        latent_shape: tuple[int, int, int] = (4, 32, 32),
        num_classes: int = 1000,
        validate: bool = True,
    ):
        flat_dim = math.prod(latent_shape)
        if latents.ndim != 2 or latents.shape[1] != flat_dim:
            raise ValueError(f"latents must be [N, {flat_dim}], got {tuple(latents.shape)}")
        if labels.shape != (latents.shape[0],):
            raise ValueError(f"labels must be [{latents.shape[0]}], got {tuple(labels.shape)}")

        self.latents = latents
        self.labels = labels.long()
        self.latent_shape = tuple(latent_shape)

        if validate:
            sub = latents[: min(len(latents), _VALIDATE_SAMPLES)].float()
            std = sub.std().item()
            if not (STD_LO < std < STD_HI):
                hint = (
                    "looks UNSCALED (scaling_factor not applied)"
                    if std > 3.0
                    else "looks DOUBLE-scaled" if std < 0.4 else "unexpected"
                )
                raise ValueError(
                    f"latent std {std:.4f} outside ({STD_LO}, {STD_HI}) -- {hint}. "
                    "The store expects scaling_factor=0.18215 applied exactly once "
                    "at encode time. Do NOT rescale at load."
                )
            if not torch.isfinite(sub).all():
                raise ValueError("non-finite values in latents")
            lo, hi = int(self.labels.min()), int(self.labels.max())
            if lo < 0 or hi >= num_classes:
                raise ValueError(f"labels in [{lo}, {hi}], expected [0, {num_classes - 1}]")

    def __len__(self) -> int:
        return self.latents.shape[0]

    @property
    def device(self) -> torch.device:
        return self.latents.device

    def batch(
        self, global_step: int, rank: int, batch_size: int, base_seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic batch for (global_step, rank).

        Returns x0 [B, *latent_shape] fp32 and labels [B] long, on the
        store's device. Same arguments always produce the same batch."""
        g = torch.Generator().manual_seed(batch_seed(base_seed, global_step, rank))
        idx = torch.randint(0, len(self), (batch_size,), generator=g).to(self.device)
        x0 = self.latents[idx].view(-1, *self.latent_shape).float()
        return x0, self.labels[idx]

    # -- construction -----------------------------------------------------

    @classmethod
    def from_files(
        cls,
        paths: list[Path],
        device: torch.device | str = "cuda",
        **kwargs,
    ) -> LatentStore:
        from safetensors import safe_open

        lat_parts, lab_parts = [], []
        for p in paths:
            with safe_open(str(p), framework="pt", device="cpu") as f:
                lat_parts.append(f.get_tensor("latents"))
                lab_parts.append(f.get_tensor("labels"))
        latents = torch.cat(lat_parts, dim=0).to(device)
        labels = torch.cat(lab_parts, dim=0).to(device)
        return cls(latents, labels, **kwargs)

    @classmethod
    def from_local(cls, directory: str | Path, device="cuda", max_files: int | None = None, **kw):
        paths = sorted(Path(directory).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"no .safetensors under {directory}")
        return cls.from_files(paths[:max_files], device=device, **kw)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "sparsetrace/dlatentzz",
        device="cuda",
        max_files: int | None = None,
        expected_total: int | None = None,
        **kw,
    ) -> LatentStore:
        """Download every latent shard and build the store. With
        max_files=None and expected_total set, asserts the full-dataset
        count -- do this once per training run, at startup."""
        from huggingface_hub import hf_hub_download, list_repo_files

        files = sorted(
            f for f in list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".safetensors")
        )
        if not files:
            raise FileNotFoundError(f"no .safetensors in {repo_id}")
        files = files[:max_files]
        paths = [Path(hf_hub_download(repo_id, f, repo_type="dataset")) for f in files]
        store = cls.from_files(paths, device=device, **kw)
        if expected_total is not None and max_files is None and len(store) != expected_total:
            raise ValueError(f"loaded {len(store):,} latents, expected {expected_total:,}")
        return store

"""src/ditflex/distributed.py -- thin DDP helpers for single-node torchrun.

Degrades to a no-op in a single process (no RANK in env), so every entry
point runs unchanged under `python x.py` and `torchrun --nproc-per-node=8`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world > 1


def setup() -> DistContext:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistContext(rank, world, local_rank, torch.device(f"cuda:{local_rank}"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(rank=0, world=1, local_rank=0, device=device)


def cleanup(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def broadcast_flag(ctx: DistContext, flag: bool) -> bool:
    """Rank 0 decides (e.g. the wall-clock deadline); everyone obeys.
    Avoids per-rank clock drift disagreeing about when to stop."""
    if not ctx.is_distributed:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=ctx.device)
    dist.broadcast(t, src=0)
    return bool(t.item() > 0.5)

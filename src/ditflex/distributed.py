"""src/ditflex/distributed.py -- thin DDP helpers for single-node torchrun.

Degrades to a no-op in a single process (no RANK in env), so every entry
point runs unchanged under ``python x.py`` and
``torchrun --nproc-per-node=8``.
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
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def broadcast_flag(ctx: DistContext, flag: bool) -> bool:
    """Broadcast rank 0's boolean decision to every rank."""
    if not ctx.is_distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=ctx.device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def broadcast_float(ctx: DistContext, value: float, src: int = 0) -> float:
    """Broadcast one float scalar from ``src`` to every rank."""
    if not ctx.is_distributed:
        return float(value)
    t = torch.tensor(float(value), dtype=torch.float64, device=ctx.device)
    dist.broadcast(t, src=src)
    return float(t.item())


def broadcast_int(ctx: DistContext, value: int, src: int = 0) -> int:
    """Broadcast one integer scalar from ``src`` to every rank."""
    if not ctx.is_distributed:
        return int(value)
    t = torch.tensor(int(value), dtype=torch.int64, device=ctx.device)
    dist.broadcast(t, src=src)
    return int(t.item())


def all_reduce_bool_and(ctx: DistContext, flag: bool) -> bool:
    """Return True only when every rank supplied True.

    This is used before backward/optimizer collectives so that one rank cannot
    abort locally while the others continue and deadlock in DDP.
    """
    if not ctx.is_distributed:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def all_reduce_mean(ctx: DistContext, value: torch.Tensor | float) -> float:
    """Return the arithmetic mean of one scalar value across all ranks."""
    if torch.is_tensor(value):
        t = value.detach().float().reshape(()).clone().to(ctx.device)
    else:
        t = torch.tensor(float(value), dtype=torch.float32, device=ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(ctx.world)
    return float(t.item())

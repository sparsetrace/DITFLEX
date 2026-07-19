"""LatentStore: shapes, validation, and the stateless sampling contract.
CPU-only -- the store is device-agnostic by construction."""

from __future__ import annotations

import pytest
import torch

from ditflex.latents import LatentStore, batch_seed


def make_store(n=256, **kw):
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(n, 4096, generator=g).bfloat16()   # std ~1: passes validation
    labels = torch.randint(0, 1000, (n,), generator=g)
    return LatentStore(latents, labels, **kw)


def test_batch_shapes_and_dtypes():
    store = make_store()
    x0, y = store.batch(global_step=0, rank=0, batch_size=8)
    assert x0.shape == (8, 4, 32, 32) and x0.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.int64


def test_sampling_is_stateless_and_deterministic():
    store = make_store()
    a = store.batch(10, 0, 16)
    b = store.batch(10, 0, 16)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ranks_and_steps_draw_different_batches():
    store = make_store()
    x_r0, _ = store.batch(10, 0, 16)
    x_r1, _ = store.batch(10, 1, 16)
    x_s11, _ = store.batch(11, 0, 16)
    assert not torch.equal(x_r0, x_r1)
    assert not torch.equal(x_r0, x_s11)


def test_seed_function_is_injective_over_realistic_ranges():
    seeds = {batch_seed(0, step, rank) for step in range(0, 2000) for rank in range(8)}
    assert len(seeds) == 2000 * 8


def test_double_scaled_latents_are_rejected():
    g = torch.Generator().manual_seed(0)
    bad = (torch.randn(256, 4096, generator=g) * 0.18215).bfloat16()
    labels = torch.zeros(256, dtype=torch.long)
    with pytest.raises(ValueError, match="DOUBLE"):
        LatentStore(bad, labels)


def test_unscaled_latents_are_rejected():
    g = torch.Generator().manual_seed(0)
    bad = (torch.randn(256, 4096, generator=g) * 5.5).bfloat16()
    labels = torch.zeros(256, dtype=torch.long)
    with pytest.raises(ValueError, match="UNSCALED"):
        LatentStore(bad, labels)


def test_wrong_flat_dim_rejected():
    with pytest.raises(ValueError, match="4096"):
        LatentStore(torch.randn(8, 1024).bfloat16(), torch.zeros(8, dtype=torch.long))

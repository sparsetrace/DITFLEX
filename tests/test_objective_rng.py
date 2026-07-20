from __future__ import annotations

from types import SimpleNamespace

import torch

from ditflex.objective import (
    DDPMObjective,
    FlowMatchingObjective,
    make_step_generator,
    objective_seed,
)


class ZeroModel(torch.nn.Module):
    def forward(self, hidden_states, timestep, class_labels):
        return SimpleNamespace(sample=torch.zeros_like(hidden_states))


def test_objective_seed_is_stable_and_namespaced():
    seed = objective_seed(0, 280_000, 0, 1_000_003)
    assert seed == objective_seed(0, 280_000, 0, 1_000_003)
    assert seed != objective_seed(0, 280_001, 0, 1_000_003)
    assert seed != objective_seed(0, 280_000, 1, 1_000_003)
    assert seed != objective_seed(0, 280_000, 0, 2_000_006)


def _loss(objective, seed_offset: int) -> torch.Tensor:
    x0 = torch.randn(8, 4, 4, 4)
    labels = torch.arange(8)
    generator = make_step_generator(
        "cpu",
        base_seed=0,
        global_step=123,
        rank=0,
        seed_offset=seed_offset,
    )
    return objective.loss(ZeroModel(), x0, labels, generator=generator)


def test_flow_objective_replays_exactly_for_same_attempt_seed():
    objective = FlowMatchingObjective(label_dropout=0.5, null_class=1000)
    first = _loss(objective, 17)
    second = _loss(objective, 17)
    # x0 is generated outside the step generator, so set the global RNG to
    # reproduce the input as well.
    torch.manual_seed(42)
    x0 = torch.randn(8, 4, 4, 4)
    labels = torch.arange(8)
    g1 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=17)
    g2 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=17)
    replay1 = objective.loss(ZeroModel(), x0, labels, generator=g1)
    replay2 = objective.loss(ZeroModel(), x0, labels, generator=g2)
    assert replay1.equal(replay2)
    assert torch.isfinite(first) and torch.isfinite(second)


def test_retry_seed_changes_flow_and_ddpm_objective():
    x0 = torch.zeros(8, 4, 4, 4)
    labels = torch.arange(8)
    for objective in (
        FlowMatchingObjective(label_dropout=0.5, null_class=1000),
        DDPMObjective(label_dropout=0.5, null_class=1000),
    ):
        g1 = make_step_generator("cpu", base_seed=0, global_step=123, rank=0, seed_offset=0)
        g2 = make_step_generator(
            "cpu", base_seed=0, global_step=123, rank=0, seed_offset=1_000_003
        )
        loss1 = objective.loss(ZeroModel(), x0, labels, generator=g1)
        loss2 = objective.loss(ZeroModel(), x0, labels, generator=g2)
        assert not loss1.equal(loss2)

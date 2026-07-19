"""EMA math checked exactly, plus copy_to and state round-trip."""

from __future__ import annotations

import torch

from ditflex.ema import EMA


def test_update_is_the_ema_recurrence():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    old = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for name, p in model.named_parameters():
        expected = 0.9 * old[name] + 0.1 * p.detach().float()
        assert torch.allclose(ema.state_dict()[name], expected, atol=1e-7)


def test_copy_to_restores_shadow():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0)
    ema.copy_to(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach().float(), ema.state_dict()[name])


def test_load_rejects_key_mismatch():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model)
    import pytest
    with pytest.raises(KeyError):
        ema.load_state_dict({"wrong": torch.zeros(1)})

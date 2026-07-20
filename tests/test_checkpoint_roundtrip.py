"""Checkpoint round-trip on CPU with a plain model: model + EMA + AdamW
state must survive save->load exactly, and config drift must be refused.
checkpoint.py is model-agnostic, so nn.Sequential is a fair proxy."""

from __future__ import annotations

import pytest
import torch

from ditflex.checkpoint import clean_state_dict, load_checkpoint, save_checkpoint
from ditflex.config import Config
from ditflex.ema import EMA


def make_trained_bits(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    ema = EMA(model, decay=0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):                       # populate optimizer state
        loss = model(torch.randn(4, 8)).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
    return model, ema, opt


def test_roundtrip_exact(tmp_path):
    cfg = Config()
    model, ema, opt = make_trained_bits(seed=0)
    save_checkpoint(tmp_path, model, ema, opt, cfg, {"step": 123, "run_history": []})

    model2, ema2, opt2 = make_trained_bits(seed=1)   # different weights/state
    state = load_checkpoint(tmp_path, model2, ema2, opt2, cfg)

    assert state["step"] == 123
    for k, v in model.state_dict().items():
        assert torch.allclose(model2.state_dict()[k], v)
    for k, v in ema.state_dict().items():
        assert torch.equal(ema2.state_dict()[k], v)
    o1, o2 = opt.state_dict(), opt2.state_dict()
    assert o1["param_groups"] == o2["param_groups"]
    for idx in o1["state"]:
        for key in o1["state"][idx]:
            assert torch.allclose(
                o2["state"][idx][key].float(), o1["state"][idx][key].float()
            ), f"optim state {idx}.{key}"


def test_config_drift_is_refused(tmp_path):
    cfg = Config()
    model, ema, opt = make_trained_bits()
    save_checkpoint(tmp_path, model, ema, opt, cfg, {"step": 1, "run_history": []})

    drifted = Config()
    drifted.train.lr = 3e-4
    with pytest.raises(ValueError, match="same experiment"):
        load_checkpoint(tmp_path, model, ema, opt, drifted)
    # ...unless explicitly allowed
    load_checkpoint(tmp_path, model, ema, opt, drifted, allow_config_change=True)


def test_clean_state_dict_strips_wrapper_prefixes():
    sd = {"_orig_mod.module.blocks.0.w": 1, "module.head.b": 2, "plain": 3}
    assert set(clean_state_dict(sd)) == {"blocks.0.w", "head.b", "plain"}


def test_candidate_validation_allows_model_buffers(tmp_path):
    from ditflex.checkpoint import copy_checkpoint, validate_checkpoint

    class WithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.register_buffer("fixed", torch.ones(3))

        def forward(self, x):
            return self.linear(x) + self.fixed

    source = tmp_path / "candidate"
    copied = tmp_path / "copied"
    model = WithBuffer()
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = Config()
    save_checkpoint(source, model, ema, optimizer, cfg, {"step": 123})

    state = validate_checkpoint(source, expected_step=123)
    assert state["step"] == 123
    copy_checkpoint(source, copied)
    assert validate_checkpoint(copied)["step"] == 123

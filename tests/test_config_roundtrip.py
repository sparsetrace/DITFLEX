"""Config must survive JSON exactly -- it is embedded in every checkpoint
and a resumed run must reconstruct the identical experiment. Pure stdlib;
runs anywhere."""

from __future__ import annotations

from ditflex.config import Config, DataConfig, ModelConfig


def test_default_roundtrip():
    cfg = Config()
    back = Config.from_json(cfg.to_json())
    assert back == cfg


def test_modified_values_survive():
    cfg = Config()
    cfg.train.objective = "flow"
    cfg.train.global_batch = 64
    cfg.model.num_layers = 6
    back = Config.from_json(cfg.to_json())
    assert back == cfg
    assert back.train.objective == "flow"
    assert back.model.num_layers == 6


def test_latent_shape_is_tuple_after_roundtrip():
    # JSON has no tuples; DataConfig.__post_init__ must restore tuple so
    # view(-1, *shape) and equality both behave.
    back = Config.from_json(Config().to_json())
    assert isinstance(back.data.latent_shape, tuple)
    assert back.data.latent_shape == (4, 32, 32)


def test_defaults_are_the_published_recipe():
    m, t = ModelConfig(), Config().train
    assert (m.num_attention_heads * m.attention_head_dim, m.num_layers) == (1024, 24)
    assert m.patch_size == 2 and m.sample_size == 32
    assert t.lr == 1e-4 and t.weight_decay == 0.0
    assert t.ema_decay == 0.9999 and t.label_dropout == 0.1
    assert t.global_batch == 256
    assert DataConfig().expected_total == 1_281_167

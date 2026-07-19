"""ditflex: DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions."""

from ditflex.attention import (
    FlexSelfAttnProcessor,
    IdentityFlexSelfAttnProcessor,
    identity_score_mod,
    reference_self_attention,
)
from ditflex.config import Config, DataConfig, HubConfig, ModelConfig, TrainConfig
from ditflex.diffusion import (
    DmapFlexSelfAttnProcessor,
    doob_score_mod,
    edge_field_score_mod,
    exact_edge_field,
    model_qk_ratios,
    qk_ratio,
    temperature_score_mod,
)
from ditflex.diffusion_model import build_dmap_model
from ditflex.ema import EMA
from ditflex.latents import LatentStore, batch_seed
from ditflex.model import build_model
from ditflex.objective import build_objective

__all__ = [
    "Config",
    "DmapFlexSelfAttnProcessor",
    "EMA",
    "doob_score_mod",
    "edge_field_score_mod",
    "exact_edge_field",
    "model_qk_ratios",
    "qk_ratio",
    "temperature_score_mod",
    "DataConfig",
    "FlexSelfAttnProcessor",
    "HubConfig",
    "IdentityFlexSelfAttnProcessor",
    "LatentStore",
    "ModelConfig",
    "TrainConfig",
    "batch_seed",
    "build_dmap_model",
    "build_model",
    "build_objective",
    "identity_score_mod",
    "reference_self_attention",
]
__version__ = "0.1.0"

"""ditflex: DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions."""

from ditflex.attention import (
    FlexSelfAttnProcessor,
    IdentityFlexSelfAttnProcessor,
    identity_score_mod,
    reference_self_attention,
)
from ditflex.config import Config, DataConfig, HubConfig, ModelConfig, TrainConfig
from ditflex.latents import LatentStore, batch_seed
from ditflex.model import build_model
from ditflex.objective import build_objective

__all__ = [
    "Config",
    "DataConfig",
    "FlexSelfAttnProcessor",
    "HubConfig",
    "IdentityFlexSelfAttnProcessor",
    "LatentStore",
    "ModelConfig",
    "TrainConfig",
    "batch_seed",
    "build_model",
    "build_objective",
    "identity_score_mod",
    "reference_self_attention",
]
__version__ = "0.1.0"

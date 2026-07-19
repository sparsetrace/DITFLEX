"""src/ditflex/model.py -- build DiT-L/2 with FlexAttention installed.

The ONLY place a model is constructed. Every entry point (training,
sampling, eval, tests) builds through here, so the Flex processor and the
config-to-architecture mapping cannot drift between them.

Processor installation walks the modules and calls the per-module
Attention.set_processor() -- NOT the model-level set_attn_processor()
convenience method, which newer diffusers removed from
DiTTransformer2DModel. The walk also counts what it touched and demands
exactly one self-attention per layer, so an architecture surprise fails
here, loudly, instead of surfacing as an uninterpretable training curve.

torch.compile does NOT happen here -- train.py compiles. Tests and the
identity gate need the uncompiled module.
"""

from __future__ import annotations

from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.attention import FlexSelfAttnProcessor, ScoreMod
from ditflex.config import ModelConfig


def install_flex_processors(model, score_mod: ScoreMod | None = None) -> int:
    """Install FlexSelfAttnProcessor on every diffusers Attention module.
    Returns the number of modules touched."""
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.set_processor(FlexSelfAttnProcessor(score_mod=score_mod))
            count += 1
    return count


def build_model(cfg: ModelConfig, score_mod: ScoreMod | None = None) -> DiTTransformer2DModel:
    """DiT at cfg's geometry, self-attention routed through FlexAttention.

    score_mod=None installs the identity baseline; passing a score_mod is
    the experiment. Nothing else changes between the two."""
    model = DiTTransformer2DModel(
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_layers=cfg.num_layers,
        sample_size=cfg.sample_size,
        patch_size=cfg.patch_size,
        num_embeds_ada_norm=cfg.num_classes + 1,   # +1: CFG null class
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
    )

    n_installed = install_flex_processors(model, score_mod)
    if n_installed != cfg.num_layers:
        raise RuntimeError(
            f"installed Flex on {n_installed} Attention modules but the config "
            f"has {cfg.num_layers} layers -- the DiT architecture is not one "
            "self-attention per block as assumed. Do not train on this."
        )

    # GUARD, not a feature: this builder produces ONLY the certified
    # baseline DiT. Variant configs must go through their own builders so
    # a dmap-labeled config can never silently yield an untied baseline.
    if getattr(cfg, "qk_mode", "amap") != "amap":
        raise ValueError(
            f"build_model builds the baseline only; qk_mode={cfg.qk_mode!r} "
            "requires ditflex.diffusion_model.build_dmap_model."
        )
    return model

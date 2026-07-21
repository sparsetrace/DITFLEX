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

QK-NORM (cfg.qk_norm, amap only; DEVIATION adopted at the 344K
migration): installs torch.nn.RMSNorm(head_dim, eps=1e-6) as norm_q and
norm_k on every Attention module. The Flex processor applies them
per-head after the head reshape (see attention.py). The parameters are
named ``...attn1.norm_q.weight`` / ``...attn1.norm_k.weight`` and are
therefore covered by EMA, checkpointing, and the migration tooling
(ditflex.migrate) by name.

torch.compile does NOT happen here -- train.py compiles. Tests and the
identity gate need the uncompiled module.
"""

from __future__ import annotations

import torch
from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.attention import QK_NORM_EPS, FlexSelfAttnProcessor, ScoreMod
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


def install_qk_norms(model, head_dim: int) -> int:
    """Install per-head RMSNorm(head_dim) as norm_q/norm_k on every
    Attention module. Returns the number of modules touched.

    Assigning the modules as attributes registers them as submodules, so
    the new parameters appear in named_parameters(), state_dict(), the
    EMA shadow, and the optimizer -- everything downstream keys by name.
    Weight init is RMSNorm's default (ones), which is NOT an identity
    transform on pre-trained Q/K (it rescales each head vector to unit
    RMS): a migrated checkpoint needs the short reduced-LR warmup
    documented in run/migrate_qknorm.py.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.norm_q = torch.nn.RMSNorm(head_dim, eps=QK_NORM_EPS)
            module.norm_k = torch.nn.RMSNorm(head_dim, eps=QK_NORM_EPS)
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

    if getattr(cfg, "qk_norm", False):
        # GUARD: qk-norm is an amap-chain deviation only. A dmap-labeled
        # config never reaches here (rejected below), but a future builder
        # calling this with a variant config must decide explicitly.
        if getattr(cfg, "qk_mode", "amap") != "amap":
            raise ValueError(
                "qk_norm=True is defined for the amap baseline only; "
                f"qk_mode={cfg.qk_mode!r} must not carry it (untied norms break "
                "R == 0; tied norms flatten the DMAP destination potential)."
            )
        n_normed = install_qk_norms(model, cfg.attention_head_dim)
        if n_normed != cfg.num_layers:
            raise RuntimeError(
                f"installed qk-norm on {n_normed} Attention modules, expected "
                f"{cfg.num_layers}."
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

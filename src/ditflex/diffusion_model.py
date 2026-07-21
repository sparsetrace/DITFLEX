"""src/ditflex/diffusion_model.py -- the DMAP-DiT: diffusion.py's
mechanism combined with the transformer.

model.py stays the untouched, gate-certified baseline builder; this
module builds the variant by calling it and applying the paper's surgery
on top:

    1) build the baseline DiT (geometry identical, Flex identity
       processors installed, layer count verified by the frozen builder);
    2) tie W_K := W_Q in every attention layer -- scores become
       q_i . q_j, symmetric, R == 0 identically through training.
       Sharing the Module (not copying weights) keeps the constraint
       exact; state_dict and EMA dedupe the shared parameters;
    3) replace every processor with DmapFlexSelfAttnProcessor --
       row-normalization of the squared-distance kernel exp(-H) with the
       surviving destination potential, plus the Coifman-Lafon Doob
       correction when cfg.dmap_alpha > 0.

QK-NORM IS REFUSED HERE, deliberately.  The amap chain adopted per-head
RMSNorm on Q/K at its 344K migration (unbounded-logit instability).  It
does not transfer to this chain: untied norm_q/norm_k would make the
effective bilinear asymmetric (destroying the R == 0 invariant that IS
the experiment), and even a tied norm forces every |q_j| toward unit
RMS, flattening the destination potential g_j = scale*|q_j|^2 whose
survival in the softmax is the defining difference between DMAP and
plain attention (Sec. 3.1).  A "normed DMAP" is a third architecture,
not a stabilized DMAP.  Pre-committed trigger recorded here: if this
chain's probe shows sustained logit growth or grad-median ratcheting in
the 200-280K range (where the amap chain's pathology developed), design
a DMAP-appropriate intervention then -- as its own documented arm.

Dependency direction: baseline -> paper, never backward. model.py and
attention.py import nothing from here or from diffusion.py.
"""

from __future__ import annotations

from dataclasses import replace

from diffusers import DiTTransformer2DModel
from diffusers.models.attention_processor import Attention

from ditflex.config import ModelConfig
from ditflex.diffusion import DmapFlexSelfAttnProcessor
from ditflex.model import build_model


def build_dmap_model(cfg: ModelConfig) -> DiTTransformer2DModel:
    if cfg.qk_mode != "dmap":
        raise ValueError(f"build_dmap_model expects qk_mode='dmap', got {cfg.qk_mode!r}")
    if getattr(cfg, "qk_norm", False):
        raise ValueError(
            "qk_norm=True is not valid for the DMAP chain: untied norms break "
            "the R == 0 symmetry; a tied norm flattens the destination "
            "potential g_j that defines the DMAP kernel. If this chain ever "
            "develops the amap chain's logit pathology, design a "
            "DMAP-appropriate intervention as its own documented arm (see "
            "module docstring)."
        )

    # The frozen builder constructs the geometry (and refuses dmap configs
    # by design), so hand it an amap-labeled copy of the same geometry.
    model = build_model(replace(cfg, qk_mode="amap"))

    n_applied = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.to_k = module.to_q
            module.set_processor(DmapFlexSelfAttnProcessor(alpha=cfg.dmap_alpha))
            n_applied += 1
    if n_applied != cfg.num_layers:
        raise RuntimeError(
            f"applied DMAP surgery to {n_applied} layers, expected {cfg.num_layers}"
        )
    return model

""" src/ditflex/__init__.py
ditflex: DiT-L/2 on ImageNet-256 latents with swappable FlexAttention score functions.
"""

from ditflex.attention import FlexSelfAttnProcessor, reference_self_attention

__all__ = ["FlexSelfAttnProcessor", "reference_self_attention"]
__version__ = "0.1.0"

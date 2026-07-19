"""The DMAP variant must be exactly what it claims: W_K is W_Q (the same
Module, not a copy), scores symmetric, R identically zero -- and the
baseline must remain untied with R > 0 at init."""

from __future__ import annotations

import pytest
import torch

from ditflex.config import ModelConfig
from ditflex.diffusion import attention_qk_ratios
from ditflex.diffusion_model import build_dmap_model
from ditflex.model import build_model


def tiny(**kw):
    return ModelConfig(
        num_attention_heads=2, attention_head_dim=8, num_layers=2,
        sample_size=8, patch_size=2, num_classes=10, **kw,
    )


def attn_modules(model):
    from diffusers.models.attention_processor import Attention
    return [m for m in model.modules() if isinstance(m, Attention)]


def test_dmap_ties_every_layer_and_r_is_zero():
    model = build_dmap_model(tiny(qk_mode="dmap"))
    for attn in attn_modules(model):
        assert attn.to_k is attn.to_q          # shared Module, not a copy
        r = attention_qk_ratios(attn)
        assert r.max().item() < 1e-6           # B = Wq^T Wq: symmetric exactly


def test_amap_baseline_stays_untied():
    model = build_model(tiny(qk_mode="amap"))
    for attn in attn_modules(model):
        assert attn.to_k is not attn.to_q
        assert attention_qk_ratios(attn).min().item() > 0.5   # random init R ~= 1


def test_tying_survives_state_dict_roundtrip():
    torch.manual_seed(0)
    m1 = build_dmap_model(tiny(qk_mode="dmap"))
    m2 = build_dmap_model(tiny(qk_mode="dmap"))
    m2.load_state_dict(m1.state_dict())
    for a1, a2 in zip(attn_modules(m1), attn_modules(m2), strict=True):
        assert torch.equal(a1.to_q.weight, a2.to_q.weight)
        assert a2.to_k is a2.to_q


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="qk_mode|diffusion_model"):
        build_model(tiny(qk_mode="cmap"))


def test_dmap_installs_processor_with_default_alpha():
    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    model = build_dmap_model(tiny(qk_mode="dmap"))
    for attn in attn_modules(model):
        assert isinstance(attn.processor, DmapFlexSelfAttnProcessor)
        assert attn.processor.alpha == 0.0


def test_baseline_builder_refuses_dmap_configs():
    """The guard in the frozen builder: a dmap config can never silently
    yield an untied baseline."""
    with pytest.raises(ValueError, match="diffusion_model"):
        build_model(tiny(qk_mode="dmap"))


def test_dmap_builder_refuses_amap_configs():
    with pytest.raises(ValueError, match="dmap"):
        build_dmap_model(tiny(qk_mode="amap"))


def _dense_dmap_reference(attn, x, alpha):
    """Textbook construction in fp64: H_ij = g_i + g_j - 2 s_ij,
    P = exp(-H), optional Doob tilt by degrees^{-alpha}, row-normalize,
    apply to V, project out."""
    heads, head_dim = attn.heads, attn.to_q.weight.shape[0] // attn.heads
    b, n, _ = x.shape
    x64 = x.double()
    q = x64 @ attn.to_q.weight.double().T + attn.to_q.bias.double()
    k = x64 @ attn.to_k.weight.double().T + attn.to_k.bias.double()
    v = x64 @ attn.to_v.weight.double().T + attn.to_v.bias.double()
    q = q.view(b, n, heads, head_dim).transpose(1, 2)
    k = k.view(b, n, heads, head_dim).transpose(1, 2)
    v = v.view(b, n, heads, head_dim).transpose(1, 2)
    s = (q @ k.transpose(-2, -1)) * attn.scale
    g = s.diagonal(dim1=-2, dim2=-1)                                   # [B,H,N]
    H = g.unsqueeze(-1) + g.unsqueeze(-2) - 2.0 * s
    P = torch.exp(-H)
    if alpha > 0:
        deg = P.sum(dim=-1)                                            # q_i
        P = P * deg.pow(-alpha).unsqueeze(-2)                          # tilt dests
    probs = P / P.sum(dim=-1, keepdim=True)
    out = (probs @ v).transpose(1, 2).reshape(b, n, heads * head_dim)
    return out @ attn.to_out[0].weight.double().T + attn.to_out[0].bias.double()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("alpha", [0.0, 1.0], ids=["alpha0", "alpha1"])
def test_dmap_processor_matches_dense_math(alpha):
    from diffusers.models.attention_processor import Attention

    from ditflex.attention import IdentityFlexSelfAttnProcessor
    from ditflex.diffusion import DmapFlexSelfAttnProcessor

    torch.manual_seed(0)
    heads, head_dim, n, c = 2, 8, 32, 16
    attn = Attention(query_dim=c, heads=heads, dim_head=head_dim, dropout=0.0, bias=True)
    attn = attn.to(device="cuda", dtype=torch.float32).eval()
    attn.to_k = attn.to_q                                # DMAP semantics: tied
    x = torch.randn(2, n, c, device="cuda")

    attn.set_processor(DmapFlexSelfAttnProcessor(alpha=alpha))
    with torch.no_grad():
        got = attn(x)
    ref = _dense_dmap_reference(attn, x, alpha)

    max_rel = ((got.double() - ref).abs().max() / (ref.abs().max() + 1e-12)).item()
    assert max_rel < 1e-4, f"flex vs dense DMAP(alpha={alpha}): max_rel={max_rel:.3e}"

    # The surviving destination potential means DMAP(0) != plain attention.
    if alpha == 0.0:
        attn.set_processor(IdentityFlexSelfAttnProcessor())
        with torch.no_grad():
            plain = attn(x)
        assert (got - plain).abs().max().item() > 1e-3, (
            "DMAP(alpha=0) equals plain attention -- the destination "
            "potential g_j is missing"
        )

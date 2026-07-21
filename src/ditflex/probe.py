"""src/ditflex/probe.py -- opt-in training diagnostics. NEVER on the production path.

Two questions, answered with data instead of pattern-matching:

  1. WHICH parameter family produces the gradient tail?
     grad_family_norms() walks named_parameters AFTER backward (grads must
     still be present) and reduces squared grad norms into the same six
     families train.py's weight-norm log already uses (qk / vo / mlp /
     ada / emb / oth). If one family's grad norm dominates and grows across
     a bounded diagnostic run, the culprit has a name in the logs.

  2. ARE attention logits growing?
     attention_logit_probe() captures every Attention module's input with
     forward pre-hooks, recomputes the QK logits EXPLICITLY from the
     module's own projections in fp32 (no fused kernel, same spirit as
     reference_self_attention), and returns per-layer |logit| max.
     Calibration: healthy DiT attention logits sit roughly 5-20; values of
     50+ that climb over a few thousand steps are the attention-logit-growth
     signature, and QK-norm is the established fix. If instead logits stay
     tame while the 'ada' grad family dominates, the adaLN modulation MLPs
     are the target and the surgery is different (per-group weight decay),
     with no new parameters and no optimizer-state migration.

Both helpers are rank-0-only by convention, use NO collectives, and are
read-only with respect to training state: safe to call on the RAW
(uncompiled, unwrapped) module in the middle of a DDP training step. The
probe forward runs the eager Flex path (the model handed in is the raw
module train.py keeps around for exactly this kind of use); at N=256 and a
small probe batch this is well under a second per stability window.

Everything here is gated behind --probe-attn-logits in train.py. With the
flag off, train.py never calls into this module and the production chain
is byte-for-byte the same behavior as before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FAMILIES = ("qk", "vo", "mlp", "ada", "emb", "oth")


def family_of(name: str) -> str:
    """Identical classification to train.py's LOG_EVERY weight-norm block.

    Kept as a duplicate on purpose: probe.py must not import from the
    training loop, and the production loop must not import its logging
    taxonomy from a diagnostics module.
    """
    if "to_q" in name or "to_k" in name:
        return "qk"
    if "to_v" in name or "to_out" in name:
        return "vo"
    if ".ff." in name:
        return "mlp"
    if "norm1" in name or "norm_out" in name or "adaln" in name.lower():
        return "ada"
    if "emb" in name or "pos_embed" in name or "proj_out" in name:
        return "emb"
    return "oth"


@torch.no_grad()
def grad_family_norms(model: torch.nn.Module) -> dict[str, float]:
    """Per-family L2 gradient norms. Call after backward, before zero_grad.

    Returns {family: ||grad||_2} with families summed in squared space, so
    the values compose exactly like the global grad norm train.py already
    logs (global**2 == sum over families of family**2, up to shared-param
    dedup in the dmap chain, where tied to_k/to_q gradients accumulate
    into the one shared parameter and are counted once, in 'qk').
    """
    squares = dict.fromkeys(FAMILIES, 0.0)
    seen: set[int] = set()  # dmap ties to_k to to_q: count shared params once
    for name, parameter in model.named_parameters():
        if parameter.grad is None or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        squares[family_of(name)] += (
            parameter.grad.detach().float().pow(2).sum().item()
        )
    return {key: value**0.5 for key, value in squares.items()}


def format_families(norms: dict[str, float]) -> str:
    return "  ".join(f"{key}={norms[key]:9.2f}" for key in FAMILIES)


def _make_capture_hook(store: dict[str, torch.Tensor], name: str):
    def hook(_module, args):
        # args[0] is hidden_states [B, N, C]; keep it for the explicit
        # logit recomputation below. detach: this is a read-only probe.
        store[name] = args[0].detach()

    return hook


@torch.no_grad()
def attention_logit_probe(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    *,
    t_value: float = 500.0,
    autocast_dtype: torch.dtype = torch.bfloat16,
    top_k: int = 5,
) -> dict:
    """Max |QK logit| per attention layer, computed explicitly in fp32.

    The forward runs under the same autocast dtype as training so captured
    inputs reflect the numerics the model actually sees; the logits
    themselves are then recomputed in fp32 from the module's own to_q/to_k
    weights (explicit matmul, no fused kernel), so the number reported is
    the mathematical logit, not a kernel artifact.

    Args:
        model:  the RAW module (uncompiled, unwrapped). Never the DDP wrapper.
        x0:     [B, 4, 32, 32] probe batch (a small slice of the current
                training batch is fine; B=8 keeps captures under ~150 MB).
        labels: [B] class labels for the same slice.
        t_value: timestep passed to the model; the DiT embedder accepts a
                float for both objectives, and mid-schedule (500) is a
                representative operating point. Logit growth, when present,
                shows up at every t.

    Returns dict with:
        per_layer: {module_name: max_abs_logit}
        max:       overall max
        argmax:    name of the layer attaining it
        top:       [(short_name, value)] for the top_k layers, descending
    """
    from diffusers.models.attention_processor import Attention

    attention_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Attention)
    ]
    if not attention_modules:
        raise ValueError("no diffusers Attention modules found on the probe model")

    captured: dict[str, torch.Tensor] = {}
    handles = [
        module.register_forward_pre_hook(_make_capture_hook(captured, name))
        for name, module in attention_modules
    ]

    was_training = model.training
    model.eval()
    try:
        t = torch.full((x0.shape[0],), float(t_value), device=x0.device)
        if x0.is_cuda and autocast_dtype != torch.float32:
            with torch.autocast("cuda", dtype=autocast_dtype):
                model(hidden_states=x0, timestep=t, class_labels=labels)
        else:
            # fp32 / TF32 training mode, or the CPU path used by unit tests:
            # run the plain forward so captures match training numerics.
            model(hidden_states=x0, timestep=t, class_labels=labels)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    per_layer: dict[str, float] = {}
    for name, module in attention_modules:
        hidden = captured[name].float()
        weight_q = module.to_q.weight.detach().float()
        weight_k = module.to_k.weight.detach().float()
        bias_q = None if module.to_q.bias is None else module.to_q.bias.detach().float()
        bias_k = None if module.to_k.bias is None else module.to_k.bias.detach().float()

        query = F.linear(hidden, weight_q, bias_q)
        key = F.linear(hidden, weight_k, bias_k)

        batch, seq_len, _ = hidden.shape
        heads = module.heads
        head_dim = query.shape[-1] // heads
        query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        logits = (query @ key.transpose(-2, -1)) * module.scale
        per_layer[name] = logits.abs().amax().item()

    ranked = sorted(per_layer.items(), key=lambda kv: kv[1], reverse=True)
    argmax_name, max_value = ranked[0]

    def short(name: str) -> str:
        # "transformer_blocks.17.attn1" -> "blk17"
        for token in name.split("."):
            if token.isdigit():
                return f"blk{token}"
        return name

    return {
        "per_layer": per_layer,
        "max": max_value,
        "argmax": short(argmax_name),
        "top": [(short(n), round(v, 2)) for n, v in ranked[:top_k]],
    }

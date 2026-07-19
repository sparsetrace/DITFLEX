"""src/ditflex/ema.py -- exponential moving average of model parameters.

Built on and updated through the RAW module (before torch.compile / DDP
wrapping). The wrappers share the same Parameter objects, so updating via
the raw reference is correct -- and it means the EMA state dict carries
clean parameter names with no `_orig_mod.` / `module.` prefixes ever.
"""

from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Load EMA weights into a model (for sampling/eval)."""
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name].to(p.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow) ^ set(sd)
        if missing:
            raise KeyError(f"EMA state mismatch on keys: {sorted(missing)[:5]} ...")
        for name, t in sd.items():
            self.shadow[name] = t.float().clone()

    def to(self, device) -> EMA:
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

"""src/ditflex/config.py -- experiment configuration as plain dataclasses.

Deliberately imports NO torch: a config must be constructible and
round-trippable anywhere -- CI on CPU, a laptop reading a checkpoint's
embedded config, the Hub viewer -- without a GPU environment.

The defaults ARE the experiment: DiT-L/2 at the published recipe
(README "Recipe" table). Anything that deviates from the published
DiT/SiT setup is marked DEVIATION in a comment and must stay in sync
with the README's known-deviations list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """DiT-L/2: width 1024 = 16 heads x 64, depth 24, patch 2 -> 458M params."""

    num_attention_heads: int = 16
    attention_head_dim: int = 64
    num_layers: int = 24
    patch_size: int = 2
    sample_size: int = 32          # 32x32 latents -> 256 tokens at patch 2
    in_channels: int = 4
    # DEVIATION: published DiT uses out_channels=8 (4 eps + 4 learned sigma,
    # trained with hybrid MSE+VLB). Our objectives are eps/velocity MSE only,
    # so sigma channels would be dead weights receiving zero gradient.
    out_channels: int = 4
    num_classes: int = 1000        # null class for CFG is index num_classes
    # "amap": standard directed QK attention (the baseline).
    # "dmap": EQ-sector / diffusion-map attention -- W_K tied to W_Q, so
    #   every score matrix is symmetric (q_i . q_j) and R == 0 identically.
    #   DEVIATIONS vs baseline: ~25M fewer params (no separate W_K), and
    #   each head's bilinear is PSD (a subfamily of symmetric).
    qk_mode: str = "amap"
    # Coifman-Lafon density-correction exponent for qk_mode="dmap".
    # 0 = pure squared-distance DMAP: softmax(2s_ij - g_j). NOTE this is
    #   NOT plain attention -- the destination potential g_j survives the
    #   row-softmax (only the source term g_i is killed).
    # 0.5 = Fokker-Planck; 1 = Laplace-Beltrami. Ignored for amap.
    dmap_alpha: float = 0.0
    # DEVIATION (adopted at the 344K migration, amap chain only): per-head
    # RMSNorm(head_dim, eps=1e-6) on Q and K before attention. Forced by
    # evidence: unbounded block-1 QK logits (8.6e6 at 312.5K) driven by
    # adaLN weight growth produced the gradient-spike instability; RMSNorm
    # bounds per-head logits by construction. INVALID for qk_mode="dmap":
    # untied norms would break the R == 0 symmetry, tied norms would
    # flatten the destination potential g_j that defines the DMAP kernel
    # (build_dmap_model refuses the combination). Old checkpoints lacking
    # this key deserialize to False, so pre-migration configs round-trip
    # unchanged.
    qk_norm: bool = False


@dataclass
class DataConfig:
    hub_repo: str = "sparsetrace/dlatentzz"
    latent_shape: tuple[int, int, int] = (4, 32, 32)
    expected_total: int = 1_281_167   # ImageNet-1k train; a constant of the dataset
    # DEVIATION (of the dataset itself, recorded here for provenance):
    # latents are posterior MODE, not sampled; no horizontal-flip pass;
    # torchvision Resize+CenterCrop rather than ADM center_crop_arr.

    def __post_init__(self):
        self.latent_shape = tuple(self.latent_shape)  # JSON round-trip: list -> tuple


@dataclass
class TrainConfig:
    objective: str = "ddpm"        # ddpm | flow
    global_batch: int = 256
    lr: float = 1e-4               # constant, no warmup (published recipe)
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    label_dropout: float = 0.1     # for classifier-free guidance
    base_seed: int = 0
    deadline_check_every: int = 500  # steps between wall-clock checks (rank 0)


@dataclass
class HubConfig:
    checkpoint_repo: str = "sparsetrace/ditflex-L2"
    archive_every_steps: int = 200_000
    # Periodic save+push cadence. On ephemeral containers a local-only save
    # protects nothing, so every periodic save uploads. At ~9.5 steps/s this
    # is ~18 min of compute at risk between saves. 0 disables (end-of-run
    # save only).
    save_every_steps: int = 10_000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            train=TrainConfig(**d["train"]),
            hub=HubConfig(**d["hub"]),
        )

    @classmethod
    def from_json(cls, s: str) -> Config:
        return cls.from_dict(json.loads(s))

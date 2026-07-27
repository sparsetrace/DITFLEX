# DDMAP  (Doob-potential DMAP)

Symmetric kinetic attention (Gram from m=(q+k)/√2) **plus a free one-body Doob
potential** φ — generalising DMAP's *tied* potential −½‖m‖² to a learned one from
a matrix independent of W_M:

    logit_ij = ⟨m_i, m_j⟩ + φ_i + φ_j
    φ_i      = diag(R W Rᵀ)_i = r_iᵀ Λ r_i,   r = R·P     (P, Λ new, ⟂ W_M)

`--potential`:
- **free** (default) — learned φ; Λ init 0 so it starts as bare-Gram attention
  and learns the potential. This is the new arm (a learnable −Δ+V / Doob kernel).
- **dmap** — φ = −½‖m‖², recovers the exact DMAP distance kernel (ablation).
- **none** — φ = 0, bare symmetric Gram attention (ablation).

φ is computed at the **context level** (a per-token scalar from activations), not
folded into a projection. Only the per-key φ_j survives softmax (φ_i cancels
along its row), so φ is a learned, geometry-derived per-key bias — a Doob
h-transform of the kernel.

## Cost / compatibility

`free` adds a potential head per attention (phi_proj [C,C] + phi_lambda [H,Dh]),
~+5% params on SiT-XL/2. DDMAP checkpoints carry these extra keys, so they can't
load back into SiT/AMAP/DMAP; warm-starting FROM those loads the shared qkv/proj
and leaves φ at init (Λ=0 ⇒ starts as bare Gram).

## Files (coupled, borrowed from /AMAP/)

- `ddmap_attention.py` — operator (`apply_ddmap`); self-test verifies
  potential='dmap' == DMAP, 'free' Λ=0 == bare Gram.
- `ddmap_common.py`, `DDMAP.py`, `sample_ddmap.py` — as the other arms.

## Run

    modal run DDMAP/DDMAP.py --stage finetune --potential free \
        --steps 40000 --save-every 10000 --sample-every 10000
    # ablations:
    modal run DDMAP/DDMAP.py --stage finetune --potential dmap  ...   # == DMAP
    modal run DDMAP/DDMAP.py --stage finetune --potential none  ...   # bare Gram
    modal run DDMAP/sample_ddmap.py --step latest --weights ema

Resolves DDMAP-own checkpoint → else base SiT (+ apply_ddmap). `--steps N` is
additive; step-0 grid snapshotted unconditionally. Push to `jcandane/DDMAP`.

## The comparison this arm buys

- **free vs dmap** — does a *learned* potential beat the geometry-tied −½‖m‖²?
  (Is DMAP's specific potential optimal, or just one choice?)
- **free vs none** — how much does *any* potential add over bare Gram?
- Ties into DAC: the connection theory wants kinetic + connection (flux) +
  **potential**; DDMAP is the potential slot, tested empirically.

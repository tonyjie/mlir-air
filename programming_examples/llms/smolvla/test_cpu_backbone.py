import numpy as np
from smolvla_backbone_weights import load_backbone_weights, SmolVLABackboneConfig
from smolvla_cpu_helpers import cpu_backbone_forward, build_prefix_mask


def test_cpu_backbone_matches_oracle():
    o = np.load("smolvla_oracle.npz")
    cfg = SmolVLABackboneConfig()
    w = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    # The language block of the 240-token prefix is right-padded to a fixed
    # tokenizer length ("pick up the cube" is ~4 real tokens out of 48 slots);
    # the real model excludes padding from attention entirely and freezes
    # RoPE position_ids at the last real position (position_ids =
    # cumsum(pad_masks) - 1). Both must be reproduced to match the oracle.
    pad_mask = o["prefix_pad_masks"]
    positions = np.cumsum(pad_mask.astype(np.int64)) - 1
    mask = build_prefix_mask(n_prefix=240, n_state=1, pad_mask=pad_mask)
    hidden = cpu_backbone_forward(
        o["prefix_embed"], w, cfg, mask, rope_base=cfg.rope_base, positions=positions
    )
    # final-norm hidden must match the oracle within f32 round-off
    cos = np.sum(hidden * o["final_norm_hidden"]) / (
        np.linalg.norm(hidden) * np.linalg.norm(o["final_norm_hidden"])
    )
    assert cos > 0.9999, f"cosine {cos} too low — CPU backbone diverges from oracle"

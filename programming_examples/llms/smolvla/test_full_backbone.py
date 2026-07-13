# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task 3.1 acceptance test (Phase 3 gate): the FULL 16-layer SmolVLA backbone
assembled from NPU kernels vs the real-model oracle.

Two checks:
  1. Per-layer DIAGNOSIS: for each layer i, cosine(NPU per_layer[i][:241],
     oracle layer_hidden[i]); print all 16; assert each > 0.98 (drift tolerance
     across depth).
  2. FINAL GATE: cosine(final_hidden[:241], oracle final_norm_hidden) > 0.99.

Attention path is selectable:
    python3 test_full_backbone.py            # Step A (cpu_attn=True, GOAL 1)
    python3 test_full_backbone.py --npu-attn # Step B (cpu_attn=False, GOAL 2)

Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_full_backbone.py
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from smolvla_backbone_weights import (
    load_backbone_weights,
    SmolVLABackboneConfig,
    generate_rope_lut,
)
from smolvla_cpu_helpers import build_prefix_mask
from smolvla_backbone_prefill import (
    compile_all_kernels,
    run_backbone_prefill,
)
from shared.infra.cache import KernelCache, Profiler

NPU_SEQ_LEN = 256  # registry kernels validated at M=256 (Task 1.1)
N_PREFIX = 240
N_STATE = 1
ORACLE_LEN = N_PREFIX + N_STATE  # 241

PER_LAYER_THRESH = 0.98
FINAL_THRESH = 0.99


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_padded_mask_and_positions(pad_mask_241):
    """Extend the 241-token prefix mask + RoPE positions to NPU_SEQ_LEN=256.
    Identical construction to test_single_block.build_padded_mask_and_positions.
    """
    mask_241 = build_prefix_mask(N_PREFIX, N_STATE, pad_mask=pad_mask_241)
    mask_256 = np.full((NPU_SEQ_LEN, NPU_SEQ_LEN), -np.inf, dtype=np.float32)
    mask_256[:ORACLE_LEN, :ORACLE_LEN] = mask_241

    pad_mask_256 = np.zeros((NPU_SEQ_LEN,), dtype=bool)
    pad_mask_256[:ORACLE_LEN] = pad_mask_241
    positions_256 = np.cumsum(pad_mask_256.astype(np.int64)) - 1
    positions_256 = np.clip(positions_256, 0, None)
    return mask_256, positions_256


def main():
    cpu_attn = "--npu-attn" not in sys.argv
    mode = "Step A (CPU attention)" if cpu_attn else "Step B (NPU attention)"
    print("=" * 70)
    print(f"SmolVLA Task 3.1: FULL 16-layer NPU backbone -- {mode}")
    print("=" * 70)

    cfg = SmolVLABackboneConfig()
    weights = load_backbone_weights("lerobot/smolvla_base", config=cfg)

    oracle_path = _HERE / "smolvla_oracle.npz"
    o = np.load(oracle_path)
    pad_mask_241 = o["prefix_pad_masks"]
    assert pad_mask_241.shape == (ORACLE_LEN,), pad_mask_241.shape
    layer_hidden = o["layer_hidden"]  # (16, 241, 960)
    final_norm_hidden = o["final_norm_hidden"]  # (241, 960)
    assert layer_hidden.shape[0] == cfg.n_layers, (
        layer_hidden.shape,
        cfg.n_layers,
    )

    mask_256, positions_256 = build_padded_mask_and_positions(pad_mask_241)

    # RoPE LUT gathered per (padding-frozen) position -- NOT a plain arange.
    max_pos = int(positions_256.max()) + 1
    full_lut = generate_rope_lut(cfg, seq_len=max_pos, dtype=bfloat16)
    rope_lut_bf16 = full_lut[positions_256]  # (256, head_dim)

    # NPU input: pad the 241-row oracle prefix embedding to 256 rows with zeros.
    x_f32 = np.zeros((NPU_SEQ_LEN, cfg.emb_dim), dtype=np.float32)
    x_f32[:ORACLE_LEN] = o["prefix_embed"]
    x_bf16 = x_f32.astype(bfloat16)

    # -- Compile kernels & run the full backbone on NPU --
    cache = KernelCache(
        "smolvla_block_kernel_cache", verbose=False, profiler=Profiler()
    )
    compile_all_kernels(cache, cfg, NPU_SEQ_LEN, cpu_attn=cpu_attn)

    final_hidden, per_layer = run_backbone_prefill(
        x_bf16,
        weights,
        cfg,
        cache,
        mask_256,
        positions_256,
        rope_lut_bf16,
        cpu_attn=cpu_attn,
        verbose=True,
    )
    assert len(per_layer) == cfg.n_layers, len(per_layer)

    # -- Per-layer diagnosis --
    print("\n" + "-" * 70)
    print("Per-layer cosine (NPU per_layer[i][:241] vs oracle layer_hidden[i])")
    print("-" * 70)
    per_layer_cos = []
    for i in range(cfg.n_layers):
        c = cosine(per_layer[i][:ORACLE_LEN], layer_hidden[i])
        per_layer_cos.append(c)
        flag = "" if c > PER_LAYER_THRESH else "  <-- BELOW THRESHOLD"
        print(f"  layer {i:2d}: cosine = {c:.6f}{flag}")

    # -- Final gate --
    cos_final = cosine(final_hidden[:ORACLE_LEN], final_norm_hidden)
    print("\n" + "=" * 70)
    print(
        f"FINAL GATE: cosine(final_hidden[:{ORACLE_LEN}], "
        f"oracle final_norm_hidden) = {cos_final:.6f}"
    )
    print("=" * 70)

    # -- Assertions (root-cause, don't loosen) --
    failed = [(i, c) for i, c in enumerate(per_layer_cos) if not c > PER_LAYER_THRESH]
    assert not failed, (
        f"Per-layer cosine below {PER_LAYER_THRESH} at layers "
        f"{[i for i, _ in failed]}: {[f'{c:.4f}' for _, c in failed]}. "
        f"The per-layer print localizes the divergence (layer-indexed weight "
        f"bug / accumulation)."
    )
    assert cos_final > FINAL_THRESH, (
        f"Final-hidden cosine {cos_final} <= {FINAL_THRESH}. Per-layer cosines: "
        f"{[f'{c:.4f}' for c in per_layer_cos]}"
    )
    print(
        f"\nPASS: all {cfg.n_layers} per-layer cosines > {PER_LAYER_THRESH}; "
        f"final gate > {FINAL_THRESH}"
    )


if __name__ == "__main__":
    main()

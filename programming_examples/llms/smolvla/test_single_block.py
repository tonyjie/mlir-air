# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task 2.1 acceptance test: ONE SmolVLA backbone layer (layer 0) assembled
from NPU kernels (RMSNorm+QKV+RoPE fused ELF, CPU non-causal attention,
O-proj+residual+FFN fused ELF) vs the real-model oracle.

Gate: cosine(NPU layer-0 output[:241], oracle layer_hidden[0]) > 0.99.

Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_single_block.py
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
from smolvla_cpu_helpers import (
    build_prefix_mask,
    noncausal_attention_reference,
    rms_norm,
    _rope_half_split,
)
from smolvla_backbone_prefill import (
    compile_all_kernels,
    run_transformer_block,
)
from shared.infra.cache import KernelCache, Profiler

NPU_SEQ_LEN = 256  # registry kernels validated at M=256 (Task 1.1)
N_PREFIX = 240
N_STATE = 1
ORACLE_LEN = N_PREFIX + N_STATE  # 241


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_padded_mask_and_positions(pad_mask_241):
    """Extend the 241-token prefix mask + RoPE positions to NPU_SEQ_LEN=256.

    The 15 extra NPU-padding rows (241..255) are treated exactly like
    tokenizer padding: fully masked out both as queries and as keys (never
    attended, and their own attention output is garbage-but-unused), and
    their RoPE position is frozen at the last real position (harmless, since
    their output rows are discarded before comparing to the oracle).
    """
    mask_241 = build_prefix_mask(N_PREFIX, N_STATE, pad_mask=pad_mask_241)
    mask_256 = np.full((NPU_SEQ_LEN, NPU_SEQ_LEN), -np.inf, dtype=np.float32)
    mask_256[:ORACLE_LEN, :ORACLE_LEN] = mask_241

    pad_mask_256 = np.zeros((NPU_SEQ_LEN,), dtype=bool)
    pad_mask_256[:ORACLE_LEN] = pad_mask_241
    positions_256 = np.cumsum(pad_mask_256.astype(np.int64)) - 1
    # Rows before the first real token would get position -1; SmolVLA's own
    # prefix never starts with padding (image/language tokens come first),
    # but clip defensively so RoPE never sees a negative position.
    positions_256 = np.clip(positions_256, 0, None)
    return mask_256, positions_256


def cpu_recompute_layer(x_f32, lw, cfg, mask_256, positions_256):
    """Recompute the same layer on CPU (F32) for per-substep diagnostics,
    using the SAME mask/positions the NPU path uses (256-padded)."""
    h = rms_norm(x_f32, lw.attn_norm, cfg.rms_norm_eps)
    q = (h @ lw.wq.astype(np.float32)).reshape(NPU_SEQ_LEN, cfg.n_heads, cfg.head_dim)
    k = (h @ lw.wk.astype(np.float32)).reshape(
        NPU_SEQ_LEN, cfg.n_kv_heads, cfg.head_dim
    )
    v = (h @ lw.wv.astype(np.float32)).reshape(
        NPU_SEQ_LEN, cfg.n_kv_heads, cfg.head_dim
    )
    q = _rope_half_split(q, positions_256, cfg.rope_base, cfg.head_dim)
    k = _rope_half_split(k, positions_256, cfg.rope_base, cfg.head_dim)
    attn_raw = noncausal_attention_reference(
        q, k, v, mask_256, cfg.n_heads, cfg.n_kv_heads
    )
    attn_raw = attn_raw.reshape(NPU_SEQ_LEN, cfg.n_heads * cfg.head_dim)
    proj = attn_raw @ lw.wo.astype(np.float32)
    x2 = x_f32 + proj
    h2 = rms_norm(x2, lw.ffn_norm, cfg.rms_norm_eps)
    g = h2 @ lw.w_gate.astype(np.float32)
    u = h2 @ lw.w_up.astype(np.float32)
    silu = g / (1.0 + np.exp(-g))
    ffn_out = x2 + (silu * u) @ lw.w_down.astype(np.float32)
    return {
        "h": h,
        "q": q,
        "k": k,
        "v": v,
        "attn_out": attn_raw,  # pre-O-proj, matches run_transformer_block's intermediates["attn_out"]
        "ffn_out": ffn_out,
    }


def main():
    print("=" * 70)
    print("SmolVLA Task 2.1: single-block NPU prefill (Step A, CPU attention)")
    print("=" * 70)

    cfg = SmolVLABackboneConfig()
    weights = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    lw0 = weights.layers[0]

    oracle_path = _HERE / "smolvla_oracle.npz"
    o = np.load(oracle_path)
    pad_mask_241 = o["prefix_pad_masks"]
    assert pad_mask_241.shape == (ORACLE_LEN,), pad_mask_241.shape

    mask_256, positions_256 = build_padded_mask_and_positions(pad_mask_241)

    # RoPE LUT: one table row per distinct position value, gathered per-token
    # by the (padding-frozen) position array -- NOT a plain arange lookup.
    max_pos = int(positions_256.max()) + 1
    full_lut = generate_rope_lut(cfg, seq_len=max_pos, dtype=bfloat16)
    rope_lut_bf16 = full_lut[positions_256]  # (256, head_dim), gathered per token

    # NPU input: pad the 241-row oracle prefix embedding to 256 rows with zeros.
    x_f32 = np.zeros((NPU_SEQ_LEN, cfg.emb_dim), dtype=np.float32)
    x_f32[:ORACLE_LEN] = o["prefix_embed"]
    x_bf16 = x_f32.astype(bfloat16)

    # -- CPU recompute (same mask/positions) for per-substep diagnostics --
    cpu_ref = cpu_recompute_layer(x_f32, lw0, cfg, mask_256, positions_256)

    # -- Compile kernels & run on NPU --
    cache = KernelCache(
        "smolvla_block_kernel_cache", verbose=False, profiler=Profiler()
    )
    compile_all_kernels(cache, cfg, NPU_SEQ_LEN, cpu_attn=True)

    npu_out, inter = run_transformer_block(
        x_bf16,
        lw0,
        rope_lut_bf16,
        cfg,
        cache,
        mask_256,
        positions_256,
        layer_idx=0,
        cpu_attn=True,
        verbose=True,
    )

    # -- Per-substep diagnostics (localize a low final cosine) --
    print("\n" + "-" * 70)
    print("Per-substep diagnostic cosines (NPU intermediate vs CPU F32 recompute)")
    print("-" * 70)

    q_roped_cpu = cpu_ref["q"].reshape(NPU_SEQ_LEN, cfg.n_heads * cfg.head_dim)
    k_roped_cpu = cpu_ref["k"].reshape(NPU_SEQ_LEN, cfg.n_kv_heads * cfg.head_dim)
    v_cpu = cpu_ref["v"].reshape(NPU_SEQ_LEN, cfg.n_kv_heads * cfg.head_dim)

    cos_q = cosine(inter["q_roped"][:ORACLE_LEN], q_roped_cpu[:ORACLE_LEN])
    cos_k = cosine(inter["k_roped"][:ORACLE_LEN], k_roped_cpu[:ORACLE_LEN])
    cos_v = cosine(inter["v"][:ORACLE_LEN], v_cpu[:ORACLE_LEN])
    print(f"  Q (post-RoPE, NPU vs CPU):  cosine = {cos_q:.6f}")
    print(f"  K (post-RoPE, NPU vs CPU):  cosine = {cos_k:.6f}")
    print(f"  V (NPU vs CPU):             cosine = {cos_v:.6f}")

    cos_attn = cosine(
        inter["attn_out"][:ORACLE_LEN],
        cpu_ref["attn_out"].reshape(NPU_SEQ_LEN, -1)[:ORACLE_LEN],
    )
    print(f"  Attn out (CPU non-causal, using NPU Q/K/V): cosine = {cos_attn:.6f}")

    cos_ffn_vs_cpurecompute = cosine(
        inter["ffn_out"][:ORACLE_LEN], cpu_ref["ffn_out"][:ORACLE_LEN]
    )
    print(
        f"  Final block out (NPU vs CPU recompute):     cosine = {cos_ffn_vs_cpurecompute:.6f}"
    )

    # -- Primary gate: NPU layer-0 output vs the real-model oracle --
    layer0_oracle = o["layer_hidden"][0]  # (241, 960)
    cos_final = cosine(npu_out[:ORACLE_LEN], layer0_oracle)
    print("\n" + "=" * 70)
    print(
        f"PRIMARY GATE: cosine(NPU layer-0 out[:{ORACLE_LEN}], oracle layer_hidden[0]) = {cos_final:.6f}"
    )
    print("=" * 70)

    assert cos_final > 0.99, (
        f"Layer-0 cosine {cos_final} <= 0.99 -- NPU single-block assembly "
        f"diverges from the oracle. Substep cosines: Q={cos_q:.4f} K={cos_k:.4f} "
        f"V={cos_v:.4f} attn={cos_attn:.4f} ffn(vs cpu recompute)={cos_ffn_vs_cpurecompute:.4f}"
    )
    print("\nPASS: layer-0 cosine > 0.99")


if __name__ == "__main__":
    main()

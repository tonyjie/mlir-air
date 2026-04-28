# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side dimension padding for Qwen3-4B on NPU2.

Background
----------
Qwen3-4B has emb_dim=2560 and hidden_dim=9728. Neither is a clean
multiple of the AIE2P shim DMA's per-dim size limit (1024). When the fused
`rms_attn_gemms` ELF shares an L2 buffer (`normed`) between RMSNorm output
and Q/K/V GEMM input at K=2560, the lowering emits a 2-D
`bd_dim_layout_array` DMA pattern that silently mis-reads the L2 data.
Phase 2 result: RMSNorm output cosine 0.999975 ✓ but Q/K/V GEMM cosines
all 0.000000 (deterministic garbage, not NaN).

Pad UP to BD-friendly dims:
    emb_dim    2560 → 3072   (= 3 × 1024 — single-dim BD; 1.20× compute inflation)
    hidden_dim 9728 → 10240  (= 10 × 1024 — single-dim BD; 1.05× FFN inflation)
    n_heads / n_kv_heads / head_dim / q_dim / kv_dim — UNCHANGED (q_dim=4096
        is independent of emb_dim in Qwen3-4B; no GQA-aware reindex needed
        like qwen2.5-1.5B's qwen25_pad)

Cost: ~20% more compute on Q/K/V/Gate/Up/Down GEMMs that touch emb_dim;
~5% more on Down K-axis. All padded weights are zero, so phantom emb/hidden
positions produce zero contributions that the zero-padded downstream weights
discard.

The one non-trivial detail: **RMSNorm weight rescaling.** RMSNorm normalizes
by `sqrt(mean(x^2) + eps)` over emb_dim. Padding x with zeros keeps the sum
unchanged but divides by a larger denominator → smaller `mean`, larger
`1/rms`. To compensate, pre-scale all RMSNorm weights by
`sqrt(orig_emb / padded_emb) = sqrt(2560/3072) ≈ 0.9128` so the post-RMSNorm
values match the original. Eps contribution is negligible at our scales
(BF16 noise floor).

Public API
----------
- `pad_weights(orig_weights, orig_config, padded_config) -> padded_weights`
- `make_padded_config(orig_config, padded_emb_dim, padded_hidden_dim) -> padded_config`
- `pad_input_embed(x, padded_emb_dim) -> padded_x`
- `slice_output(padded_block_out, orig_emb_dim) -> sliced_block_out`
"""

from copy import copy

import numpy as np
from ml_dtypes import bfloat16

from qwen3_4b_weights import LayerWeights, LlamaConfig, LlamaWeights


def make_padded_config(
    orig_config,
    padded_emb_dim=3072,
    padded_hidden_dim=10240,
):
    """Build a LlamaConfig with padded emb_dim / hidden_dim.

    n_heads / n_kv_heads / head_dim / q_dim / kv_dim are UNCHANGED — Qwen3-4B
    has q_dim=4096 ≠ emb_dim=2560, so emb_dim padding doesn't require Q-head
    structure changes (unlike qwen2.5-1.5B which had q_dim==emb_dim).
    """
    padded = copy(orig_config)
    padded.emb_dim = padded_emb_dim
    padded.hidden_dim = padded_hidden_dim
    return padded


def _pad_2d(weight, target_in, target_out, dtype=bfloat16):
    arr = np.asarray(weight, dtype=dtype)
    in_orig, out_orig = arr.shape
    assert in_orig <= target_in and out_orig <= target_out, (
        f"can't pad ({in_orig}, {out_orig}) to ({target_in}, {target_out}): "
        f"shrinking would lose information"
    )
    if in_orig == target_in and out_orig == target_out:
        return arr
    out = np.zeros((target_in, target_out), dtype=dtype)
    out[:in_orig, :out_orig] = arr
    return out


def _scale_rmsnorm_weight(weight, orig_dim, padded_dim, dtype=bfloat16):
    """Pre-scale RMSNorm weight by sqrt(orig_dim/padded_dim) for the first
    orig_dim entries; zeros for [orig_dim:padded_dim].

    Compensates for RMSNorm's larger denominator after zero-padding x.
    """
    arr = np.asarray(weight, dtype=np.float32)
    factor = float(np.sqrt(orig_dim / padded_dim))
    out = np.zeros((padded_dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr * factor
    return out.astype(dtype)


def pad_weights(orig_weights, orig_config, padded_config, dtype=bfloat16):
    """Produce a LlamaWeights with all tensors zero-padded to padded_config.

    Padding rules (Qwen3-4B; simpler than qwen25_pad because q_dim is
    independent of emb_dim and there's no QKV bias):
    - embed_table: zero-pad columns to padded_emb (vocab × p_emb).
    - final_norm + per-layer attn_norm + ffn_norm: RMS-pre-scale to padded_emb.
    - wq / wk / wv: zero-pad input axis (emb) to padded_emb. Output axis
      (q_dim / kv_dim) UNCHANGED.
    - wo: zero-pad output axis (emb) to padded_emb. Input axis (q_dim) UNCHANGED.
    - q_norm / k_norm: head_dim unchanged → no padding needed.
    - w_gate / w_up: zero-pad input axis (emb) to padded_emb AND output axis
      (hidden) to padded_hidden.
    - w_down: zero-pad input axis (hidden) to padded_hidden AND output axis
      (emb) to padded_emb.
    - lm_head: tied to embed → reuse padded embed_table.
    """
    o_emb = orig_config.emb_dim
    p_emb = padded_config.emb_dim
    o_hid = orig_config.hidden_dim
    p_hid = padded_config.hidden_dim
    n_heads = orig_config.n_heads
    n_kv = orig_config.n_kv_heads
    head_dim = orig_config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    vocab = orig_config.vocab_size

    assert padded_config.n_heads == n_heads, "Qwen3-4B padding keeps n_heads"
    assert padded_config.n_kv_heads == n_kv
    assert padded_config.head_dim == head_dim

    # Embedding table: pad output emb dim with zeros.
    embed = np.asarray(orig_weights.embed_table, dtype=dtype)
    embed_padded = np.zeros((vocab, p_emb), dtype=dtype)
    embed_padded[:, :o_emb] = embed

    final_norm_padded = _scale_rmsnorm_weight(
        orig_weights.final_norm, o_emb, p_emb, dtype
    )

    layers_padded = []
    for lw in orig_weights.layers:
        new_lw = LayerWeights(
            attn_norm=_scale_rmsnorm_weight(lw.attn_norm, o_emb, p_emb, dtype),
            wq=_pad_2d(lw.wq, p_emb, q_dim, dtype),
            wk=_pad_2d(lw.wk, p_emb, kv_dim, dtype),
            wv=_pad_2d(lw.wv, p_emb, kv_dim, dtype),
            q_norm=np.asarray(lw.q_norm, dtype=dtype),  # head_dim unchanged
            k_norm=np.asarray(lw.k_norm, dtype=dtype),  # head_dim unchanged
            wo=_pad_2d(lw.wo, q_dim, p_emb, dtype),
            ffn_norm=_scale_rmsnorm_weight(lw.ffn_norm, o_emb, p_emb, dtype),
            w_gate=_pad_2d(lw.w_gate, p_emb, p_hid, dtype),
            w_up=_pad_2d(lw.w_up, p_emb, p_hid, dtype),
            w_down=_pad_2d(lw.w_down, p_hid, p_emb, dtype),
        )
        layers_padded.append(new_lw)

    return LlamaWeights(
        embed_table=embed_padded,
        layers=layers_padded,
        final_norm=final_norm_padded,
        lm_head=embed_padded,  # tied
    )


def pad_input_embed(x, padded_emb_dim, dtype=bfloat16):
    """Pad an (seq_len, orig_emb_dim) tensor to (seq_len, padded_emb_dim) with zeros."""
    arr = np.asarray(x, dtype=dtype)
    seq_len, o_emb = arr.shape
    if o_emb == padded_emb_dim:
        return arr
    out = np.zeros((seq_len, padded_emb_dim), dtype=dtype)
    out[:, :o_emb] = arr
    return out


def slice_output(padded_block_out, orig_emb_dim):
    """Slice (seq_len, padded_emb_dim) back to (seq_len, orig_emb_dim)."""
    arr = np.asarray(padded_block_out)
    if arr.shape[-1] == orig_emb_dim:
        return arr
    return arr[..., :orig_emb_dim]


# ---------------------------------------------------------------------------
# Sanity test: padded CPU forward must match orig CPU forward (cos > 0.99999)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_THIS_DIR))

    import qwen3_4b_reference as qref
    from qwen3_4b_weights import load_weights, generate_rope_lut

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--padded-emb", type=int, default=3072)
    parser.add_argument("--padded-hidden", type=int, default=10240)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    orig_config = LlamaConfig()
    print(f"Loading weights ({args.model})...")
    orig_weights = load_weights(args.model, config=orig_config)

    padded_config = make_padded_config(
        orig_config,
        padded_emb_dim=args.padded_emb,
        padded_hidden_dim=args.padded_hidden,
    )
    print(
        f"Padding emb {orig_config.emb_dim}→{padded_config.emb_dim}, "
        f"hidden {orig_config.hidden_dim}→{padded_config.hidden_dim}..."
    )
    padded_weights = pad_weights(orig_weights, orig_config, padded_config)

    # Build canonical input
    np.random.seed(42)
    token_ids = np.random.randint(0, orig_config.vocab_size, args.seq_len)

    # Orig forward (single block)
    orig_embed = np.asarray(orig_weights.embed_table, dtype=np.float32)
    orig_x = orig_embed[token_ids]
    orig_rope = generate_rope_lut(
        config=orig_config, seq_len=args.seq_len, dtype=np.float32
    )
    orig_out, _ = qref.transformer_block(
        orig_x, orig_weights.layers[0], orig_rope, orig_config
    )

    # Padded forward (single block)
    padded_embed = np.asarray(padded_weights.embed_table, dtype=np.float32)
    padded_x = padded_embed[token_ids]
    padded_rope = generate_rope_lut(
        config=padded_config, seq_len=args.seq_len, dtype=np.float32
    )
    padded_out_full, _ = qref.transformer_block(
        padded_x, padded_weights.layers[0], padded_rope, padded_config
    )
    sliced = slice_output(padded_out_full, orig_config.emb_dim)

    a = orig_out.astype(np.float64).flatten()
    b = sliced.astype(np.float64).flatten()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    print(f"\nPadded vs orig single-block (CPU F32):")
    print(f"  cosine  : {cos:.9f}  (gate ≥ 0.999998)")
    print(f"  max_abs : {max_abs:.6f}")

    if cos >= 0.999998:
        print(f"\nPADDING SANITY: PASS")
        sys.exit(0)
    else:
        print(f"\nPADDING SANITY: FAIL")
        sys.exit(1)

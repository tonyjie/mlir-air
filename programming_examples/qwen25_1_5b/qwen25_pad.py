# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side dimension padding for Qwen2.5-1.5B on NPU2.

Background
----------
Qwen2.5-1.5B has emb_dim=1536 and hidden_dim=8960. Neither is a clean
multiple of the AIE2P shim DMA's per-dim size limit (1024). They force the
lowering pass to emit 2-D `bd_dim_layout_array[<size=512, stride=768>, ...]`
DMA patterns, which inflate the BD count enough to exhaust the per-channel
BD pool when 6+ launches are stitched into one ELF (rms_gemms_rope) or 8+
launches (o_ffn). Result at seq_len=2048: "Allocator exhausted available
buffer descriptor IDs" during AIE lowering.

Pad UP to BD-friendly dims:
    emb_dim    1536 → 2048   (= 2 × 1024 — single-dim BD)
    hidden_dim 8960 → 9216   (= 9 × 1024 — single-dim BD)
    kv_dim      256 → 256    (already fits with tile_n=64, herd_n=4)
    n_heads      12 →   16   (forced by emb_dim/head_dim = 2048/128 = 16)

Cost: ~33% more compute on Q/O GEMMs, ~3% more on FFN. All padded weights
are zero, so phantom Q heads and phantom FFN units produce zero contributions
that the zero-padded downstream weights then discard.

The one non-trivial detail: **RMSNorm weight rescaling.** RMSNorm normalizes
by `sqrt(mean(x^2) + eps)` over emb_dim. Padding x with zeros keeps the sum
unchanged but divides by a larger denominator → smaller `mean`, larger
`1/rms`. To compensate, pre-scale all RMSNorm weights by
`sqrt(orig_emb / padded_emb) ≈ 0.866` so the post-RMSNorm values match the
original. Eps contribution is negligible at our scales (BF16 noise floor).

Public API
----------
- `pad_weights(orig_weights, orig_config, padded_config) -> padded_weights`
- `make_padded_config(orig_config, padded_emb_dim, padded_hidden_dim,
                      padded_n_heads) -> padded_config`
- `slice_output(padded_block_out, orig_emb_dim) -> sliced_block_out`
"""

from copy import copy

import numpy as np
from ml_dtypes import bfloat16

from qwen25_weights import LayerWeights, LlamaConfig, LlamaWeights


def make_padded_config(
    orig_config,
    padded_emb_dim=2048,
    padded_hidden_dim=9216,
    padded_n_heads=None,
):
    """Build a LlamaConfig with padded emb_dim / hidden_dim / n_heads.

    n_heads padded so that padded_n_heads * head_dim == padded_emb_dim.
    Other fields (head_dim, n_kv_heads, vocab, rope_base, qkv_bias, dtype)
    are inherited unchanged.
    """
    if padded_n_heads is None:
        padded_n_heads = padded_emb_dim // orig_config.head_dim
    assert padded_n_heads * orig_config.head_dim == padded_emb_dim, (
        f"padded_n_heads={padded_n_heads} * head_dim={orig_config.head_dim} != "
        f"padded_emb_dim={padded_emb_dim}"
    )
    assert padded_n_heads % orig_config.n_kv_heads == 0, (
        f"padded_n_heads={padded_n_heads} not divisible by "
        f"n_kv_heads={orig_config.n_kv_heads} — would break GQA"
    )

    padded = copy(orig_config)
    padded.emb_dim = padded_emb_dim
    padded.hidden_dim = padded_hidden_dim
    padded.n_heads = padded_n_heads
    return padded


def _pad_2d(weight, target_in, target_out, dtype=bfloat16):
    """Zero-pad a (in, out) weight matrix to (target_in, target_out)."""
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


def _pad_1d(vec, target_dim, dtype=bfloat16):
    """Zero-pad a 1-D vector to length target_dim."""
    arr = np.asarray(vec, dtype=dtype)
    if arr.shape[0] == target_dim:
        return arr
    out = np.zeros((target_dim,), dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _scale_rmsnorm_weight(weight, orig_dim, padded_dim, dtype=bfloat16):
    """Pre-scale RMSNorm weight by sqrt(orig_dim/padded_dim) for first orig_dim
    entries; zeros for [orig_dim:padded_dim].

    Compensates for RMSNorm's larger denominator after zero-padding x. See the
    derivation in the module docstring.
    """
    arr = np.asarray(weight, dtype=np.float32)
    factor = float(np.sqrt(orig_dim / padded_dim))
    out = np.zeros((padded_dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr * factor
    return out.astype(dtype)


def _gqa_reindex_qhead_axis(
    orig_per_head, n_kv_heads, group_orig, group_padded, head_dim
):
    """Reorder a per-head tensor so that orig heads are placed at padded
    positions that preserve GQA mapping.

    Concretely: for each KV group g in 0..n_kv_heads:
      padded heads [g*group_padded : g*group_padded + group_orig] = orig heads
        [g*group_orig : (g+1)*group_orig]   — REAL data
      padded heads [g*group_padded + group_orig : (g+1)*group_padded] = 0
        — phantom (zero), maps to the same KV head

    Args:
        orig_per_head: numpy array shape (..., n_heads_orig*head_dim, ...) where
            the head axis is the LAST axis (Q bias) or SECOND axis (Q weight,
            shape (in, n_heads*head_dim)).
            We expect the caller to have flattened the head axis into a
            single dim of size n_heads_orig*head_dim. We treat that dim as
            consecutive head_dim-wide blocks per head.
        n_kv_heads, group_orig, group_padded, head_dim: GQA dims.

    Returns:
        New numpy array with the head axis size = n_kv_heads * group_padded *
        head_dim, containing real data at the right positions and zeros for
        phantom heads.

    NOTE: we operate only on the LAST axis here. The caller decides which
    axis is the head axis and passes the array shaped accordingly.
    """
    arr = np.asarray(orig_per_head)
    orig_head_dim = n_kv_heads * group_orig * head_dim
    padded_head_dim = n_kv_heads * group_padded * head_dim
    assert (
        arr.shape[-1] == orig_head_dim
    ), f"expected last dim {orig_head_dim}, got {arr.shape[-1]}"
    out_shape = arr.shape[:-1] + (padded_head_dim,)
    out = np.zeros(out_shape, dtype=arr.dtype)
    for g in range(n_kv_heads):
        for h_local in range(group_orig):
            orig_off = (g * group_orig + h_local) * head_dim
            pad_off = (g * group_padded + h_local) * head_dim
            out[..., pad_off : pad_off + head_dim] = arr[
                ..., orig_off : orig_off + head_dim
            ]
    return out


def _gqa_reindex_qhead_axis_first(
    orig_per_head, n_kv_heads, group_orig, group_padded, head_dim
):
    """Like _gqa_reindex_qhead_axis but operates on the FIRST axis.

    Used for wo whose shape is (n_heads*head_dim, output_dim) — the head axis
    is the FIRST axis.
    """
    arr = np.asarray(orig_per_head)
    orig_head_dim = n_kv_heads * group_orig * head_dim
    padded_head_dim = n_kv_heads * group_padded * head_dim
    assert (
        arr.shape[0] == orig_head_dim
    ), f"expected first dim {orig_head_dim}, got {arr.shape[0]}"
    out_shape = (padded_head_dim,) + arr.shape[1:]
    out = np.zeros(out_shape, dtype=arr.dtype)
    for g in range(n_kv_heads):
        for h_local in range(group_orig):
            orig_off = (g * group_orig + h_local) * head_dim
            pad_off = (g * group_padded + h_local) * head_dim
            out[pad_off : pad_off + head_dim] = arr[orig_off : orig_off + head_dim]
    return out


def pad_weights(orig_weights, orig_config, padded_config, dtype=bfloat16):
    """Produce a LlamaWeights with all tensors zero-padded to padded_config.

    Padding rules:
    - Embeddings + final RMSNorm: zero-pad / RMS-pre-scale on emb_dim.
    - Per-layer wq/bq/wo: GQA-aware reindexing on the Q-head axis (insert
      phantom heads INSIDE each KV group, not at the end) so each KV head
      still owns the same orig Q heads it did before.
    - wk/wv/bk/bv: kv_dim unchanged (n_kv_heads is held constant), so just
      zero-pad the input axis on wk/wv.
    - w_gate/w_up: zero-pad input axis to padded emb_dim, output axis to
      padded hidden_dim.
    - w_down: zero-pad input axis (hidden_dim) and output axis (emb_dim).

    Tied lm_head reuses the padded embed_table.
    """
    o_emb = orig_config.emb_dim
    p_emb = padded_config.emb_dim
    o_hid = orig_config.hidden_dim
    p_hid = padded_config.hidden_dim
    n_kv = orig_config.n_kv_heads
    n_kv_dim = n_kv * orig_config.head_dim
    head_dim = orig_config.head_dim
    group_orig = orig_config.n_heads // n_kv
    group_padded = padded_config.n_heads // n_kv
    vocab = orig_config.vocab_size

    assert (
        padded_config.n_kv_heads == n_kv
    ), "GQA-aware reindex requires n_kv_heads to stay the same"
    assert n_kv * group_padded * head_dim == p_emb, (
        f"padded emb_dim={p_emb} must equal n_kv_heads*group_padded*head_dim="
        f"{n_kv * group_padded * head_dim}"
    )

    embed = np.asarray(orig_weights.embed_table, dtype=dtype)
    embed_padded = np.zeros((vocab, p_emb), dtype=dtype)
    embed_padded[:, :o_emb] = embed

    final_norm_padded = _scale_rmsnorm_weight(
        orig_weights.final_norm, o_emb, p_emb, dtype
    )

    layers_padded = []
    for lw in orig_weights.layers:
        # wq: orig shape (o_emb=1536, n_heads_orig*head_dim=1536). Pad input
        # axis to p_emb, then GQA-reindex the OUTPUT (head) axis.
        wq_in_padded = _pad_2d(lw.wq, p_emb, o_emb, dtype)
        wq_padded = _gqa_reindex_qhead_axis(
            wq_in_padded, n_kv, group_orig, group_padded, head_dim
        )
        assert wq_padded.shape == (p_emb, p_emb), wq_padded.shape

        # bq: orig shape (n_heads_orig*head_dim=1536,). GQA-reindex on the
        # only axis.
        bq_padded = _gqa_reindex_qhead_axis(
            np.asarray(lw.bq, dtype=dtype), n_kv, group_orig, group_padded, head_dim
        )

        # wo: orig shape (n_heads_orig*head_dim=1536, o_emb=1536).
        # GQA-reindex the FIRST axis, then zero-pad the OUTPUT axis to p_emb.
        wo_qheaded = _gqa_reindex_qhead_axis_first(
            np.asarray(lw.wo, dtype=dtype), n_kv, group_orig, group_padded, head_dim
        )
        wo_padded = _pad_2d(wo_qheaded, p_emb, p_emb, dtype)

        new_lw = LayerWeights(
            attn_norm=_scale_rmsnorm_weight(lw.attn_norm, o_emb, p_emb, dtype),
            wq=wq_padded,
            wk=_pad_2d(lw.wk, p_emb, n_kv_dim, dtype),
            wv=_pad_2d(lw.wv, p_emb, n_kv_dim, dtype),
            bq=bq_padded,
            bk=np.asarray(lw.bk, dtype=dtype),  # KV dim unchanged
            bv=np.asarray(lw.bv, dtype=dtype),
            wo=wo_padded,
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


def slice_output(padded_block_out, orig_emb_dim):
    """Slice a padded (seq_len, padded_emb_dim) tensor back to (seq_len, orig_emb_dim)."""
    return np.ascontiguousarray(padded_block_out[:, :orig_emb_dim])

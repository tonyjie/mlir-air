# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prefill orchestration helpers shared across LLM deployments.

Wraps the per-layer NPU loop + final RMSNorm + LM Head with three flavors:
- npu_full_prefill              : Phase 4 baseline (no KV extraction)
- run_npu_full_prefill          : Phase 3 with optional per-layer capture
                                  for diagnostic mode
- npu_prefill_with_kv_extraction: Phase 6 inference runner — extracts K/V
                                  from rms_gemms_rope intermediates so decode
                                  can pick up where prefill left off

All three take `ref_module` so the per-model CPU reference's `rms_norm` /
`transformer_block` are used for the final RMSNorm + (when capturing
intermediates) the per-layer CPU reference comparison.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from llama3_prefill import run_transformer_block


def embed_and_pad(prompt, tokenizer, weights, seq_len):
    """Tokenize, embed, and pad a prompt to seq_len.

    Returns (x_bf16, token_ids_int64, real_len). Padding uses tokenizer.eos_token_id
    (or 0 if absent).
    """
    token_ids = tokenizer.encode(prompt)
    real_len = len(token_ids)
    if real_len < seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = token_ids + [pad] * (seq_len - real_len)
    token_ids = np.array(token_ids[:seq_len], dtype=np.int64)
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[token_ids]
    return x_f32.astype(bfloat16), token_ids, real_len


def npu_full_prefill(
    x_bf16, weights, config, cache, rope_lut_bf16, ref_module, cpu_attn=False
):
    """Run all N layers + final RMSNorm + CPU LM Head; return logits + timings.

    Phase 4 baseline path — does NOT extract K/V (that's
    `npu_prefill_with_kv_extraction` for the end-to-end inference runner).
    """
    t0 = time.time()
    for layer_idx in range(config.n_layers):
        x_bf16, _ = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=layer_idx,
            verify=False,
            cpu_attn=cpu_attn,
            verbose=False,
        )
    npu_layer_time = time.time() - t0

    t0 = time.time()
    x_f32 = np.asarray(x_bf16, dtype=np.float32)
    x_normed = ref_module.rms_norm(x_f32, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x_normed @ lm_head.T
    cpu_lm_head_time = time.time() - t0

    return logits, npu_layer_time, cpu_lm_head_time


def run_npu_full_prefill(
    input_ids,
    weights,
    config,
    cache,
    rope_lut_bf16,
    ref_module,
    cpu_attn=False,
    capture_intermediates=False,
    verbose=False,
):
    """Phase 3 NPU full prefill — supports per-layer capture for drift analysis.

    Returns (logits, per_layer_outputs_or_None, npu_layer_time).
    """
    seq_len = len(input_ids)
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[input_ids]
    x_bf16 = x_f32.astype(bfloat16)

    per_layer = [] if capture_intermediates else None
    t0 = time.time()
    for layer_idx in range(config.n_layers):
        x_bf16, _ = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=layer_idx,
            verify=False,
            cpu_attn=cpu_attn,
            verbose=verbose,
        )
        if capture_intermediates:
            per_layer.append(np.asarray(x_bf16, dtype=np.float32).copy())
    npu_time = time.time() - t0

    x_f32_out = np.asarray(x_bf16, dtype=np.float32)
    x_normed = ref_module.rms_norm(x_f32_out, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x_normed @ lm_head.T

    return logits, per_layer, npu_time


def run_cpu_full_prefill(
    input_ids, weights, config, rope_lut_f32, ref_module, capture_intermediates=False
):
    """CPU reference full prefill — exposes per-layer outputs for drift analysis."""
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table_f32[input_ids]

    per_layer = [] if capture_intermediates else None
    for layer_idx in range(config.n_layers):
        x, _ = ref_module.transformer_block(
            x, weights.layers[layer_idx], rope_lut_f32, config
        )
        if capture_intermediates:
            per_layer.append(x.copy())

    x = ref_module.rms_norm(x, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x @ lm_head.T
    return logits, per_layer


def npu_prefill_with_kv_extraction(
    token_ids,
    weights,
    config,
    prefill_cache,
    rope_lut_bf16,
    max_seq,
    cpu_attn=False,
):
    """Run NPU prefill; extract per-layer K (post-RoPE) and V into KV cache.

    Used by the end-to-end inference runner (Phase 6 product) to feed the
    decode loop without re-running CPU prefill.

    Returns (x_last_layer_bf16, k_cache, v_cache).
    """
    seq_len = len(token_ids)
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    embed_f32 = weights.embed_table[token_ids].astype(np.float32)
    x_bf16 = embed_f32.astype(bfloat16)

    for layer_idx in range(config.n_layers):
        x_bf16, intermediates = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            verify=False,
            cpu_attn=cpu_attn,
            verbose=False,
        )
        k_roped = (
            np.asarray(intermediates["k_roped"], dtype=bfloat16)
            .reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
        )
        v_raw = (
            np.asarray(intermediates["v"], dtype=bfloat16)
            .reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
        )
        k_cache[layer_idx, :, :seq_len, :] = k_roped
        v_cache[layer_idx, :, :seq_len, :] = v_raw

    return x_bf16, k_cache, v_cache

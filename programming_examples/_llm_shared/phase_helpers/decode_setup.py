# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decode-side helpers shared across LLM deployments.

Functions:
- pre_transpose_decode_weights : pre-transpose Q/K/V/O/Gate/Up/Down for decode GEMV
- npu_lm_head_gemv             : NPU LM Head GEMV invocation (8-partition)
- seed_kv_cache_via_cpu_prefill: CPU reference prefill that populates per-layer
                                 KV cache for decode tests (Phase 5 path)

Pattern: callers pass `ref_module` (the per-model CPU reference, e.g.
`smollm2_reference` or `llama32_3b_reference`) so the same helper drives
any deployment.
"""

import numpy as np
from ml_dtypes import bfloat16


def pre_transpose_decode_weights(weights, config):
    """Pre-transpose per-layer weights for decode GEMV layout (idempotent).

    Decode GEMV expects W transposed vs the prefill GEMM layout. Caches the
    transposed copies on the LayerWeights objects so subsequent calls are
    free. Sets `weights._decode_weights_transposed = True` as a guard.
    """
    emb_dim = config.emb_dim
    kv_dim = config.n_kv_heads * config.head_dim
    hidden_dim = config.hidden_dim
    if hasattr(weights, "_decode_weights_transposed"):
        return
    for i, lw in enumerate(weights.layers):
        lw._wq_t = np.ascontiguousarray(
            lw.wq.astype(bfloat16).reshape(emb_dim, emb_dim).T
        )
        lw._wk_t = np.ascontiguousarray(
            lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
        )
        lw._wv_t = np.ascontiguousarray(
            lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
        )
        lw._wo_t = np.ascontiguousarray(
            lw.wo.astype(bfloat16).reshape(emb_dim, emb_dim).T
        )
        lw._wgate_t = np.ascontiguousarray(
            lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
        )
        lw._wup_t = np.ascontiguousarray(
            lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
        )
        lw._wdown_t = np.ascontiguousarray(
            lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
        )
        lw._layer_idx = i
    weights._decode_weights_transposed = True


def npu_lm_head_gemv(decode_cache, weights, config, x_normed_bf16):
    """Run NPU LM Head GEMV (8-partition × 16384) on a single token's hidden state.

    Output is concatenated across the 8 partitions and clipped to vocab_size
    (handles the small zero-pad when vocab is not exactly 8 × 16384, e.g.
    smollm2's vocab=49152 or llama-family vocab=128256).

    Caller must have called `llama3_inference._preload_decode_weights` first
    to populate `weights._lm_weight_parts_gemv`.
    """
    # Late import to keep this module light; llama3_inference is the source of
    # truth for the LM Head GEMV partition constants and backend kwargs.
    import llama3_inference

    lm_inputs = [x_normed_bf16]
    for p in range(llama3_inference._LM_N_PARTITIONS):
        lm_inputs.append(weights._lm_weight_parts_gemv[p])
        lm_inputs.append(np.zeros(llama3_inference._LM_N_PART, dtype=bfloat16))
    results = decode_cache.load_and_run(
        "lm_head_gemv",
        llama3_inference._LM_GEMV_BACKEND,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(llama3_inference._LM_N_PARTITIONS)],
        static_input_indices={
            1 + 2 * p for p in range(llama3_inference._LM_N_PARTITIONS)
        },
        intermediate_indices={
            2 + 2 * p for p in range(llama3_inference._LM_N_PARTITIONS)
        },
    )
    logits = np.concatenate(results, axis=0)[: config.vocab_size]
    return logits


def seed_kv_cache_via_cpu_prefill(
    weights, config, prompt_token_ids, rope_lut_f32, max_seq, ref_module
):
    """Run CPU reference prefill on the prompt; populate per-layer KV cache.

    Used by Phase 5 decode tests to seed the KV cache without depending on
    NPU prefill (which is validated separately by Phase 4). The end-to-end
    inference runner uses `prefill_runner.npu_prefill_with_kv_extraction`
    instead.

    Args:
        weights: LlamaWeights from the model's weights.load_weights()
        config: LlamaConfig
        prompt_token_ids: list/array of token IDs from tokenizer.encode(prompt)
        rope_lut_f32: F32 RoPE LUT (seq_len, head_dim)
        max_seq: KV cache capacity (positions)
        ref_module: the per-model CPU reference module (e.g.
                    `smollm2_reference` or `llama32_3b_reference`). Must
                    expose `transformer_block(x, layer_weights, rope_lut, config)`
                    and `rms_norm(x, weight, eps)`.

    Returns:
        (k_cache, v_cache, last_hidden_normed_f32): K/V caches shaped
        (n_layers, n_kv_heads, max_seq, head_dim) bfloat16, plus the
        F32 RMS-normed last-position hidden state for first-token LM head.
    """
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table_f32[np.array(prompt_token_ids, dtype=np.int64)]
    prompt_len = len(prompt_token_ids)

    k_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    for i in range(n_layers):
        x, intermediates = ref_module.transformer_block(
            x, weights.layers[i], rope_lut_f32, config
        )
        k_per_pos = intermediates["k_roped"].reshape(prompt_len, n_kv_heads, head_dim)
        v_per_pos = intermediates["v"].reshape(prompt_len, n_kv_heads, head_dim)
        k_cache[i, :, :prompt_len, :] = k_per_pos.transpose(1, 0, 2).astype(bfloat16)
        v_cache[i, :, :prompt_len, :] = v_per_pos.transpose(1, 0, 2).astype(bfloat16)

    x_normed = ref_module.rms_norm(x, weights.final_norm)
    return k_cache, v_cache, x_normed[-1]

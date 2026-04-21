# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5 — Qwen3-0.6B NPU decode block + generate loop.

**Phase A**: all decode ops moved to NPU (per-leaf), residual adds stay on host
(microsecond cost, plus eltwise_add at n=1024 hits a Peano llc crash —
deferred follow-up). Phase B will fuse leaves into multi-launch ELFs.

Per-token decode pipeline (per layer):
  1. NPU rms_attn_gemvs ELF (4 internal launches) → normed, q, k, v
  2. NPU qknorm_q (weighted_rms_norm M=16 N=128) → q_normed
  3. NPU qknorm_k (weighted_rms_norm M=8  N=128) → k_normed
  4. NPU rope_q (rope_halfsplit nrows=16, hd=128) → q_roped
  5. NPU rope_k (rope_halfsplit nrows=8,  hd=128) → k_roped
  6. Host decode_attention_cpu with KV cache → attn_out
  7. NPU O GEMV (m=emb_dim, k=q_dim)
  8. Host residual add (proj + x → res1; cheap, ~µs)
  9. NPU rms_1d (FFN RMSNorm)
 10. NPU Gate GEMV / Up GEMV
 11. NPU silu_and_mul (n=hidden_dim) — replaces the host SiLU*up
 12. NPU Down GEMV
 13. Host residual add (down + res1 → block_out)

After 28 layers:
  - Host final RMSNorm
  - NPU LM head GEMV (Step 4 ELF, 10 × 16384 partitions)

All NPU kernels validated standalone in qwen3_kernel_registry_test.py.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
# NOTE: do NOT add _EXAMPLES_DIR/"weighted_rms_norm" — it would shadow the
# `weighted_rms_norm` package import used by llama3 helpers.
for _p in (
    _EXAMPLES_DIR,
    _EXAMPLES_DIR / "llama3",
    _EXAMPLES_DIR / "matrix_vector_multiplication" / "bf16",
    _THIS_DIR,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen3_weights import LlamaConfig, generate_rope_lut

import llama3_prefill as _lp
from llama3_prefill import KernelCache, prepare_air_project
from llama3_decode import decode_attention_cpu

from multi_launch.rms_attn_gemvs_qknorm_rope_qwen3 import (
    build_rms_attn_gemvs_qknorm_rope_qwen3_module,
)
from multi_launch.o_gemv_ffn_silu_qwen3 import build_o_gemv_ffn_silu_qwen3_module

# ---------------------------------------------------------------------------
# Backend kwargs (matvec needs these for correctness — Step 1 finding)
# ---------------------------------------------------------------------------

_GEMV_BACKEND_BASE = {
    "omit_while_true_loop": False,
    "omit_pingpong": True,
    "runtime_loop_tiling_sizes": [4, 4],
    "use_lock_race_condition_fix": True,
    "output_format": "elf",
}
# Phase B fused ELFs (only these are used by the production decode path):
_RMS_ATTN_GEMVS_QKNORM_ROPE_BACKEND = {
    **_GEMV_BACKEND_BASE,
    "instance_name": "rms_attn_gemvs_qknorm_rope",
}
_O_GEMV_FFN_SILU_BACKEND = {
    **_GEMV_BACKEND_BASE,
    "instance_name": "o_gemv_ffn_silu",
}
_LM_HEAD_BACKEND = {**_GEMV_BACKEND_BASE, "instance_name": "lm_head_gemv"}


# ---------------------------------------------------------------------------
# Compile decode kernels (one-time)
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config):
    """Compile the Qwen3 decode NPU kernels — Phase B fused-ELF design.

    3 ELFs total (down from per-leaf 8):
      - rms_attn_gemvs_qknorm_rope (8 launches)
      - o_gemv_ffn_silu (8 launches, 3-K matvec rename)
      - lm_head_gemv (10 partitions)
    """
    from llama3.multi_launch_builder.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )
    from _llm_shared.kernel_builder.external_kernels import (
        compile_mv_og,
        compile_mv_dg_qwen3,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n=== Compiling Qwen3 decode kernels (Phase B fused) ===")
    # The 3-K rename in o_gemv_ffn_silu needs both renamed .o copies.
    compile_mv_og(tile_m=8)
    compile_mv_dg_qwen3(tile_m=8)

    if "rms_attn_gemvs_qknorm_rope" not in cache.artifacts:
        print("  [rms_attn_gemvs_qknorm_rope] building (8-launch fused)...")
        cache.compile_and_cache(
            "rms_attn_gemvs_qknorm_rope",
            build_rms_attn_gemvs_qknorm_rope_qwen3_module(
                emb_dim=emb_dim,
                q_dim=q_dim,
                kv_dim=kv_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            ),
            _RMS_ATTN_GEMVS_QKNORM_ROPE_BACKEND,
        )
    else:
        print("  [rms_attn_gemvs_qknorm_rope] cached")

    if "o_gemv_ffn_silu" not in cache.artifacts:
        print("  [o_gemv_ffn_silu] building (8-launch fused, 3-K rename)...")
        cache.compile_and_cache(
            "o_gemv_ffn_silu",
            build_o_gemv_ffn_silu_qwen3_module(
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                o_in_dim=q_dim,
            ),
            _O_GEMV_FFN_SILU_BACKEND,
        )
    else:
        print("  [o_gemv_ffn_silu] cached")

    if "lm_head_gemv" not in cache.artifacts:
        print(f"  [lm_head_gemv] building (10 × 16384, K={emb_dim})...")
        cache.compile_and_cache(
            "lm_head_gemv",
            build_lm_head_gemv_module(
                emb_dim=emb_dim,
                n_partitions=10,
                n_part=16384,
                tile_m=16,
                m_input=16,
                herd_m=8,
            ),
            _LM_HEAD_BACKEND,
        )
    else:
        print("  [lm_head_gemv] cached")

    cache._save_manifest()
    print(f"  → {len(cache.artifacts)} kernels in cache: {sorted(cache.artifacts)}")


# ---------------------------------------------------------------------------
# Pre-transpose decode weights once per `weights` (mirrors llama3 pattern at
# llama3_decode.py:480+). Stored as attributes on each LayerWeights so the
# decode block doesn't pay np.ascontiguousarray cost on every call.
# ---------------------------------------------------------------------------


def _ensure_decode_weights_transposed(layer_weights, config):
    """Pre-transpose weights to (out, in) form expected by matvec; cache on lw."""
    if getattr(layer_weights, "_decode_t_done", False):
        return
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    layer_weights._wq_t = np.ascontiguousarray(
        np.asarray(layer_weights.wq, dtype=bfloat16).T
    ).reshape(q_dim, emb_dim)
    layer_weights._wk_t = np.ascontiguousarray(
        np.asarray(layer_weights.wk, dtype=bfloat16).T
    ).reshape(kv_dim, emb_dim)
    layer_weights._wv_t = np.ascontiguousarray(
        np.asarray(layer_weights.wv, dtype=bfloat16).T
    ).reshape(kv_dim, emb_dim)
    layer_weights._wo_t = np.ascontiguousarray(
        np.asarray(layer_weights.wo, dtype=bfloat16).T
    ).reshape(emb_dim, q_dim)
    layer_weights._wgate_t = np.ascontiguousarray(
        np.asarray(layer_weights.w_gate, dtype=bfloat16).T
    ).reshape(hidden_dim, emb_dim)
    layer_weights._wup_t = np.ascontiguousarray(
        np.asarray(layer_weights.w_up, dtype=bfloat16).T
    ).reshape(hidden_dim, emb_dim)
    layer_weights._wdown_t = np.ascontiguousarray(
        np.asarray(layer_weights.w_down, dtype=bfloat16).T
    ).reshape(emb_dim, hidden_dim)
    layer_weights._decode_t_done = True


# Per-layer arg-list cache. Keyed by f"{kernel_name}_L{layer_idx}". Each
# entry is a list[np.ndarray] sized to the kernel's func arg count, with
# weights/intermediates pre-allocated and dynamic slots set to None.
# Mutated in place per call.
_DECODE_ARG_CACHE = {}


# ---------------------------------------------------------------------------
# Per-block decode (one token, one layer)
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
    layer_idx,
):
    """One transformer block for one decode token.

    x_bf16: (emb_dim,) input
    k_cache_layer/v_cache_layer: (n_kv_heads, max_seq, head_dim) — host KV cache
    current_pos: int, current token position (for KV cache write + masking)
    rope_lut_bf16: (max_seq, head_dim) half-split LUT
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    # Per-layer arg-list cache: avoid rebuilding the 17-arg list (and the
    # ~MB of np.ascontiguousarray weight transposes) on every call. We mutate
    # the dynamic slots (x_in, lut_q, lut_k) in place; static weights and
    # intermediate buffers are reused across calls.
    _arg_cache = _DECODE_ARG_CACHE
    rms_key = f"rms_attn_gemvs_qknorm_rope_L{layer_idx}"
    if rms_key not in _arg_cache:
        # First call: pre-transpose weights and allocate intermediate/output
        # numpy arrays once. KernelCache.load_and_run will then write static
        # inputs to BOs only on the first call (skipped thereafter via
        # static_input_indices).
        _ensure_decode_weights_transposed(layer_weights, config)
        _arg_cache[rms_key] = [
            None,  # 0: x_in (dynamic — overwritten per call)
            np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
            layer_weights._wq_t,  # (q_dim, emb_dim)
            layer_weights._wk_t,  # (kv_dim, emb_dim)
            layer_weights._wv_t,  # (kv_dim, emb_dim)
            np.asarray(layer_weights.q_norm, dtype=bfloat16).reshape(head_dim),
            np.asarray(layer_weights.k_norm, dtype=bfloat16).reshape(head_dim),
            None,  # 7: lut_q (dynamic)
            None,  # 8: lut_k (dynamic)
            np.zeros(emb_dim, dtype=bfloat16),  # 9: normed
            np.zeros(q_dim, dtype=bfloat16),  # 10: q
            np.zeros(kv_dim, dtype=bfloat16),  # 11: k
            np.zeros(q_dim, dtype=bfloat16),  # 12: q_normed
            np.zeros(kv_dim, dtype=bfloat16),  # 13: k_normed
            np.zeros(kv_dim, dtype=bfloat16),  # 14: v
            np.zeros(q_dim, dtype=bfloat16),  # 15: q_roped
            np.zeros(kv_dim, dtype=bfloat16),  # 16: k_roped
        ]

    # Update only the dynamic args (x_in, lut_q, lut_k) per call.
    rms_args = _arg_cache[rms_key]
    rms_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(emb_dim)
    lut_row = np.asarray(rope_lut_bf16[current_pos], dtype=bfloat16).reshape(head_dim)
    rms_args[7] = np.ascontiguousarray(
        np.tile(lut_row, n_heads).reshape(n_heads * head_dim)
    )
    rms_args[8] = np.ascontiguousarray(
        np.tile(lut_row, n_kv_heads).reshape(n_kv_heads * head_dim)
    )

    res = _lp._run_cached(
        cache,
        "rms_attn_gemvs_qknorm_rope",
        _RMS_ATTN_GEMVS_QKNORM_ROPE_BACKEND,
        *rms_args,
        output_indices=[14, 15, 16],
        # weights + per-layer norm weights are static. LUTs change per
        # decode position — exclude them so they get re-written each token.
        static_input_indices={1, 2, 3, 4, 5, 6},
        intermediate_indices={9, 10, 11, 12, 13, 14, 15, 16},
        bo_key=rms_key,
    )
    v = res[14].reshape(kv_dim)
    q_roped = res[15].reshape(q_dim)
    k_roped = res[16].reshape(kv_dim)

    # ---- 2. Update KV cache + CPU attention (host) ----
    k_per_head = k_roped.reshape(n_kv_heads, head_dim)
    v_per_head = v.reshape(n_kv_heads, head_dim)
    k_cache_layer[:, current_pos, :] = k_per_head
    v_cache_layer[:, current_pos, :] = v_per_head
    attn_out = decode_attention_cpu(
        q_roped,
        k_cache_layer,
        v_cache_layer,
        current_pos,
        n_heads,
        n_kv_heads,
        head_dim,
    )  # (q_dim,) bf16

    # ---- 3. NPU o_gemv_ffn_silu (8-launch fused, 3-K matvec rename) ----
    # O + add + RMSNorm + Gate + Up + SiLU+Mul + Down + add → block_out.
    offn_key = f"o_gemv_ffn_silu_L{layer_idx}"
    if offn_key not in _arg_cache:
        _ensure_decode_weights_transposed(layer_weights, config)
        _arg_cache[offn_key] = [
            None,  # 0: attn_out (dynamic)
            layer_weights._wo_t,  # 1: wo (q_dim → emb_dim)
            np.zeros(emb_dim, dtype=bfloat16),  # 2: proj
            None,  # 3: x_res (dynamic)
            np.zeros(emb_dim, dtype=bfloat16),  # 4: res1
            np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # 5
            np.zeros(emb_dim, dtype=bfloat16),  # 6: normed2
            layer_weights._wgate_t,  # 7
            np.zeros(hidden_dim, dtype=bfloat16),  # 8: gate
            layer_weights._wup_t,  # 9
            np.zeros(hidden_dim, dtype=bfloat16),  # 10: up
            np.zeros(hidden_dim, dtype=bfloat16),  # 11: swiglu
            layer_weights._wdown_t,  # 12
            np.zeros(emb_dim, dtype=bfloat16),  # 13: down
            np.zeros(emb_dim, dtype=bfloat16),  # 14: out
        ]

    offn_args = _arg_cache[offn_key]
    offn_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(q_dim)
    offn_args[3] = np.asarray(x_bf16, dtype=bfloat16).reshape(emb_dim)

    res = _lp._run_cached(
        cache,
        "o_gemv_ffn_silu",
        _O_GEMV_FFN_SILU_BACKEND,
        *offn_args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},  # weights
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=offn_key,
    )
    return res[14].reshape(emb_dim)


# ---------------------------------------------------------------------------
# NPU LM head: x_normed → logits[VOCAB]
# ---------------------------------------------------------------------------


_LM_PARTITIONS = 10
_LM_PART_SIZE = 16384


def npu_lm_head(cache, x_bf16, weights, config):
    """Run the 10-partition LM head GEMV ELF, slice down to vocab_size."""
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    # Pad lm_head from (vocab, emb_dim) to (10*16384, emb_dim) along axis 0.
    # Cache the partition tensors so we don't re-pad on every call.
    if not hasattr(npu_lm_head, "_part_weights"):
        lm_full_t = np.ascontiguousarray(
            np.asarray(weights.lm_head, dtype=bfloat16)
        )  # (vocab, emb_dim) — already (out, in) from loader
        padded = _LM_PARTITIONS * _LM_PART_SIZE
        if padded > vocab:
            pad_rows = np.zeros((padded - vocab, emb_dim), dtype=bfloat16)
            lm_full_t = np.concatenate([lm_full_t, pad_rows], axis=0)
        parts = []
        for p in range(_LM_PARTITIONS):
            parts.append(
                np.ascontiguousarray(
                    lm_full_t[p * _LM_PART_SIZE : (p + 1) * _LM_PART_SIZE]
                ).reshape(_LM_PART_SIZE, emb_dim)
            )
        npu_lm_head._part_weights = parts

    parts = npu_lm_head._part_weights
    outs = [np.zeros(_LM_PART_SIZE, dtype=bfloat16) for _ in range(_LM_PARTITIONS)]
    args = [np.asarray(x_bf16, dtype=bfloat16).reshape(emb_dim)]
    for w, o in zip(parts, outs):
        args.append(w)
        args.append(o)
    output_idxs = [2 + 2 * p for p in range(_LM_PARTITIONS)]
    static_idxs = {1 + 2 * p for p in range(_LM_PARTITIONS)}
    res = _lp._run_cached(
        cache,
        "lm_head_gemv",
        _LM_HEAD_BACKEND,
        *args,
        output_indices=output_idxs,
        static_input_indices=static_idxs,
        intermediate_indices=set(output_idxs),
        bo_key="lm_head_gemv",
    )
    logits_padded = np.concatenate(
        [res[2 + 2 * p].reshape(_LM_PART_SIZE) for p in range(_LM_PARTITIONS)]
    )
    return logits_padded[:vocab]


# ---------------------------------------------------------------------------
# Generate loop
# ---------------------------------------------------------------------------


def generate(
    seed_token_ids,
    n_tokens,
    weights,
    config,
    cache,
    max_seq=512,
    npu_lm=True,
    verbose=False,
):
    """Greedy decode `n_tokens` tokens after the seed prompt.

    seed_token_ids: list[int] of the prompt tokens (real, not padded). Their
        K/V cache must already be populated by a prior NPU prefill — this
        function performs ONLY decode (one token at a time).

    NOTE: for this MVP we run a CPU prefill on the seed (cheap at small
    n_tokens) to populate the KV cache. The final inference runner can wire
    NPU prefill + KV extraction.
    """
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    emb_dim = config.emb_dim
    rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq, dtype=bfloat16)

    # --- KV cache: per-layer (n_kv_heads, max_seq, head_dim) ---
    k_cache = [
        np.zeros((n_kv_heads, max_seq, head_dim), dtype=bfloat16)
        for _ in range(n_layers)
    ]
    v_cache = [
        np.zeros((n_kv_heads, max_seq, head_dim), dtype=bfloat16)
        for _ in range(n_layers)
    ]

    # --- CPU prefill for the seed (populates KV cache) ---
    import qwen3_reference

    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)
    seed_x = embed_table[np.array(seed_token_ids, dtype=np.int64)]
    x_cpu = seed_x.copy()
    for li in range(n_layers):
        # transformer_block returns (block_out, intermediates) — extract K/V
        x_cpu, inter = qwen3_reference.transformer_block(
            x_cpu, weights.layers[li], rope_lut_f32, config
        )
        # K_roped / V at each position [0..len(seed_token_ids)-1]
        k_roped = inter["k_roped"]  # (seq, kv_dim)
        v = inter["v"]  # (seq, kv_dim)
        seq = k_roped.shape[0]
        k_cache[li][:, :seq, :] = (
            k_roped.reshape(seq, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        v_cache[li][:, :seq, :] = (
            v.reshape(seq, n_kv_heads, head_dim).transpose(1, 0, 2).astype(bfloat16)
        )

    # First token from seed prefill (using the CPU-computed last-position hidden)
    x_last = x_cpu[len(seed_token_ids) - 1]
    norm_w = np.asarray(weights.final_norm, dtype=np.float32)
    rms = np.sqrt(np.mean(x_last * x_last) + config.rms_norm_eps)
    x_normed = ((x_last / rms) * norm_w).astype(bfloat16)
    if npu_lm:
        logits0 = npu_lm_head(cache, x_normed, weights, config)
    else:
        logits0 = (
            np.asarray(x_normed, np.float32) @ np.asarray(weights.lm_head, np.float32).T
        )
    next_id = int(np.argmax(logits0))

    decoded = list(seed_token_ids) + [next_id]
    times = []
    cur_pos = len(seed_token_ids)  # next decode token writes here

    for step in range(n_tokens - 1):
        t0 = time.time()
        x = embed_table[next_id].astype(bfloat16)
        for li in range(n_layers):
            x = run_decode_block(
                x,
                weights.layers[li],
                cache,
                config,
                k_cache[li],
                v_cache[li],
                cur_pos,
                rope_lut_bf16,
                layer_idx=li,
            )
        # Final RMSNorm + LM head
        x_f32 = x.astype(np.float32)
        rms = np.sqrt(np.mean(x_f32 * x_f32) + config.rms_norm_eps)
        x_normed = ((x_f32 / rms) * norm_w).astype(bfloat16)
        if npu_lm:
            logits = npu_lm_head(cache, x_normed, weights, config)
        else:
            logits = x_f32 @ np.asarray(weights.lm_head, np.float32).T

        next_id = int(np.argmax(logits))
        decoded.append(next_id)
        cur_pos += 1
        times.append(time.time() - t0)
        if verbose:
            print(f"  step {step}: id={next_id}  ({times[-1]*1000:.1f} ms)")
        if cur_pos >= max_seq:
            print(f"  WARN: hit max_seq={max_seq}, stopping decode")
            break

    return decoded, times


# ---------------------------------------------------------------------------
# Preload — pre-fire both fused decode ELFs once per layer (with dummy I/O)
# so that:
#   1. Per-layer BO sets get allocated up front (out of the hot decode loop).
#   2. Static-input weights get DMA'd to NPU BOs once (subsequent calls skip
#      the write via static_input_indices).
# Mirrors `llama3_prefill.py:preload_static_inputs` and the warmup dance in
# `llama3_decode.py:_decode_weights_preloaded_to_bos`.
# Also pre-loads the LM head GEMV's 10 partition weights.
# ---------------------------------------------------------------------------


def preload_decode_weights(cache, weights, config):
    """One-time warmup: allocate per-layer BOs + write static weights.

    After this returns, every per-layer `_run_cached(...)` call inside
    `run_decode_block` sees first_call=False and skips the weight-BO write
    (only the dynamic args get DMA'd each call).
    """
    if getattr(weights, "_qwen3_decode_preloaded", False):
        return
    n_layers = config.n_layers
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\nPreloading Qwen3 decode weights into per-layer BOs (×{n_layers})...")
    t0 = time.time()
    dummy_lut_q = np.zeros(q_dim, dtype=bfloat16)
    dummy_lut_k = np.zeros(kv_dim, dtype=bfloat16)
    for li in range(n_layers):
        lw = weights.layers[li]
        _ensure_decode_weights_transposed(lw, config)

        # Fire ELF 1 with dummy x_in/luts — populates BOs for this layer.
        _lp._run_cached(
            cache,
            "rms_attn_gemvs_qknorm_rope",
            _RMS_ATTN_GEMVS_QKNORM_ROPE_BACKEND,
            np.zeros(emb_dim, dtype=bfloat16),  # 0 x_in
            np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),  # 1 norm_w
            lw._wq_t,
            lw._wk_t,
            lw._wv_t,  # 2-4 weights
            np.asarray(lw.q_norm, dtype=bfloat16).reshape(head_dim),  # 5
            np.asarray(lw.k_norm, dtype=bfloat16).reshape(head_dim),  # 6
            dummy_lut_q,
            dummy_lut_k,  # 7-8 luts
            np.zeros(emb_dim, dtype=bfloat16),  # 9
            np.zeros(q_dim, dtype=bfloat16),  # 10
            np.zeros(kv_dim, dtype=bfloat16),  # 11
            np.zeros(q_dim, dtype=bfloat16),  # 12
            np.zeros(kv_dim, dtype=bfloat16),  # 13
            np.zeros(kv_dim, dtype=bfloat16),  # 14
            np.zeros(q_dim, dtype=bfloat16),  # 15
            np.zeros(kv_dim, dtype=bfloat16),  # 16
            output_indices=[14, 15, 16],
            static_input_indices={1, 2, 3, 4, 5, 6},
            intermediate_indices={9, 10, 11, 12, 13, 14, 15, 16},
            bo_key=f"rms_attn_gemvs_qknorm_rope_L{li}",
        )

        # Fire ELF 2 with dummy attn_out/x_res.
        _lp._run_cached(
            cache,
            "o_gemv_ffn_silu",
            _O_GEMV_FFN_SILU_BACKEND,
            np.zeros(q_dim, dtype=bfloat16),  # 0 attn_out
            lw._wo_t,  # 1
            np.zeros(emb_dim, dtype=bfloat16),  # 2
            np.zeros(emb_dim, dtype=bfloat16),  # 3 x_res
            np.zeros(emb_dim, dtype=bfloat16),  # 4
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # 5
            np.zeros(emb_dim, dtype=bfloat16),  # 6
            lw._wgate_t,  # 7
            np.zeros(hidden_dim, dtype=bfloat16),  # 8
            lw._wup_t,  # 9
            np.zeros(hidden_dim, dtype=bfloat16),  # 10
            np.zeros(hidden_dim, dtype=bfloat16),  # 11
            lw._wdown_t,  # 12
            np.zeros(emb_dim, dtype=bfloat16),  # 13
            np.zeros(emb_dim, dtype=bfloat16),  # 14
            output_indices=[14],
            static_input_indices={1, 5, 7, 9, 12},
            intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
            bo_key=f"o_gemv_ffn_silu_L{li}",
        )

    # LM head: warm BOs + static-load all 10 partitions.
    if not hasattr(npu_lm_head, "_part_weights"):
        # Force the partition-weight cache to populate by running once.
        npu_lm_head(cache, np.zeros(emb_dim, dtype=bfloat16), weights, config)

    weights._qwen3_decode_preloaded = True
    print(
        f"  Preload done: {time.time()-t0:.2f}s ({n_layers} layers × 2 ELFs + LM head)"
    )


# ---------------------------------------------------------------------------
# decode_loop_from_kv — entry point for inference runner that ALREADY did
# NPU prefill and extracted per-layer K/V. Skips the redundant CPU prefill.
# ---------------------------------------------------------------------------


def decode_loop_from_kv(
    seed_token_ids,
    first_decoded_id,
    n_more_tokens,
    k_per_layer_seqfirst,
    v_per_layer_seqfirst,
    weights,
    config,
    cache,
    max_seq=512,
    npu_lm=True,
    verbose=False,
):
    """Decode n_more_tokens after the prompt + first_decoded_id, given KV
    cache contents already populated by NPU prefill.

    k_per_layer_seqfirst, v_per_layer_seqfirst: lists of length n_layers,
        each shape (real_seq, n_kv_heads * head_dim) BF16 (the seq-first
        layout the prefill block produces).

    Returns (decoded_ids, per_token_times). decoded_ids includes the seed
    + first_decoded_id + n_more_tokens-1 newly decoded tokens.
    """
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    real_seq = len(seed_token_ids)
    rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq, dtype=bfloat16)

    # Allocate full KV cache and fill the seed positions from the supplied
    # seq-first K/V (heads-first layout: n_kv_heads, max_seq, head_dim).
    k_cache, v_cache = [], []
    for li in range(n_layers):
        kc = np.zeros((n_kv_heads, max_seq, head_dim), dtype=bfloat16)
        vc = np.zeros((n_kv_heads, max_seq, head_dim), dtype=bfloat16)
        k_seq = np.asarray(k_per_layer_seqfirst[li], dtype=bfloat16)[:real_seq]
        v_seq = np.asarray(v_per_layer_seqfirst[li], dtype=bfloat16)[:real_seq]
        kc[:, :real_seq, :] = k_seq.reshape(real_seq, n_kv_heads, head_dim).transpose(
            1, 0, 2
        )
        vc[:, :real_seq, :] = v_seq.reshape(real_seq, n_kv_heads, head_dim).transpose(
            1, 0, 2
        )
        k_cache.append(kc)
        v_cache.append(vc)

    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    norm_w = np.asarray(weights.final_norm, dtype=np.float32)

    decoded = list(seed_token_ids) + [first_decoded_id]
    next_id = first_decoded_id
    cur_pos = real_seq  # the next decode token writes into KV cache here
    times = []

    for step in range(n_more_tokens - 1):
        t0 = time.time()
        x = embed_table[next_id].astype(bfloat16)
        for li in range(n_layers):
            x = run_decode_block(
                x,
                weights.layers[li],
                cache,
                config,
                k_cache[li],
                v_cache[li],
                cur_pos,
                rope_lut_bf16,
                layer_idx=li,
            )
        x_f32 = x.astype(np.float32)
        rms = np.sqrt(np.mean(x_f32 * x_f32) + config.rms_norm_eps)
        x_normed = ((x_f32 / rms) * norm_w).astype(bfloat16)
        if npu_lm:
            logits = npu_lm_head(cache, x_normed, weights, config)
        else:
            logits = x_f32 @ np.asarray(weights.lm_head, np.float32).T

        next_id = int(np.argmax(logits))
        decoded.append(next_id)
        cur_pos += 1
        times.append(time.time() - t0)
        if verbose:
            print(f"  step {step}: id={next_id}  ({times[-1]*1000:.1f} ms)")
        if cur_pos >= max_seq:
            print(f"  WARN: hit max_seq={max_seq}, stopping decode")
            break

    return decoded, times

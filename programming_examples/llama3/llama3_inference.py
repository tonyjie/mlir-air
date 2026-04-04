#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B BF16 Inference on MLIR-AIR (NPU2).

Unified script: NPU prefill + NPU decode.
- Prefill: runs full prompt through 16 transformer layers on NPU (~1.92s)
- Decode: generates tokens one at a time using GEMV kernels on NPU (~351ms/tok)

Usage:
    cd build_peano
    python3 ../llama3_inference.py --compile-only
    python3 ../llama3_inference.py --run-only --n-tokens 10 --profile
    python3 ../llama3_inference.py --run-only --n-tokens 100 --profile
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama3_weights import LlamaConfig, load_weights, generate_rope_lut
from llama3_prefill import (
    KernelCache,
    compile_all_kernels,
    run_transformer_block,
    _run_cached,
    _SIMPLE_BACKEND,
)
from llama3_decode import (
    compile_decode_kernels,
    run_decode_block,
    decode_attention_cpu,
    _GEMV_K2048_BACKEND,
    _GEMV_K8192_BACKEND,
)

# ---------------------------------------------------------------------------
# NPU Prefill with KV cache extraction
# ---------------------------------------------------------------------------


def run_npu_prefill(
    token_ids,
    weights,
    config,
    prefill_cache,
    rope_lut_bf16,
    max_seq,
    cpu_attn=True,
    profile=False,
):
    """Run NPU prefill and extract KV cache for decode.

    Returns:
        prefill_token: int — first predicted token ID
        k_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        v_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        prompt_len: actual prompt length (before padding)
    """
    seq_len = len(token_ids)
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    # Pre-allocate KV cache
    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    # Token embedding
    embed_f32 = weights.embed_table[token_ids].astype(np.float32)
    x_bf16 = embed_f32.astype(bfloat16)
    x_f32 = embed_f32.copy()

    print(f"Running NPU prefill ({config.n_layers} layers, seq_len={seq_len})...")
    t_prefill_start = time.time()

    # Run 16 transformer layers on NPU, collecting KV cache
    for layer_idx in range(config.n_layers):
        layer_t0 = time.perf_counter() if profile else None

        x_bf16, x_f32, intermediates = run_transformer_block(
            x_bf16,
            x_f32,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            prefill_cache,
            layer_idx=layer_idx,
            verify=False,
            cpu_attn=cpu_attn,
        )

        # Extract KV cache from intermediates
        # k_roped: (seq_len, kv_dim) — post-RoPE K
        # v: (seq_len, kv_dim) — raw V projection
        k_roped = intermediates["k_roped"]
        v_raw = intermediates["v"]

        k_cache[layer_idx, :, :seq_len, :] = (
            k_roped.astype(bfloat16)
            .reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
        )
        v_cache[layer_idx, :, :seq_len, :] = (
            v_raw.astype(bfloat16)
            .reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
        )

        if profile:
            layer_t = time.perf_counter() - layer_t0
            print(f"  Layer {layer_idx:2d}: {layer_t*1000:.0f}ms")

    # Final RMSNorm (NPU)
    x_in = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)
    w_in = np.asarray(weights.final_norm, dtype=bfloat16).reshape(emb_dim)
    y_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(prefill_cache, "rmsnorm", _SIMPLE_BACKEND, x_in, w_in, y_out)
    x_normed = results[-1].reshape(seq_len, emb_dim)

    # LM Head (NPU — 8-partition multi-launch ELF)
    vocab_size = weights.lm_head.shape[0]
    n_part = 16384
    n_partitions = 8

    if not hasattr(weights, "_lm_weight_parts"):
        weights._lm_weight_parts = []
        for p in range(n_partitions):
            n_start = p * n_part
            n_end = min(n_start + n_part, vocab_size)
            n_actual = n_end - n_start
            w = np.zeros((emb_dim, n_part), dtype=bfloat16)
            w[:, :n_actual] = np.ascontiguousarray(
                weights.lm_head[n_start:n_end, :].T
            ).astype(bfloat16)
            weights._lm_weight_parts.append(w)

    _LM_HEAD_BACKEND = {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "lm_head",
    }
    x_lm = np.asarray(x_normed, dtype=bfloat16).reshape(seq_len, emb_dim)
    lm_inputs = [x_lm]
    output_indices = []
    for p in range(n_partitions):
        lm_inputs.append(weights._lm_weight_parts[p])
        lm_inputs.append(np.zeros((seq_len, n_part), dtype=bfloat16))
        output_indices.append(2 + 2 * p)

    results = _run_cached(
        prefill_cache,
        "lm_head",
        _LM_HEAD_BACKEND,
        *lm_inputs,
        output_indices=output_indices,
        static_input_indices=(
            {1 + 2 * p for p in range(n_partitions)}
            | {2 + 2 * p for p in range(n_partitions)}
        ),
    )

    logits = np.zeros((seq_len, vocab_size), dtype=bfloat16)
    for p in range(n_partitions):
        out_idx = 2 + 2 * p
        n_start = p * n_part
        n_end = min(n_start + n_part, vocab_size)
        logits[:, n_start:n_end] = results[out_idx].reshape(seq_len, n_part)[
            :, : n_end - n_start
        ]
    logits_f32 = logits.astype(np.float32)

    # Find actual prompt length and predict first token
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_len = len([t for t in token_ids if t != _tok.eos_token_id])
    pred_pos = prompt_len - 1
    prefill_token = int(np.argmax(logits_f32[pred_pos]))

    t_prefill = time.time() - t_prefill_start
    print(f"NPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}")

    return prefill_token, k_cache, v_cache, prompt_len


# ---------------------------------------------------------------------------
# Full inference: NPU prefill + NPU decode
# ---------------------------------------------------------------------------


def generate(
    prompt_tokens,
    weights,
    config,
    prefill_cache,
    decode_cache,
    rope_lut_bf16,
    n_tokens=10,
    profile=False,
    cpu_attn=True,
):
    """Run NPU prefill + NPU decode generation.

    Args:
        prompt_tokens: list of token IDs (padded to seq_len)
        weights: LlamaWeights
        config: LlamaConfig
        prefill_cache: KernelCache for prefill kernels
        decode_cache: KernelCache for decode kernels
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT
        n_tokens: number of tokens to generate
        profile: print per-token timing
        cpu_attn: use CPU attention for prefill
    """
    seq_len = len(prompt_tokens)
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    max_seq = seq_len + n_tokens

    print(f"\n{'='*60}")
    print(f"LLAMA Inference: prompt_len={seq_len}, n_tokens={n_tokens}")
    print(f"{'='*60}\n")

    # --- Phase 1: NPU Prefill ---
    prefill_token, k_cache, v_cache, prompt_len = run_npu_prefill(
        prompt_tokens,
        weights,
        config,
        prefill_cache,
        rope_lut_bf16,
        max_seq,
        cpu_attn=cpu_attn,
        profile=profile,
    )

    # --- Phase 2: NPU Decode ---
    # Pre-transpose weights for GEMV (done once)
    kv_dim = n_kv_heads * head_dim
    hidden_dim = config.hidden_dim
    if not hasattr(weights, "_decode_weights_transposed"):
        print("Pre-transposing weights for GEMV...")
        for lw in weights.layers:
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
        weights._decode_weights_transposed = True
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    generated_tokens = [prefill_token]
    current_pos = prompt_len  # Next position after actual prompt
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    print(f"\nDecoding {n_tokens} tokens...")
    t_decode_start = time.time()

    for token_idx in range(n_tokens):
        t_token_start = time.perf_counter()

        # Run 16 transformer blocks
        x = x_decode.copy()
        for layer_idx in range(config.n_layers):
            x = run_decode_block(
                x,
                weights.layers[layer_idx],
                decode_cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )

        # Final RMSNorm + LM Head (CPU)
        from llama3_reference import rms_norm

        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, emb_dim),
            weights.final_norm.astype(np.float32),
        )
        logits = x_normed @ weights.lm_head.astype(np.float32).T
        next_token = int(np.argmax(logits[0]))

        t_token = time.perf_counter() - t_token_start

        generated_tokens.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if profile:
            print(f"  Token {token_idx}: id={next_token}, time={t_token*1000:.0f}ms")

    t_decode = time.time() - t_decode_start

    print(f"\nGenerated {n_tokens} tokens in {t_decode:.2f}s")
    print(f"Tokens/second: {n_tokens / t_decode:.2f}")
    print(f"Time/token: {t_decode / n_tokens * 1000:.0f}ms")

    return generated_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-1B Inference (NPU)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--cpu-attn", action="store_true", help="Use CPU attention for prefill"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
    )
    args = parser.parse_args()

    # Copy external kernel .o files to cwd
    import shutil
    from pathlib import Path

    for src_name, search_paths in [
        (
            "mv.o",
            [
                Path(__file__).parent.parent
                / "matrix_vector_multiplication"
                / "bf16"
                / "build_peano"
                / "mv.o",
            ],
        ),
        (
            "rope.o",
            [
                Path(__file__).parent.parent
                / "rope_lut"
                / "test_llama_dims"
                / "build_peano"
                / "rope.o",
            ],
        ),
        (
            "silu_and_mul.o",
            [
                Path(__file__).parent
                / "ffn_swiglu"
                / "build_peano"
                / "air_project"
                / "silu_and_mul.o",
                Path(__file__).parent
                / "build_peano"
                / "air_project"
                / "silu_and_mul.o",
            ],
        ),
        (
            "attn_npu2.o",
            [
                Path(__file__).parent.parent
                / "flash_attention"
                / "kernel_fusion_based"
                / "build_peano"
                / "attn.o",
            ],
        ),
    ]:
        if not Path(src_name).exists():
            for src_path in search_paths:
                if src_path.exists():
                    shutil.copy2(src_path, src_name)
                    break

    config = LlamaConfig()
    seq_len = 2048

    # Separate caches for prefill and decode (use absolute paths matching
    # the standalone scripts' defaults)
    llama_dir = Path(__file__).resolve().parent
    prefill_cache = KernelCache(str(llama_dir / "kernel_cache"), verbose=args.verbose)
    decode_cache = KernelCache("decode_kernel_cache", verbose=args.verbose)

    if not args.run_only:
        print("Compiling prefill kernels...")
        compile_all_kernels(prefill_cache, config, seq_len, cpu_attn=args.cpu_attn)
        print("\nCompiling decode kernels...")
        compile_decode_kernels(decode_cache, config)

    if args.compile_only:
        sys.exit(0)

    if args.run_only:
        prefill_cache.load_manifest()
        decode_cache.load_manifest()

    # Load weights and tokenizer
    print("\nLoading weights...")
    weights = load_weights("meta-llama/Llama-3.2-1B")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_len_actual = len(prompt_tokens)

    # Pad to seq_len
    if len(prompt_tokens) < seq_len:
        prompt_tokens = prompt_tokens + [tokenizer.eos_token_id] * (
            seq_len - len(prompt_tokens)
        )

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    generated = generate(
        prompt_tokens,
        weights,
        config,
        prefill_cache,
        decode_cache,
        rope_lut_bf16,
        n_tokens=args.n_tokens,
        profile=args.profile,
        cpu_attn=args.cpu_attn,
    )

    # Decode tokens
    print(f"\n{'='*60}")
    print(f"Generated text:")
    print(f"{'='*60}")
    all_tokens = prompt_tokens[:prompt_len_actual] + generated
    print(tokenizer.decode(all_tokens))

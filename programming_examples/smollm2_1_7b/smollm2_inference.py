# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B — end-to-end NPU inference (NPU prefill + NPU decode).

Pipeline:
    embed (CPU)
    --- prefill (per prompt) ---
    -> 24x [rms_gemms_rope (NPU 6-launch ELF) -> NPU FlashAttention -> o_ffn (NPU 8-launch ELF)]
       extracting per-layer K (post-RoPE) and V into KV cache from the rms_gemms_rope intermediates
    -> final RMSNorm at last prompt position only (CPU, single vector)
    -> NPU LM Head GEMV at last prompt position (reuses the decode lm_head_gemv kernel — vector in)
    -> argmax -> first generated token
    --- decode (per token) ---
    -> 24x [rms_gemv_rope (NPU 6-launch ELF) -> CPU attention (with KV cache) -> o_gemv_ffn (NPU 8-launch ELF)]
       updating KV cache at current_pos
    -> final RMSNorm (CPU, single vector)
    -> NPU LM Head GEMV
    -> argmax -> next token

Modeled on `llama3_inference.run_npu_prefill` + `generate`. Differences for SmolLM2:
  - 24 layers (vs 16); MHA n_kv_heads=32 (vs GQA n_kv_heads=8)
  - Tied embeddings (lm_head shares embed_tokens)
  - vocab=49152 with NPU LM Head GEMV using zero-padded 8-partition layout
  - Reuses the **decode** lm_head_gemv kernel for the prefill first-token prediction
    (we only need logits at one position; no need for the prefill lm_head GEMM)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
for p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smollm2_weights import LlamaConfig, load_weights, generate_rope_lut
import smollm2_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    preload_prefill_weights,
)
from llama3_decode import compile_decode_kernels, run_decode_block
import llama3_inference  # _preload_decode_weights, _LM_N_PARTITIONS/_PART, _LM_GEMV_BACKEND
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

from smollm2_phase2_test import compile_block_kernels
from smollm2_phase5_test import _pre_transpose_decode_weights, _npu_lm_head_gemv


def npu_prefill_with_kv_extraction(
    token_ids, weights, config, prefill_cache, rope_lut_bf16, max_seq, cpu_attn=False
):
    """Run 24-layer NPU prefill; extract per-layer K (post-RoPE) and V into KV cache.

    Returns:
        x_last_layer_bf16: (seq_len, emb_dim) — final transformer-layer output
        k_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        v_cache: (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
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
        # Reshape (seq_len, n_kv_heads*head_dim) -> (n_kv_heads, seq_len, head_dim)
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


def main():
    parser = argparse.ArgumentParser(
        description="SmolLM2-1.7B end-to-end NPU inference (prefill + decode)"
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number of decode tokens to generate (default: 10)",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Prefill seq_len AND decode max position (default: 2048; "
        "reuses Phase 4's prefill kernel cache)",
    )
    parser.add_argument("--prefill-cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--decode-cache-dir", type=str, default="decode_kernel_cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use CPU attention in prefill instead of NPU FA",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print per-token decode timings"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile prefill + decode kernels and exit",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    config = LlamaConfig()
    print(f"SmolLM2 end-to-end NPU inference")
    print(
        f"  layers={config.n_layers}, vocab={config.vocab_size}, attn={'CPU' if args.cpu_attn else 'NPU FA'}"
    )
    print(f"  seq_len={args.seq_len}, n_tokens={args.n_tokens}")

    # ---- Compile (cache hits if previously built) ----
    print("\n[setup] Kernel caches...")
    t = time.time()
    prepare_air_project()

    prefill_cache_dir = _THIS_DIR / args.prefill_cache_dir
    decode_cache_dir = _THIS_DIR / args.decode_cache_dir
    prefill_cache = KernelCache(cache_dir=str(prefill_cache_dir))
    decode_cache = KernelCache(cache_dir=str(decode_cache_dir))
    if (prefill_cache_dir / "manifest.json").exists():
        prefill_cache.load_manifest()
    if (decode_cache_dir / "manifest.json").exists():
        decode_cache.load_manifest()

    compile_all_external_kernels(head_dim=config.head_dim)
    compile_block_kernels(prefill_cache, config, args.seq_len, cpu_attn=args.cpu_attn)
    needed_decode = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
    if not all(k in decode_cache.artifacts for k in needed_decode):
        compile_decode_kernels(decode_cache, config)
    t_compile = time.time() - t
    print(f"  Compile / cache load: {t_compile:.1f}s")

    if args.compile_only:
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    # ---- Load weights ----
    print("\n[setup] Weights...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    t_load = time.time() - t
    print(f"  Weight load: {t_load:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )

    # ---- Pre-load BOs (prefill + decode + LM Head GEMV) ----
    print("\n[setup] Pre-loading BOs...")
    t = time.time()
    _pre_transpose_decode_weights(weights, config)
    preload_prefill_weights(weights, config, prefill_cache, args.seq_len, rope_lut_bf16)
    llama3_inference._preload_decode_weights(decode_cache, weights, config)
    t_preload = time.time() - t
    print(f"  BO preload: {t_preload:.1f}s")

    # ---- Tokenize ----
    prompt_tokens = tokenizer.encode(args.prompt)
    real_len = len(prompt_tokens)
    if real_len > args.seq_len:
        prompt_tokens = prompt_tokens[: args.seq_len]
        real_len = args.seq_len
    if real_len < args.seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = prompt_tokens + [pad] * (args.seq_len - real_len)
    else:
        token_ids = prompt_tokens
    token_ids = np.array(token_ids, dtype=np.int64)
    pred_pos = real_len - 1

    print(f"\n{'='*60}")
    print(f"Inference on prompt: {args.prompt!r}")
    print(f"  ({real_len} real tokens; padded to {args.seq_len})")
    print(f"{'='*60}")

    # ---- NPU PREFILL ----
    print(f"\n[1/3] NPU prefill ({config.n_layers} layers)...")
    t = time.time()
    x_prefill, k_cache, v_cache = npu_prefill_with_kv_extraction(
        token_ids,
        weights,
        config,
        prefill_cache,
        rope_lut_bf16,
        max_seq=args.seq_len,
        cpu_attn=args.cpu_attn,
    )
    t_prefill = time.time() - t
    print(
        f"  NPU prefill: {t_prefill:.2f}s  ({t_prefill/config.n_layers*1000:.0f} ms/layer)"
    )

    # ---- First-token prediction (NPU LM Head GEMV at last prompt position) ----
    t = time.time()
    last_hidden = np.asarray(x_prefill, dtype=np.float32)[pred_pos : pred_pos + 1]
    last_normed_bf16 = (
        smollm2_reference.rms_norm(last_hidden, weights.final_norm)
        .flatten()
        .astype(bfloat16)
    )
    first_logits = _npu_lm_head_gemv(decode_cache, weights, config, last_normed_bf16)
    first_token = int(np.argmax(first_logits))
    t_first_lm_head = time.time() - t
    print(
        f"  First LM Head GEMV: {t_first_lm_head*1000:.0f} ms  -> "
        f"{tokenizer.decode([first_token])!r} (id={first_token})"
    )

    # ---- NPU DECODE LOOP ----
    print(f"\n[2/3] NPU decode loop ({args.n_tokens} tokens)...")
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    generated = list(prompt_tokens) + [first_token]
    decode_times = []
    current_token = first_token

    for token_idx in range(args.n_tokens - 1):
        current_pos = len(generated) - 1
        if current_pos >= args.seq_len:
            print(f"  Hit seq_len cap at pos={current_pos}, stopping")
            break

        x_in_bf16 = embed_table_f32[current_token].astype(bfloat16)

        t = time.time()
        x_bf16 = x_in_bf16
        for layer_idx in range(config.n_layers):
            x_bf16 = run_decode_block(
                x_bf16,
                weights.layers[layer_idx],
                decode_cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )
        x_f32 = np.asarray(x_bf16, dtype=np.float32).reshape(1, config.emb_dim)
        x_normed_bf16 = (
            smollm2_reference.rms_norm(x_f32, weights.final_norm)
            .flatten()
            .astype(bfloat16)
        )
        next_logits = _npu_lm_head_gemv(decode_cache, weights, config, x_normed_bf16)
        next_token = int(np.argmax(next_logits))
        decode_times.append(time.time() - t)

        if args.profile:
            print(
                f"  Tok {token_idx+1:2d} pos={current_pos:3d}  "
                f"{tokenizer.decode([next_token])!r:<14s}  {decode_times[-1]*1000:.0f} ms"
            )

        generated.append(next_token)
        current_token = next_token

    # ---- Summary ----
    print(f"\n[3/3] Done.\n{'='*60}")
    print(f"Generated text:")
    print(f"  {tokenizer.decode(generated)!r}")
    print()
    print(f"Timings (one-time setup):")
    print(f"  Compile / cache load : {t_compile:.1f}s")
    print(f"  Weight load          : {t_load:.1f}s")
    print(f"  BO preload           : {t_preload:.1f}s")
    print(f"Timings (per inference):")
    print(
        f"  NPU prefill (24L)    : {t_prefill:.2f}s ({t_prefill/config.n_layers*1000:.0f} ms/layer)"
    )
    print(f"  First LM Head GEMV   : {t_first_lm_head*1000:.0f} ms")
    if decode_times:
        avg_ms = float(np.mean(decode_times)) * 1000
        med_ms = float(np.median(decode_times)) * 1000
        n_dec = len(decode_times)
        print(
            f"  Decode {n_dec} tokens     : avg={avg_ms:.0f} ms/token, median={med_ms:.0f} ms/token  ({1000/avg_ms:.1f} tok/s)"
        )
    print(
        f"  Total inference wall : {t_prefill + t_first_lm_head + sum(decode_times):.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

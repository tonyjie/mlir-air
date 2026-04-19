# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 5 — decode performance for Qwen2.5-1.5B on NPU2.

CPU prefill seeds the KV cache (Phase 5 design, per llama3 pattern), then
NPU decode loop with KV cache. Decode uses ORIG shapes (emb=1536,
hidden=8960, n_heads=12) — at M=1 the BD-pool exhaustion that forced
prefill padding doesn't apply, so we keep decode at the natural shapes
for simplicity and speed.

Pattern application (5/5):
  1. Multi-launch merging      INHERITED (rms_gemv_rope=6, o_gemv_ffn=8)
  2. Static weight BOs         INHERITED (preload_decode_weights)
  3. NPU LM Head GEMV          APPLIED (10×16384, vocab=151936)
  4. Extern kernel rename      APPLIED (mv_k8960.o for Down GEMV K=8960)
  5. CPU->NPU op promotion     PARTIAL (attention stays CPU, llama3 design)
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
for _p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen25_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen25_reference

from llama3_prefill import KernelCache, prepare_air_project
from llama3_decode import run_decode_block

from _llm_shared.phase_helpers.decode_setup import (
    pre_transpose_decode_weights,
    seed_kv_cache_via_cpu_prefill,
)

from qwen25_bias import (
    install_qkv_bias_wrapper,
    register_layer_bias,
    precompute_rope_bias,
    set_decode_position,
)
from qwen25_decode_setup import (
    compile_qwen25_decode_kernels,
    preload_qwen25_lm_head,
    qwen25_npu_lm_head_gemv,
)


def _register_decode_biases(weights, config, rope_lut_bf16, max_seq):
    """Pre-RoPE bias for ALL n_layers, sized to max_seq."""
    for i, lw in enumerate(weights.layers):
        bq_roped = precompute_rope_bias(
            lw.bq, rope_lut_bf16, config.n_heads, config.head_dim, max_seq
        )
        bk_roped = precompute_rope_bias(
            lw.bk, rope_lut_bf16, config.n_kv_heads, config.head_dim, max_seq
        )
        register_layer_bias(i, bq_roped, bk_roped, lw.bv)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Phase 5 decode perf")
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=20,
        help="Number of decode tokens to generate (default: 20)",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--max-seq",
        type=int,
        default=128,
        help="Max KV-cache positions (default: 128)",
    )
    parser.add_argument("--cache-dir", type=str, default="decode_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--cpu-verify", action="store_true", default=True)
    parser.add_argument("--no-cpu-verify", dest="cpu_verify", action="store_false")
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=2,
        help="Discard first N tokens for timing average",
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"Qwen2.5-1.5B (orig shapes for decode): n_layers={config.n_layers}, "
        f"emb={config.emb_dim}, hidden={config.hidden_dim}, n_heads={config.n_heads}, "
        f"n_kv_heads={config.n_kv_heads}, head_dim={config.head_dim}, "
        f"vocab={config.vocab_size}"
    )

    if args.compile_only:
        prepare_air_project()
        cache = KernelCache(cache_dir=str(_THIS_DIR / args.cache_dir))
        if (_THIS_DIR / args.cache_dir / "manifest.json").exists():
            cache.load_manifest()
        compile_qwen25_decode_kernels(cache, config)
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    print("\nLoading weights...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.max_seq, dtype=bfloat16
    )
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)

    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_len = len(prompt_tokens)
    print(f"\nPrompt: {args.prompt!r} ({prompt_len} tokens)")

    print("\nRunning CPU reference prefill (one-time, seeds KV cache)...")
    t = time.time()
    k_cache, v_cache, x_normed_first = seed_kv_cache_via_cpu_prefill(
        weights,
        config,
        prompt_tokens,
        rope_lut_f32,
        args.max_seq,
        qwen25_reference,
    )
    print(f"  CPU prefill: {time.time()-t:.1f}s")
    lm_head_f32 = np.asarray(weights.lm_head, dtype=np.float32)
    logits_first = x_normed_first @ lm_head_f32.T
    first_npu_token = int(np.argmax(logits_first))
    print(
        f"  First generated token (from CPU prefill): "
        f"{tokenizer.decode([first_npu_token])!r} (id={first_npu_token})"
    )

    prepare_air_project()
    cache = KernelCache(cache_dir=str(_THIS_DIR / args.cache_dir))
    if (_THIS_DIR / args.cache_dir / "manifest.json").exists():
        cache.load_manifest()
        print(f"\nLoaded existing decode cache: {sorted(cache.artifacts.keys())}")

    needed = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
    if not all(k in cache.artifacts for k in needed):
        print("\nCompiling decode kernels for Qwen2.5-1.5B shapes...")
        t = time.time()
        compile_qwen25_decode_kernels(cache, config)
        print(f"  Compile: {time.time()-t:.1f}s")
    else:
        print("\nDecode kernels already cached — skipping compile.")

    install_qkv_bias_wrapper()
    print(
        f"\nRegistering Qwen2 QKV bias for all {config.n_layers} layers (max_seq={args.max_seq})..."
    )
    _register_decode_biases(weights, config, rope_lut_bf16, args.max_seq)

    print("\nPre-transposing decode weights...")
    pre_transpose_decode_weights(weights, config)

    print(f"Pre-loading LM head BOs (10 partitions × {16384})...")
    t = time.time()
    preload_qwen25_lm_head(cache, weights, config)
    print(f"  Preload: {time.time()-t:.1f}s")

    print(f"\n{'='*60}")
    print(f"DECODE LOOP — generating {args.n_tokens} tokens")
    print(f"{'='*60}")

    generated = list(prompt_tokens) + [first_npu_token]
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    npu_token_times_full = []
    matches = 0
    n_decoded = 0
    current_token = first_npu_token

    for token_idx in range(args.n_tokens - 1):
        current_pos = len(generated) - 1
        if current_pos >= args.max_seq:
            print(f"  Hit max_seq={args.max_seq}, stopping")
            break

        x_in_bf16 = embed_table_f32[current_token].astype(bfloat16)

        t0 = time.time()
        x_bf16 = x_in_bf16
        for layer_idx in range(config.n_layers):
            # Mark layer-id on weights so run_decode_block uses per-layer BO key.
            weights.layers[layer_idx]._layer_idx = layer_idx
            # Tell the bias wrapper which position this rms_gemv_rope is at.
            set_decode_position(current_pos)
            x_bf16 = run_decode_block(
                x_bf16,
                weights.layers[layer_idx],
                cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )
        x_f32 = np.asarray(x_bf16, dtype=np.float32).reshape(1, config.emb_dim)
        x_normed_bf16 = (
            qwen25_reference.rms_norm(x_f32, weights.final_norm)
            .flatten()
            .astype(bfloat16)
        )
        npu_logits = qwen25_npu_lm_head_gemv(cache, weights, config, x_normed_bf16)
        next_npu = int(np.argmax(npu_logits))
        npu_t = time.time() - t0
        npu_token_times_full.append(npu_t)

        match_str = "?"
        cpu_t = 0.0
        if args.cpu_verify:
            t0 = time.time()
            cpu_x = embed_table_f32[np.array(generated, dtype=np.int64)]
            for layer_idx in range(config.n_layers):
                cpu_x, _ = qwen25_reference.transformer_block(
                    cpu_x, weights.layers[layer_idx], rope_lut_f32, config
                )
            cpu_normed = qwen25_reference.rms_norm(cpu_x, weights.final_norm)
            cpu_logits = cpu_normed[-1] @ lm_head_f32.T
            next_cpu = int(np.argmax(cpu_logits))
            cpu_t = time.time() - t0
            match = next_npu == next_cpu
            matches += int(match)
            match_str = "YES" if match else f"NO  CPU={tokenizer.decode([next_cpu])!r}"
        n_decoded += 1

        npu_token_str = tokenizer.decode([next_npu])
        print(
            f"  Tok {token_idx+1:2d} pos={current_pos:3d}  "
            f"NPU={next_npu:6d} {npu_token_str!r:<14s}  "
            f"NPU_t={npu_t*1000:6.1f}ms  CPU_t={cpu_t:5.1f}s  {match_str}"
        )

        generated.append(next_npu)
        current_token = next_npu

    # Clear decode position so prefill (if reused later) isn't affected.
    set_decode_position(None)

    print(f"\n{'='*60}")
    print(f"Phase 5 — decode perf summary")
    print(f"{'='*60}")
    print(f"  Generated text: {tokenizer.decode(generated)!r}")
    print()

    if npu_token_times_full:
        warmup = min(args.n_warmup, max(0, len(npu_token_times_full) - 1))
        steady = npu_token_times_full[warmup:]
        avg_full_ms = float(np.mean(npu_token_times_full)) * 1000
        avg_steady_ms = float(np.mean(steady)) * 1000 if steady else 0.0
        median_steady_ms = float(np.median(steady)) * 1000 if steady else 0.0
        tps = 1000.0 / avg_steady_ms if avg_steady_ms > 0 else 0
        print(f"  Tokens generated:  {n_decoded}")
        print(f"  Latency (all):     avg = {avg_full_ms:.1f} ms/token")
        print(f"  Latency (steady, skip first {warmup}):")
        print(f"      avg = {avg_steady_ms:.1f} ms/token  ({tps:.1f} tok/s)")
        print(f"      median = {median_steady_ms:.1f} ms/token")

    if args.cpu_verify:
        print(f"  Top-1 NPU/CPU match: {matches}/{n_decoded}  (gate: ≥80%)")
        match_pct = matches / n_decoded if n_decoded > 0 else 0
        passed_match = match_pct >= 0.80
    else:
        passed_match = True

    print()
    print(f"  Pattern application:")
    print(
        f"    [INHERITED] 1. Multi-launch merging      (rms_gemv_rope=6, o_gemv_ffn=8)"
    )
    print(f"    [INHERITED] 2. Static weight BOs         (decode preload)")
    print(f"    [APPLIED  ] 3. NPU LM Head GEMV          (10×16384, vocab=151936)")
    print(
        f"    [APPLIED  ] 4. Extern kernel rename      (mv_k8960.o for Down GEMV K=8960)"
    )
    print(
        f"    [PARTIAL  ] 5. CPU->NPU op promotion     (attention stays CPU per llama3 design)"
    )

    passed = passed_match and (n_decoded > 0)
    print(f"\n  Phase 5: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

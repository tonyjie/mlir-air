# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 5 — decode performance for SmolLM2-1.7B on NPU2.

Decode pipeline (per-token):
    embed (CPU)
    -> 24x [rms_gemv_rope (NPU 6-launch ELF) -> CPU attention -> o_gemv_ffn (NPU 8-launch ELF)]
    -> final RMSNorm (CPU)
    -> LM Head GEMV (NPU, 8-partition, last 5 zero-padded for vocab=49152)
    -> argmax

Setup:
  CPU prefill on the prompt seeds the KV cache (per-layer K/V), then NPU
  decode generates tokens one at a time, validating each next-token against
  CPU reference next-token.

Pattern application (per `optimize-decode-perf`):
  1. Multi-launch merging:   INHERITED (rms_gemv_rope=6, o_gemv_ffn=8 launches)
  2. Static weight BOs:      INHERITED (decode preload sets static_indices)
  3. NPU LM Head GEMV:       APPLIED (zero-padded 8-partition; right-sizing
                             to 3 partitions left as Phase 6 lesson item)
  4. Extern kernel rename:   INHERITED (mv_k8192.o for Down GEMV K=8192)
  5. CPU->NPU op promotion:  PARTIAL (attention stays on CPU per llama3 design;
                             final RMSNorm stays on CPU — single-token cost
                             is negligible)
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

from llama3_prefill import KernelCache, prepare_air_project
from llama3_decode import compile_decode_kernels, run_decode_block
import llama3_inference  # for _preload_decode_weights, _LM_N_PARTITIONS, _LM_N_PART, _LM_GEMV_BACKEND
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels


def _seed_kv_cache_via_cpu_prefill(
    weights, config, prompt_token_ids, rope_lut_f32, max_seq
):
    """Run CPU reference prefill on the prompt; populate per-layer KV cache.

    Returns:
        k_cache, v_cache: arrays of shape (n_layers, n_kv_heads, max_seq, head_dim)
        last_hidden_normed_f32: (emb_dim,) F32 — final-RMSNorm of last prompt position
    """
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    emb_dim = config.emb_dim

    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table_f32[
        np.array(prompt_token_ids, dtype=np.int64)
    ]  # (prompt_len, emb_dim)
    prompt_len = len(prompt_token_ids)

    k_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    for i in range(n_layers):
        x, intermediates = smollm2_reference.transformer_block(
            x, weights.layers[i], rope_lut_f32, config
        )
        # Stash K (post-RoPE) and V (pre-attention) in cache layout
        k_per_pos = intermediates["k_roped"].reshape(prompt_len, n_kv_heads, head_dim)
        v_per_pos = intermediates["v"].reshape(prompt_len, n_kv_heads, head_dim)
        k_cache[i, :, :prompt_len, :] = k_per_pos.transpose(1, 0, 2).astype(bfloat16)
        v_cache[i, :, :prompt_len, :] = v_per_pos.transpose(1, 0, 2).astype(bfloat16)

    x_normed = smollm2_reference.rms_norm(x, weights.final_norm)
    return k_cache, v_cache, x_normed[-1]


def _pre_transpose_decode_weights(weights, config):
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


def _npu_lm_head_gemv(decode_cache, weights, config, x_normed_bf16):
    """Run NPU LM Head GEMV; return concatenated logits clipped to vocab_size."""
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


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-1.7B Phase 5 decode perf")
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number of decode tokens to generate (default: 10)",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--max-seq", type=int, default=128, help="Max KV-cache positions (default: 128)"
    )
    parser.add_argument("--cache-dir", type=str, default="decode_kernel_cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--cpu-verify", action="store_true", default=True)
    parser.add_argument("--no-cpu-verify", dest="cpu_verify", action="store_false")
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=2,
        help="Discard first N tokens for timing average (NPU warmup)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile decode kernels and exit (skip weight load + decode)",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    config = LlamaConfig()
    print(f"SmolLM2 config: {config}")

    # Compile kernels first so --compile-only can exit before the 9s weight load.
    if args.compile_only:
        prepare_air_project()
        cache_dir = _THIS_DIR / args.cache_dir
        decode_cache = KernelCache(cache_dir=str(cache_dir))
        if (cache_dir / "manifest.json").exists():
            decode_cache.load_manifest()
        needed = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
        if not all(k in decode_cache.artifacts for k in needed):
            print("\nCompiling decode kernels for SmolLM2 shapes...")
            compile_decode_kernels(decode_cache, config)
        else:
            print("\nDecode kernels already cached.")
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

    # CPU prefill to seed KV cache + initial generated token
    print("\nRunning CPU reference prefill (one-time, seeds KV cache)...")
    t = time.time()
    k_cache, v_cache, x_normed_first = _seed_kv_cache_via_cpu_prefill(
        weights, config, prompt_tokens, rope_lut_f32, args.max_seq
    )
    print(f"  CPU prefill: {time.time()-t:.1f}s")
    lm_head_f32 = np.asarray(weights.lm_head, dtype=np.float32)
    logits_first = x_normed_first @ lm_head_f32.T
    first_npu_token = int(np.argmax(logits_first))
    print(
        f"  First generated token (from CPU prefill): "
        f"{tokenizer.decode([first_npu_token])!r} (id={first_npu_token})"
    )

    # Set up decode kernel cache + compile
    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    decode_cache = KernelCache(cache_dir=str(cache_dir))
    if (cache_dir / "manifest.json").exists():
        decode_cache.load_manifest()
        print(
            f"\nLoaded existing decode cache: {sorted(decode_cache.artifacts.keys())}"
        )

    needed = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
    if not all(k in decode_cache.artifacts for k in needed):
        print("\nCompiling decode kernels for SmolLM2 shapes...")
        t = time.time()
        compile_decode_kernels(decode_cache, config)
        print(f"  Compile: {time.time()-t:.1f}s")
    else:
        print("\nDecode kernels already cached — skipping compile.")

    # Pre-transpose weights for decode
    print("\nPre-transposing decode weights...")
    _pre_transpose_decode_weights(weights, config)

    # Pre-load decode BOs (config-driven; LM Head zero-pads to 8 partitions)
    print("Pre-loading decode BOs (24 layers + LM Head GEMV)...")
    t = time.time()
    llama3_inference._preload_decode_weights(decode_cache, weights, config)
    print(f"  Preload: {time.time()-t:.1f}s")

    # Decode loop
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
        current_pos = len(generated) - 1  # position where the NEW K/V will be written
        if current_pos >= args.max_seq:
            print(f"  Hit max_seq={args.max_seq}, stopping")
            break

        # NPU decode for current_token (predicts next token)
        x_in_bf16 = embed_table_f32[current_token].astype(bfloat16)

        t0 = time.time()
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
        # Final RMSNorm (CPU — single vector, ~µs)
        x_f32 = np.asarray(x_bf16, dtype=np.float32).reshape(1, config.emb_dim)
        x_normed_bf16 = (
            smollm2_reference.rms_norm(x_f32, weights.final_norm)
            .flatten()
            .astype(bfloat16)
        )
        # NPU LM Head GEMV
        npu_logits = _npu_lm_head_gemv(decode_cache, weights, config, x_normed_bf16)
        next_npu = int(np.argmax(npu_logits))
        npu_t = time.time() - t0
        npu_token_times_full.append(npu_t)

        # CPU verification (slow — full prefill on growing sequence)
        match_str = "?"
        cpu_t = 0.0
        if args.cpu_verify:
            t0 = time.time()
            cpu_x = embed_table_f32[np.array(generated, dtype=np.int64)]
            for layer_idx in range(config.n_layers):
                cpu_x, _ = smollm2_reference.transformer_block(
                    cpu_x, weights.layers[layer_idx], rope_lut_f32, config
                )
            cpu_normed = smollm2_reference.rms_norm(cpu_x, weights.final_norm)
            cpu_logits = cpu_normed[-1] @ lm_head_f32.T
            next_cpu = int(np.argmax(cpu_logits))
            cpu_t = time.time() - t0
            match = next_npu == next_cpu
            matches += int(match)
            match_str = "✓" if match else f"✗ CPU={tokenizer.decode([next_cpu])!r}"
        n_decoded += 1

        npu_token_str = tokenizer.decode([next_npu])
        print(
            f"  Tok {token_idx+1:2d} pos={current_pos:3d}  "
            f"NPU={next_npu:5d} {npu_token_str!r:<14s}  "
            f"NPU_t={npu_t*1000:6.1f}ms  "
            f"CPU_t={cpu_t:5.1f}s  {match_str}"
        )

        generated.append(next_npu)
        current_token = next_npu

    # Summary
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
    print(
        f"    [INHERITED] 2. Static weight BOs         (decode preload sets static_indices)"
    )
    print(
        f"    [APPLIED  ] 3. NPU LM Head GEMV          (8-partition zero-padded for vocab=49152)"
    )
    print(f"    [INHERITED] 4. Extern kernel rename      (mv_k8192.o for Down GEMV)")
    print(
        f"    [PARTIAL  ] 5. CPU->NPU op promotion     (attention stays CPU per llama3 design)"
    )

    passed = passed_match and (n_decoded > 0)
    print(f"\n  Phase 5: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

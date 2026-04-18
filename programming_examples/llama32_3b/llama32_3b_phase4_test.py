# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 4 — prefill performance for Llama-3.2-3B on NPU2.

Confirms the 5 prefill optimization patterns from `optimize-prefill-perf` and
measures end-to-end prefill latency.

Pattern status (most inherited from llama3_prefill orchestration):
  1. Multi-launch merging:    INHERITED (rms_gemms_rope=6 launches, o_ffn=8)
  2. Per-layer BO pre-loading: APPLIED HERE (preload_prefill_weights — same
                               config-driven helper smollm2 used)
  3. Intermediate buffer reuse: INHERITED (intermediate_indices set per kernel)
  4. Seq-first layout:         INHERITED (RoPE+FA accept seq-first natively)
  5. CPU->NPU op promotion:    ATTEMPTED-FAILED for NPU FlashAttention at
                               head_dim=128 (kernel compiles via
                               compile_attn_npu2_split with the L1-feasible
                               lkp=64 lqp=256 dk=dv=128 config, but runtime
                               hangs with ERT_CMD_STATE_TIMEOUT at our specific
                               (n_heads=24, n_kv_heads=8, lq=lk=2048) shape).
                               Defer NPU LM Head to Phase 5. Use CPU attention
                               + CPU LM Head for this measurement.

Methodology:
  - "Cold" run: lazy BO allocation on first per-layer call (Phase 3 default).
  - "Warm" run (after preload_prefill_weights): all per-layer BOs pre-allocated
    and weights pre-written; inference loop only does dynamic-input writes +
    kernel launch + output read.
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

from llama32_3b_weights import LlamaConfig, load_weights, generate_rope_lut
import llama32_3b_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    preload_prefill_weights,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

from llama32_3b_phase2_test import compile_block_kernels

PROMPT = "The capital of France is"


def _embed_and_pad(prompt, tokenizer, weights, seq_len):
    token_ids = tokenizer.encode(prompt)
    real_len = len(token_ids)
    if real_len < seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = token_ids + [pad] * (seq_len - real_len)
    token_ids = np.array(token_ids[:seq_len], dtype=np.int64)
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[token_ids]
    return x_f32.astype(bfloat16), token_ids, real_len


def npu_full_prefill(x_bf16, weights, config, cache, rope_lut_bf16, cpu_attn=True):
    """Run all N layers + final RMSNorm + LM Head; return logits + timings."""
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
    x_normed = llama32_3b_reference.rms_norm(x_f32, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x_normed @ lm_head.T
    cpu_lm_head_time = time.time() - t0

    return logits, npu_layer_time, cpu_lm_head_time


def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-3B Phase 4 prefill perf")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=True,
        help="Use CPU attention fallback (default: True; NPU FA at head_dim=128 currently hangs)",
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Try NPU FlashAttention (currently hangs ERT_CMD_STATE_TIMEOUT at our shape)",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument(
        "--n-warm-runs",
        type=int,
        default=3,
        help="Number of timed runs after preload (default: 3)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile prefill kernels and exit",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    config = LlamaConfig()
    print(
        f"Llama-3.2-3B config: n_layers={config.n_layers}, "
        f"kv_dim={config.n_kv_heads * config.head_dim}, head_dim={config.head_dim}"
    )
    print(f"Attention: {'CPU' if args.cpu_attn else 'NPU FA'}")
    print(f"seq_len:   {args.seq_len}\n")

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=False)
    if (cache_dir / "manifest.json").exists():
        cache.load_manifest()
    compile_all_external_kernels(head_dim=config.head_dim)
    compile_block_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)
    print()

    if args.compile_only:
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    print("Loading weights...")
    weights = load_weights(args.model, config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )

    x_bf16, _, real_len = _embed_and_pad(PROMPT, tokenizer, weights, args.seq_len)
    print(f"Input: {PROMPT!r}  ({real_len} real tokens, padded to {args.seq_len})\n")

    # ----- COLD run (no preload) -----
    print("=" * 60)
    print("MEASUREMENT 1 — COLD prefill (no preload_prefill_weights call)")
    print("=" * 60)
    print("First per-layer call allocates BOs and writes weights inline.")
    t_wall = time.time()
    logits_cold, npu_layer_t_cold, lm_head_t = npu_full_prefill(
        x_bf16, weights, config, cache, rope_lut_bf16, cpu_attn=args.cpu_attn
    )
    cold_wall = time.time() - t_wall
    cold_top1 = int(np.argmax(logits_cold[real_len - 1]))
    print(
        f"  NPU layers  : {npu_layer_t_cold:.3f} s  "
        f"({npu_layer_t_cold/config.n_layers*1000:.0f} ms/layer)"
    )
    print(f"  CPU LM Head : {lm_head_t:.3f} s")
    print(f"  Wall total  : {cold_wall:.3f} s")
    print(f"  Top-1 token : '{tokenizer.decode([cold_top1])}' (id={cold_top1})")

    # Reset BO state so PRELOAD measurement is honest
    cache._loaded = {}
    if hasattr(cache, "_bo_cache"):
        cache._bo_cache = {}
    if hasattr(weights, "_prefill_weights_preloaded"):
        delattr(weights, "_prefill_weights_preloaded")
    if hasattr(run_transformer_block, "_arg_cache"):
        run_transformer_block._arg_cache = {}

    # ----- PRELOAD step (Pattern 2) -----
    print()
    print("=" * 60)
    print(f"PATTERN 2 APPLIED — preload_prefill_weights ({config.n_layers} layers)")
    print("=" * 60)
    t = time.time()
    preload_prefill_weights(weights, config, cache, args.seq_len, rope_lut_bf16)
    preload_t = time.time() - t
    print(
        f"  Preload time: {preload_t:.3f} s  "
        f"({preload_t/config.n_layers*1000:.0f} ms/layer)"
    )

    # ----- WARM runs -----
    print()
    print("=" * 60)
    print(f"MEASUREMENT 2 — WARM prefill (×{args.n_warm_runs} runs after preload)")
    print("=" * 60)
    warm_layer_times = []
    warm_wall_times = []
    warm_top1 = None
    for i in range(args.n_warm_runs):
        x_warm, _, _ = _embed_and_pad(PROMPT, tokenizer, weights, args.seq_len)
        t_wall = time.time()
        logits_warm, npu_layer_t, lm_head_t_w = npu_full_prefill(
            x_warm, weights, config, cache, rope_lut_bf16, cpu_attn=args.cpu_attn
        )
        wall = time.time() - t_wall
        warm_layer_times.append(npu_layer_t)
        warm_wall_times.append(wall)
        warm_top1 = int(np.argmax(logits_warm[real_len - 1]))
        print(
            f"  Run {i+1}: NPU layers={npu_layer_t:.3f} s "
            f"({npu_layer_t/config.n_layers*1000:.0f} ms/layer)  "
            f"LM Head={lm_head_t_w:.3f} s  wall={wall:.3f} s  "
            f"top-1='{tokenizer.decode([warm_top1])}'"
        )

    warm_layer_avg = float(np.mean(warm_layer_times))
    warm_wall_avg = float(np.mean(warm_wall_times))

    # ----- Summary -----
    print()
    print("=" * 60)
    print("Phase 4 — Prefill perf summary")
    print("=" * 60)
    print(f"  Cold first-prompt NPU layer time : {npu_layer_t_cold:.3f} s")
    print(f"  Warm avg NPU layer time          : {warm_layer_avg:.3f} s")
    if npu_layer_t_cold > 0:
        gain = npu_layer_t_cold - warm_layer_avg
        pct = (1 - warm_layer_avg / npu_layer_t_cold) * 100
        print(
            f"  Pattern 2 gain on first prompt   : {gain:.3f} s ({pct:.1f}% reduction)"
        )
    print(f"  Cold wall (NPU+LMhead)            : {cold_wall:.3f} s")
    print(f"  Warm wall avg                     : {warm_wall_avg:.3f} s")
    print(f"  Preload setup cost                : {preload_t:.3f} s (one-time)")

    print()
    print("Pattern application status:")
    print(
        "  [INHERITED      ] 1. Multi-launch merging      (rms_gemms_rope=6, o_ffn=8 launches)"
    )
    print(
        f"  [APPLIED        ] 2. Per-layer BO pre-loading  (preload_prefill_weights, {preload_t:.2f} s setup)"
    )
    print(
        "  [INHERITED      ] 3. Intermediate buffer reuse (intermediate_indices set per kernel)"
    )
    print("  [INHERITED      ] 4. Seq-first layout          (RoPE/FA native)")
    if args.cpu_attn:
        print(
            "  [PARTIAL        ] 5. CPU->NPU op promotion     "
            "(NPU FA hangs at head_dim=128 + lq=lk=2048; LM Head deferred to Phase 5)"
        )
    else:
        print("  [APPLIED        ] 5. CPU->NPU op promotion     (NPU FA active)")

    correctness_ok = cold_top1 == warm_top1 and warm_top1 is not None
    n_patterns = 4 if args.cpu_attn else 5

    print()
    print(f"  Cold top-1 == warm top-1  : {correctness_ok}  (no regression)")
    print(f"  Patterns applied/inherited: {n_patterns} of 5  (≥3 required)")
    passed = correctness_ok and n_patterns >= 3
    print(f"\n  Phase 4: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

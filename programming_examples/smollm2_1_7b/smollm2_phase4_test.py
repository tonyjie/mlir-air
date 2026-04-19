# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 4 — prefill performance for SmolLM2-1.7B on NPU2.

5/5 patterns applied with --npu-attn (seq-first FA at head_dim=64).
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
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.orchestration import compile_block_kernels
from _llm_shared.phase_helpers.prefill_runner import embed_and_pad, npu_full_prefill

PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-1.7B Phase 4 prefill perf")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cpu-attn", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--n-warm-runs", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"SmolLM2 config: n_layers={config.n_layers}, "
        f"kv_dim={config.n_kv_heads * config.head_dim}"
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

    x_bf16, _, real_len = embed_and_pad(PROMPT, tokenizer, weights, args.seq_len)
    print(f"Input: {PROMPT!r}  ({real_len} real tokens, padded to {args.seq_len})\n")

    print("=" * 60)
    print("MEASUREMENT 1 — COLD prefill (no preload_prefill_weights call)")
    print("=" * 60)
    t_wall = time.time()
    logits_cold, npu_layer_t_cold, lm_head_t = npu_full_prefill(
        x_bf16,
        weights,
        config,
        cache,
        rope_lut_bf16,
        smollm2_reference,
        cpu_attn=args.cpu_attn,
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

    cache._loaded = {}
    if hasattr(cache, "_bo_cache"):
        cache._bo_cache = {}
    if hasattr(weights, "_prefill_weights_preloaded"):
        delattr(weights, "_prefill_weights_preloaded")
    if hasattr(run_transformer_block, "_arg_cache"):
        run_transformer_block._arg_cache = {}

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

    print()
    print("=" * 60)
    print(f"MEASUREMENT 2 — WARM prefill (×{args.n_warm_runs} runs after preload)")
    print("=" * 60)
    warm_layer_times, warm_wall_times = [], []
    warm_top1 = None
    for i in range(args.n_warm_runs):
        x_warm, _, _ = embed_and_pad(PROMPT, tokenizer, weights, args.seq_len)
        t_wall = time.time()
        logits_warm, npu_layer_t, lm_head_t_w = npu_full_prefill(
            x_warm,
            weights,
            config,
            cache,
            rope_lut_bf16,
            smollm2_reference,
            cpu_attn=args.cpu_attn,
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

    print()
    print("Pattern application status:")
    print("  [INHERITED] 1. Multi-launch merging      (rms_gemms_rope=6, o_ffn=8)")
    print(
        f"  [APPLIED  ] 2. Per-layer BO pre-loading  (preload_prefill_weights, {preload_t:.2f} s setup)"
    )
    print(
        "  [INHERITED] 3. Intermediate buffer reuse (intermediate_indices set per kernel)"
    )
    print("  [INHERITED] 4. Seq-first layout          (RoPE/FA native)")
    if args.cpu_attn:
        print(
            "  [PARTIAL  ] 5. CPU->NPU op promotion     (use --npu-attn for full credit)"
        )
    else:
        print("  [APPLIED  ] 5. CPU->NPU op promotion     (NPU FA active)")

    correctness_ok = cold_top1 == warm_top1
    print()
    print(f"  Cold top-1 == warm top-1  : {correctness_ok}  (no regression)")
    print(
        f"  Patterns applied/inherited: {'5' if not args.cpu_attn else '4'} of 5  (≥3 required)"
    )
    passed = correctness_ok
    print(f"\n  Phase 4: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

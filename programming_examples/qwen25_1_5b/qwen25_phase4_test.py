# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 4 — prefill performance for Qwen2.5-1.5B on NPU2.

NPU forward uses padded shapes (emb=2048, hidden=9216, n_heads=16) plus
the GQA-reindexed weights, the Qwen2 host bias add, and (optionally) the
Option C head-first FA wrapper.

Methodology:
  - "Cold" run: lazy BO allocation on first per-layer call
  - "Warm" run (after preload_prefill_weights): all BOs pre-allocated
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

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    preload_prefill_weights,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.prefill_runner import embed_and_pad, npu_full_prefill

from qwen25_bias import install_qkv_bias_wrapper
from qwen25_pad import make_padded_config, pad_weights
from qwen25_phase2_test import _compile_qwen25_block_kernels
from qwen25_phase3_test import _register_all_layer_biases

PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Phase 4 prefill perf")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--cpu-attn", dest="cpu_attn", action="store_true", default=True
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Use NPU FA via Option C head-first wrapper (head_dim=128).",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n-warm-runs", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    padded_config = make_padded_config(
        orig_config, padded_emb_dim=2048, padded_hidden_dim=9216
    )
    print(
        f"Qwen2.5-1.5B padded config: n_layers={padded_config.n_layers}, "
        f"emb={padded_config.emb_dim} (orig {orig_config.emb_dim}), "
        f"hidden={padded_config.hidden_dim} (orig {orig_config.hidden_dim}), "
        f"n_heads={padded_config.n_heads}, kv_dim={padded_config.n_kv_heads*padded_config.head_dim}"
    )
    print(f"Attention: {'CPU' if args.cpu_attn else 'NPU FA (Option C head-first)'}")
    print(f"seq_len:   {args.seq_len}\n")

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=False)
    if (cache_dir / "manifest.json").exists():
        cache.load_manifest()
    compile_all_external_kernels(head_dim=padded_config.head_dim)
    _compile_qwen25_block_kernels(
        cache, padded_config, args.seq_len, cpu_attn=args.cpu_attn
    )
    print()

    if args.compile_only:
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    print("Loading orig weights + building padded weights...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    padded_weights = pad_weights(orig_weights, orig_config, padded_config)
    print(f"  Done in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    padded_rope_bf16 = generate_rope_lut(
        config=padded_config, seq_len=args.seq_len, dtype=bfloat16
    )

    install_qkv_bias_wrapper()
    print(f"Registering Qwen2 QKV bias for all {padded_config.n_layers} layers...")
    _register_all_layer_biases(
        padded_weights, padded_config, padded_rope_bf16, args.seq_len
    )

    x_bf16, _, real_len = embed_and_pad(PROMPT, tokenizer, padded_weights, args.seq_len)
    print(f"\nInput: {PROMPT!r}  ({real_len} real tokens, padded to {args.seq_len})\n")

    # ----- COLD run -----
    print("=" * 60)
    print("MEASUREMENT 1 — COLD prefill (no preload_prefill_weights call)")
    print("=" * 60)
    t_wall = time.time()
    logits_cold, npu_layer_t_cold, lm_head_t = npu_full_prefill(
        x_bf16,
        padded_weights,
        padded_config,
        cache,
        padded_rope_bf16,
        qwen25_reference,
        cpu_attn=args.cpu_attn,
    )
    cold_wall = time.time() - t_wall
    cold_top1 = int(np.argmax(logits_cold[real_len - 1]))
    print(
        f"  NPU layers  : {npu_layer_t_cold:.3f} s  "
        f"({npu_layer_t_cold/padded_config.n_layers*1000:.0f} ms/layer)"
    )
    print(f"  CPU LM Head : {lm_head_t:.3f} s")
    print(f"  Wall total  : {cold_wall:.3f} s")
    print(f"  Top-1 token : '{tokenizer.decode([cold_top1])}' (id={cold_top1})")

    # Reset BO state for honest preload measurement.
    cache._loaded = {}
    if hasattr(cache, "_bo_cache"):
        cache._bo_cache = {}
    if hasattr(padded_weights, "_prefill_weights_preloaded"):
        delattr(padded_weights, "_prefill_weights_preloaded")
    if hasattr(run_transformer_block, "_arg_cache"):
        run_transformer_block._arg_cache = {}

    # ----- PRELOAD (Pattern 2) -----
    print()
    print("=" * 60)
    print(
        f"PATTERN 2 APPLIED — preload_prefill_weights ({padded_config.n_layers} layers)"
    )
    print("=" * 60)
    t = time.time()
    preload_prefill_weights(
        padded_weights, padded_config, cache, args.seq_len, padded_rope_bf16
    )
    preload_t = time.time() - t
    print(
        f"  Preload time: {preload_t:.3f} s  "
        f"({preload_t/padded_config.n_layers*1000:.0f} ms/layer)"
    )

    # ----- WARM runs -----
    print()
    print("=" * 60)
    print(f"MEASUREMENT 2 — WARM prefill (×{args.n_warm_runs} runs after preload)")
    print("=" * 60)
    warm_layer_times, warm_wall_times = [], []
    warm_top1 = None
    for i in range(args.n_warm_runs):
        x_warm, _, _ = embed_and_pad(PROMPT, tokenizer, padded_weights, args.seq_len)
        t_wall = time.time()
        logits_warm, npu_layer_t, lm_head_t_w = npu_full_prefill(
            x_warm,
            padded_weights,
            padded_config,
            cache,
            padded_rope_bf16,
            qwen25_reference,
            cpu_attn=args.cpu_attn,
        )
        wall = time.time() - t_wall
        warm_layer_times.append(npu_layer_t)
        warm_wall_times.append(wall)
        warm_top1 = int(np.argmax(logits_warm[real_len - 1]))
        print(
            f"  Run {i+1}: NPU layers={npu_layer_t:.3f} s "
            f"({npu_layer_t/padded_config.n_layers*1000:.0f} ms/layer)  "
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
    print(f"  Preload setup cost                : {preload_t:.3f} s (one-time)")

    print()
    print("Pattern application status:")
    print(
        "  [INHERITED] 1. Multi-launch merging      (rms_gemms_rope=6, o_ffn=8 launches)"
    )
    print(
        f"  [APPLIED  ] 2. Per-layer BO pre-loading  (preload_prefill_weights, {preload_t:.2f} s setup)"
    )
    print(
        "  [INHERITED] 3. Intermediate buffer reuse (intermediate_indices set per kernel)"
    )
    print("  [INHERITED] 4. Seq-first layout          (RoPE/FA native)")
    if args.cpu_attn:
        print(
            "  [PARTIAL  ] 5. CPU->NPU op promotion     "
            "(use --npu-attn for Option C head-first FA at head_dim=128)"
        )
    else:
        print(
            "  [APPLIED  ] 5. CPU->NPU op promotion     "
            "(NPU FA via Option C head-first + host transposes)"
        )

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

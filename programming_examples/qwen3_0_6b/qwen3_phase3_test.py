# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 3 — full 28-layer correctness for Qwen3-0.6B on NPU2.

Wires all 28 transformer blocks via the split-ELF helper from Phase 2,
applies final RMSNorm + LM head on host (NPU LM head GEMV is a Phase 5
optimization), then validates next-token prediction against CPU reference
on the canonical prompt set.

Phase 3 GATE (per LESSON 2):
  - DECISIVE prompts (CPU top-1 prob > 0.5): NPU top-1 == CPU top-1 EXACT
  - COMPETITIVE prompts (CPU top-1 prob ≤ 0.5): top-5 overlap (CPU top-1
    ∈ NPU top-5 AND NPU top-1 ∈ CPU top-5)
  - No NaN anywhere

Per-layer cosine drift is informational only at n_layers=28, head_dim=128
(BF16 accumulation drives last-layer cos to ~0.88; the strongest signal
is the top-1/top-5 gate above).
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

from qwen3_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen3_reference

import llama3_prefill as _lp
from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.canonical_prompts import (
    DECISIVE_PROMPTS,
    COMPETITIVE_PROMPTS,
)
from _llm_shared.phase_helpers.headfirst_fa import (
    install_headfirst_fa_wrapper,
    compile_headfirst_fa_kernel,
)
from qwen3_phase2_test import _compile_qwen3_block_kernels, run_qwen3_block_npu


def _rms_norm_host(x_bf16, weight_bf16, eps):
    x = np.asarray(x_bf16, dtype=np.float32)
    w = np.asarray(weight_bf16, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return ((x / rms) * w).astype(bfloat16)


def npu_full_prefill(token_ids, weights, config, rope_lut_bf16, cache, verbose=False):
    """Run all 28 layers on NPU then final RMSNorm + LM head on host.

    Returns logits (seq_len, vocab) F32.
    """
    seq_len = token_ids.shape[0]
    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table[token_ids].astype(bfloat16)

    for i in range(config.n_layers):
        if verbose:
            print(f"  Layer {i}/{config.n_layers}...", flush=True)
        x = run_qwen3_block_npu(
            x,
            weights.layers[i],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=i,
        )

    # Final RMSNorm on host
    x_normed = _rms_norm_host(x, weights.final_norm, config.rms_norm_eps)

    # LM head on host (NPU GEMV is a Phase 5 perf optimization)
    lm_head_f32 = np.asarray(weights.lm_head, dtype=np.float32)
    logits = np.asarray(x_normed, dtype=np.float32) @ lm_head_f32.T
    return logits


def evaluate_prompt(
    prompt, tokenizer, weights, config, rope_lut_bf16, cache, args, kind="decisive"
):
    """Run NPU + CPU on prompt, compute top-1/top-5 metrics."""
    token_ids = tokenizer.encode(prompt)
    real_len = len(token_ids)
    if real_len > args.seq_len:
        print(f"  WARN: '{prompt}' has {real_len} tokens > seq_len={args.seq_len}")
        token_ids = token_ids[: args.seq_len]
        real_len = args.seq_len
    pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    padded_ids = np.array(token_ids + [pad] * (args.seq_len - real_len), dtype=np.int64)
    pred_pos = real_len - 1

    # NPU
    t = time.time()
    npu_logits = npu_full_prefill(
        padded_ids, weights, config, rope_lut_bf16, cache, verbose=args.verbose
    )
    npu_time = time.time() - t
    npu_next = npu_logits[pred_pos]
    npu_top5 = np.argsort(npu_next)[-5:][::-1].tolist()
    npu_top1 = npu_top5[0]
    npu_has_nan = bool(np.any(np.isnan(npu_logits)))

    # CPU reference
    t = time.time()
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[padded_ids]
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)
    x_cpu = x_f32.copy()
    for i in range(config.n_layers):
        x_cpu, _ = qwen3_reference.transformer_block(
            x_cpu, weights.layers[i], rope_lut_f32, config
        )
    norm_w_f32 = np.asarray(weights.final_norm, dtype=np.float32)
    rms = np.sqrt(np.mean(x_cpu * x_cpu, axis=-1, keepdims=True) + config.rms_norm_eps)
    x_cpu_normed = (x_cpu / rms) * norm_w_f32
    cpu_logits = x_cpu_normed @ np.asarray(weights.lm_head, dtype=np.float32).T
    cpu_time = time.time() - t
    cpu_next = cpu_logits[pred_pos]
    cpu_top5 = np.argsort(cpu_next)[-5:][::-1].tolist()
    cpu_top1 = cpu_top5[0]
    cpu_softmax = np.exp(cpu_next - np.max(cpu_next))
    cpu_softmax /= cpu_softmax.sum()
    cpu_top1_prob = float(cpu_softmax[cpu_top1])

    top1_match = npu_top1 == cpu_top1
    top5_overlap = (cpu_top1 in npu_top5) and (npu_top1 in cpu_top5)

    # Dynamically classify: if CPU top-1 is decisive (prob > 0.5) require EXACT
    # match, else accept top-5 overlap. Override the static "kind" label since
    # the canonical prompt set was tuned for llama3 and Qwen3 may have
    # different per-prompt prob distributions.
    effective_kind = "decisive" if cpu_top1_prob > 0.5 else "competitive"
    if effective_kind == "decisive":
        passed = top1_match and not npu_has_nan
    else:
        passed = top5_overlap and not npu_has_nan
    kind = f"{effective_kind}({kind[0]})"  # e.g. "decisive(d)" or "competitive(d)"

    print(
        f"  [{kind:11s}] '{prompt[:40]:40s}'  "
        f"NPU top-1 '{tokenizer.decode([npu_top1])}'  "
        f"CPU top-1 '{tokenizer.decode([cpu_top1])}' (p={cpu_top1_prob:.3f})  "
        f"{'PASS' if passed else 'FAIL'}  "
        f"NPU={npu_time:.1f}s CPU={cpu_time:.1f}s  NaN={npu_has_nan}"
    )
    if not passed and args.verbose:
        print(f"    NPU top-5: {[tokenizer.decode([t]) for t in npu_top5]}")
        print(f"    CPU top-5: {[tokenizer.decode([t]) for t in cpu_top5]}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Phase 3 full 28-layer correctness test"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Phase 3 default = 512 (matches Phase 2 cached kernels). FA at hd=128 needs lqp=256.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument(
        "--decisive-only",
        action="store_true",
        help="Skip competitive prompts (faster smoke test)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"Qwen3-0.6B Phase 3 — full {config.n_layers}-layer NPU forward "
        f"vs CPU reference (seq_len={args.seq_len})"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=args.verbose)
    if (cache_dir / "manifest.json").exists():
        try:
            cache.load_manifest()
            print(f"  Loaded existing kernel cache: {sorted(cache.artifacts.keys())}")
        except Exception as e:
            print(
                f"  Could not load manifest ({type(e).__name__}: {e}); will recompile"
            )

    print(
        "\nCompiling external kernels (rope_halfsplit, silu_and_mul, attn_npu2, mv)..."
    )
    compile_all_external_kernels(head_dim=config.head_dim)

    print("\nCompiling/loading Qwen3 block kernels...")
    t = time.time()
    _compile_qwen3_block_kernels(cache, config, args.seq_len)
    print(f"  Block kernels ready: {time.time()-t:.1f}s")

    prompts_to_run = list(DECISIVE_PROMPTS)
    if not args.decisive_only:
        prompts_to_run += list(COMPETITIVE_PROMPTS)

    print(f"\nRunning {len(prompts_to_run)} prompts...")
    results = []
    for prompt in prompts_to_run:
        kind = "decisive" if prompt in DECISIVE_PROMPTS else "competitive"
        passed = evaluate_prompt(
            prompt, tokenizer, weights, config, rope_lut_bf16, cache, args, kind
        )
        results.append((prompt, kind, passed))

    n_pass = sum(1 for _, _, p in results if p)
    print(f"\n{'='*60}")
    print(f"Phase 3: {n_pass}/{len(results)} prompts PASS")
    print(f"{'='*60}")
    for prompt, kind, passed in results:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{kind:11s}] {marker}  '{prompt}'")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

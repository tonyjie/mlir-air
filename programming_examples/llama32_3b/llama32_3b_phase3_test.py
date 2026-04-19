# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 3 — full-model correctness for Llama-3.2-3B on NPU2.

All N layers + final RMSNorm + (CPU) LM Head, with the LESSON 2 gate
(decisive prompts top-1 match + competitive prompts top-5 overlap).

Phase 3 gate:
    Decisive prompts (CPU top-1 p > 0.5): NPU top-1 == CPU top-1
    Competitive prompts (CPU top-1 p ≤ 0.5): top-5 overlap
    No NaN
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

from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.canonical_prompts import CANONICAL_PROMPTS
from _llm_shared.phase_helpers.orchestration import (
    compile_block_kernels,
    evaluate_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-3B Phase 3 full-model test")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=True,
        help="Use CPU attention fallback (default: True)",
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Use NPU FlashAttention (Option C head-first wrapper at head_dim=128)",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Capture per-layer outputs for cosine-sim drift analysis (uses 2x DRAM)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prompts", nargs="+", default=None, help="Override the canonical prompt set"
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    config = LlamaConfig()
    print(
        f"Llama-3.2-3B config: n_layers={config.n_layers}, emb_dim={config.emb_dim}, "
        f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads} "
        f"(GQA group={config.n_heads // config.n_kv_heads}), "
        f"head_dim={config.head_dim}, hidden_dim={config.hidden_dim}, "
        f"vocab_size={config.vocab_size}, rope_base={config.rope_base}"
    )
    print(
        f"Attention path: {'CPU fallback' if args.cpu_attn else 'NPU FlashAttention'}"
    )
    print(f"Diagnostic per-layer mode: {args.diagnostic}")

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=args.verbose)
    if (cache_dir / "manifest.json").exists():
        cache.load_manifest()
        print(f"\nLoaded existing kernel cache: {sorted(cache.artifacts.keys())}")
    compile_all_external_kernels(head_dim=config.head_dim)
    compile_block_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)

    prompts = args.prompts if args.prompts is not None else CANONICAL_PROMPTS

    print(f"\n{'='*60}")
    print(f"Phase 3 — Full-model correctness on {len(prompts)} prompt(s)")
    print(f"{'='*60}")

    results = []
    for prompt in prompts:
        results.append(
            evaluate_prompt(
                prompt,
                tokenizer,
                weights,
                config,
                cache,
                rope_lut_bf16,
                rope_lut_f32,
                args.seq_len,
                args.cpu_attn,
                args.diagnostic,
                args.verbose,
                llama32_3b_reference,
            )
        )

    print(f"\n{'='*60}")
    print(f"Phase 3 — Summary")
    print(f"{'='*60}")
    n_match = sum(r["top1_match"] for r in results)
    any_nan = any(r["has_nan_npu"] for r in results)
    min_layer_cos = None
    if args.diagnostic:
        all_layer_cos = [
            cos for r in results for (_, cos, _) in (r["per_layer_cos"] or [])
        ]
        if all_layer_cos:
            min_layer_cos = min(all_layer_cos)

    decisive_results = [r for r in results if r["decisive"]]
    competitive_results = [r for r in results if not r["decisive"]]
    decisive_match = sum(r["top1_match"] for r in decisive_results)
    competitive_overlap = sum(r["top5_overlap"] for r in competitive_results)

    for r in results:
        if r["decisive"]:
            marker = "PASS" if r["top1_match"] else "FAIL"
            cls = "decisive  "
        else:
            marker = "PASS" if r["top5_overlap"] else "FAIL"
            cls = "competitive"
        print(
            f"  [{marker}] [{cls}] {r['prompt']!r:<32s}  "
            f"NPU={r['npu_token']!r:<10s} CPU={r['cpu_token']!r:<10s} "
            f"cpu_p={r['cpu_top1_p']:.3f}  corr={r['logits_corr']:.4f}"
        )

    print(
        f"\n  Strict top-1 match (all prompts): {n_match}/{len(prompts)}  "
        f"(skill default gate: {len(prompts)}/{len(prompts)})"
    )
    print(
        f"  Decisive prompts (CPU p>0.5) top-1 match: "
        f"{decisive_match}/{len(decisive_results)}  (gate: all)"
    )
    print(
        f"  Competitive prompts top-5 overlap: "
        f"{competitive_overlap}/{len(competitive_results)}  (gate: all)"
    )
    if min_layer_cos is not None:
        print(f"  Min per-layer cosine_sim: {min_layer_cos:.6f}  (informational)")
    print(f"  Any NaN in NPU: {any_nan}  (gate: False)")

    passed = (
        decisive_match == len(decisive_results)
        and competitive_overlap == len(competitive_results)
        and not any_nan
    )
    print(f"\n  Phase 3: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

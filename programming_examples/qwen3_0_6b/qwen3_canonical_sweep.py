# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run all 6 canonical prompts through the full NPU pipeline (Phase B fused
prefill + fused decode) and verify per-prompt that:

  - the first decoded token (NPU) is in the CPU reference's top-5
  - if the prompt is "decisive" (CPU top-1 prob > 0.5) require an exact match
  - if "competitive" (CPU top-1 prob ≤ 0.5) only require top-5 overlap

This is the production-grade Phase 3 gate, re-run with the new NPU decode.
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
from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.canonical_prompts import (
    DECISIVE_PROMPTS,
    COMPETITIVE_PROMPTS,
)
import qwen3_decode
import qwen3_reference
from qwen3_phase4_test import npu_full_prefill
from qwen3_phase2_test import _compile_qwen3_block_kernels


def _cpu_first_token_logits(seed_token_ids, weights, config):
    """Run full CPU forward at the seed length; return logits at the last
    real position (next-token prediction)."""
    seq_len = len(seed_token_ids)
    rope_lut_f32 = np.asarray(
        generate_rope_lut(config=config, seq_len=seq_len, dtype=bfloat16),
        dtype=np.float32,
    )
    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table[np.array(seed_token_ids, dtype=np.int64)]
    for li in range(config.n_layers):
        x, _ = qwen3_reference.transformer_block(
            x, weights.layers[li], rope_lut_f32, config
        )
    norm_w = np.asarray(weights.final_norm, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config.rms_norm_eps)
    x = (x / rms) * norm_w
    lm = np.asarray(weights.lm_head, dtype=np.float32)
    return (x @ lm.T)[-1]  # (vocab,) at last real position


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 NPU canonical-prompt sweep (full NPU pipeline)"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    # Cache-dir naming MUST match the seq_len the cache was built at (LESSON L1).
    parser.add_argument("--prefill-cache", default="prefill_kernel_cache_2048")
    parser.add_argument("--decode-cache", default="decode_kernel_cache")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(f"Loading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  {time.time()-t:.1f}s")

    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )

    prepare_air_project()
    prefill_cache = KernelCache(
        cache_dir=str(_THIS_DIR / args.prefill_cache), verbose=args.verbose
    )
    if (_THIS_DIR / args.prefill_cache / "manifest.json").exists():
        prefill_cache.load_manifest()

    print("\nCompiling external kernels...")
    compile_all_external_kernels(head_dim=config.head_dim)

    print(f"\nCompiling/loading prefill kernels (seq_len={args.seq_len})...")
    _compile_qwen3_block_kernels(prefill_cache, config, args.seq_len)

    decode_cache = KernelCache(
        cache_dir=str(_THIS_DIR / args.decode_cache), verbose=args.verbose
    )
    if (_THIS_DIR / args.decode_cache / "manifest.json").exists():
        decode_cache.load_manifest()
    qwen3_decode.compile_decode_kernels(decode_cache, config)
    qwen3_decode.preload_decode_weights(decode_cache, weights, config)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    prompts = list(DECISIVE_PROMPTS) + list(COMPETITIVE_PROMPTS)
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0

    print(f"\n{'='*80}")
    print(f"Canonical-prompt sweep: NPU prefill + NPU LM head, vs CPU reference")
    print(f"{'='*80}")

    n_pass_total = 0
    for prompt in prompts:
        seed = tok.encode(prompt)
        real_len = len(seed)
        if real_len > args.seq_len:
            print(f"  WARN: skipping over-long prompt: {prompt!r}")
            continue
        padded = seed + [pad] * (args.seq_len - real_len)
        padded_ids = np.array(padded[: args.seq_len], dtype=np.int64)

        # NPU prefill + NPU LM head for first decoded token
        npu_hidden = npu_full_prefill(
            padded_ids,
            weights,
            config,
            rope_lut_bf16,
            prefill_cache,
        )
        norm_w = np.asarray(weights.final_norm, dtype=np.float32)
        x_in = np.asarray(npu_hidden[real_len - 1], dtype=np.float32)
        rms = np.sqrt(np.mean(x_in * x_in) + config.rms_norm_eps)
        x_normed_bf16 = ((x_in / rms) * norm_w).astype(bfloat16)
        npu_logits = qwen3_decode.npu_lm_head(
            decode_cache, x_normed_bf16, weights, config
        )
        npu_top1 = int(np.argmax(npu_logits))

        # CPU reference
        cpu_logits = _cpu_first_token_logits(seed, weights, config)
        cpu_softmax = np.exp(cpu_logits - cpu_logits.max())
        cpu_softmax /= cpu_softmax.sum()
        cpu_top1 = int(np.argmax(cpu_logits))
        cpu_top1_prob = float(cpu_softmax[cpu_top1])
        cpu_top5 = np.argsort(cpu_logits)[-5:][::-1].tolist()

        # Dynamic decisive/competitive classification (LESSON 3)
        decisive = cpu_top1_prob > 0.5
        if decisive:
            passed = npu_top1 == cpu_top1
            verdict = "PASS" if passed else "FAIL"
            kind = "decisive (top-1 strict)"
        else:
            passed = (cpu_top1 in np.argsort(npu_logits)[-5:].tolist()) and (
                npu_top1 in cpu_top5
            )
            verdict = "PASS" if passed else "FAIL"
            kind = "competitive (top-5 overlap)"

        if passed:
            n_pass_total += 1

        print(
            f"\n  prompt: {prompt!r}"
            f"\n    CPU top-1: '{tok.decode([cpu_top1])}' (id={cpu_top1}, p={cpu_top1_prob:.3f}) — {kind}"
            f"\n    NPU top-1: '{tok.decode([npu_top1])}' (id={npu_top1})"
            f"\n    {verdict}"
        )

    print(f"\n{'='*80}")
    print(f"Canonical sweep result: {n_pass_total}/{len(prompts)} PASS")
    print(f"{'='*80}")
    return 0 if n_pass_total == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())

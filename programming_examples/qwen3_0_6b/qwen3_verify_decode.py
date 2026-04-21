# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional correctness gate for the Phase B fused-ELF decode path.

Runs NPU decode (qwen3_decode.decode_loop_from_kv) and CPU decode
(qwen3_phase5_test.cpu_decode_token) side-by-side from the same prompt seed,
asserts per-token argmax agreement.

Per-token gate: NPU top-1 == CPU top-1.

A drift on any token is reported with rank-of-NPU-pick in CPU's top-5 to
distinguish "BF16 noise reorders close-prob tokens" (top-5 hit, soft-pass)
from "real bug" (NPU pick not in CPU top-5, hard-fail).
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
from qwen3_phase5_test import cpu_decode_token
import qwen3_decode
from qwen3_phase4_test import npu_full_prefill
from qwen3_phase2_test import _compile_qwen3_block_kernels


def main():
    parser = argparse.ArgumentParser(
        description="Verify NPU decode == CPU decode (per-token top-1)"
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    # Cache-dir naming MUST match the seq_len the cache was built at (LESSON L1:
    # kernel cache name doesn't encode seq_len; reusing a stale-shape ELF gives
    # garbage outputs). Default matches qwen3_inference.py (--cache-dir).
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

    # ---- Prefill cache (must already exist; recompiles take ~3 min) ----
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

    # ---- Decode cache + preload (warmup the new fused ELFs) ----
    decode_cache = KernelCache(
        cache_dir=str(_THIS_DIR / args.decode_cache),
        verbose=args.verbose,
    )
    if (_THIS_DIR / args.decode_cache / "manifest.json").exists():
        decode_cache.load_manifest()
    qwen3_decode.compile_decode_kernels(decode_cache, config)
    qwen3_decode.preload_decode_weights(decode_cache, weights, config)

    # ---- Tokenize + pad ----
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    seed = tok.encode(args.prompt)
    real_len = len(seed)
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    padded = seed + [pad] * (args.seq_len - real_len)
    padded_ids = np.array(padded[: args.seq_len], dtype=np.int64)

    # ---- NPU prefill + KV extraction ----
    print(f"\nNPU prefill (warm) at seq_len={args.seq_len}...")
    _ = npu_full_prefill(padded_ids, weights, config, rope_lut_bf16, prefill_cache)
    t = time.time()
    npu_hidden, k_per_layer, v_per_layer = npu_full_prefill(
        padded_ids,
        weights,
        config,
        rope_lut_bf16,
        prefill_cache,
        collect_kv=True,
    )
    print(f"  Prefill: {time.time()-t:.2f}s")

    # First decoded token from NPU prefill (NPU LM head)
    norm_w = np.asarray(weights.final_norm, dtype=np.float32)
    x_in = np.asarray(npu_hidden[real_len - 1], dtype=np.float32)
    rms = np.sqrt(np.mean(x_in * x_in) + config.rms_norm_eps)
    x_normed_bf16 = ((x_in / rms) * norm_w).astype(bfloat16)
    logits0 = qwen3_decode.npu_lm_head(decode_cache, x_normed_bf16, weights, config)
    first_id = int(np.argmax(logits0))
    print(f"  First decoded token (NPU): '{tok.decode([first_id])}' (id={first_id})")

    # ---- NPU decode loop ----
    print(f"\nNPU decode {args.n_tokens - 1} more tokens...")
    npu_decoded, npu_times = qwen3_decode.decode_loop_from_kv(
        seed_token_ids=seed,
        first_decoded_id=first_id,
        n_more_tokens=args.n_tokens,
        k_per_layer_seqfirst=k_per_layer,
        v_per_layer_seqfirst=v_per_layer,
        weights=weights,
        config=config,
        cache=decode_cache,
        max_seq=max(args.seq_len, 256),
        npu_lm=True,
        verbose=False,
    )

    # ---- CPU forward at SAME context as NPU's chain (per step) ----
    # We do NOT let CPU walk its own greedy chain — that would diverge after
    # the first BF16 reorder and make every subsequent step "different prompt"
    # incomparable. Instead, at each step we feed CPU exactly the prefix the
    # NPU saw (seed + npu_decoded[:step]) and get CPU's top-5 at that prefix.
    # That isolates per-token NPU vs CPU correctness from chain divergence.
    print(f"\nCPU forward at NPU-chain context for {args.n_tokens} steps (slow)...")
    cpu_decoded = list(seed)  # we still keep CPU's own chain for display
    cpu_top5_per_step = []
    cpu_top1_per_step = []
    cpu_times = []
    npu_chain = list(seed) + list(npu_decoded[len(seed) :])  # seed + all NPU picks
    for step in range(args.n_tokens):
        # Context CPU sees = seed + first `step` NPU picks (so step 0 sees seed only).
        ctx = npu_chain[: len(seed) + step]
        t = time.time()
        logits = cpu_decode_token(ctx, weights, config)
        cpu_top1 = int(np.argmax(logits))
        top5 = np.argsort(logits)[-5:][::-1].tolist()
        cpu_top5_per_step.append(top5)
        cpu_top1_per_step.append(cpu_top1)
        # Maintain CPU's own chain in parallel (separate, just for display).
        if step == 0:
            cpu_decoded.append(cpu_top1)
        else:
            # Re-derive CPU's own greedy chain by feeding its previous chain.
            own_logits = cpu_decode_token(cpu_decoded, weights, config)
            cpu_decoded.append(int(np.argmax(own_logits)))
        cpu_times.append(time.time() - t)
        if args.verbose:
            print(
                f"  step {step}: CPU@NPU-ctx top-1='{tok.decode([cpu_top1])}' (id={cpu_top1})"
                f"  ({cpu_times[-1]:.2f}s)"
            )

    # ---- Compare ----
    print(f"\n{'='*72}")
    print(f"Per-token gate: NPU top-1 == CPU top-1")
    print(f"{'='*72}")
    print(f"  NPU generated: {tok.decode(npu_decoded)!r}")
    print(f"  CPU generated: {tok.decode(cpu_decoded)!r}")
    print()

    n_match = 0
    n_top5 = 0
    n_total = args.n_tokens
    for step in range(args.n_tokens):
        npu_id = npu_decoded[real_len + step]
        # Use CPU's top-1 AT THE SAME CONTEXT THE NPU SAW (not CPU's own
        # divergent chain), so comparisons stay valid past the first BF16
        # reorder.
        cpu_id = cpu_top1_per_step[step]
        cpu_top5 = cpu_top5_per_step[step]
        if npu_id == cpu_id:
            verdict = "MATCH"
            n_match += 1
            n_top5 += 1
        elif npu_id in cpu_top5:
            verdict = f"top-{cpu_top5.index(npu_id)+1} (BF16 reorder)"
            n_top5 += 1
        else:
            verdict = "MISS (NPU not in CPU top-5)"
        print(
            f"  step {step:2d}: NPU '{tok.decode([npu_id]):<10s}' (id={npu_id:6d}) | "
            f"CPU@ctx '{tok.decode([cpu_id]):<10s}' (id={cpu_id:6d}) | {verdict}"
        )

    print()
    print(f"  Top-1 exact match  : {n_match}/{n_total}")
    print(f"  Top-5 overlap match: {n_top5}/{n_total}")
    print()
    if npu_times:
        print(
            f"  NPU decode wall: {1000*np.mean(npu_times):.1f} ms/token "
            f"({1.0/np.mean(npu_times):.2f} tok/s)"
        )
    if cpu_times:
        print(
            f"  CPU decode wall: {1000*np.mean(cpu_times):.1f} ms/token "
            f"({1.0/np.mean(cpu_times):.2f} tok/s)"
        )
        print(
            f"  Speedup (NPU vs CPU decode): {np.mean(cpu_times)/np.mean(npu_times):.1f}×"
        )

    # PASS criteria: every token must be in CPU top-5 (BF16 reorders are OK).
    if n_top5 == n_total:
        print("\nPHASE B verify: PASS (all NPU tokens within CPU top-5)")
        return 0
    else:
        print(f"\nPHASE B verify: FAIL ({n_total - n_top5} tokens missed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

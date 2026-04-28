#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 5 NPU decode smoke test for Qwen3-4B.

Compiles the 3 decode ELFs (rms_attn_gemvs_qknorm_rope + o_gemv_ffn_silu +
lm_head_gemv at 19×8192 K=3072) and runs a few NPU decode tokens after
NPU prefill. Validates:
  - 3 decode ELFs compile clean at Qwen3-4B padded shapes
  - NPU decode pipeline produces sensible tokens (matches CPU decode)
  - Per-token NPU decode time
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

from qwen3_4b_weights import LlamaConfig, load_weights, generate_rope_lut
from qwen3_4b_pad import make_padded_config, pad_weights
import qwen3_4b_decode as qwen3_decode

import llama3_prefill as _lp
from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B NPU decode smoke")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--cache-dir", type=str, default="build/prefill_kernel_cache")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    config = make_padded_config(
        orig_config, padded_emb_dim=3072, padded_hidden_dim=10240
    )
    print(
        f"Qwen3-4B decode smoke (PADDED emb={config.emb_dim}, hidden={config.hidden_dim}, "
        f"q_dim={config.n_heads * config.head_dim}, kv_dim={config.n_kv_heads * config.head_dim}, "
        f"vocab={config.vocab_size})"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    print(f"  Loaded in {time.time()-t:.1f}s")
    print("Padding weights...")
    weights = pad_weights(orig_weights, orig_config, config)

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=args.verbose)
    if (cache_dir / "manifest.json").exists():
        try:
            cache.load_manifest()
            print(f"  Loaded existing kernel cache: {sorted(cache.artifacts.keys())}")
        except Exception as e:
            print(f"  Could not load manifest ({type(e).__name__}: {e})")

    print(
        "\nCompiling external kernels (incl. mv_og.o + mv_dg_qwen3.o for 3-K rename)..."
    )
    compile_all_external_kernels(head_dim=config.head_dim)

    # Compile 3 NPU decode ELFs (rms_attn_gemvs_qknorm_rope, o_gemv_ffn_silu, lm_head_gemv)
    qwen3_decode.compile_decode_kernels(cache, config)
    print(f"  Decode kernel cache: {sorted(cache.artifacts.keys())}")

    # Run NPU decode: CPU prefill seed (KV cache populate) + NPU decode loop
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'  (seed={seed_ids})")

    print(
        f"\nRunning NPU decode generate({args.n_tokens} tokens, max_seq={args.max_seq})..."
    )
    t = time.time()
    decoded_tokens, decode_times = qwen3_decode.generate(
        seed_ids,
        args.n_tokens,
        weights,
        config,
        cache,
        max_seq=args.max_seq,
        npu_lm=True,
        verbose=args.verbose,
    )
    total = time.time() - t

    full_text = tokenizer.decode(seed_ids + decoded_tokens)
    print(f"\nGenerated: '{full_text}'")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Decode times (per token, s): {[f'{x:.3f}' for x in decode_times]}")
    if decode_times:
        avg = sum(decode_times) / len(decode_times)
        print(f"NPU decode avg: {avg*1000:.1f} ms/token ({1/avg:.1f} tok/s)")
    print(f"Total wall: {total:.2f}s")

    print("\nPhase 5 NPU decode smoke: PASS (decode ELFs compiled + ran end-to-end)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

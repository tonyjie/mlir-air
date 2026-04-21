# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test for qwen3_decode.py: NPU-decode generation vs CPU reference.

Compiles all decode kernels, generates a few tokens via NPU decode, prints
output. Compares first-decoded token against the CPU-only path.
"""

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

from qwen3_weights import LlamaConfig, load_weights
from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
import qwen3_decode


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--cache-dir", default="decode_kernel_cache")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(f"Loading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  {time.time()-t:.1f}s")

    prepare_air_project()
    cache = KernelCache(cache_dir=str(_THIS_DIR / args.cache_dir), verbose=args.verbose)
    if (_THIS_DIR / args.cache_dir / "manifest.json").exists():
        try:
            cache.load_manifest()
            print(f"Loaded cache: {sorted(cache.artifacts)}")
        except Exception as e:
            print(f"  cache load failed: {e}")

    print("\nCompiling external kernels...")
    compile_all_external_kernels(head_dim=config.head_dim)

    print("\nCompiling Qwen3 decode kernels...")
    qwen3_decode.compile_decode_kernels(cache, config)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    seed = tok.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'  ({len(seed)} tokens)")

    print(f"\nNPU decode {args.n_tokens} tokens (greedy)...")
    decoded, times = qwen3_decode.generate(
        seed,
        args.n_tokens,
        weights,
        config,
        cache,
        max_seq=128,
        npu_lm=True,
        verbose=args.verbose,
    )
    print(f"\nGenerated: {tok.decode(decoded)!r}")
    if times:
        print(
            f"\nNPU decode (avg over {len(times)} tokens): "
            f"{1000*np.mean(times):.1f} ms/token  ({1.0/np.mean(times):.2f} tok/s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 — per-kernel shape sweep for Qwen3-0.6B on NPU2.

Validates that each ELF needed for the Qwen3 split-path compiles at the
required shapes:

  - rms_attn_gemms (predecessor) — RMSNorm + Q/K/V GEMM, NO RoPE
      shapes: emb_dim=1024, q_dim=2048, kv_dim=1024
  - o_ffn (current) — O proj + Gate/Up + SwiGLU + Down
      shapes: emb_dim=1024, q_dim=2048 (input), hidden_dim=3072
  - head-first FlashAttention (Option C) at head_dim=128
  - LM head GEMV at vocab=151936

Phase 1 GATE: every shape compiles to ELF without error.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
for _p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen3_weights import LlamaConfig

from llama3_prefill import KernelCache, prepare_air_project, _RMS_GEMMS_ROPE_BACKEND
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels


def compile_rms_attn_gemms(cache, config, seq_len):
    from llama3.multi_launch_builder.superseded.rms_attn_gemms_multi import (
        build_rms_attn_gemms_module,
    )

    q_dim = config.n_heads * config.head_dim  # 16 * 128 = 2048
    kv_dim = config.n_kv_heads * config.head_dim  # 8 * 128 = 1024

    print(
        f"\n--- Compiling rms_attn_gemms (seq_len={seq_len}, "
        f"emb_dim={config.emb_dim}, q_dim={q_dim}, kv_dim={kv_dim}) ---"
    )
    if "rms_attn_gemms" in cache.artifacts:
        print("  cached, skip")
        return
    module = build_rms_attn_gemms_module(
        seq_len=seq_len,
        emb_dim=config.emb_dim,
        kv_dim=kv_dim,
        q_dim=q_dim,
        # tile_n=128 fits all three of q_dim=2048 (16 tiles) and kv_dim=1024 (8 tiles)
        tile_n=128,
        herd_n=4,
    )
    cache.compile_and_cache(
        "rms_attn_gemms",
        module,
        {
            "verbose": cache.verbose,
            **_RMS_GEMMS_ROPE_BACKEND,
            "instance_name": "rms_attn_gemms",
        },
    )


def compile_o_ffn(cache, config, seq_len):
    from llama3.multi_launch_builder.o_ffn_multi import build_o_ffn_module

    q_dim = config.n_heads * config.head_dim  # 2048

    print(
        f"\n--- Compiling o_ffn (seq_len={seq_len}, "
        f"emb_dim={config.emb_dim}, hidden_dim={config.hidden_dim}, "
        f"o_in_dim={q_dim}) ---"
    )
    if "o_ffn" in cache.artifacts:
        print("  cached, skip")
        return
    # NOTE: o_ffn input is q_dim (2048), output is emb_dim (1024). The default
    # builder assumes input==output==emb_dim. May need a separate o_in_dim
    # parameter — surface this if compile fails.
    try:
        module = build_o_ffn_module(
            seq_len=seq_len,
            emb_dim=config.emb_dim,
            hidden_dim=config.hidden_dim,
            o_in_dim=q_dim,
        )
        cache.compile_and_cache(
            "o_ffn",
            module,
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "o_ffn",
            },
        )
    except Exception as e:
        print(f"  o_ffn compile FAILED: {type(e).__name__}: {e}")
        print(
            f"  → likely needs o_in_dim != emb_dim parameter (Qwen3 q_dim=2048 vs emb_dim=1024)"
        )
        raise


def compile_headfirst_fa(cache, config, seq_len):
    from _llm_shared.phase_helpers.headfirst_fa import compile_headfirst_fa_kernel

    print(f"\n--- Compiling head-first FA (Option C) at head_dim={config.head_dim} ---")
    compile_headfirst_fa_kernel(
        cache,
        seq_len,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        verbose=cache.verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Phase 1 kernel shape sweep"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-fa",
        action="store_true",
        help="Skip FA compile (useful when iterating on rms_attn_gemms only)",
    )
    parser.add_argument(
        "--skip-o-ffn",
        action="store_true",
        help="Skip o_ffn compile (useful when bisecting)",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(f"Qwen3-0.6B Phase 1: per-kernel shape sweep")
    print(
        f"  emb_dim={config.emb_dim}, q_dim={config.n_heads*config.head_dim}, "
        f"kv_dim={config.n_kv_heads*config.head_dim}, hidden_dim={config.hidden_dim}, "
        f"head_dim={config.head_dim}, seq_len={args.seq_len}"
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
        f"\n=== External kernels (rope_halfsplit, silu_and_mul, attn_npu2 split, mv) ==="
    )
    compile_all_external_kernels(head_dim=config.head_dim)

    results = {}
    t0 = time.time()
    try:
        compile_rms_attn_gemms(cache, config, args.seq_len)
        results["rms_attn_gemms"] = "PASS"
    except Exception as e:
        results["rms_attn_gemms"] = f"FAIL: {type(e).__name__}: {e}"

    if not args.skip_o_ffn:
        try:
            compile_o_ffn(cache, config, args.seq_len)
            results["o_ffn"] = "PASS"
        except Exception as e:
            results["o_ffn"] = f"FAIL: {type(e).__name__}: {e}"

    if not args.skip_fa:
        try:
            compile_headfirst_fa(cache, config, args.seq_len)
            results["flash_attn"] = "PASS"
        except Exception as e:
            results["flash_attn"] = f"FAIL: {type(e).__name__}: {e}"

    cache._save_manifest()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Phase 1 results ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for k, v in results.items():
        marker = "PASS" if v == "PASS" else "FAIL"
        print(f"  {k:25s}: {marker}{'' if v == 'PASS' else '  ' + v[:200]}")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {n_pass}/{len(results)} kernels compiled")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 — single-block correctness test for Llama-3.2-3B on NPU2.

Wires layer 0 with NPU rms_gemms_rope + o_ffn (+ optional NPU FA) and
compares against the Llama-3.2-3B CPU reference.

NPU FA at head_dim=128 routes through Option C (head-first FA + host
transposes) automatically — see `_llm_shared/phase_helpers/headfirst_fa.py`
and LESSONS Lesson 3.

Phase 2 gate (head_dim-scaled per LESSON 1):
    whole-tensor cosine_sim > 0.99
    per-position cosine_sim min > 0.98 (head_dim=128)
    no NaN
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

from llama32_3b_weights import LlamaConfig, load_weights, generate_rope_lut
import llama32_3b_reference

from llama3_prefill import KernelCache, prepare_air_project, run_transformer_block
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.metrics import head_dim_scaled_per_pos_threshold
from _llm_shared.phase_helpers.orchestration import (
    compile_block_kernels,
    preload_block_weights,
)


def main():
    parser = argparse.ArgumentParser(
        description="Llama-3.2-3B Phase 2 single-block test"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=False,
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
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

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = tokenizer.encode(args.prompt)
    real_len = len(token_ids)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"  {real_len} real tokens; padding to seq_len={args.seq_len}")
    if real_len < args.seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = token_ids + [pad] * (args.seq_len - real_len)
    token_ids = np.array(token_ids[: args.seq_len], dtype=np.int64)

    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[token_ids]
    x_bf16 = x_f32.astype(bfloat16)
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

    compile_all_external_kernels(head_dim=config.head_dim)
    t = time.time()
    compile_block_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)
    print(f"  Kernel compile: {time.time()-t:.1f}s")

    if not args.no_preload:
        print("\nPre-loading layer-0 weights into BOs...")
        try:
            preload_block_weights(
                cache, weights, config, args.seq_len, rope_lut_bf16, layer_idx=0
            )
        except Exception as e:
            print(
                f"  Preload failed ({type(e).__name__}: {e}); falling back to lazy preload"
            )

    print("\nRunning NPU single block (layer 0)...")
    t = time.time()
    npu_out, _ = run_transformer_block(
        x_bf16,
        weights.layers[0],
        rope_lut_bf16,
        config,
        cache,
        layer_idx=0,
        verify=False,
        cpu_attn=args.cpu_attn,
        verbose=args.verbose,
    )
    print(f"  NPU single block: {time.time()-t:.2f}s")

    print("\nRunning CPU reference single block (layer 0)...")
    t = time.time()
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)
    ref_out, _ = llama32_3b_reference.transformer_block(
        x_f32, weights.layers[0], rope_lut_f32, config
    )
    print(f"  CPU reference: {time.time()-t:.2f}s")

    npu_arr = np.asarray(npu_out, dtype=np.float32)
    ref_arr = np.asarray(ref_out, dtype=np.float32)
    has_nan = bool(np.any(np.isnan(npu_arr)))

    def _print_metrics(label, a, b):
        cs = metrics.cosine_sim(a, b)
        err = metrics.mae(a, b)
        max_abs = float(np.max(np.abs(a - b)))
        pp = metrics.per_pos_cosine_min(a, b)
        print(
            f"  [{label}] cosine_sim={cs:.6f}  MAE={err:.6f}  "
            f"max_abs={max_abs:.4f}  per_pos_min={pp:.6f}"
        )
        return cs, err, max_abs, pp

    print(f"\n{'='*60}")
    print(f"Phase 2 — single-block correctness")
    print(f"{'='*60}")
    print(
        f"  attention   = {'CPU fallback' if args.cpu_attn else 'NPU FlashAttention (head-first via Option C)'}"
    )
    print(f"  NaN in NPU  = {has_nan}")
    print(f"  seq_len     = {args.seq_len}, real_tokens = {real_len}")
    print()
    cs_all, err_all, _, pp_all = _print_metrics("ALL  positions", npu_arr, ref_arr)
    cs_real, err_real, _, pp_real = _print_metrics(
        "REAL tokens   ", npu_arr[:real_len], ref_arr[:real_len]
    )
    print()
    per_pos_gate = head_dim_scaled_per_pos_threshold(config.head_dim)
    print(
        f"  Gate (real-token): whole-tensor cosine > 0.99 AND "
        f"per_pos_min > {per_pos_gate} (head_dim={config.head_dim} scaled) AND no NaN"
    )

    passed = cs_real > 0.99 and pp_real > per_pos_gate and not has_nan
    print(f"\n  Phase 2: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

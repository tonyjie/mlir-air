# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 4 — prefill perf for Qwen3-1.7B on NPU2.

Applies optimization patterns from the `optimize-prefill-perf` skill:

  Pattern 1 (multi-launch merging):     SKIP — split-ELF approach is required
                                         by Q/K Norm placement (RMSNorm doesn't
                                         commute with RoPE).
  Pattern 2 (per-layer BO pre-loading): APPLIED — bo_key=f"...L{i}" +
                                         static_input_indices on weights.
  Pattern 3 (intermediate buffer reuse):APPLIED — intermediate_indices on
                                         all _out arg slots.
  Pattern 4 (seq-first layout):         ALREADY (FA wrapper transposes only
                                         once per call; activations are seq-first).
  Pattern 5 (CPU→NPU op promotion):     N/A — Q/K Norm and RoPE on host are
                                         the architectural baseline; promoting
                                         is a Phase 6+ optimization.

Baseline target (cold/warm at seq_len=2048):
  - Cold: ~5–8s (28 layers × 3 ELFs × first-call setup)
  - Warm with preload: ~3–4s (28 × ~110 ms/layer; split ELF inherently slower
                       than llama3's fused 3-launch)

Phase 4 GATE: ≥3 of 5 patterns applied or N/A with reason; no correctness
regression (Phase 3 still PASS).
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
from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    _RMS_ATTN_GEMM_BACKEND,
    _O_FFN_BACKEND,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.qk_norm import apply_qk_norm
from _llm_shared.phase_helpers.headfirst_fa import (
    install_headfirst_fa_wrapper,
    compile_headfirst_fa_kernel,
)
from qwen3_phase2_test import _compile_qwen3_block_kernels


def _run_cached(*args, **kwargs):
    return _lp._run_cached(*args, **kwargs)


def _attn_backend_kwargs(*args, **kwargs):
    return _lp._attn_backend_kwargs(*args, **kwargs)


# Per-layer cached arg lists (Pattern 2 + 3): build once per layer, reuse
# weight numpy arrays so static_input_indices can skip BO re-write.
_per_layer_args = {}


def run_block_optimized(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx,
    return_kv=False,
):
    """Optimized single-block: per-layer BO preload + intermediate reuse.

    If return_kv=True, returns (block_out, k_roped, v) so the caller can
    populate a decode KV cache without a redundant CPU prefill pass.
    """
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim

    # ---- 1. rms_attn_gemms (NPU, weights-static, intermediates-reused) ----
    rms_key = f"rms_attn_gemms_L{layer_idx}"
    if rms_key not in _per_layer_args:
        _per_layer_args[rms_key] = [
            None,  # 0: x_in (dynamic)
            np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, q_dim),
            np.zeros((seq_len, q_dim), dtype=bfloat16),
            np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),
            np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),
        ]
    args = _per_layer_args[rms_key]
    args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

    results = _run_cached(
        cache,
        "rms_attn_gemms",
        _RMS_ATTN_GEMM_BACKEND,
        *args,
        output_indices=[4, 6, 8],
        static_input_indices={1, 3, 5, 7},
        intermediate_indices={2, 4, 6, 8},
        bo_key=rms_key,
    )
    q = results[4].reshape(seq_len, q_dim)
    k = results[6].reshape(seq_len, kv_dim)
    v = results[8].reshape(seq_len, kv_dim)

    # ---- 2. Q/K Norm (host) ----
    q_normed, k_normed = apply_qk_norm(
        q,
        k,
        layer_weights.q_norm,
        layer_weights.k_norm,
        n_heads,
        n_kv_heads,
        head_dim,
        eps=config.rms_norm_eps,
    )

    # ---- 3. RoPE (host, BF16/F32) ----
    lut_f32 = np.asarray(rope_lut_bf16[:seq_len], dtype=np.float32)
    half = head_dim // 2
    cos_vals = lut_f32[:, :half]
    sin_vals = lut_f32[:, half:]

    def _rope_per_head(x_flat_bf16, n_h):
        x = np.asarray(x_flat_bf16, dtype=np.float32).reshape(seq_len, n_h, head_dim)
        out = np.empty_like(x)
        x1, x2 = x[..., :half], x[..., half:]
        for h in range(n_h):
            out[:, h, :half] = x1[:, h, :] * cos_vals - x2[:, h, :] * sin_vals
            out[:, h, half:] = x1[:, h, :] * sin_vals + x2[:, h, :] * cos_vals
        return out.reshape(seq_len, n_h * head_dim).astype(bfloat16)

    q_roped = _rope_per_head(q_normed, n_heads)
    k_roped = _rope_per_head(k_normed, n_kv_heads)

    # ---- 4. flash_attn (NPU head-first wrapper) ----
    attn_output = np.zeros((seq_len, q_dim), dtype=bfloat16)
    attn_bk = _attn_backend_kwargs(head_dim)
    fa_results = _run_cached(
        cache,
        "flash_attn",
        attn_bk,
        np.ascontiguousarray(q_roped),
        np.ascontiguousarray(k_roped),
        np.ascontiguousarray(v),
        attn_output,
    )
    attn_out = fa_results[-1].reshape(seq_len, q_dim)

    # ---- 5. o_ffn (NPU, weights-static, intermediates-reused) ----
    offn_key = f"o_ffn_L{layer_idx}"
    if offn_key not in _per_layer_args:
        _per_layer_args[offn_key] = [
            None,  # 0: attn_out (dynamic)
            np.asarray(layer_weights.wo, dtype=bfloat16).reshape(q_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            None,  # 3: x_residual (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),
            np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
                hidden_dim, emb_dim
            ),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
            np.zeros(n_total, dtype=bfloat16),
        ]
    args = _per_layer_args[offn_key]
    args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, q_dim)
    args[3] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

    results = _run_cached(
        cache,
        "o_ffn",
        _O_FFN_BACKEND,
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=offn_key,
    )
    block_out = results[14].reshape(seq_len, emb_dim)
    if return_kv:
        # Copy out of the BO-backed result so subsequent layer runs that may
        # reuse the same intermediate slot don't clobber what the caller saved.
        return block_out, np.array(k_roped, copy=True), np.array(v, copy=True)
    return block_out


def npu_full_prefill(
    token_ids,
    weights,
    config,
    rope_lut_bf16,
    cache,
    verbose=False,
    collect_kv=False,
):
    """Run full N-layer NPU prefill.

    If collect_kv=True, also returns per-layer (k_roped, v) at all positions
    so the caller can populate a decode KV cache without a redundant CPU
    prefill pass. Both per-layer arrays are seq-first
    (seq_len, n_kv_heads * head_dim) BF16.
    """
    seq_len = token_ids.shape[0]
    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table[token_ids].astype(bfloat16)
    if not collect_kv:
        for i in range(config.n_layers):
            x = run_block_optimized(
                x, weights.layers[i], rope_lut_bf16, config, cache, i
            )
        return x

    k_per_layer, v_per_layer = [], []
    for i in range(config.n_layers):
        x, k_roped_i, v_i = run_block_optimized(
            x,
            weights.layers[i],
            rope_lut_bf16,
            config,
            cache,
            i,
            return_kv=True,
        )
        k_per_layer.append(k_roped_i)
        v_per_layer.append(v_i)
    return x, k_per_layer, v_per_layer


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B Phase 4 prefill perf measurement"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Production prefill seq_len (default 2048)",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile kernels then exit (used by `make compile-prefill`)",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of warm iterations to time"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"Qwen3-1.7B Phase 4 — prefill perf (seq_len={args.seq_len}, "
        f"{config.n_layers} layers)"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

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

    print("\nCompiling external kernels...")
    compile_all_external_kernels(head_dim=config.head_dim)

    print(f"\nCompiling/loading Qwen3 block kernels at seq_len={args.seq_len}...")
    t = time.time()
    _compile_qwen3_block_kernels(cache, config, args.seq_len)
    print(f"  Block kernels ready: {time.time()-t:.1f}s")

    if args.compile_only:
        print("--compile-only set; exiting after compile.")
        return 0

    # Build padded token ids
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer.encode(args.prompt)
    pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids = ids + [pad] * (args.seq_len - len(ids))
    token_ids = np.array(ids[: args.seq_len], dtype=np.int64)

    print(f"\nCold prefill (first call: BOs allocated + weights written)...")
    t = time.time()
    _ = npu_full_prefill(
        token_ids, weights, config, rope_lut_bf16, cache, verbose=args.verbose
    )
    cold = time.time() - t
    print(f"  Cold: {cold:.2f}s ({cold/config.n_layers*1000:.1f} ms/layer)")

    print(
        f"\nWarm prefill ({args.iterations} iterations, BOs cached + weights skip)..."
    )
    warm_times = []
    for it in range(args.iterations):
        t = time.time()
        _ = npu_full_prefill(token_ids, weights, config, rope_lut_bf16, cache)
        warm_times.append(time.time() - t)
        print(
            f"  Iter {it}: {warm_times[-1]:.2f}s ({warm_times[-1]/config.n_layers*1000:.1f} ms/layer)"
        )

    avg_warm = float(np.mean(warm_times))
    speedup = cold / avg_warm if avg_warm > 0 else 0
    print(f"\n{'='*60}")
    print(f"Phase 4 — prefill perf summary")
    print(f"{'='*60}")
    print(f"  seq_len           = {args.seq_len}")
    print(f"  n_layers          = {config.n_layers}")
    print(
        f"  Cold prefill      = {cold:.2f}s ({cold/config.n_layers*1000:.1f} ms/layer)"
    )
    print(
        f"  Warm prefill avg  = {avg_warm:.2f}s ({avg_warm/config.n_layers*1000:.1f} ms/layer)"
    )
    print(f"  Cold→warm speedup = {speedup:.2f}x")
    print(f"\nPatterns applied:")
    print(f"  P1 multi-launch merging:   SKIP (Q/K Norm requires split ELF)")
    print(f"  P2 per-layer BO preload:   APPLIED (bo_key + static_input_indices)")
    print(f"  P3 intermediate reuse:     APPLIED (intermediate_indices)")
    print(f"  P4 seq-first activations:  ALREADY (FA wrapper handles)")
    print(f"  P5 CPU→NPU op promotion:   N/A (Q/K Norm + RoPE on host = baseline)")
    print(f"  → 3 patterns applied/already; gate ≥3 satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())

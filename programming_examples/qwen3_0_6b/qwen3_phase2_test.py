# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 — single-block correctness for Qwen3-0.6B on NPU2.

Wires layer 0 with the split-ELF approach:
  1. NPU rms_attn_gemms (predecessor builder, no RoPE) — produces normed, q, k, v
  2. Host apply_qk_norm — per-head RMSNorm on q and k (Qwen3 NEW)
  3. Host RoPE — bypasses predecessor rope_qk_multi (interleaved LUT mismatch)
  4. NPU flash_attn (head-first wrapper for head_dim=128)
  5. NPU o_ffn (with o_in_dim=q_dim=2048; emb_dim=1024)

Compares against the Qwen3 CPU reference (`qwen3_reference.transformer_block`).

Phase 2 gate (head_dim-scaled per LESSON 1):
    whole-tensor cosine_sim > 0.99 (real-token positions)
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

from qwen3_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen3_reference

import llama3_prefill as _lp
from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    _RMS_ATTN_GEMM_BACKEND,
    _O_FFN_BACKEND,
)


def _run_cached(*args, **kwargs):
    """Indirect call so the head-first FA wrapper's monkey-patch on
    `_lp._run_cached` is honored (a `from ... import _run_cached` would
    snapshot the original, bypassing the patch)."""
    return _lp._run_cached(*args, **kwargs)


def _attn_backend_kwargs(*args, **kwargs):
    """Same indirection for _attn_backend_kwargs (also patched by wrapper)."""
    return _lp._attn_backend_kwargs(*args, **kwargs)


from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.metrics import head_dim_scaled_per_pos_threshold
from _llm_shared.phase_helpers.qk_norm import apply_qk_norm
from _llm_shared.phase_helpers.headfirst_fa import (
    install_headfirst_fa_wrapper,
    compile_headfirst_fa_kernel,
)


def _compile_qwen3_block_kernels(cache, config, seq_len):
    """Compile rms_attn_gemms (split, no RoPE) + o_ffn + head-first FA."""
    from llama3.multi_launch_builder.superseded.rms_attn_gemms_multi import (
        build_rms_attn_gemms_module,
    )
    from llama3.multi_launch_builder.o_ffn_multi import build_o_ffn_module

    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim

    if "rms_attn_gemms" not in cache.artifacts:
        print(
            f"  Compiling rms_attn_gemms (seq_len={seq_len}, "
            f"emb_dim={config.emb_dim}, q_dim={q_dim}, kv_dim={kv_dim})..."
        )
        module = build_rms_attn_gemms_module(
            seq_len=seq_len,
            emb_dim=config.emb_dim,
            kv_dim=kv_dim,
            q_dim=q_dim,
            tile_n=128,
            herd_n=4,
        )
        cache.compile_and_cache(
            "rms_attn_gemms",
            module,
            {"verbose": cache.verbose, **_RMS_ATTN_GEMM_BACKEND},
        )
    else:
        print("  rms_attn_gemms cached")

    if "o_ffn" not in cache.artifacts:
        print(
            f"  Compiling o_ffn (seq_len={seq_len}, "
            f"emb_dim={config.emb_dim}, hidden_dim={config.hidden_dim}, "
            f"o_in_dim={q_dim})..."
        )
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
    else:
        print("  o_ffn cached")

    install_headfirst_fa_wrapper()
    compile_headfirst_fa_kernel(
        cache,
        seq_len,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        verbose=cache.verbose,
    )

    cache._save_manifest()


def run_qwen3_block_npu(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    cpu_attn=False,
    cpu_o_ffn=False,
):
    """Run a single Qwen3 block on NPU using split rms_attn_gemms + host Q/K Norm + host RoPE + NPU FA + NPU o_ffn."""
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim

    # 1. rms_attn_gemms (NPU): RMSNorm + Q/K/V GEMMs (NO RoPE)
    rms_args = [
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0: x_in
        np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(
            emb_dim
        ),  # 1: norm_w
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2: normed
        np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, q_dim),  # 3: wq
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 4: q_out
        np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 5: wk
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6: k_out
        np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 7: wv
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8: v_out
    ]
    results = _run_cached(
        cache,
        "rms_attn_gemms",
        _RMS_ATTN_GEMM_BACKEND,
        *rms_args,
        output_indices=[4, 6, 8],
        intermediate_indices={2, 4, 6, 8},
    )
    q = results[4].reshape(seq_len, q_dim)
    k = results[6].reshape(seq_len, kv_dim)
    v = results[8].reshape(seq_len, kv_dim)

    # 2. apply_qk_norm (host)
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

    # 3. RoPE on host (BF16 inside, F32 internally)
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

    # 4. flash_attn (NPU head-first wrapper, or CPU fallback for bisect)
    if cpu_attn:
        attn_out = qwen3_reference.attention_reference(
            np.asarray(q_roped, dtype=np.float32),
            np.asarray(k_roped, dtype=np.float32),
            np.asarray(v, dtype=np.float32),
            n_heads,
            n_kv_heads,
        ).astype(bfloat16)
    else:
        q_attn = np.ascontiguousarray(q_roped)
        k_attn = np.ascontiguousarray(k_roped)
        v_attn = np.ascontiguousarray(v)
        attn_output = np.zeros((seq_len, q_dim), dtype=bfloat16)
        attn_bk = _attn_backend_kwargs(head_dim)
        results = _run_cached(
            cache,
            "flash_attn",
            attn_bk,
            q_attn,
            k_attn,
            v_attn,
            attn_output,
        )
        attn_out = results[-1].reshape(seq_len, q_dim)

    # 5. o_ffn (NPU): O proj + Residual + FFN RMSNorm + Gate/Up + SwiGLU + Down + Residual
    offn_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, q_dim),  # 0: attn_out
        np.asarray(layer_weights.wo, dtype=bfloat16).reshape(q_dim, emb_dim),  # 1: wo
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2: proj
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),  # 3: x_residual
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4: res1
        np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(
            emb_dim
        ),  # 5: ffn_norm_w
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 6: normed2
        np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
            emb_dim, hidden_dim
        ),  # 7: w_gate
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 8: gate
        np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(
            emb_dim, hidden_dim
        ),  # 9: w_up
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 10: up
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 11: swiglu
        np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
            hidden_dim, emb_dim
        ),  # 12: w_down
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 13: down
        np.zeros(n_total, dtype=bfloat16),  # 14: output (1D)
    ]
    results = _run_cached(
        cache,
        "o_ffn",
        _O_FFN_BACKEND,
        *offn_args,
        output_indices=[14],
    )
    block_out = results[14].reshape(seq_len, emb_dim)
    return block_out


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Phase 2 single-block test (NPU)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Must be ≥ 256 for FA at head_dim=128 (lqp=256).",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="(unused for now; placeholder for future BO preload)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only the CPU baseline + Q/K Norm validation (skip NPU)",
    )
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Bisect: replace NPU FA with CPU attention reference",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"Qwen3-0.6B config: n_layers={config.n_layers}, "
        f"emb_dim={config.emb_dim}, n_heads={config.n_heads}, "
        f"n_kv_heads={config.n_kv_heads} (GQA group={config.n_heads // config.n_kv_heads}), "
        f"head_dim={config.head_dim}, hidden_dim={config.hidden_dim}, "
        f"vocab={config.vocab_size}, qkv_bias={config.qkv_bias}, qk_norm={config.qk_norm}"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = tokenizer.encode(args.prompt)
    real_len = len(token_ids)
    print(f"\nPrompt: '{args.prompt}'  ({real_len} real tokens)")
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
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)

    print("\nRunning CPU reference single block (layer 0)...")
    t = time.time()
    ref_out, _ = qwen3_reference.transformer_block(
        x_f32, weights.layers[0], rope_lut_f32, config
    )
    print(f"  CPU reference: {time.time()-t:.2f}s")
    ref_arr = np.asarray(ref_out, dtype=np.float32)

    if args.cpu_only:
        print("\n[--cpu-only] Skipping NPU run.")
        return 0

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

    print("\nCompiling Qwen3 block kernels...")
    t = time.time()
    _compile_qwen3_block_kernels(cache, config, args.seq_len)
    print(f"  Block kernel compile: {time.time()-t:.1f}s")

    # ---- Bisect: NPU rms_attn_gemms vs CPU reference ----
    print("\n--- Bisect: NPU rms_attn_gemms (normed, q, k, v) vs CPU ---")
    seq_len = args.seq_len
    emb_dim = config.emb_dim
    n_heads, n_kv_heads, head_dim = config.n_heads, config.n_kv_heads, config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    lw = weights.layers[0]
    rms_args = [
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),
        np.zeros((seq_len, q_dim), dtype=bfloat16),
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
        np.zeros((seq_len, kv_dim), dtype=bfloat16),
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
        np.zeros((seq_len, kv_dim), dtype=bfloat16),
    ]
    bisect_results = _run_cached(
        cache,
        "rms_attn_gemms",
        _RMS_ATTN_GEMM_BACKEND,
        *rms_args,
        output_indices=[2, 4, 6, 8],
        intermediate_indices={2, 4, 6, 8},
    )
    npu_normed = bisect_results[2].reshape(seq_len, emb_dim)
    npu_q = bisect_results[4].reshape(seq_len, q_dim)
    npu_k = bisect_results[6].reshape(seq_len, kv_dim)
    npu_v = bisect_results[8].reshape(seq_len, kv_dim)

    eps = config.rms_norm_eps
    x_in_f32 = np.asarray(x_bf16, dtype=np.float32)
    norm_w_f32 = np.asarray(lw.attn_norm, dtype=np.float32)
    rms = np.sqrt(np.mean(x_in_f32 * x_in_f32, axis=-1, keepdims=True) + eps)
    cpu_normed = (x_in_f32 / rms) * norm_w_f32
    cpu_q = cpu_normed @ np.asarray(lw.wq, dtype=np.float32)
    cpu_k = cpu_normed @ np.asarray(lw.wk, dtype=np.float32)
    cpu_v = cpu_normed @ np.asarray(lw.wv, dtype=np.float32)

    for name, n_arr, c_arr in [
        ("normed", npu_normed, cpu_normed),
        ("q     ", npu_q, cpu_q),
        ("k     ", npu_k, cpu_k),
        ("v     ", npu_v, cpu_v),
    ]:
        n_f = np.asarray(n_arr, dtype=np.float32)
        cs = metrics.cosine_sim(n_f, c_arr)
        ma = float(np.max(np.abs(n_f - c_arr)))
        am = float(np.mean(np.abs(n_f - c_arr)))
        print(f"  [{name}] cosine={cs:.6f}  max_abs={ma:.4f}  mean_abs={am:.4f}")

    print(f"\nRunning NPU single block (layer 0, cpu_attn={args.cpu_attn})...")
    t = time.time()
    npu_out = run_qwen3_block_npu(
        x_bf16,
        weights.layers[0],
        rope_lut_bf16,
        config,
        cache,
        layer_idx=0,
        cpu_attn=args.cpu_attn,
    )
    print(f"  NPU single block: {time.time()-t:.2f}s")

    npu_arr = np.asarray(npu_out, dtype=np.float32)
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
    print(f"Phase 2 — single-block correctness (NPU vs CPU reference)")
    print(f"{'='*60}")
    print(f"  attention   = NPU FlashAttention (head-first via Option C)")
    print(f"  RoPE        = HOST (BF16) — predecessor rope_qk_multi LUT incompatible")
    print(f"  Q/K Norm    = HOST (apply_qk_norm)")
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

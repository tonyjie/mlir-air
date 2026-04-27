# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end Qwen3-0.6B inference on NPU2.

NPU prefill (Phase 4 optimized path: per-layer BO preload + intermediate
reuse + head-first FA at head_dim=128) + decode loop.

Decode mode (`--decode {npu,cpu}`):
  - **npu** (default): full NPU decode via 3 fused multi-launch ELFs
    (rms_attn_gemvs_qknorm_rope, o_gemv_ffn_silu, lm_head_gemv) +
    host-side optimizations (pre-transposed weights cached on weights,
    per-layer arg-list cache, BO preload). ~95 ms/token (~10.5 tok/s),
    at parity with llama3-1B. See docs/development_progress/phase_b_fusion.md.
  - **cpu**: legacy CPU decode (~1.23 s/token). Kept for verification.

Usage:
    python3 qwen3_inference.py --compile-only
    python3 qwen3_inference.py --n-tokens 30 --prompt "The capital of France is"
    python3 qwen3_inference.py --decode cpu          # legacy slow path
    python3 qwen3_inference.py --profile
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
from _llm_shared.phase_helpers.headfirst_fa import install_headfirst_fa_wrapper
from qwen3_phase2_test import _compile_qwen3_block_kernels
from qwen3_phase4_test import npu_full_prefill
from qwen3_phase5_test import cpu_decode_token
import qwen3_decode


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Qwen3-0.6B NPU prefill + CPU decode"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-tokens", type=int, default=30)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", type=str, default="build/prefill_kernel_cache_2048")
    parser.add_argument(
        "--decode-cache-dir",
        type=str,
        default="build/decode_kernel_cache",
        help="Separate cache dir for decode kernels (NPU decode mode only)",
    )
    parser.add_argument(
        "--decode",
        choices=["npu", "cpu"],
        default="npu",
        help="Decode backend: npu (default, ~95 ms/token, 10.5 tok/s) or cpu (~1.23 s/token)",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Compare NPU prefill vs CPU F32 reference: per-layer K/V cache "
            "cosine + max_err + mean_err for all 28 layers, plus final logits "
            "cosine at pred_pos and top-1 match. Mirrors llama3 / llama32_3b "
            "/ smollm2 / qwen25 --verify. Adds ~30s wall (CPU forward)."
        ),
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(
        f"Qwen3-0.6B end-to-end NPU inference "
        f"(seq_len={args.seq_len}, {args.n_tokens} decode tokens)"
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

    print(f"\nCompiling/loading Qwen3 block kernels (seq_len={args.seq_len})...")
    t = time.time()
    _compile_qwen3_block_kernels(cache, config, args.seq_len)
    print(f"  Block kernels ready: {time.time()-t:.1f}s")

    # ---- Compile decode cache up-front so we can use NPU LM head for the
    # first-token logits (and avoid a second CPU prefill in the decode loop). ----
    if args.decode == "npu":
        print("\nCompiling Qwen3 decode kernels...")
        decode_cache = KernelCache(
            cache_dir=str(_THIS_DIR / args.decode_cache_dir),
            verbose=args.verbose,
        )
        if (_THIS_DIR / args.decode_cache_dir / "manifest.json").exists():
            try:
                decode_cache.load_manifest()
            except Exception as e:
                print(f"  decode cache load failed: {e}")
        qwen3_decode.compile_decode_kernels(decode_cache, config)
        # Preload all decode weights into per-layer BOs (out of the hot loop):
        # this fires both fused ELFs once per layer with dummy inputs so that
        # subsequent decode-block calls skip the weight-write via
        # static_input_indices. ~28 layers × 2 ELFs ≈ 1–2 s one-time cost.
        qwen3_decode.preload_decode_weights(decode_cache, weights, config)
    else:
        decode_cache = None

    if args.compile_only:
        print("\n--compile-only set; exiting after compile.")
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_ids = tokenizer.encode(args.prompt)
    real_len = len(base_ids)
    if real_len > args.seq_len:
        print(
            f"WARN: prompt has {real_len} tokens > seq_len={args.seq_len}, truncating"
        )
        base_ids = base_ids[: args.seq_len]
        real_len = args.seq_len
    pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    padded = base_ids + [pad] * (args.seq_len - real_len)
    padded_ids = np.array(padded[: args.seq_len], dtype=np.int64)

    # Warmup (preload BOs + first compile pass) — discards K/V
    print(f"\nWarming up NPU prefill...")
    t = time.time()
    _ = npu_full_prefill(padded_ids, weights, config, rope_lut_bf16, cache)
    print(f"  Warmup: {time.time()-t:.2f}s")

    # NPU prefill (timed) — collect per-layer K/V so decode can skip CPU re-prefill.
    print(f"\nNPU prefill (single pass, with KV extraction)...")
    t = time.time()
    if args.decode == "npu":
        npu_hidden, k_per_layer, v_per_layer = npu_full_prefill(
            padded_ids,
            weights,
            config,
            rope_lut_bf16,
            cache,
            collect_kv=True,
        )
    else:
        npu_hidden = npu_full_prefill(padded_ids, weights, config, rope_lut_bf16, cache)
        k_per_layer = v_per_layer = None
    t_prefill = time.time() - t
    print(
        f"  Prefill: {t_prefill:.2f}s ({t_prefill/config.n_layers*1000:.1f} ms/layer)"
    )

    # First decoded token: host RMSNorm of last position + NPU LM head GEMV.
    norm_w = np.asarray(weights.final_norm, dtype=np.float32)
    x_in = np.asarray(npu_hidden[real_len - 1], dtype=np.float32)
    rms = np.sqrt(np.mean(x_in * x_in) + config.rms_norm_eps)
    x_normed_bf16 = ((x_in / rms) * norm_w).astype(bfloat16)
    if args.decode == "npu":
        t_lm = time.time()
        logits0 = qwen3_decode.npu_lm_head(decode_cache, x_normed_bf16, weights, config)
        if args.verbose:
            print(f"  First-token NPU LM head: {(time.time()-t_lm)*1000:.1f} ms")
    else:
        lm_head = np.asarray(weights.lm_head, dtype=np.float32)
        logits0 = (x_in / rms) * norm_w @ lm_head.T  # CPU
    next_id = int(np.argmax(logits0))

    decoded_ids = list(base_ids) + [next_id]
    if args.verbose:
        print(f"  First token: '{tokenizer.decode([next_id])}' (id={next_id})")

    if args.verify:
        print(f"\n{'='*60}")
        print(
            f"Verification: NPU prefill vs CPU F32 reference ({config.n_layers} layers)"
        )
        print(f"{'='*60}")
        from qwen3_reference import (
            transformer_block as cpu_block,
            rms_norm as cpu_rms_norm,
        )

        rope_lut_f32 = rope_lut_bf16[: args.seq_len].astype(np.float32)
        x_cpu = weights.embed_table[padded_ids].astype(np.float32)
        n_kv_heads = config.n_kv_heads
        head_dim = config.head_dim
        n_layer_warns = 0
        for li in range(config.n_layers):
            x_cpu, cpu_intermediates = cpu_block(
                x_cpu, weights.layers[li], rope_lut_f32, config
            )
            cpu_k = (
                cpu_intermediates["k_roped"]
                .astype(np.float32)
                .reshape(args.seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            cpu_v = (
                cpu_intermediates["v"]
                .astype(np.float32)
                .reshape(args.seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            # NPU k_per_layer[li] / v_per_layer[li] are seq-first
            # (seq_len, n_kv_heads * head_dim) bf16 — reshape to match.
            npu_k = (
                np.asarray(k_per_layer[li], dtype=np.float32)
                .reshape(args.seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )
            npu_v = (
                np.asarray(v_per_layer[li], dtype=np.float32)
                .reshape(args.seq_len, n_kv_heads, head_dim)
                .transpose(1, 0, 2)
            )

            k_corr = np.corrcoef(npu_k.flatten(), cpu_k.flatten())[0, 1]
            v_corr = np.corrcoef(npu_v.flatten(), cpu_v.flatten())[0, 1]
            k_maxerr = float(np.max(np.abs(npu_k - cpu_k)))
            v_maxerr = float(np.max(np.abs(npu_v - cpu_v)))
            k_meanerr = float(np.mean(np.abs(npu_k - cpu_k)))
            v_meanerr = float(np.mean(np.abs(npu_v - cpu_v)))

            k_status = "OK" if k_corr > 0.99 else "WARN"
            v_status = "OK" if v_corr > 0.99 else "WARN"
            n_layer_warns += int(k_status == "WARN") + int(v_status == "WARN")
            print(
                f"  Layer {li:2d} K_cache: [{k_status}] corr={k_corr:.6f}, "
                f"max_err={k_maxerr:.4f}, mean_err={k_meanerr:.4f}"
            )
            print(
                f"  Layer {li:2d} V_cache: [{v_status}] corr={v_corr:.6f}, "
                f"max_err={v_maxerr:.4f}, mean_err={v_meanerr:.4f}"
            )

        # Final RMSNorm + LM Head on CPU side, compare logits at pred_pos
        x_cpu_normed = cpu_rms_norm(
            x_cpu,
            weights.final_norm.astype(np.float32),
            eps=config.rms_norm_eps,
        )
        cpu_logits_pred = (
            x_cpu_normed[real_len - 1] @ weights.lm_head.astype(np.float32).T
        )
        cpu_pred = int(np.argmax(cpu_logits_pred))
        npu_logits_f32 = np.asarray(logits0, dtype=np.float32)
        logit_corr = float(np.corrcoef(npu_logits_f32, cpu_logits_pred)[0, 1])
        logit_maxerr = float(np.max(np.abs(npu_logits_f32 - cpu_logits_pred)))
        logit_meanerr = float(np.mean(np.abs(npu_logits_f32 - cpu_logits_pred)))
        print(
            f"\n  Logits (pos {real_len-1}): corr={logit_corr:.6f}, "
            f"max_err={logit_maxerr:.4f}, mean_err={logit_meanerr:.4f}"
        )
        print(f"  NPU top-1: {next_id} ({tokenizer.decode([next_id])!r})")
        print(f"  CPU top-1: {cpu_pred} ({tokenizer.decode([cpu_pred])!r})")
        print(f"  Match: {'YES' if next_id == cpu_pred else 'NO'}")
        print(
            f"  Per-layer warnings: {n_layer_warns} (informational; BF16 K/V drift across deep stacks is expected)"
        )

    if args.decode == "npu":
        print(f"\nNPU decode loop ({args.n_tokens - 1} more tokens, greedy)...")
        # decode_loop_from_kv consumes the K/V we already extracted from NPU
        # prefill — no second CPU prefill needed.
        decoded_ids, decode_times = qwen3_decode.decode_loop_from_kv(
            seed_token_ids=base_ids,
            first_decoded_id=next_id,
            n_more_tokens=args.n_tokens,
            k_per_layer_seqfirst=k_per_layer,
            v_per_layer_seqfirst=v_per_layer,
            weights=weights,
            config=config,
            cache=decode_cache,
            max_seq=max(args.seq_len, 256),
            npu_lm=True,
            verbose=args.verbose,
        )
        decode_label = "NPU decode (avg)"
    else:
        # Legacy CPU decode loop (greedy)
        print(f"\nCPU decode loop ({args.n_tokens - 1} more tokens, greedy)...")
        decode_times = []
        for step in range(args.n_tokens - 1):
            t = time.time()
            logits = cpu_decode_token(decoded_ids, weights, config)
            next_id = int(np.argmax(logits))
            decoded_ids.append(next_id)
            decode_times.append(time.time() - t)
            if args.verbose:
                print(
                    f"  step {step}: '{tokenizer.decode([next_id])}' "
                    f"({decode_times[-1]:.2f}s)"
                )
            if next_id == tokenizer.eos_token_id:
                print(f"  EOS at step {step}; stopping.")
                break
        decode_label = "CPU decode (avg)"

    print(f"\n{'='*60}")
    print(f"Generated text:")
    print(f"{'='*60}")
    print(f"  {tokenizer.decode(decoded_ids)!r}")

    if args.profile or True:  # always show perf
        print(f"\n{'='*60}")
        print(f"Perf:")
        print(f"{'='*60}")
        print(
            f"  NPU prefill (warm): {t_prefill:.2f}s "
            f"({t_prefill/config.n_layers*1000:.1f} ms/layer)"
        )
        if decode_times:
            avg = float(np.mean(decode_times))
            print(f"  {decode_label}:   {avg:.2f}s/token  ({1.0/avg:.2f} tok/s)")
            print(f"  Decode tokens:      {len(decode_times) + 1}")
        print()
        if args.decode == "npu":
            print(
                "Note: NPU decode uses 3 fused multi-launch ELFs "
                "(rms_attn_gemvs_qknorm_rope, o_gemv_ffn_silu, lm_head_gemv) "
                "= 57 NPU calls/token. Host-side BO preload + per-layer arg "
                "cache + pre-transposed weights eliminate the per-call Python "
                "overhead — see docs/development_progress/phase_b_fusion.md."
            )
        else:
            print(
                "Note: --decode cpu is the legacy slow path; default is --decode npu."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

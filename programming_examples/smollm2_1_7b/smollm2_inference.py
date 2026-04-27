# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B end-to-end NPU inference (NPU prefill + NPU decode)."""

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

from smollm2_weights import LlamaConfig, load_weights, generate_rope_lut
import smollm2_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    preload_prefill_weights,
)
from llama3_decode import compile_decode_kernels, run_decode_block
import llama3_inference
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.orchestration import compile_block_kernels
from _llm_shared.phase_helpers.decode_setup import (
    pre_transpose_decode_weights,
    npu_lm_head_gemv,
)
from _llm_shared.phase_helpers.prefill_runner import npu_prefill_with_kv_extraction


def main():
    parser = argparse.ArgumentParser(
        description="SmolLM2-1.7B end-to-end NPU inference (prefill + decode)"
    )
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--prefill-cache-dir", type=str, default="build/prefill_kernel_cache")
    parser.add_argument("--decode-cache-dir", type=str, default="build/decode_kernel_cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use CPU attention in prefill instead of NPU FA",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Compare NPU prefill vs CPU F32 reference: per-layer K/V cache "
            "cosine + max_err + mean_err for all 24 layers, plus final logits "
            "cosine at pred_pos and top-1 match. Mirrors llama3_inference.py "
            "--verify. Runs CPU forward in parallel — adds ~30s wall."
        ),
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    config = LlamaConfig()
    print(f"SmolLM2 end-to-end NPU inference")
    print(
        f"  layers={config.n_layers}, vocab={config.vocab_size}, "
        f"attn={'CPU' if args.cpu_attn else 'NPU FA'}"
    )
    print(f"  seq_len={args.seq_len}, n_tokens={args.n_tokens}")

    print("\n[setup] Kernel caches...")
    t = time.time()
    prepare_air_project()

    prefill_cache_dir = _THIS_DIR / args.prefill_cache_dir
    decode_cache_dir = _THIS_DIR / args.decode_cache_dir
    prefill_cache = KernelCache(cache_dir=str(prefill_cache_dir))
    decode_cache = KernelCache(cache_dir=str(decode_cache_dir))
    if (prefill_cache_dir / "manifest.json").exists():
        prefill_cache.load_manifest()
    if (decode_cache_dir / "manifest.json").exists():
        decode_cache.load_manifest()

    compile_all_external_kernels(head_dim=config.head_dim)
    compile_block_kernels(prefill_cache, config, args.seq_len, cpu_attn=args.cpu_attn)
    needed_decode = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
    if not all(k in decode_cache.artifacts for k in needed_decode):
        compile_decode_kernels(decode_cache, config)
    t_compile = time.time() - t
    print(f"  Compile / cache load: {t_compile:.1f}s")

    if args.compile_only:
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    print("\n[setup] Weights...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    t_load = time.time() - t
    print(f"  Weight load: {t_load:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )

    print("\n[setup] Pre-loading BOs...")
    t = time.time()
    pre_transpose_decode_weights(weights, config)
    preload_prefill_weights(weights, config, prefill_cache, args.seq_len, rope_lut_bf16)
    llama3_inference._preload_decode_weights(decode_cache, weights, config)
    t_preload = time.time() - t
    print(f"  BO preload: {t_preload:.1f}s")

    prompt_tokens = tokenizer.encode(args.prompt)
    real_len = len(prompt_tokens)
    if real_len > args.seq_len:
        prompt_tokens = prompt_tokens[: args.seq_len]
        real_len = args.seq_len
    if real_len < args.seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = prompt_tokens + [pad] * (args.seq_len - real_len)
    else:
        token_ids = prompt_tokens
    token_ids = np.array(token_ids, dtype=np.int64)
    pred_pos = real_len - 1

    print(f"\n{'='*60}")
    print(f"Inference on prompt: {args.prompt!r}")
    print(f"  ({real_len} real tokens; padded to {args.seq_len})")
    print(f"{'='*60}")

    print(f"\n[1/3] NPU prefill ({config.n_layers} layers)...")
    t = time.time()
    x_prefill, k_cache, v_cache = npu_prefill_with_kv_extraction(
        token_ids,
        weights,
        config,
        prefill_cache,
        rope_lut_bf16,
        max_seq=args.seq_len,
        cpu_attn=args.cpu_attn,
    )
    t_prefill = time.time() - t
    print(
        f"  NPU prefill: {t_prefill:.2f}s  "
        f"({t_prefill/config.n_layers*1000:.0f} ms/layer)"
    )

    t = time.time()
    last_hidden = np.asarray(x_prefill, dtype=np.float32)[pred_pos : pred_pos + 1]
    last_normed_bf16 = (
        smollm2_reference.rms_norm(last_hidden, weights.final_norm)
        .flatten()
        .astype(bfloat16)
    )
    first_logits = npu_lm_head_gemv(decode_cache, weights, config, last_normed_bf16)
    first_token = int(np.argmax(first_logits))
    t_first_lm_head = time.time() - t
    print(
        f"  First LM Head GEMV: {t_first_lm_head*1000:.0f} ms  -> "
        f"{tokenizer.decode([first_token])!r} (id={first_token})"
    )

    if args.verify:
        print(f"\n{'='*60}")
        print(
            f"Verification: NPU prefill vs CPU F32 reference ({config.n_layers} layers)"
        )
        print(f"{'='*60}")
        from smollm2_reference import transformer_block as cpu_block

        rope_lut_f32 = rope_lut_bf16[: args.seq_len].astype(np.float32)
        x_cpu = weights.embed_table[token_ids].astype(np.float32)
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
            npu_k = k_cache[li, :, : args.seq_len, :].astype(np.float32)
            npu_v = v_cache[li, :, : args.seq_len, :].astype(np.float32)

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

        x_cpu_normed = smollm2_reference.rms_norm(
            x_cpu, weights.final_norm.astype(np.float32)
        )
        cpu_logits_pred = x_cpu_normed[pred_pos] @ weights.lm_head.astype(np.float32).T
        cpu_pred = int(np.argmax(cpu_logits_pred))
        npu_logits_f32 = np.asarray(first_logits, dtype=np.float32)
        logit_corr = float(np.corrcoef(npu_logits_f32, cpu_logits_pred)[0, 1])
        logit_maxerr = float(np.max(np.abs(npu_logits_f32 - cpu_logits_pred)))
        logit_meanerr = float(np.mean(np.abs(npu_logits_f32 - cpu_logits_pred)))
        print(
            f"\n  Logits (pos {pred_pos}): corr={logit_corr:.6f}, "
            f"max_err={logit_maxerr:.4f}, mean_err={logit_meanerr:.4f}"
        )
        print(f"  NPU top-1: {first_token} ({tokenizer.decode([first_token])!r})")
        print(f"  CPU top-1: {cpu_pred} ({tokenizer.decode([cpu_pred])!r})")
        print(f"  Match: {'YES' if first_token == cpu_pred else 'NO'}")
        print(
            f"  Per-layer warnings: {n_layer_warns} (informational; BF16 K/V drift across deep stacks is expected)"
        )

    print(f"\n[2/3] NPU decode loop ({args.n_tokens} tokens)...")
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    generated = list(prompt_tokens) + [first_token]
    decode_times = []
    current_token = first_token

    for token_idx in range(args.n_tokens - 1):
        current_pos = len(generated) - 1
        if current_pos >= args.seq_len:
            print(f"  Hit seq_len cap at pos={current_pos}, stopping")
            break

        x_in_bf16 = embed_table_f32[current_token].astype(bfloat16)
        t = time.time()
        x_bf16 = x_in_bf16
        for layer_idx in range(config.n_layers):
            x_bf16 = run_decode_block(
                x_bf16,
                weights.layers[layer_idx],
                decode_cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )
        x_f32 = np.asarray(x_bf16, dtype=np.float32).reshape(1, config.emb_dim)
        x_normed_bf16 = (
            smollm2_reference.rms_norm(x_f32, weights.final_norm)
            .flatten()
            .astype(bfloat16)
        )
        next_logits = npu_lm_head_gemv(decode_cache, weights, config, x_normed_bf16)
        next_token = int(np.argmax(next_logits))
        decode_times.append(time.time() - t)

        if args.profile:
            print(
                f"  Tok {token_idx+1:2d} pos={current_pos:3d}  "
                f"{tokenizer.decode([next_token])!r:<14s}  "
                f"{decode_times[-1]*1000:.0f} ms"
            )

        generated.append(next_token)
        current_token = next_token

    print(f"\n[3/3] Done.\n{'='*60}")
    print(f"Generated text:")
    print(f"  {tokenizer.decode(generated)!r}")
    print()
    print(f"Timings (one-time setup):")
    print(f"  Compile / cache load : {t_compile:.1f}s")
    print(f"  Weight load          : {t_load:.1f}s")
    print(f"  BO preload           : {t_preload:.1f}s")
    print(f"Timings (per inference):")
    print(
        f"  NPU prefill ({config.n_layers}L)    : {t_prefill:.2f}s "
        f"({t_prefill/config.n_layers*1000:.0f} ms/layer)"
    )
    print(f"  First LM Head GEMV   : {t_first_lm_head*1000:.0f} ms")
    if decode_times:
        avg_ms = float(np.mean(decode_times)) * 1000
        med_ms = float(np.median(decode_times)) * 1000
        n_dec = len(decode_times)
        print(
            f"  Decode {n_dec} tokens     : avg={avg_ms:.0f} ms/token, "
            f"median={med_ms:.0f} ms/token  ({1000/avg_ms:.1f} tok/s)"
        )
    print(
        f"  Total inference wall : {t_prefill + t_first_lm_head + sum(decode_times):.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

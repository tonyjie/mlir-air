# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 3 — full-model correctness for SmolLM2-1.7B on NPU2.

Wires all 24 transformer layers through `run_transformer_block`, runs final
RMSNorm + LM Head on CPU (deferred NPU LM Head to Phase 5), and verifies that
the NPU top-1 prediction matches the CPU reference (which itself matches HF
to logits-corr 0.99999978, per Phase 0).

Phase 3 gate:
    Top-1 NPU prediction matches CPU reference for >= 3/3 canonical prompts
    Per-layer cosine_sim > 0.95 for all layers
    No NaN anywhere
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
for p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smollm2_weights import LlamaConfig, load_weights, generate_rope_lut
import smollm2_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

# Reuse Phase 2's compile helper (skip-if-cached behavior)
from smollm2_phase2_test import compile_block_kernels

CANONICAL_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The sky is",
]


def _per_pos_cosine_min(a, b):
    """Minimum per-position cosine sim across the seq dim."""
    a2 = np.asarray(a, dtype=np.float32).reshape(a.shape[0], -1)
    b2 = np.asarray(b, dtype=np.float32).reshape(b.shape[0], -1)
    cos = (a2 * b2).sum(axis=-1) / (
        np.linalg.norm(a2, axis=-1) * np.linalg.norm(b2, axis=-1) + 1e-12
    )
    return float(cos.min())


def _whole_cosine(a, b):
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    return float(
        np.dot(a_flat, b_flat)
        / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )


def run_npu_full_prefill(
    input_ids,
    weights,
    config,
    cache,
    rope_lut_bf16,
    cpu_attn=False,
    capture_intermediates=False,
    verbose=False,
):
    """Full 24-layer NPU prefill + final RMSNorm + LM Head (both CPU)."""
    seq_len = len(input_ids)

    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x_f32 = embed_table_f32[input_ids]  # (seq_len, emb_dim)
    x_bf16 = x_f32.astype(bfloat16)

    per_layer = [] if capture_intermediates else None
    t0 = time.time()
    for layer_idx in range(config.n_layers):
        x_bf16, _ = run_transformer_block(
            x_bf16,
            weights.layers[layer_idx],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=layer_idx,
            verify=False,
            cpu_attn=cpu_attn,
            verbose=verbose,
        )
        if capture_intermediates:
            per_layer.append(np.asarray(x_bf16, dtype=np.float32).copy())
    npu_time = time.time() - t0

    # Final RMSNorm + LM Head on CPU (Phase 5 will move LM Head to NPU)
    x_f32_out = np.asarray(x_bf16, dtype=np.float32)
    x_normed = smollm2_reference.rms_norm(x_f32_out, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x_normed @ lm_head.T  # (seq_len, vocab)

    return logits, per_layer, npu_time


def run_cpu_full_prefill(
    input_ids,
    weights,
    config,
    rope_lut_f32,
    capture_intermediates=False,
):
    """CPU reference full prefill — exposes per-layer outputs."""
    embed_table_f32 = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table_f32[input_ids]

    per_layer = [] if capture_intermediates else None
    for layer_idx in range(config.n_layers):
        x, _ = smollm2_reference.transformer_block(
            x, weights.layers[layer_idx], rope_lut_f32, config
        )
        if capture_intermediates:
            per_layer.append(x.copy())

    x = smollm2_reference.rms_norm(x, weights.final_norm)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x @ lm_head.T
    return logits, per_layer


def evaluate_prompt(
    prompt,
    tokenizer,
    weights,
    config,
    cache,
    rope_lut_bf16,
    rope_lut_f32,
    seq_len,
    cpu_attn,
    diagnostic,
    verbose,
):
    """Run NPU + CPU full prefill for one prompt; return result dict."""
    token_ids = tokenizer.encode(prompt)
    real_len = len(token_ids)
    if real_len > seq_len:
        token_ids = token_ids[:seq_len]
        real_len = seq_len
    elif real_len < seq_len:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = token_ids + [pad] * (seq_len - real_len)
    input_ids = np.array(token_ids, dtype=np.int64)
    pred_pos = real_len - 1

    print(f"\n--- Prompt: {prompt!r} ---")
    print(f"  {real_len} real tokens; pred at position {pred_pos}")

    # NPU
    print("  Running NPU full prefill (24 layers)...")
    npu_logits, npu_per_layer, npu_time = run_npu_full_prefill(
        input_ids,
        weights,
        config,
        cache,
        rope_lut_bf16,
        cpu_attn=cpu_attn,
        capture_intermediates=diagnostic,
        verbose=verbose,
    )
    print(
        f"    NPU prefill: {npu_time:.2f}s  ({npu_time/config.n_layers*1000:.0f} ms/layer)"
    )

    # CPU reference
    print("  Running CPU reference full prefill (24 layers)...")
    t = time.time()
    cpu_logits, cpu_per_layer = run_cpu_full_prefill(
        input_ids,
        weights,
        config,
        rope_lut_f32,
        capture_intermediates=diagnostic,
    )
    cpu_time = time.time() - t
    print(f"    CPU reference: {cpu_time:.1f}s")

    # Compare top-1
    npu_top1 = int(np.argmax(npu_logits[pred_pos]))
    cpu_top1 = int(np.argmax(cpu_logits[pred_pos]))
    npu_token = tokenizer.decode([npu_top1])
    cpu_token = tokenizer.decode([cpu_top1])
    top1_match = npu_top1 == cpu_top1

    # Logits correlation (sanity)
    logits_corr = _whole_cosine(npu_logits[pred_pos], cpu_logits[pred_pos])
    has_nan_npu = bool(np.any(np.isnan(npu_logits[pred_pos])))

    print(f"  Top-1 NPU:  '{npu_token}' (id={npu_top1})")
    print(f"  Top-1 CPU:  '{cpu_token}' (id={cpu_top1})")
    print(f"  Match:      {'YES' if top1_match else 'NO'}")
    print(f"  Logits cos: {logits_corr:.6f}")
    print(f"  NaN in NPU: {has_nan_npu}")

    per_layer_cos = None
    if diagnostic and npu_per_layer is not None:
        per_layer_cos = []
        for li, (npu_l, cpu_l) in enumerate(zip(npu_per_layer, cpu_per_layer)):
            cos = _whole_cosine(npu_l, cpu_l)
            per_pos = _per_pos_cosine_min(npu_l, cpu_l)
            per_layer_cos.append((li, cos, per_pos))
        print("  Per-layer cosine_sim (whole / per-pos min):")
        for li, cos, pp in per_layer_cos:
            flag = "" if cos > 0.95 else "  <-- DRIFT"
            print(f"    Layer {li:2d}: {cos:.6f}  / {pp:.6f}{flag}")

    return {
        "prompt": prompt,
        "npu_top1": npu_top1,
        "cpu_top1": cpu_top1,
        "npu_token": npu_token,
        "cpu_token": cpu_token,
        "top1_match": top1_match,
        "logits_corr": logits_corr,
        "has_nan_npu": has_nan_npu,
        "per_layer_cos": per_layer_cos,
        "real_len": real_len,
        "pred_pos": pred_pos,
        "npu_time_s": npu_time,
        "cpu_time_s": cpu_time,
    }


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-1.7B Phase 3 full-model test")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=False,
        help="Use CPU attention fallback (default: False, use NPU FA)",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Capture per-layer outputs for cosine-sim drift analysis (uses 2x DRAM)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prompts", nargs="+", default=None, help="Override the canonical prompt set"
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    config = LlamaConfig()
    print(
        f"SmolLM2 config: n_layers={config.n_layers}, emb_dim={config.emb_dim}, "
        f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads} (MHA), "
        f"head_dim={config.head_dim}, hidden_dim={config.hidden_dim}, "
        f"vocab_size={config.vocab_size}, rope_base={config.rope_base}"
    )
    print(
        f"Attention path: {'CPU fallback' if args.cpu_attn else 'NPU FlashAttention'}"
    )
    print(f"Diagnostic per-layer mode: {args.diagnostic}")

    # Load weights
    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    weights = load_weights(args.model, config=config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    # Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # RoPE LUT
    rope_lut_bf16 = generate_rope_lut(
        config=config, seq_len=args.seq_len, dtype=bfloat16
    )
    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)

    # Kernel cache (compile if not already cached from Phase 2)
    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=args.verbose)
    if (cache_dir / "manifest.json").exists():
        cache.load_manifest()
        print(f"\nLoaded existing kernel cache: {sorted(cache.artifacts.keys())}")
    compile_all_external_kernels(head_dim=config.head_dim)
    compile_block_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)

    prompts = args.prompts if args.prompts is not None else CANONICAL_PROMPTS

    print(f"\n{'='*60}")
    print(f"Phase 3 — Full-model correctness on {len(prompts)} prompt(s)")
    print(f"{'='*60}")

    results = []
    for prompt in prompts:
        results.append(
            evaluate_prompt(
                prompt,
                tokenizer,
                weights,
                config,
                cache,
                rope_lut_bf16,
                rope_lut_f32,
                args.seq_len,
                args.cpu_attn,
                args.diagnostic,
                args.verbose,
            )
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"Phase 3 — Summary")
    print(f"{'='*60}")
    n_match = sum(r["top1_match"] for r in results)
    any_nan = any(r["has_nan_npu"] for r in results)
    min_layer_cos = None
    if args.diagnostic:
        all_layer_cos = [
            cos for r in results for (_, cos, _) in (r["per_layer_cos"] or [])
        ]
        if all_layer_cos:
            min_layer_cos = min(all_layer_cos)

    for r in results:
        marker = "PASS" if r["top1_match"] else "FAIL"
        print(
            f"  [{marker}] {r['prompt']!r:<40s}  "
            f"NPU={r['npu_token']!r:<12s} CPU={r['cpu_token']!r:<12s} "
            f"corr={r['logits_corr']:.4f}"
        )

    print(
        f"\n  Top-1 match: {n_match}/{len(prompts)}  (gate: {len(prompts)}/{len(prompts)})"
    )
    if min_layer_cos is not None:
        print(f"  Min per-layer cosine_sim: {min_layer_cos:.6f}  (gate: > 0.95)")
    print(f"  Any NaN in NPU: {any_nan}  (gate: False)")

    passed = (
        n_match == len(prompts)
        and not any_nan
        and (min_layer_cos is None or min_layer_cos > 0.95)
    )
    print(f"\n  Phase 3: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

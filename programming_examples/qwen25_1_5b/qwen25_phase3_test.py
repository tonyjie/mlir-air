# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 3 — full-model correctness for Qwen2.5-1.5B on NPU2.

All 28 layers + final RMSNorm + (CPU) LM Head, with the LESSON 2 gate
(decisive prompts top-1 match + competitive prompts top-5 overlap).

Two pieces of qwen25-specific infra are wired in for every layer:
- `qwen25_bias`: host-side QKV-bias add via _run_cached monkey-patch
- `qwen25_pad`:  GQA-aware reindexed padding to 2048/9216 dims

NPU runs the padded-shape forward (kernels at emb=2048, hidden=9216);
CPU reference runs the original-shape forward (emb=1536, hidden=8960).
The padded LM-head logits collapse to original-vocab logits because:
  - Padded x[:, 1536:] = 0 throughout (zero-pad propagation)
  - Padded lm_head (= padded embed_table) has zeros in cols [1536:]

Phase 3 gate (LESSON 2):
    Decisive prompts (CPU top-1 p > 0.5): NPU top-1 == CPU top-1
    Competitive prompts (CPU top-1 p ≤ 0.5): top-5 overlap
    No NaN
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

from qwen25_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen25_reference

from llama3_prefill import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.canonical_prompts import CANONICAL_PROMPTS
from _llm_shared.phase_helpers.prefill_runner import (
    run_npu_full_prefill,
    run_cpu_full_prefill,
)
from _llm_shared.phase_helpers.metrics import whole_cosine, per_pos_cosine_min

from qwen25_bias import (
    install_qkv_bias_wrapper,
    register_layer_bias,
    precompute_rope_bias,
    clear_layer_bias,
)
from qwen25_pad import make_padded_config, pad_weights, slice_output

# Reuse the Phase 2 compile helper (it auto-detects padded vs unpadded shapes
# and picks the right tile config).
from qwen25_phase2_test import _compile_qwen25_block_kernels


def _register_all_layer_biases(padded_weights, padded_config, padded_rope_lut, seq_len):
    """Register Qwen2 QKV bias (RoPE-rotated) for ALL n_layers."""
    clear_layer_bias()
    for i, lw in enumerate(padded_weights.layers):
        bq_roped = precompute_rope_bias(
            lw.bq,
            padded_rope_lut,
            padded_config.n_heads,
            padded_config.head_dim,
            seq_len,
        )
        bk_roped = precompute_rope_bias(
            lw.bk,
            padded_rope_lut,
            padded_config.n_kv_heads,
            padded_config.head_dim,
            seq_len,
        )
        register_layer_bias(i, bq_roped, bk_roped, lw.bv)


def _evaluate_prompt_qwen25(
    prompt,
    tokenizer,
    padded_weights,
    orig_weights,
    padded_config,
    orig_config,
    cache,
    padded_rope_bf16,
    orig_rope_f32,
    seq_len,
    cpu_attn,
    diagnostic,
    verbose,
):
    """Per-prompt eval with split NPU (padded) vs CPU (orig) weights."""
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

    # NPU: padded shapes throughout (28 layers + final RMSNorm + LM head all
    # work because zero padding propagates through).
    print(f"  Running NPU full prefill ({padded_config.n_layers} layers, padded)...")
    npu_logits, npu_per_layer, npu_time = run_npu_full_prefill(
        input_ids,
        padded_weights,
        padded_config,
        cache,
        padded_rope_bf16,
        qwen25_reference,
        cpu_attn=cpu_attn,
        capture_intermediates=diagnostic,
        verbose=verbose,
    )
    print(
        f"    NPU prefill: {npu_time:.2f}s "
        f"({npu_time/padded_config.n_layers*1000:.0f} ms/layer)"
    )

    # CPU: orig shapes (true Qwen2.5 reference).
    print(
        f"  Running CPU reference full prefill ({orig_config.n_layers} layers, orig)..."
    )
    t = time.time()
    cpu_logits, cpu_per_layer = run_cpu_full_prefill(
        input_ids,
        orig_weights,
        orig_config,
        orig_rope_f32,
        qwen25_reference,
        capture_intermediates=diagnostic,
    )
    cpu_time = time.time() - t
    print(f"    CPU reference: {cpu_time:.1f}s")

    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    npu_lg = npu_logits[pred_pos]
    cpu_lg = cpu_logits[pred_pos]
    npu_top5 = list(np.argsort(npu_lg)[-5:][::-1])
    cpu_top5 = list(np.argsort(cpu_lg)[-5:][::-1])
    npu_top1 = int(npu_top5[0])
    cpu_top1 = int(cpu_top5[0])
    npu_token = tokenizer.decode([npu_top1])
    cpu_token = tokenizer.decode([cpu_top1])
    top1_match = npu_top1 == cpu_top1

    npu_p = _softmax(npu_lg.astype(np.float64))
    cpu_p = _softmax(cpu_lg.astype(np.float64))
    cpu_top1_p = float(cpu_p[cpu_top1])
    npu_top1_p = float(npu_p[npu_top1])
    decisive = cpu_top1_p > 0.5
    cpu_in_npu5 = cpu_top1 in npu_top5
    npu_in_cpu5 = npu_top1 in cpu_top5
    top5_overlap = cpu_in_npu5 and npu_in_cpu5

    logits_corr = whole_cosine(npu_lg, cpu_lg)
    has_nan_npu = bool(np.any(np.isnan(npu_lg)))

    cls = "decisive" if decisive else "competitive"
    print(f"  Top-1 NPU:  '{npu_token}' (id={npu_top1}, p={npu_top1_p:.3f})")
    print(f"  Top-1 CPU:  '{cpu_token}' (id={cpu_top1}, p={cpu_top1_p:.3f})  [{cls}]")
    print(f"  Top-1 match: {'YES' if top1_match else 'NO'}")
    print(f"  Top-5 overlap (cpu_top1∈npu5 AND npu_top1∈cpu5): {top5_overlap}")
    print(f"  Logits cos: {logits_corr:.6f}")
    print(f"  NaN in NPU: {has_nan_npu}")

    per_layer_cos = None
    if diagnostic and npu_per_layer is not None:
        per_layer_cos = []
        for li, (npu_l, cpu_l) in enumerate(zip(npu_per_layer, cpu_per_layer)):
            # Slice padded NPU output to compare against unpadded CPU output.
            npu_l_sliced = slice_output(npu_l, orig_config.emb_dim).astype(np.float32)
            cos = whole_cosine(npu_l_sliced, cpu_l)
            per_pos = per_pos_cosine_min(npu_l_sliced[:real_len], cpu_l[:real_len])
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
        "cpu_top1_p": cpu_top1_p,
        "npu_top1_p": npu_top1_p,
        "decisive": decisive,
        "top5_overlap": top5_overlap,
        "logits_corr": logits_corr,
        "has_nan_npu": has_nan_npu,
        "per_layer_cos": per_layer_cos,
        "real_len": real_len,
        "pred_pos": pred_pos,
        "npu_time_s": npu_time,
        "cpu_time_s": cpu_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Phase 3 full-model test")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=True,
        help="Use CPU attention fallback (default)",
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Use NPU FlashAttention via Option C wrapper.",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Capture per-layer outputs for cosine drift analysis (uses 2x DRAM)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompts", nargs="+", default=None)
    args = parser.parse_args()

    os.chdir(_THIS_DIR)

    orig_config = LlamaConfig()
    print(
        f"Qwen2.5-1.5B orig config: n_layers={orig_config.n_layers}, "
        f"emb_dim={orig_config.emb_dim}, n_heads={orig_config.n_heads}, "
        f"n_kv_heads={orig_config.n_kv_heads}, head_dim={orig_config.head_dim}, "
        f"hidden_dim={orig_config.hidden_dim}, vocab={orig_config.vocab_size}"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    print(f"  Loaded orig in {time.time()-t:.1f}s")

    padded_config = make_padded_config(
        orig_config, padded_emb_dim=2048, padded_hidden_dim=9216
    )
    print(
        f"PADDED config: emb={padded_config.emb_dim}, hidden={padded_config.hidden_dim}, "
        f"n_heads={padded_config.n_heads} (group_padded="
        f"{padded_config.n_heads // padded_config.n_kv_heads})"
    )
    t = time.time()
    padded_weights = pad_weights(orig_weights, orig_config, padded_config)
    print(
        f"  Built padded weights in {time.time()-t:.1f}s "
        f"(layer0.wq={padded_weights.layers[0].wq.shape})"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    padded_rope_bf16 = generate_rope_lut(
        config=padded_config, seq_len=args.seq_len, dtype=bfloat16
    )
    orig_rope_f32 = np.asarray(
        generate_rope_lut(config=orig_config, seq_len=args.seq_len, dtype=bfloat16),
        dtype=np.float32,
    )

    prepare_air_project()
    cache_dir = _THIS_DIR / args.cache_dir
    cache = KernelCache(cache_dir=str(cache_dir), verbose=args.verbose)
    if (cache_dir / "manifest.json").exists():
        cache.load_manifest()
        print(f"\nLoaded existing kernel cache: {sorted(cache.artifacts.keys())}")
    compile_all_external_kernels(head_dim=padded_config.head_dim)
    _compile_qwen25_block_kernels(
        cache, padded_config, args.seq_len, cpu_attn=args.cpu_attn
    )

    install_qkv_bias_wrapper()
    print(f"\nRegistering Qwen2 QKV bias for all {padded_config.n_layers} layers...")
    _register_all_layer_biases(
        padded_weights, padded_config, padded_rope_bf16, args.seq_len
    )

    prompts = args.prompts if args.prompts is not None else CANONICAL_PROMPTS

    print(f"\n{'='*60}")
    print(f"Phase 3 — Full-model correctness on {len(prompts)} prompt(s)")
    print(f"{'='*60}")

    results = []
    for prompt in prompts:
        results.append(
            _evaluate_prompt_qwen25(
                prompt,
                tokenizer,
                padded_weights,
                orig_weights,
                padded_config,
                orig_config,
                cache,
                padded_rope_bf16,
                orig_rope_f32,
                args.seq_len,
                args.cpu_attn,
                args.diagnostic,
                args.verbose,
            )
        )

    print(f"\n{'='*60}")
    print(f"Phase 3 — Summary")
    print(f"{'='*60}")
    n_match = sum(r["top1_match"] for r in results)
    any_nan = any(r["has_nan_npu"] for r in results)
    decisive_results = [r for r in results if r["decisive"]]
    competitive_results = [r for r in results if not r["decisive"]]
    decisive_match = sum(r["top1_match"] for r in decisive_results)
    competitive_overlap = sum(r["top5_overlap"] for r in competitive_results)

    for r in results:
        if r["decisive"]:
            marker = "PASS" if r["top1_match"] else "FAIL"
            cls = "decisive  "
        else:
            marker = "PASS" if r["top5_overlap"] else "FAIL"
            cls = "competitive"
        print(
            f"  [{marker}] [{cls}] {r['prompt']!r:<32s}  "
            f"NPU={r['npu_token']!r:<10s} CPU={r['cpu_token']!r:<10s} "
            f"cpu_p={r['cpu_top1_p']:.3f}  corr={r['logits_corr']:.4f}"
        )

    print(f"\n  Strict top-1 match (all prompts): {n_match}/{len(prompts)}")
    print(
        f"  Decisive prompts (CPU p>0.5) top-1 match: "
        f"{decisive_match}/{len(decisive_results)}  (gate: all)"
    )
    print(
        f"  Competitive prompts top-5 overlap: "
        f"{competitive_overlap}/{len(competitive_results)}  (gate: all)"
    )
    print(f"  Any NaN in NPU: {any_nan}  (gate: False)")

    passed = (
        decisive_match == len(decisive_results)
        and competitive_overlap == len(competitive_results)
        and not any_nan
    )
    print(f"\n  Phase 3: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

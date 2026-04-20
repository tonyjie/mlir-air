# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 / Phase 3 orchestration helpers shared across LLM deployments.

- compile_block_kernels: compile rms_gemms_rope + o_ffn (+ flash_attn). Routes
                         to head-first FA at head_dim=128 (LESSON 3 / Option C).
- preload_block_weights: layer-0 BO preload (Phase 2 single-block test)
- evaluate_prompt      : Phase 3 per-prompt evaluator with the LESSON 2
                         decisive vs competitive gate
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from llama3_prefill import (
    _RMS_GEMMS_ROPE_BACKEND,
    _O_FFN_BACKEND,
    run_transformer_block,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

from .metrics import cosine_sim, mae, per_pos_cosine_min, whole_cosine
from .headfirst_fa import (
    HEAD_DIM_128_FA_CONFIG,
    compile_headfirst_fa_kernel,
    install_headfirst_fa_wrapper,
)


def compile_block_kernels(cache, config, seq_len, cpu_attn=True):
    """Compile the rms_gemms_rope + o_ffn (+ optional flash_attn) ELFs.

    Routes head_dim=128 NPU FA through Option C (head-first kernel + host
    transpose wrapper). For head_dim ≤ 64, uses the standard seq-first FA.

    Skips kernels that are already in `cache.artifacts` (cache-hit path).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling per-block kernels (seq_len={seq_len})...")
    print(
        f"  config: emb_dim={emb_dim}, kv_dim={kv_dim} "
        f"(GQA group={n_heads // n_kv_heads}), n_heads={n_heads}, head_dim={head_dim}"
    )
    print(f"  cpu_attn={cpu_attn}")
    print(f"{'='*60}\n")

    from llama3.multi_launch_builder.rms_gemms_rope_multi import (
        build_rms_gemms_rope_module,
    )
    from llama3.multi_launch_builder.o_ffn_multi import build_o_ffn_module

    if "rms_gemms_rope" not in cache.artifacts:
        cache.compile_and_cache(
            "rms_gemms_rope",
            build_rms_gemms_rope_module(
                seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
            ),
            {"verbose": cache.verbose, **_RMS_GEMMS_ROPE_BACKEND},
        )
    else:
        print("  rms_gemms_rope already cached, skipping compile")

    if "o_ffn" not in cache.artifacts:
        cache.compile_and_cache(
            "o_ffn",
            build_o_ffn_module(seq_len, emb_dim, hidden_dim),
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "o_ffn",
            },
        )
    else:
        print("  o_ffn already cached, skipping compile")

    # FlashAttention compile (only if --npu-attn requested)
    if not cpu_attn:
        # At head_dim ≥ 128 use Option C (head-first FA + host transposes).
        # At head_dim ≤ 64 the seq-first FA works fine and is preferred (no
        # transpose overhead).
        if head_dim >= 128:
            install_headfirst_fa_wrapper()  # idempotent
            compile_headfirst_fa_kernel(
                cache, seq_len, n_heads, n_kv_heads, head_dim, verbose=cache.verbose
            )
        else:
            # Standard seq-first FA path (llama3 / smollm2)
            from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
                build_module as build_attn_seq,
            )

            if "flash_attn" not in cache.artifacts:
                cache.compile_and_cache(
                    "flash_attn",
                    build_attn_seq(
                        lk=seq_len,
                        lkp=head_dim,
                        lq=seq_len,
                        lqp=256,
                        dk=head_dim,
                        dv=head_dim,
                        num_q_tiles=4,
                        num_cascade_stages=4,
                        num_heads=n_heads,
                        num_kv_heads=n_kv_heads,
                        causal=True,
                    ),
                    {
                        "verbose": cache.verbose,
                        "omit_while_true_loop": False,  # enable_shared_buffers=True at lkp==dk
                        "omit_pingpong": "all",
                        "runtime_loop_tiling_sizes": [1, 1],
                        "output_format": "elf",
                        "instance_name": "attention_bf16",
                    },
                )

    cache._save_manifest()
    print(f"\nCompiled {len(cache.artifacts)} kernels to {cache.cache_dir}/")


def preload_block_weights(cache, weights, config, seq_len, rope_lut_bf16, layer_idx=0):
    """Pre-load layer-`layer_idx` weights into per-layer BOs (Phase 2 single-block).

    Mirrors llama3's `preload_prefill_weights` but scoped to ONE layer.
    Issues a real warm-up `_run_cached` call per kernel so the per-layer BOs
    get allocated AND the weights get written via the same code path that
    `run_transformer_block` will use later (with `bo_key=f"<kernel>_L<idx>"`
    + `static_input_indices`). On the second call, weights skip BO write.

    Earlier versions called `cache.preload_static_inputs(name, kwargs,
    list_of_tuples)` which silently fell back to lazy preload because that
    API actually expects a flat dict — the runner survived but Pattern 2
    (BO pre-loading) was degraded for Phase 2 tests. Caught by
    evaluate-deployment audit on qwen25_1_5b 2026-04-19. Fixed to use the
    same warm-up pattern as `llama3_prefill.preload_prefill_weights`.
    """
    from llama3_prefill import _run_cached, run_transformer_block

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim

    lw = weights.layers[layer_idx]
    rope_lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    rope_lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # Also populate run_transformer_block's per-layer arg cache so the
    # subsequent run_transformer_block call reuses the same arrays.
    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    rms_args = [
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0 x_in (dynamic)
        np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2 intermediate
        np.asarray(lw.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4 intermediate
        np.asarray(lw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6 intermediate
        np.asarray(lw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8 intermediate/output
        rope_lut_q,
        rope_lut_k,
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 11 q_roped
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 12 k_roped
    ]
    _arg_cache[f"rms_gemms_rope_L{layer_idx}"] = rms_args
    _run_cached(
        cache,
        "rms_gemms_rope",
        _RMS_GEMMS_ROPE_BACKEND,
        *rms_args,
        output_indices=[8, 11, 12],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=f"rms_gemms_rope_L{layer_idx}",
    )

    offn_args = [
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # arg0 attn_out (dynamic)
        np.asarray(lw.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 3 residual (dynamic)
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4
        np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 6
        np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 8
        np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 10
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 11
        np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 13
        np.zeros(n_total, dtype=bfloat16),  # 14 output
    ]
    _arg_cache[f"o_ffn_L{layer_idx}"] = offn_args
    _run_cached(
        cache,
        "o_ffn",
        _O_FFN_BACKEND,
        *offn_args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=f"o_ffn_L{layer_idx}",
    )


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
    ref_module,
):
    """Phase 3 per-prompt evaluator with the LESSON 2 decisive/competitive gate.

    Returns a result dict with top-1/top-5/decisive/match fields used by the
    Phase 3 summary.
    """
    from .prefill_runner import run_npu_full_prefill, run_cpu_full_prefill

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

    print(f"  Running NPU full prefill ({config.n_layers} layers)...")
    npu_logits, npu_per_layer, npu_time = run_npu_full_prefill(
        input_ids,
        weights,
        config,
        cache,
        rope_lut_bf16,
        ref_module,
        cpu_attn=cpu_attn,
        capture_intermediates=diagnostic,
        verbose=verbose,
    )
    print(
        f"    NPU prefill: {npu_time:.2f}s "
        f"({npu_time/config.n_layers*1000:.0f} ms/layer)"
    )

    print(f"  Running CPU reference full prefill ({config.n_layers} layers)...")
    t = time.time()
    cpu_logits, cpu_per_layer = run_cpu_full_prefill(
        input_ids,
        weights,
        config,
        rope_lut_f32,
        ref_module,
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
    decisive = cpu_top1_p > 0.5  # LESSON 2 classification
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
            cos = whole_cosine(npu_l, cpu_l)
            per_pos = per_pos_cosine_min(npu_l, cpu_l)
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

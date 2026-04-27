# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 — single-block correctness test for Qwen2.5-0.5B on NPU2.

Mirrors `qwen25_1_5b/qwen25_phase2_test.py` with three differences:

1. **head_dim=64** (vs 1.5B's 128) → standard seq-first FA path; **no
   Option C head-first wrapper installed**. Saves the host transpose
   pair around FA.
2. **Smaller dim padding**: emb 896 → 1024 (1×1024), hidden 4864 → 5120
   (5×1024). n_h pads 14 → 16 (= 1024/64). Group_padded = 16/2 = 8 (vs
   orig 7) — qwen25_pad's GQA-aware reindex path handles this.
3. **Smaller tile configs**: rms_gemms_rope tile_n=32 herd_n=4 (so
   K/V's N=128 fits 32×4=128), o_ffn defaults work since padded emb=1024
   and hidden=5120 are both 1024-aligned.

QKV bias handled identically: bias-free `rms_gemms_rope` ELF runs, then
host adds pre-RoPE'd biases via `qwen25_bias` wrapper (RoPE linearity).

Phase 2 gate (per `single-block-validation` skill, head_dim-scaled):
    whole-tensor cosine_sim > 0.99
    per-position cosine_sim min > 0.99 (head_dim=64)
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
# sys.path: programming_examples/, programming_examples/llama3/,
# programming_examples/qwen25_1_5b/ (for qwen25_pad and qwen25_bias),
# this dir.
for _p in (
    _EXAMPLES_DIR,
    _EXAMPLES_DIR / "llama3",
    _EXAMPLES_DIR / "qwen25_1_5b",
    _THIS_DIR,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen25_0_5b_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen25_0_5b_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    _RMS_GEMMS_ROPE_BACKEND,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.metrics import head_dim_scaled_per_pos_threshold

# Reuse qwen25_1_5b's bias + pad helpers (model-agnostic, just take Configs).
# Imported via sys.path above; the helpers' internal imports of
# `qwen25_weights` resolve to qwen25_1_5b/qwen25_weights.py — that's fine
# because the only types they touch (LlamaConfig / LayerWeights / LlamaWeights
# dataclass shape) are structurally identical to ours (we kept the names
# and field set).
from qwen25_bias import (
    install_qkv_bias_wrapper,
    register_layer_bias,
    precompute_rope_bias,
)
from qwen25_pad import make_padded_config, pad_weights, slice_output

# ---------------------------------------------------------------------------
# Qwen2.5-0.5B-tuned compile + preload
# ---------------------------------------------------------------------------


def _compile_qwen25_0_5b_block_kernels(cache, config, seq_len, cpu_attn=False):
    """Compile rms_gemms_rope + o_ffn (+ optional FA) at Qwen2.5-0.5B tile config.

    Uses the SHARED multi-launch builders unchanged but overrides tile
    parameters to fit Qwen2.5-0.5B's narrow K/V (N=128) and small
    hidden_dim (5120 padded). FA is the standard seq-first path
    (head_dim=64 — no Option C wrapper).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling Qwen2.5-0.5B per-block kernels (seq_len={seq_len})...")
    print(
        f"  emb_dim={emb_dim}, kv_dim={kv_dim} "
        f"(GQA group={n_heads // n_kv_heads}), head_dim={head_dim}, "
        f"hidden_dim={hidden_dim}"
    )
    print(f"  cpu_attn={cpu_attn}")
    print(f"{'='*60}\n")

    from llama3.multi_launch_builder.rms_gemms_rope_multi import (
        build_rms_gemms_rope_module,
    )
    from llama3.multi_launch_builder.o_ffn_multi import build_o_ffn_module

    if "rms_gemms_rope" not in cache.artifacts:
        # Padded emb=1024, kv_dim=128 (orig 128, n_kv unchanged at 2):
        # tile_n must divide all of Q (N=1024), K (N=128), V (N=128).
        # tile_n=32, herd_n=4 → Q 1024/128=8 ✓, K/V 128/128=1 ✓.
        module = build_rms_gemms_rope_module(
            seq_len=seq_len,
            emb_dim=emb_dim,
            kv_dim=kv_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            tile_n=32,
            herd_n=4,
        )
        cache.compile_and_cache(
            "rms_gemms_rope",
            module,
            {"verbose": cache.verbose, **_RMS_GEMMS_ROPE_BACKEND},
        )
    else:
        print("  rms_gemms_rope already cached, skipping compile")

    if "o_ffn" not in cache.artifacts:
        # Padded emb=1024 + hidden=5120:
        #   O GEMM: M=2048 K=1024 N=1024 — default tile_n=64 herd_n=4 (1024/256=4 ✓)
        #   Gate/Up: M=2048 K=1024 N=5120 — default tile_n=64 herd_n=4 (5120/256=20 ✓)
        #   Down:   M=2048 K=5120 N=1024 — default tile_n=64 herd_n=4 (4 ✓)
        # SwiGLU: hidden=5120, herd_x=8 → swiglu_tile_n=640 (5120/(640*8)=1 ✓)
        o_module = build_o_ffn_module(
            seq_len=seq_len,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            swiglu_tile_n=640,
        )
        cache.compile_and_cache(
            "o_ffn",
            o_module,
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "o_ffn",
            },
        )
    else:
        print("  o_ffn already cached, skipping compile")

    if not cpu_attn and "flash_attn" not in cache.artifacts:
        # head_dim=64 → standard seq-first FA path (NOT Option C).
        # FA at n_h=16 (padded), n_kv=2, hd=64 — 16 % num_heads_per_unroll(2) = 0 ✓
        from _llm_shared.kernel_builder.external_kernels import compile_attn_npu2

        compile_attn_npu2(head_dim=head_dim)

        from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module as build_attn,
        )

        lkp = head_dim  # 64 (enable_shared_buffers when lkp == dk)
        lqp = 256  # 256/4 = 64 = lkp = causal-mask requirement
        enable_shared_buffers = lkp == head_dim
        cache.compile_and_cache(
            "flash_attn",
            build_attn(
                lk=seq_len,
                lkp=lkp,
                lq=seq_len,
                lqp=lqp,
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
                "omit_while_true_loop": not enable_shared_buffers,
                "omit_pingpong": "all",
                "runtime_loop_tiling_sizes": [1, 1],
                "output_format": "elf",
                "instance_name": "attention_bf16",
            },
        )
    elif not cpu_attn:
        print("  flash_attn already cached, skipping compile")

    cache._save_manifest()
    print(f"\nCompiled {len(cache.artifacts)} kernels to {cache.cache_dir}/")


def _preload_qwen25_0_5b_block_weights(
    cache, weights, config, seq_len, rope_lut_bf16, layer_idx=0
):
    """Preload layer-`layer_idx` weights into per-layer BOs AND register bias."""
    from _llm_shared.phase_helpers.orchestration import preload_block_weights

    preload_block_weights(cache, weights, config, seq_len, rope_lut_bf16, layer_idx)

    lw = weights.layers[layer_idx]
    bq_roped = precompute_rope_bias(
        lw.bq, rope_lut_bf16, config.n_heads, config.head_dim, seq_len
    )
    bk_roped = precompute_rope_bias(
        lw.bk, rope_lut_bf16, config.n_kv_heads, config.head_dim, seq_len
    )
    register_layer_bias(layer_idx, bq_roped, bk_roped, lw.bv)
    print(
        f"  Registered Qwen2 QKV bias for layer {layer_idx}: "
        f"bq_roped={bq_roped.shape}, bk_roped={bk_roped.shape}, bv={lw.bv.shape}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B Phase 2 single-block test"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Phase 2 default = 512 (M=seq_len ≥ tile_m*herd_m=512 minimum). Use 2048 for full-shape test.",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=False,
        help="Use CPU attention fallback",
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Use NPU FlashAttention seq-first (default; no Option C since hd=64)",
    )
    parser.add_argument("--cache-dir", type=str, default="build/prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument(
        "--pad",
        dest="pad",
        action="store_true",
        default=None,
        help="Pad emb 896→1024 + hidden 4864→5120. Default: auto-on at seq_len ≥ 1024.",
    )
    parser.add_argument(
        "--no-pad",
        dest="pad",
        action="store_false",
        help="Force unpadded path (smaller seq_len only).",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    print(
        f"Qwen2.5-0.5B config: n_layers={orig_config.n_layers}, "
        f"emb_dim={orig_config.emb_dim}, n_heads={orig_config.n_heads}, "
        f"n_kv_heads={orig_config.n_kv_heads} "
        f"(GQA group={orig_config.n_heads // orig_config.n_kv_heads}), "
        f"head_dim={orig_config.head_dim}, hidden_dim={orig_config.hidden_dim}, "
        f"vocab={orig_config.vocab_size}, rope_base={orig_config.rope_base}, "
        f"qkv_bias={orig_config.qkv_bias}"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    # Default: always pad for Qwen2.5-0.5B. Both emb=896 and hidden=4864
    # break default tile_k_l2 in o_ffn O GEMM (K=896 not divisible). Padding
    # to 1024/5120 fixes this and dodges BD blowup at seq_len=2048.
    use_pad = args.pad if args.pad is not None else True
    if use_pad:
        # 896 → 1024 (1×1024), 4864 → 5120 (5×1024).
        # n_h: 14 → 16 (= 1024/64); group_padded = 16/2 = 8 (was 7).
        config = make_padded_config(
            orig_config,
            padded_emb_dim=1024,
            padded_hidden_dim=5120,
        )
        print(
            f"\nPADDED config: emb_dim={config.emb_dim} (was {orig_config.emb_dim}), "
            f"hidden_dim={config.hidden_dim} (was {orig_config.hidden_dim}), "
            f"n_heads={config.n_heads} (was {orig_config.n_heads})"
        )
        weights = pad_weights(orig_weights, orig_config, config)
        print(
            f"  Padded weights: embed_table={weights.embed_table.shape}, "
            f"layer0.wq={weights.layers[0].wq.shape}, "
            f"layer0.w_gate={weights.layers[0].w_gate.shape}"
        )
    else:
        config = orig_config
        weights = orig_weights

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
    _compile_qwen25_0_5b_block_kernels(
        cache, config, args.seq_len, cpu_attn=args.cpu_attn
    )
    print(f"  Kernel compile: {time.time()-t:.1f}s")

    install_qkv_bias_wrapper()

    if not args.no_preload:
        print("\nPre-loading layer-0 weights into BOs and registering bias...")
        try:
            _preload_qwen25_0_5b_block_weights(
                cache, weights, config, args.seq_len, rope_lut_bf16, layer_idx=0
            )
        except Exception as e:
            print(
                f"  Preload failed ({type(e).__name__}: {e}); falling back to "
                "lazy preload — registering bias anyway"
            )
            lw = weights.layers[0]
            bq_roped = precompute_rope_bias(
                lw.bq, rope_lut_bf16, config.n_heads, config.head_dim, args.seq_len
            )
            bk_roped = precompute_rope_bias(
                lw.bk, rope_lut_bf16, config.n_kv_heads, config.head_dim, args.seq_len
            )
            register_layer_bias(0, bq_roped, bk_roped, lw.bv)

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

    print("\nRunning CPU reference single block (layer 0) — UNPADDED...")
    t = time.time()
    orig_x_f32 = np.asarray(orig_weights.embed_table, dtype=np.float32)[token_ids]
    orig_rope_lut_bf16 = generate_rope_lut(
        config=orig_config, seq_len=args.seq_len, dtype=bfloat16
    )
    rope_lut_f32 = np.asarray(orig_rope_lut_bf16, dtype=np.float32)
    ref_out, _ = qwen25_0_5b_reference.transformer_block(
        orig_x_f32, orig_weights.layers[0], rope_lut_f32, orig_config
    )
    print(f"  CPU reference: {time.time()-t:.2f}s")

    npu_arr = np.asarray(npu_out, dtype=np.float32)
    if use_pad:
        npu_arr = slice_output(npu_arr, orig_config.emb_dim).astype(np.float32)
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
    print("Phase 2 — single-block correctness")
    print(f"{'='*60}")
    print(
        f"  attention   = "
        f"{'CPU fallback' if args.cpu_attn else 'NPU FlashAttention seq-first (hd=64, no Option C)'}"
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

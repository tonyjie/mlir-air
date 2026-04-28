# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 — single-block correctness test for Qwen2.5-3B on NPU2.

Mirrors qwen25_1_5b/qwen25_phase2_test.py with shape adapted for 3B:

- emb_dim=2048 already 1024-aligned → NO emb padding needed (vs 1.5B's
  1536→2048 pad). Only hidden_dim padded 11008→11264 (11×1024).
- n_heads stays at 16 (no GQA reindex needed since emb unchanged).
- head_dim=128 → Option C head-first FA wrapper (same as 1.5B).
- QKV bias via host-side post-RoPE add (qwen25_bias.py, same pattern).
- W1 watch: same GQA g=8 + n_kv=2 as qwen25_0_5b → expect same NPU FA
  per-position cos drop (~0.94). Paper-relevant: tests if W1 is
  hd-modulated (0.5B was hd=64, this is hd=128).

Phase 2 gate (head_dim=128 scaled per LESSON 1):
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
for _p in (
    _EXAMPLES_DIR,
    _EXAMPLES_DIR / "llama3",
    _EXAMPLES_DIR / "qwen25_1_5b",
    _THIS_DIR,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen25_3b_weights import LlamaConfig, load_weights, generate_rope_lut
import qwen25_3b_reference

from llama3_prefill import (
    KernelCache,
    prepare_air_project,
    run_transformer_block,
    _RMS_GEMMS_ROPE_BACKEND,
)
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers import metrics
from _llm_shared.phase_helpers.metrics import head_dim_scaled_per_pos_threshold
from _llm_shared.phase_helpers.headfirst_fa import (
    install_headfirst_fa_wrapper,
    compile_headfirst_fa_kernel,
)

from qwen25_bias import (
    install_qkv_bias_wrapper,
    register_layer_bias,
    precompute_rope_bias,
)
from qwen25_pad import make_padded_config, pad_weights, slice_output


def _compile_qwen25_3b_block_kernels(cache, config, seq_len, cpu_attn=False):
    """Compile rms_gemms_rope + o_ffn (+ Option C FA) at Qwen2.5-3B tile config.

    emb_dim=2048 padded to 2048 (no change), hidden_dim=11008 padded to 11264.
    Tile config: same pattern as qwen25_1_5b PADDED path (tile_n=64 herd_n=4).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling Qwen2.5-3B per-block kernels (seq_len={seq_len})...")
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
        # tile_n=64 herd_n=4 → Q (N=2048): 2048/256=8 ✓; K/V (N=256): 256/256=1 ✓.
        # Default herd_m=8 (mirror qwen25_1_5b PADDED known-good recipe at hidden=9216).
        # Our hidden=12288 (padded from 11008) is also 1024-aligned — should fit
        # default BD pool at seq_len=2048.
        module = build_rms_gemms_rope_module(
            seq_len=seq_len,
            emb_dim=emb_dim,
            kv_dim=kv_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            tile_n=64,
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
        # Padded emb=2048, hidden=11264. First attempt with default tiles
        # hit "BD chain with unassigned IDs" (Allocator exhausted) at hidden=11264 +
        # 8-launch ELF + herd_m=8. Mirror qwen25_1_5b's UNpadded fallback recipe
        # (halve channel pressure via herd_m=4) plus smaller swiglu_tile_n=704
        # (2 iters instead of 1 — more launches but less L1/launch).
        o_module = build_o_ffn_module(
            seq_len=seq_len,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
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

    if not cpu_attn:
        # head_dim=128 → Option C head-first FA wrapper (same as qwen25_1_5b).
        install_headfirst_fa_wrapper()
        compile_headfirst_fa_kernel(
            cache, seq_len, n_heads, n_kv_heads, head_dim, verbose=cache.verbose
        )

    cache._save_manifest()
    print(f"\nCompiled {len(cache.artifacts)} kernels to {cache.cache_dir}/")


def _preload_qwen25_3b_block_weights(
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


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-3B Phase 2 single-block test")
    parser.add_argument("--seq-len", type=int, default=2048)
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
        help="Use NPU FA via Option C head-first wrapper (default; hd=128)",
    )
    parser.add_argument("--cache-dir", type=str, default="build/prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    print(
        f"Qwen2.5-3B config: n_layers={orig_config.n_layers}, "
        f"emb_dim={orig_config.emb_dim}, n_heads={orig_config.n_heads}, "
        f"n_kv_heads={orig_config.n_kv_heads} "
        f"(GQA group={orig_config.n_heads // orig_config.n_kv_heads}), "
        f"head_dim={orig_config.head_dim}, hidden_dim={orig_config.hidden_dim}, "
        f"vocab={orig_config.vocab_size}, qkv_bias={orig_config.qkv_bias}"
    )

    print(f"\nLoading weights from {args.model}...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    print(f"  Loaded in {time.time()-t:.1f}s")

    # Padding: emb=2048 already 1024-aligned (no change), hidden 11008→11264.
    # Padding strategy: emb=2048 already 1024-aligned (no change).
    # hidden=11008 → 12288 (12×1024). 12288 is bigger than 11264 but uses
    # default tile config that mirrors qwen25_1_5b PADDED known-good recipe.
    # Cost: 12288/11008 = 11.6% extra FFN compute (vs 11264's 2.3%).
    config = make_padded_config(
        orig_config,
        padded_emb_dim=2048,
        padded_hidden_dim=12288,
    )
    print(
        f"\nPADDED config: emb_dim={config.emb_dim} (was {orig_config.emb_dim}, "
        f"unchanged), hidden_dim={config.hidden_dim} (was {orig_config.hidden_dim}), "
        f"n_heads={config.n_heads}"
    )
    weights = pad_weights(orig_weights, orig_config, config)
    print(
        f"  Padded weights: embed_table={weights.embed_table.shape}, "
        f"layer0.wq={weights.layers[0].wq.shape}, "
        f"layer0.w_gate={weights.layers[0].w_gate.shape}"
    )

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
    _compile_qwen25_3b_block_kernels(
        cache, config, args.seq_len, cpu_attn=args.cpu_attn
    )
    print(f"  Kernel compile: {time.time()-t:.1f}s")

    install_qkv_bias_wrapper()

    if not args.no_preload:
        print("\nPre-loading layer-0 weights into BOs and registering bias...")
        try:
            _preload_qwen25_3b_block_weights(
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
    ref_out, _ = qwen25_3b_reference.transformer_block(
        orig_x_f32, orig_weights.layers[0], rope_lut_f32, orig_config
    )
    print(f"  CPU reference: {time.time()-t:.2f}s")

    npu_arr = np.asarray(npu_out, dtype=np.float32)
    # Slice padded NPU output back to orig emb_dim (here orig=padded so no-op,
    # but kept for symmetry with qwen25_1_5b path).
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
        f"{'CPU fallback' if args.cpu_attn else 'NPU FA Option C head-first (hd=128)'}"
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

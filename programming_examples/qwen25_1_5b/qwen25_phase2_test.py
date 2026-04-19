# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 2 — single-block correctness test for Qwen2.5-1.5B on NPU2.

Wires layer 0 with NPU rms_gemms_rope (bias-free ELF + host bias post-add via
`qwen25_bias`) and NPU o_ffn, optionally NPU FlashAttention via Option C.
Compares against the Qwen2.5 CPU reference.

Phase 2 gate (head_dim-scaled per LESSON 1):
    whole-tensor cosine_sim > 0.99
    per-position cosine_sim min > 0.98 (head_dim=128)
    no NaN

Tile-config notes (from Phase 1 audit):
- `rms_gemms_rope` is built with `tile_n=64` so all three of Q (N=1536),
  K (N=256), V (N=256) satisfy `N % (tile_n * herd_n) == 0` with default
  herd_n=4.
- `o_ffn` is built with `gate_tile_n=64` and `swiglu_tile_n=2240` to fit
  hidden_dim=8960 (8960 = 2240 × 4; 8960 = 64 × 140).

QKV bias: Qwen2's per-projection bias is added on the host AFTER the bias-free
ELF runs. RoPE's linearity makes this exact: `RoPE(q + bq) = RoPE(q) + RoPE(bq)`.
The `qwen25_bias.install_qkv_bias_wrapper` monkey-patches `_run_cached` to
inject the precomputed pre-RoPE'd bias into rms_gemms_rope outputs.
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

# ---------------------------------------------------------------------------
# Qwen2.5-tuned compile + preload
# ---------------------------------------------------------------------------


def _compile_qwen25_block_kernels(cache, config, seq_len, cpu_attn=True):
    """Compile rms_gemms_rope + o_ffn (+ optional FA) at Qwen2.5 tile config.

    Uses the SHARED multi-launch builders unchanged but overrides the tile
    parameters to fit Qwen2.5's narrow K/V (N=256) and odd hidden_dim (8960).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling Qwen2.5-1.5B per-block kernels (seq_len={seq_len})...")
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

    # Tile config notes:
    # - Padded path (emb_dim=2048, hidden_dim=9216): default tiles work — both
    #   dims are clean multiples of 1024 so DMAs use single-dim BDs.
    # - Unpadded path (emb_dim=1536, hidden_dim=8960): tile_n=64 (fits K/V's
    #   N=256), herd_m=4 (halve channels) — only viable at seq_len ≤ 1024 due
    #   to BD pool exhaustion at seq_len=2048.
    is_padded = (emb_dim == 2048) and (hidden_dim == 9216)
    if "rms_gemms_rope" not in cache.artifacts:
        if is_padded:
            module = build_rms_gemms_rope_module(
                seq_len=seq_len,
                emb_dim=emb_dim,
                kv_dim=kv_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                # default tile_n=128, herd_n=4 → 512.
                #   Q: 2048/(128*4)=4 ✓
                #   K/V: 256/512=0.5 ✗ — still doesn't fit K/V even when padded.
                # Drop tile_n=64, herd_n=4 → 256 fits all three.
                tile_n=64,
                herd_n=4,
            )
        else:
            # Unpadded fallback (seq_len ≤ 1024).
            module = build_rms_gemms_rope_module(
                seq_len=seq_len,
                emb_dim=emb_dim,
                kv_dim=kv_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                tile_n=64,
                herd_n=4,
                herd_m=4,
                rope_herd_x=4,
            )
        cache.compile_and_cache(
            "rms_gemms_rope",
            module,
            {"verbose": cache.verbose, **_RMS_GEMMS_ROPE_BACKEND},
        )
    else:
        print("  rms_gemms_rope already cached, skipping compile")

    if "o_ffn" not in cache.artifacts:
        if is_padded:
            # hidden_dim=9216 = 9 × 1024 — clean BDs with default tiles.
            o_module = build_o_ffn_module(
                seq_len=seq_len,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
            )
        else:
            o_module = build_o_ffn_module(
                seq_len=seq_len,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                gate_tile_n=64,
                swiglu_tile_n=2240,
                o_herd_m=4,
                gate_herd_m=4,
                down_herd_m=4,
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
        # head_dim=128 → Option C head-first FA wrapper (same as llama32_3b).
        install_headfirst_fa_wrapper()
        compile_headfirst_fa_kernel(
            cache, seq_len, n_heads, n_kv_heads, head_dim, verbose=cache.verbose
        )

    cache._save_manifest()
    print(f"\nCompiled {len(cache.artifacts)} kernels to {cache.cache_dir}/")


def _preload_qwen25_block_weights(
    cache, weights, config, seq_len, rope_lut_bf16, layer_idx=0
):
    """Preload layer-`layer_idx` weights into per-layer BOs AND register bias.

    Mirrors `_llm_shared.phase_helpers.orchestration.preload_block_weights`
    for the rms_gemms_rope/o_ffn arg layout, then precomputes Qwen2 bias
    tensors for the same layer and registers them with the bias wrapper.
    """
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
        description="Qwen2.5-1.5B Phase 2 single-block test"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help=(
            "Phase 2 single-block default = 512. Constraints: (a) "
            "M=seq_len must be ≥ tile_m*herd_m=512 or GEMM produces "
            "garbage; (b) at seq_len=2048 the 6-launch ELF's BDs exhaust "
            "the shim-channel pool — Phase 3 prerequisite."
        ),
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
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
        help="Use NPU FlashAttention via Option C head-first wrapper (head_dim=128).",
    )
    parser.add_argument("--cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument(
        "--pad",
        dest="pad",
        action="store_true",
        default=None,
        help=(
            "Pad emb_dim 1536→2048 and hidden_dim 8960→9216 host-side "
            "to dodge the BD-allocator blowup at seq_len=2048. Default: "
            "auto-enabled at seq_len ≥ 1024 (where the unpadded path "
            "doesn't compile)."
        ),
    )
    parser.add_argument(
        "--no-pad",
        dest="pad",
        action="store_false",
        help="Force the unpadded path (works only at seq_len ≤ 1024).",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    print(
        f"Qwen2.5-1.5B config: n_layers={orig_config.n_layers}, "
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

    # Pad-mode auto-enable: required at seq_len ≥ 1024 (BD blowup).
    use_pad = args.pad if args.pad is not None else (args.seq_len >= 1024)
    if use_pad:
        config = make_padded_config(
            orig_config,
            padded_emb_dim=2048,
            padded_hidden_dim=9216,
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
    _compile_qwen25_block_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)
    print(f"  Kernel compile: {time.time()-t:.1f}s")

    # Install Qwen2 QKV bias wrapper (idempotent; no-op for layers we don't register).
    install_qkv_bias_wrapper()

    if not args.no_preload:
        print("\nPre-loading layer-0 weights into BOs and registering bias...")
        try:
            _preload_qwen25_block_weights(
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
    # CPU reference uses the ORIGINAL (unpadded) config and weights so the
    # comparison is against the true Qwen2.5 forward, not the padded variant.
    t = time.time()
    orig_x_f32 = np.asarray(orig_weights.embed_table, dtype=np.float32)[token_ids]
    orig_rope_lut_bf16 = generate_rope_lut(
        config=orig_config, seq_len=args.seq_len, dtype=bfloat16
    )
    rope_lut_f32 = np.asarray(orig_rope_lut_bf16, dtype=np.float32)
    ref_out, _ = qwen25_reference.transformer_block(
        orig_x_f32, orig_weights.layers[0], rope_lut_f32, orig_config
    )
    print(f"  CPU reference: {time.time()-t:.2f}s")

    # Slice padded NPU output back to orig_emb_dim for apples-to-apples.
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
    print(f"Phase 2 — single-block correctness")
    print(f"{'='*60}")
    print(
        f"  attention   = "
        f"{'CPU fallback' if args.cpu_attn else 'NPU FlashAttention (head-first via Option C)'}"
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

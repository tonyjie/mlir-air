# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-1.5B end-to-end NPU inference (NPU prefill + NPU decode).

Pipeline:
    embed (CPU)
    --- prefill (per prompt) — PADDED shapes (emb=2048, hidden=9216) ---
    -> 28x [rms_gemms_rope (NPU 6-launch ELF, bias added on host) ->
            {NPU FA via Option C | CPU attn} -> o_ffn (NPU 8-launch ELF)]
       extracting per-layer K (post-RoPE) and V into KV cache
    -> slice last hidden state from padded emb_dim back to orig emb_dim
    -> CPU final RMSNorm at last prompt position (orig shapes)
    -> NPU LM Head GEMV (10×16384 partition, vocab=151936)
    -> argmax -> first generated token

    --- decode (per token) — ORIG shapes (emb=1536, hidden=8960) ---
    -> 28x [rms_gemv_rope (NPU 6-launch ELF, bias added on host
            via set_decode_position(pos)) ->
            CPU attention (KV cache) -> o_gemv_ffn (NPU 8-launch ELF)]
    -> CPU final RMSNorm + NPU LM Head GEMV -> argmax

Why split shapes between prefill (padded) and decode (orig):
    Prefill at seq_len=2048 needs padding to dodge the BD-pool exhaustion
    on emb_dim=1536 (LESSON 3) — done via `qwen25_pad.pad_weights` with
    GQA-aware reindexing (LESSON 4). Decode at M=1 doesn't have that
    blowup, so we keep orig shapes for simplicity. The KV cache layout
    (n_kv_heads × max_seq × head_dim) is invariant under emb_dim padding,
    so the cache built during padded prefill is consumed by orig decode
    without any conversion.
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
    preload_prefill_weights,
)
from llama3_decode import run_decode_block

from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
from _llm_shared.phase_helpers.decode_setup import pre_transpose_decode_weights
from _llm_shared.phase_helpers.prefill_runner import npu_prefill_with_kv_extraction

from qwen25_bias import install_qkv_bias_wrapper, set_decode_position
from qwen25_pad import make_padded_config, pad_weights, slice_output
from qwen25_phase2_test import _compile_qwen25_block_kernels
from qwen25_phase3_test import _register_all_layer_biases
from qwen25_phase5_test import _register_decode_biases
from qwen25_decode_setup import (
    compile_qwen25_decode_kernels,
    qwen25_npu_lm_head_gemv,
    preload_qwen25_lm_head,
)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B end-to-end NPU inference (prefill + decode)"
    )
    parser.add_argument("--n-tokens", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--prefill-cache-dir", type=str, default="prefill_kernel_cache")
    parser.add_argument("--decode-cache-dir", type=str, default="decode_kernel_cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--cpu-attn",
        dest="cpu_attn",
        action="store_true",
        default=False,
        help="Use CPU attention in prefill (default: NPU FA via Option C)",
    )
    parser.add_argument(
        "--npu-attn",
        dest="cpu_attn",
        action="store_false",
        help="Use NPU FA Option C head-first wrapper (default).",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print per-token decode timings"
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    orig_config = LlamaConfig()
    padded_config = make_padded_config(
        orig_config, padded_emb_dim=2048, padded_hidden_dim=9216
    )
    print("Qwen2.5-1.5B end-to-end NPU inference")
    print(
        f"  prefill (padded): emb={padded_config.emb_dim}, hidden={padded_config.hidden_dim}, "
        f"n_heads={padded_config.n_heads} (GQA-reindexed, group_padded=8)"
    )
    print(
        f"  decode (orig):    emb={orig_config.emb_dim}, hidden={orig_config.hidden_dim}, "
        f"n_heads={orig_config.n_heads}, vocab={orig_config.vocab_size}"
    )
    print(
        f"  attn={'CPU' if args.cpu_attn else 'NPU FA (Option C)'}, "
        f"seq_len={args.seq_len}, n_tokens={args.n_tokens}"
    )

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

    compile_all_external_kernels(head_dim=padded_config.head_dim)
    _compile_qwen25_block_kernels(
        prefill_cache, padded_config, args.seq_len, cpu_attn=args.cpu_attn
    )
    needed_decode = ["rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"]
    if not all(k in decode_cache.artifacts for k in needed_decode):
        compile_qwen25_decode_kernels(decode_cache, orig_config)
    t_compile = time.time() - t
    print(f"  Compile / cache load: {t_compile:.1f}s")

    if args.compile_only:
        print("--compile-only: kernels compiled, exiting before weight load.")
        return 0

    print("\n[setup] Weights...")
    t = time.time()
    orig_weights = load_weights(args.model, config=orig_config)
    padded_weights = pad_weights(orig_weights, orig_config, padded_config)
    t_load = time.time() - t
    print(f"  Weight load + pad: {t_load:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    padded_rope_bf16 = generate_rope_lut(
        config=padded_config, seq_len=args.seq_len, dtype=bfloat16
    )
    decode_rope_bf16 = generate_rope_lut(
        config=orig_config, seq_len=args.seq_len, dtype=bfloat16
    )

    print("\n[setup] Pre-loading BOs and bias registry...")
    t = time.time()
    install_qkv_bias_wrapper()
    # Prefill bias (padded shapes): registered first; will be OVERRIDDEN
    # in-place when we register decode bias (orig shapes) below. The
    # cache.load_and_run patch dispatches by kernel name (rms_gemms_rope
    # vs rms_gemv_rope), but the registered bias TENSORS are shared by
    # layer_idx — we need different shapes for prefill (padded n_heads=16)
    # vs decode (orig n_heads=12). Solution: register prefill biases,
    # run prefill (uses them), then re-register decode biases for the
    # decode loop. See _PHASE_LIVE_BIAS comment below.
    _register_all_layer_biases(
        padded_weights, padded_config, padded_rope_bf16, args.seq_len
    )
    pre_transpose_decode_weights(orig_weights, orig_config)
    preload_prefill_weights(
        padded_weights, padded_config, prefill_cache, args.seq_len, padded_rope_bf16
    )
    preload_qwen25_lm_head(decode_cache, orig_weights, orig_config)
    t_preload = time.time() - t
    print(f"  BO preload + bias: {t_preload:.1f}s")

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

    print(f"\n[1/3] NPU prefill ({padded_config.n_layers} layers, padded shapes)...")
    t = time.time()
    x_prefill, k_cache, v_cache = npu_prefill_with_kv_extraction(
        token_ids,
        padded_weights,
        padded_config,
        prefill_cache,
        padded_rope_bf16,
        max_seq=args.seq_len,
        cpu_attn=args.cpu_attn,
    )
    t_prefill = time.time() - t
    print(
        f"  NPU prefill: {t_prefill:.2f}s  "
        f"({t_prefill/padded_config.n_layers*1000:.0f} ms/layer)"
    )

    # Slice padded last hidden state back to orig emb_dim for the LM head
    # GEMV (which is built for orig shapes).
    t = time.time()
    last_hidden_padded = np.asarray(x_prefill, dtype=np.float32)[
        pred_pos : pred_pos + 1
    ]
    last_hidden = slice_output(last_hidden_padded, orig_config.emb_dim)
    last_normed_bf16 = (
        qwen25_reference.rms_norm(last_hidden, orig_weights.final_norm)
        .flatten()
        .astype(bfloat16)
    )
    first_logits = qwen25_npu_lm_head_gemv(
        decode_cache, orig_weights, orig_config, last_normed_bf16
    )
    first_token = int(np.argmax(first_logits))
    t_first_lm_head = time.time() - t
    print(
        f"  First LM Head GEMV: {t_first_lm_head*1000:.0f} ms  -> "
        f"{tokenizer.decode([first_token])!r} (id={first_token})"
    )

    # Re-register bias for decode (orig n_heads=12, sized to seq_len for full
    # context window).
    print(f"\n[setup] Re-registering bias for decode (orig shapes)...")
    _register_decode_biases(orig_weights, orig_config, decode_rope_bf16, args.seq_len)

    print(f"\n[2/3] NPU decode loop ({args.n_tokens} tokens)...")
    embed_table_f32 = np.asarray(orig_weights.embed_table, dtype=np.float32)
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
        for layer_idx in range(orig_config.n_layers):
            orig_weights.layers[layer_idx]._layer_idx = layer_idx
            set_decode_position(current_pos)
            x_bf16 = run_decode_block(
                x_bf16,
                orig_weights.layers[layer_idx],
                decode_cache,
                orig_config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                decode_rope_bf16,
            )
        x_f32 = np.asarray(x_bf16, dtype=np.float32).reshape(1, orig_config.emb_dim)
        x_normed_bf16 = (
            qwen25_reference.rms_norm(x_f32, orig_weights.final_norm)
            .flatten()
            .astype(bfloat16)
        )
        next_logits = qwen25_npu_lm_head_gemv(
            decode_cache, orig_weights, orig_config, x_normed_bf16
        )
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

    set_decode_position(None)

    print(f"\n[3/3] Done.\n{'='*60}")
    print(f"Generated text:")
    print(f"  {tokenizer.decode(generated)!r}")
    print()
    print(f"Timings (one-time setup):")
    print(f"  Compile / cache load : {t_compile:.1f}s")
    print(f"  Weight load + pad    : {t_load:.1f}s")
    print(f"  BO preload + bias    : {t_preload:.1f}s")
    print(f"Timings (per inference):")
    print(
        f"  NPU prefill ({padded_config.n_layers}L)    : {t_prefill:.2f}s "
        f"({t_prefill/padded_config.n_layers*1000:.0f} ms/layer)"
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

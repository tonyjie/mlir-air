#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache.
Runs prefill first to populate KV cache, then decodes token-by-token.

Usage:
    cd build_peano
    python3 ../llama3_decode.py --compile-only
    python3 ../llama3_decode.py --run-only --n-tokens 10 --profile
    python3 ../llama3_decode.py --run-only --n-tokens 1 --verify
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama3_weights import LlamaConfig, load_weights, generate_rope_lut
from llama3_prefill import KernelCache, prepare_air_project, _build_gemm_module

# ---------------------------------------------------------------------------
# GEMV module builder (wraps matrix_vector_multiplication)
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "matrix_vector_multiplication", "bf16"
    ),
)
from matvec import build_module as build_gemv_module

# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------

# Backend kwargs for different kernel types
_GEMV_K2048_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "",  # ping-pong ON for K=2048
    "runtime_loop_tiling_sizes": [16, 16],
    "use_lock_race_condition_fix": False,
}

_GEMV_K8192_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",  # ping-pong OFF for K=8192 (L1 too tight)
    "runtime_loop_tiling_sizes": [16, 16],
    "use_lock_race_condition_fix": False,
}

_SIMPLE_BACKEND = {"omit_while_true_loop": False}


def compile_decode_kernels(cache, config):
    """Compile all unique decode kernel configs."""
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling decode kernels...")
    print(f"{'='*60}\n")

    # 1. QKV GEMV multi-launch: Q + K + V in one ELF (3 launches)
    from llama3.multi_launch_builder.rms_qkv_gemv_multi import build_rms_qkv_gemv_module

    cache.compile_and_cache(
        "qkv_gemv",
        build_rms_qkv_gemv_module(emb_dim, kv_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "qkv_gemv",
            **_GEMV_K2048_BACKEND,
        },
    )

    # 2. O GEMV + Add multi-launch: O projection + residual (2 launches)
    from llama3.multi_launch_builder.o_gemv_add_multi import build_o_gemv_add_module

    cache.compile_and_cache(
        "o_gemv_add",
        build_o_gemv_add_module(emb_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "o_gemv_add",
            **_GEMV_K2048_BACKEND,
        },
    )

    # 3. Gate + Up GEMV multi-launch (2 launches)
    from llama3.multi_launch_builder.ffn_gemv_multi import build_gate_up_gemv_module

    cache.compile_and_cache(
        "gate_up_gemv",
        build_gate_up_gemv_module(emb_dim, hidden_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "gate_up_gemv",
            **_GEMV_K2048_BACKEND,
        },
    )

    # 4. Down GEMV (single, K=8192)
    cache.compile_and_cache(
        "gemv_down",
        build_gemv_module(emb_dim, hidden_dim, 2, 1, 8, bfloat16, bfloat16),
        {"verbose": cache.verbose, **_GEMV_K8192_BACKEND},
    )

    # 5. RMSNorm: M=1, N=2048
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    cache.compile_and_cache(
        "rmsnorm",
        build_rms(1, emb_dim, bfloat16, 16),
        {"verbose": cache.verbose, **_SIMPLE_BACKEND},
    )

    # 6. Eltwise Add: n=2048
    from eltwise_add.eltwise_add import build_module as build_add

    cache.compile_and_cache(
        "add",
        build_add(emb_dim, 1024, bfloat16, vector_size=16, herd_x=1, herd_y=1),
        {"verbose": cache.verbose, **_SIMPLE_BACKEND},
    )

    # 7. RoPE Q: (32, 64)
    from rope_lut.rope_lut import build_module as build_rope

    cache.compile_and_cache(
        "rope_q",
        build_rope(n_heads, head_dim, bfloat16),
        {"verbose": cache.verbose, **_SIMPLE_BACKEND},
    )

    # 8. RoPE K: (8, 64)
    cache.compile_and_cache(
        "rope_k",
        build_rope(n_kv_heads, head_dim, bfloat16),
        {"verbose": cache.verbose, **_SIMPLE_BACKEND},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(
    q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim
):
    """Single-query attention with KV cache.

    Args:
        q: (emb_dim,) — query vector for current token
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1]
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values [0:current_pos+1]
        current_pos: current token position (0-indexed)
        n_heads: number of Q heads (32)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (64)

    Returns:
        attn_out: (emb_dim,) — attention output
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)  # (n_kv, seq, hd)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale  # (seq,)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]  # (hd,)

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
):
    """Run one transformer block for a single decode token.

    Args:
        x_bf16: (emb_dim,) input token embedding
        layer_weights: LayerWeights for this layer
        cache: KernelCache
        config: LlamaConfig
        k_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's K cache
        v_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's V cache
        current_pos: current token position
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT

    Returns:
        output: (emb_dim,) — block output
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = None  # Set by caller via layer_weights._layer_idx
    if hasattr(layer_weights, "_layer_idx"):
        layer_idx = layer_weights._layer_idx

    def _run(name, backend, *inputs, **kwargs):
        return cache.load_and_run(name, backend, *inputs, **kwargs)

    _QKV_BACKEND = {
        "output_format": "elf",
        "instance_name": "qkv_gemv",
        **_GEMV_K2048_BACKEND,
    }
    _OA_BACKEND = {
        "output_format": "elf",
        "instance_name": "o_gemv_add",
        **_GEMV_K2048_BACKEND,
    }
    _GU_BACKEND = {
        "output_format": "elf",
        "instance_name": "gate_up_gemv",
        **_GEMV_K2048_BACKEND,
    }

    # 1. RMSNorm (pre-attn)
    x_in = x_bf16.reshape(1, emb_dim).astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    y_out = np.zeros((1, emb_dim), dtype=bfloat16)
    results = _run("rmsnorm", _SIMPLE_BACKEND, x_in, w_norm, y_out)
    normed = results[-1].flatten().astype(bfloat16)

    # 2-4. Q/K/V GEMV (3-launch ELF, single XRT call)
    wq = layer_weights._wq_t
    wk = layer_weights._wk_t
    wv = layer_weights._wv_t
    q_out = np.zeros(emb_dim, dtype=bfloat16)
    k_out = np.zeros(kv_dim, dtype=bfloat16)
    v_out = np.zeros(kv_dim, dtype=bfloat16)

    results = _run(
        "qkv_gemv",
        _QKV_BACKEND,
        normed,
        wq,
        q_out,
        wk,
        k_out,
        wv,
        v_out,
        output_indices=[2, 4, 6],
    )
    q = results[2].astype(bfloat16)
    k = results[4].astype(bfloat16)
    v = results[6].astype(bfloat16)

    # 5-6. RoPE on Q and K (single position)
    q_heads = q.reshape(n_heads, head_dim)  # (32, 64)
    k_heads = k.reshape(n_kv_heads, head_dim)  # (8, 64)

    # LUT for current position only
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, 64)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)  # (32, 64)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)  # (8, 64)

    q_roped_out = np.zeros(n_heads * head_dim, dtype=bfloat16)
    k_roped_out = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

    results = _run("rope_q", _SIMPLE_BACKEND, q_heads.flatten(), lut_q, q_roped_out)
    q_roped = results[-1].reshape(n_heads, head_dim).astype(bfloat16)
    results = _run("rope_k", _SIMPLE_BACKEND, k_heads.flatten(), lut_k, k_roped_out)
    k_roped = results[-1].reshape(n_kv_heads, head_dim).astype(bfloat16)

    # Update KV cache
    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # 7. CPU Attention
    attn_out = decode_attention_cpu(
        q_roped.flatten(),
        k_cache_layer,
        v_cache_layer,
        current_pos,
        n_heads,
        n_kv_heads,
        head_dim,
    )

    # 8-9. O GEMV + Residual Add (2-launch ELF)
    wo = layer_weights._wo_t
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = x_bf16.flatten().astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    results = _run(
        "o_gemv_add",
        _OA_BACKEND,
        wo,
        attn_out,
        proj_buf,
        x_residual,
        res1_buf,
        output_indices=[4],
    )
    res1 = results[4].astype(bfloat16)

    # 10. RMSNorm (pre-FFN)
    x_in2 = res1.reshape(1, emb_dim).astype(bfloat16)
    w_norm2 = layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16)
    y_out2 = np.zeros((1, emb_dim), dtype=bfloat16)
    results = _run("rmsnorm", _SIMPLE_BACKEND, x_in2, w_norm2, y_out2)
    normed2 = results[-1].flatten().astype(bfloat16)

    # 11-12. Gate + Up GEMV (2-launch ELF)
    w_gate = layer_weights._wgate_t
    w_up = layer_weights._wup_t
    gate_out = np.zeros(hidden_dim, dtype=bfloat16)
    up_out = np.zeros(hidden_dim, dtype=bfloat16)

    results = _run(
        "gate_up_gemv",
        _GU_BACKEND,
        w_gate,
        normed2,
        gate_out,
        w_up,
        up_out,
        output_indices=[2, 4],
    )
    gate = results[2].astype(bfloat16)
    up = results[4].astype(bfloat16)

    # 13. SiLU × mul (CPU for now — decode size is tiny, ~8192 elements)
    gate_f32 = gate.astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-gate_f32))
    swiglu = (gate_f32 * sigmoid * up.astype(np.float32)).astype(bfloat16)

    # 14. Down GEMV
    w_down = layer_weights._wdown_t
    down_out = np.zeros(emb_dim, dtype=bfloat16)
    results = _run(
        "gemv_down",
        _GEMV_K8192_BACKEND,
        w_down,
        swiglu,
        down_out,
    )
    down = results[-1].astype(bfloat16)

    # 15. Residual Add
    c_add2 = np.zeros(emb_dim, dtype=bfloat16)
    results = _run("add", _SIMPLE_BACKEND, res1, down, c_add2)
    output = results[-1].astype(bfloat16)

    return output


# ---------------------------------------------------------------------------
# Full decode generation
# ---------------------------------------------------------------------------


def generate(
    prompt_tokens,
    weights,
    config,
    cache,
    rope_lut_bf16,
    n_tokens=10,
    profile=False,
    verify=False,
):
    """Run prefill + multi-token decode generation.

    Args:
        prompt_tokens: list of token IDs
        weights: LlamaWeights
        config: LlamaConfig
        cache: KernelCache
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT
        n_tokens: number of tokens to generate
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens

    print(f"\n{'='*60}")
    print(f"LLAMA Decode: prompt_len={seq_len}, n_tokens={n_tokens}")
    print(f"{'='*60}\n")

    # --- Prefill phase (CPU reference for KV cache) ---
    print("Running CPU prefill for KV cache...")
    from llama3_reference import transformer_block as cpu_block, rms_norm

    t_prefill_start = time.time()

    x_embed = weights.embed_table[prompt_tokens].astype(bfloat16)
    rope_lut_f32 = rope_lut_bf16[:seq_len].astype(np.float32)

    # KV cache: stores post-RoPE K and raw V
    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    x = x_embed.astype(np.float32)
    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        x_out, intermediates = cpu_block(x, lw, rope_lut_f32, config)

        # Use post-RoPE K from CPU reference intermediates
        k_roped = intermediates["k_roped"].astype(np.float32)  # (seq, kv_dim)
        v_raw = intermediates["v"].astype(np.float32)  # (seq, kv_dim)

        # Store in cache: (n_kv_heads, seq, head_dim)
        k_cache[layer_idx, :, :seq_len, :] = (
            k_roped.reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        v_cache[layer_idx, :, :seq_len, :] = (
            v_raw.reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        x = x_out

    # Final norm + LM Head (CPU)
    x_normed = rms_norm(x, weights.final_norm.astype(np.float32))
    logits = x_normed @ weights.lm_head.astype(np.float32).T
    # Find actual prompt length (before EOS padding)
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_len = len([t for t in prompt_tokens if t != _tok.eos_token_id])
    pred_pos = prompt_len - 1
    prefill_token = int(np.argmax(logits[pred_pos]))

    t_prefill = time.time() - t_prefill_start
    print(f"Prefill done in {t_prefill:.2f}s. First decode token: {prefill_token}")

    # The decode starts from the prefill's predicted token
    # (not from the hidden state — we embed the predicted token)

    # Pre-transpose all weights for GEMV (done once, not per-token)
    kv_dim = n_kv_heads * head_dim
    hidden_dim = config.hidden_dim
    if not hasattr(weights, "_decode_weights_transposed"):
        print("Pre-transposing weights for GEMV...")
        for lw in weights.layers:
            lw._wq_t = np.ascontiguousarray(
                lw.wq.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wo_t = np.ascontiguousarray(
                lw.wo.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wgate_t = np.ascontiguousarray(
                lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wup_t = np.ascontiguousarray(
                lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wdown_t = np.ascontiguousarray(
                lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
            )
        weights._decode_weights_transposed = True
    # Tag layers with index for per-layer kernel name isolation
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # --- Decode phase ---
    generated_tokens = [prefill_token]
    current_pos = prompt_len  # Next position to fill (after actual prompt)

    # Use the prefill_token as input to first decode step
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    print(f"\nDecoding {n_tokens} tokens...")
    t_decode_start = time.time()

    for token_idx in range(n_tokens):
        t_token_start = time.perf_counter()

        # Run 16 transformer blocks
        x = x_decode.copy()
        for layer_idx in range(config.n_layers):
            x = run_decode_block(
                x,
                weights.layers[layer_idx],
                cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )

        # Final RMSNorm (CPU)
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, emb_dim),
            weights.final_norm.astype(np.float32),
        )

        # LM Head (CPU for now)
        logits = x_normed @ weights.lm_head.astype(np.float32).T  # (1, vocab)
        next_token = int(np.argmax(logits[0]))

        t_token = time.perf_counter() - t_token_start

        generated_tokens.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if profile:
            print(f"  Token {token_idx}: id={next_token}, time={t_token*1000:.0f}ms")

    t_decode = time.time() - t_decode_start

    print(f"\nGenerated {n_tokens} tokens in {t_decode:.2f}s")
    print(f"Tokens/second: {n_tokens / t_decode:.2f}")
    print(f"Time/token: {t_decode / n_tokens * 1000:.0f}ms")

    return generated_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-1B Decode")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
    )
    args = parser.parse_args()

    # Copy external kernel .o files to cwd so aircc can find them.
    # aircc's prepare_air_project() copies *.o from cwd to air_project/.
    import shutil
    from pathlib import Path

    mv_src = (
        Path(__file__).parent.parent
        / "matrix_vector_multiplication"
        / "bf16"
        / "build_peano"
        / "mv.o"
    )
    if mv_src.exists() and not Path("mv.o").exists():
        shutil.copy2(mv_src, "mv.o")
    rope_src = (
        Path(__file__).parent.parent
        / "rope_lut"
        / "test_llama_dims"
        / "build_peano"
        / "rope.o"
    )
    if rope_src.exists() and not Path("rope.o").exists():
        shutil.copy2(rope_src, "rope.o")
    config = LlamaConfig()

    cache_dir = "decode_kernel_cache"
    cache = KernelCache(cache_dir, verbose=args.verbose)

    if not args.run_only:
        compile_decode_kernels(cache, config)

    if args.compile_only:
        sys.exit(0)

    if args.run_only:
        cache.load_manifest()

    # Load weights and tokenizer
    print("\nLoading weights...")
    weights = load_weights("meta-llama/Llama-3.2-1B")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_tokens = tokenizer.encode(args.prompt)
    # Pad to seq_len=2048
    seq_len = 2048
    if len(prompt_tokens) < seq_len:
        prompt_tokens = prompt_tokens + [tokenizer.eos_token_id] * (
            seq_len - len(prompt_tokens)
        )

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    generated = generate(
        prompt_tokens,
        weights,
        config,
        cache,
        rope_lut_bf16,
        n_tokens=args.n_tokens,
        profile=args.profile,
        verify=args.verify,
    )

    # Decode tokens
    print(f"\n{'='*60}")
    print(f"Generated text:")
    print(f"{'='*60}")
    all_tokens = prompt_tokens[: len(tokenizer.encode(args.prompt))] + generated
    print(tokenizer.decode(all_tokens))

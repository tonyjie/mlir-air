# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2-style QKV bias support for the shared rms_gemms_rope multi-launch ELF.

Background
----------
Qwen2-family models add a 1-D bias to each Q/K/V projection BEFORE RoPE:

    q[s, h*hd:(h+1)*hd] = (normed[s] @ wq)[h*hd:(h+1)*hd] + bq[h*hd:(h+1)*hd]
    k[s, h*hd:(h+1)*hd] = (normed[s] @ wk)[h*hd:(h+1)*hd] + bk[h*hd:(h+1)*hd]
    v[s, h*hd:(h+1)*hd] = (normed[s] @ wv)[h*hd:(h+1)*hd] + bv[h*hd:(h+1)*hd]
    q_roped = RoPE(q),  k_roped = RoPE(k)

The shared `rms_gemms_rope` ELF (`llama3/multi_launch_builder/`) was designed
for bias-free LlamaForCausalLM models and emits the bias-free path:
    q_roped = RoPE(normed @ wq)

To avoid forking the shared multi-launch builder, we exploit RoPE's
**linearity** (it's a per-position 2-D rotation, hence linear):

    RoPE(q + bq) = RoPE(q) + RoPE(bq)

So we can run the ELF unmodified to get `q_roped_unbiased`, then add the
**pre-RoPE'd** bias on the host:

    q_roped_qwen2 = q_roped_unbiased + RoPE(broadcast(bq, seq_len))

The pre-RoPE'd biases (`bq_roped`, `bk_roped`) are precomputed once per layer
at preload time and stored in the layer registry; they're tiny (~6 MB Q, ~1 MB
K per layer).

V doesn't get RoPE, so V's bias is just a 1-D broadcast-add.

This file installs a `_run_cached` monkey-patch (same pattern as
`_llm_shared/phase_helpers/headfirst_fa.py`) that intercepts every
`rms_gemms_rope` invocation, looks up the per-layer biases via the
`bo_key=f"rms_gemms_rope_L{layer_idx}"` convention, and adds them to the
ELF's outputs.

Public API
----------
- `precompute_rope_bias(b, rope_lut, n_heads, head_dim, seq_len)` — RoPE-rotate
  a 1-D Q or K bias into a (seq_len, n_heads*head_dim) tensor.
- `register_layer_bias(layer_idx, bq_roped, bk_roped, bv)` — store per-layer
  pre-RoPE'd Q/K bias and V bias for the wrapper to consume.
- `clear_layer_bias()` — empty the registry (tests, between deployments).
- `install_qkv_bias_wrapper()` — install the idempotent `_run_cached` patch.
"""

import os
import numpy as np
from ml_dtypes import bfloat16

# Per-layer registry: layer_idx -> dict(bq_roped, bk_roped, bv)
# Populated by `register_layer_bias`, consumed by the patched `_run_cached`.
_LAYER_BIAS = {}

_INSTALLED = False


def precompute_rope_bias(b, rope_lut, n_heads, head_dim, seq_len):
    """Apply RoPE to a broadcast-over-positions copy of a 1-D Q or K bias.

    Args:
        b:        (n_heads * head_dim,) bias vector (numpy, any float dtype).
        rope_lut: (>= seq_len, head_dim) RoPE LUT in half-split [cos|sin]
                  layout (matches `rope_halfsplit.cc` and qwen25_weights.py).
        n_heads:  number of attention heads this bias belongs to (Q or KV).
        head_dim: per-head dimension.
        seq_len:  prefill sequence length.

    Returns:
        (seq_len, n_heads * head_dim) bfloat16 tensor — the per-position
        RoPE'd bias to add to the ELF's q_roped / k_roped output.
    """
    b_f32 = np.asarray(b, dtype=np.float32).reshape(n_heads, head_dim)
    lut = np.asarray(rope_lut[:seq_len], dtype=np.float32)
    half = head_dim // 2
    cos = lut[:, :half]  # (seq_len, half)
    sin = lut[:, half:]  # (seq_len, half)

    out = np.empty((seq_len, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        # Broadcast b_f32[h] across all seq_len positions.
        x1 = np.broadcast_to(b_f32[h, :half], (seq_len, half))  # (seq_len, half)
        x2 = np.broadcast_to(b_f32[h, half:], (seq_len, half))
        out[:, h, :half] = x1 * cos - x2 * sin
        out[:, h, half:] = x1 * sin + x2 * cos
    return out.reshape(seq_len, n_heads * head_dim).astype(bfloat16)


def register_layer_bias(layer_idx, bq_roped, bk_roped, bv):
    """Register pre-RoPE'd Q/K bias and the raw V bias for one layer.

    The patched `_run_cached("rms_gemms_rope", ..., bo_key="..._L{layer_idx}")`
    looks up these tensors and adds them to the corresponding ELF outputs.

    Args:
        layer_idx: integer layer index (matches the bo_key suffix).
        bq_roped:  (seq_len, n_heads * head_dim) bf16 — from
                   precompute_rope_bias(layer_weights.bq, ...)
        bk_roped:  (seq_len, n_kv_heads * head_dim) bf16 — from
                   precompute_rope_bias(layer_weights.bk, ...)
        bv:        (n_kv_heads * head_dim,) bf16 — V bias (no RoPE; broadcasts
                   over positions).
    """
    _LAYER_BIAS[int(layer_idx)] = {
        "bq_roped": np.ascontiguousarray(bq_roped),
        "bk_roped": np.ascontiguousarray(bk_roped),
        "bv": np.ascontiguousarray(np.asarray(bv, dtype=bfloat16)),
    }


def clear_layer_bias():
    """Empty the per-layer bias registry (tests, deployment switches)."""
    _LAYER_BIAS.clear()


# Decode position: set by the per-token loop before invoking run_decode_block.
# Used by the load_and_run patch when it sees a `rms_gemv_rope` call to slice
# the precomputed bq_roped / bk_roped to the right position.
_DECODE_POS = None


def set_decode_position(pos):
    """Tell the decode bias wrapper which position the next rms_gemv_rope is at."""
    global _DECODE_POS
    _DECODE_POS = int(pos) if pos is not None else None


def _apply_bias_to_results(results, bias, pos=None):
    """Add registered bias to (v, q_roped, k_roped) at results[8],[11],[12].

    For decode (pos given): slice bq_roped/bk_roped to [pos:pos+1].
    For prefill (pos=None): use the full bq_roped/bk_roped tensors.
    """
    bq_roped = bias["bq_roped"]
    bk_roped = bias["bk_roped"]
    bv = bias["bv"]
    if pos is not None:
        bq_roped = bq_roped[pos : pos + 1]
        bk_roped = bk_roped[pos : pos + 1]

    v = np.asarray(results[8])
    q_roped = np.asarray(results[11])
    k_roped = np.asarray(results[12])
    seq_dim = bq_roped.shape[0]
    q2 = q_roped.reshape(seq_dim, -1)
    k2 = k_roped.reshape(seq_dim, -1)
    v2 = v.reshape(seq_dim, -1)
    q2_new = (q2.astype(np.float32) + bq_roped.astype(np.float32)).astype(bfloat16)
    k2_new = (k2.astype(np.float32) + bk_roped.astype(np.float32)).astype(bfloat16)
    v2_new = (v2.astype(np.float32) + bv.astype(np.float32)).astype(bfloat16)
    results = list(results)
    results[8] = v2_new.reshape(v.shape)
    results[11] = q2_new.reshape(q_roped.shape)
    results[12] = k2_new.reshape(k_roped.shape)
    return results


def install_qkv_bias_wrapper():
    """Monkey-patch KernelCache.load_and_run to apply Qwen2 QKV bias.

    Catches both prefill (`rms_gemms_rope`, full bq_roped) and decode
    (`rms_gemv_rope`, sliced to current position via `set_decode_position`).

    Idempotent. Safe to leave installed across deployments — only fires for
    layers that have explicitly registered bias via `register_layer_bias`.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from llama3_prefill import KernelCache as _KC

    _orig_load_and_run = _KC.load_and_run

    def _patched_load_and_run(self, name, backend_kwargs, *inputs, **kwargs):
        results = _orig_load_and_run(self, name, backend_kwargs, *inputs, **kwargs)
        if name not in ("rms_gemms_rope", "rms_gemv_rope"):
            return results

        bo_key = kwargs.get("bo_key")
        if not bo_key:
            return results
        suffix = f"{name}_L"
        if not bo_key.startswith(suffix):
            return results
        try:
            layer_idx = int(bo_key.split("_L", 1)[1])
        except (ValueError, IndexError):
            return results
        bias = _LAYER_BIAS.get(layer_idx)
        if bias is None:
            if os.environ.get("QWEN25_BIAS_DEBUG"):
                print(
                    f"  [qwen25_bias] no bias registered for layer {layer_idx}; "
                    f"keys={list(_LAYER_BIAS.keys())}",
                    flush=True,
                )
            return results

        # Decode path: slice to current position.
        pos = _DECODE_POS if name == "rms_gemv_rope" else None
        if name == "rms_gemv_rope" and pos is None:
            raise RuntimeError(
                "qwen25_bias: rms_gemv_rope intercepted but no decode position "
                "set. Call qwen25_bias.set_decode_position(pos) before each "
                "run_decode_block invocation."
            )
        if os.environ.get("QWEN25_BIAS_DEBUG"):
            print(
                f"  [qwen25_bias] applying bias for {name} layer {layer_idx} pos={pos}",
                flush=True,
            )
        return _apply_bias_to_results(results, bias, pos=pos)

    _KC.load_and_run = _patched_load_and_run
    _INSTALLED = True

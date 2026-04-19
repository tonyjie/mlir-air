# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-1.5B Weight Loader.

Loads Qwen2.5-1.5B (Qwen2ForCausalLM) weights from HuggingFace safetensors
into numpy arrays usable by MLIR-AIR kernels.

Differences from the LlamaForCausalLM loader (`llama3_weights.py`,
`smollm2_weights.py`) handled here:

  - **QKV bias** (q_proj.bias / k_proj.bias / v_proj.bias) — Qwen2-family only.
    Loaded into `LayerWeights.bq / bk / bv`. Bias for o_proj is absent
    (matches Qwen2 convention).
  - 28 layers, head_dim=128, GQA group=6 (12 Q / 2 KV heads).
  - vocab=151936, rope_base=1e6.
  - tie_word_embeddings=True → same fallback path as Llama-3.2-1B.
  - rms_norm_eps=1e-6 (vs Llama's 1e-5) — used by the CPU reference.

Weight convention:
    HuggingFace stores linear weights as (out_features, in_features).
    Our GEMM convention is y = x @ W, so we transpose to (in_features, out_features).
    Biases are 1-D (out_features,) and not transposed.
"""

import os
import glob as glob_module
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    """Qwen2.5-1.5B model hyperparameters (Qwen2ForCausalLM architecture).

    Class kept named `LlamaConfig` so existing helpers (`llama3_inference`,
    `_llm_shared` infra) accept it unchanged. The `qkv_bias` and
    `rms_norm_eps` fields are Qwen2-specific extensions.
    """

    n_layers: int = 28
    emb_dim: int = 1536
    n_heads: int = 12
    head_dim: int = 128
    n_kv_heads: int = 2  # GQA: 6 Q heads per KV head
    hidden_dim: int = 8960
    vocab_size: int = 151936
    rope_base: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = True
    dtype: np.dtype = bfloat16


# ---------------------------------------------------------------------------
# Per-layer weight container
# ---------------------------------------------------------------------------


@dataclass
class LayerWeights:
    """Weight matrices + biases for a single Qwen2 transformer layer.

    All projection weights follow y = x @ W convention (in, out).
    Biases (Qwen2-only) are 1-D (out,). o_proj has no bias in Qwen2.

    Attributes:
        attn_norm:  (emb_dim,)                    RMSNorm weight for attention
        wq:         (emb_dim, n_heads*head_dim)
        wk:         (emb_dim, n_kv_heads*head_dim)
        wv:         (emb_dim, n_kv_heads*head_dim)
        bq:         (n_heads*head_dim,)           Q bias  (NEW vs llama3)
        bk:         (n_kv_heads*head_dim,)        K bias  (NEW vs llama3)
        bv:         (n_kv_heads*head_dim,)        V bias  (NEW vs llama3)
        wo:         (emb_dim, emb_dim)            O projection (no bias)
        ffn_norm:   (emb_dim,)                    RMSNorm weight for FFN
        w_gate:     (emb_dim, hidden_dim)
        w_up:       (emb_dim, hidden_dim)
        w_down:     (hidden_dim, emb_dim)
    """

    attn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    bq: np.ndarray
    bk: np.ndarray
    bv: np.ndarray
    wo: np.ndarray
    ffn_norm: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


# ---------------------------------------------------------------------------
# Full model weight container
# ---------------------------------------------------------------------------


@dataclass
class LlamaWeights:
    """All weights for a Qwen2.5-1.5B model.

    Naming kept as LlamaWeights for downstream-helper compatibility.
    """

    embed_table: np.ndarray  # (151936, 1536)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (1536,)
    lm_head: np.ndarray = None  # tied to embed_table


# ---------------------------------------------------------------------------
# HuggingFace name mapping
# ---------------------------------------------------------------------------

# Standard Qwen2/Llama parameter naming. Bias entries are added below.
_HF_LAYER_MAP = {
    "input_layernorm.weight": ("attn_norm", False),
    "self_attn.q_proj.weight": ("wq", True),
    "self_attn.k_proj.weight": ("wk", True),
    "self_attn.v_proj.weight": ("wv", True),
    "self_attn.o_proj.weight": ("wo", True),
    "post_attention_layernorm.weight": ("ffn_norm", False),
    "mlp.gate_proj.weight": ("w_gate", True),
    "mlp.up_proj.weight": ("w_up", True),
    "mlp.down_proj.weight": ("w_down", True),
}

# Qwen2 QKV bias entries — 1-D, no transpose.
_HF_LAYER_BIAS_MAP = {
    "self_attn.q_proj.bias": "bq",
    "self_attn.k_proj.bias": "bk",
    "self_attn.v_proj.bias": "bv",
}


# ---------------------------------------------------------------------------
# Safetensors loading helpers (mirrored from llama3_weights.py)
# ---------------------------------------------------------------------------


def _resolve_safetensor_files(model_path: str) -> List[str]:
    if os.path.isdir(model_path):
        pattern = os.path.join(model_path, "*.safetensors")
        files = sorted(glob_module.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        return files

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        model_path,
        allow_patterns=["*.safetensors", "*.json"],
    )
    pattern = os.path.join(local_dir, "*.safetensors")
    files = sorted(glob_module.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found after downloading {model_path}"
        )
    return files


def _load_tensor(file_handle, key: str, dtype) -> np.ndarray:
    tensor = file_handle.get_tensor(key)
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.astype(dtype)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_weights(
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[LlamaConfig] = None,
) -> LlamaWeights:
    """Load Qwen2.5-1.5B weights from safetensors into numpy arrays."""
    from safetensors import safe_open

    if config is None:
        config = LlamaConfig()

    safetensor_files = _resolve_safetensor_files(model_name_or_path)

    key_to_file = {}
    for filepath in safetensor_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    # --- embedding table ---
    embed_key = "model.embed_tokens.weight"
    if embed_key not in key_to_file:
        raise KeyError(f"Missing weight: {embed_key}")
    with safe_open(key_to_file[embed_key], framework="numpy") as f:
        embed_table = _load_tensor(f, embed_key, dtype)
    assert embed_table.shape == (config.vocab_size, config.emb_dim), (
        f"embed_table shape mismatch: expected "
        f"({config.vocab_size}, {config.emb_dim}), got {embed_table.shape}"
    )

    # --- per-layer weights + Qwen2 QKV biases ---
    layers = []
    for layer_idx in range(config.n_layers):
        layer_tensors = {}

        for hf_suffix, (field_name, needs_transpose) in _HF_LAYER_MAP.items():
            hf_key = f"model.layers.{layer_idx}.{hf_suffix}"
            if hf_key not in key_to_file:
                raise KeyError(f"Missing weight for layer {layer_idx}: {hf_key}")
            with safe_open(key_to_file[hf_key], framework="numpy") as f:
                tensor = _load_tensor(f, hf_key, dtype)
            if needs_transpose:
                tensor = np.ascontiguousarray(tensor.T)
            layer_tensors[field_name] = tensor

        # Qwen2-specific: q/k/v biases.
        for hf_suffix, field_name in _HF_LAYER_BIAS_MAP.items():
            hf_key = f"model.layers.{layer_idx}.{hf_suffix}"
            if hf_key not in key_to_file:
                if config.qkv_bias:
                    raise KeyError(
                        f"Layer {layer_idx} missing expected QKV bias: {hf_key}"
                    )
                # If qkv_bias is False but loader still constructs zeros, use them.
                expected_dim = (
                    config.n_heads * config.head_dim
                    if field_name == "bq"
                    else config.n_kv_heads * config.head_dim
                )
                layer_tensors[field_name] = np.zeros((expected_dim,), dtype=dtype)
                continue
            with safe_open(key_to_file[hf_key], framework="numpy") as f:
                layer_tensors[field_name] = _load_tensor(f, hf_key, dtype)

        layer = LayerWeights(**layer_tensors)

        # Sanity-check shapes.
        assert layer.attn_norm.shape == (
            config.emb_dim,
        ), f"Layer {layer_idx} attn_norm: {layer.attn_norm.shape}"
        assert layer.wq.shape == (
            config.emb_dim,
            config.n_heads * config.head_dim,
        ), f"Layer {layer_idx} wq: {layer.wq.shape}"
        assert layer.wk.shape == (
            config.emb_dim,
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} wk: {layer.wk.shape}"
        assert layer.wv.shape == (
            config.emb_dim,
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} wv: {layer.wv.shape}"
        assert layer.bq.shape == (
            config.n_heads * config.head_dim,
        ), f"Layer {layer_idx} bq: {layer.bq.shape}"
        assert layer.bk.shape == (
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} bk: {layer.bk.shape}"
        assert layer.bv.shape == (
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} bv: {layer.bv.shape}"
        assert layer.wo.shape == (
            config.emb_dim,
            config.emb_dim,
        ), f"Layer {layer_idx} wo: {layer.wo.shape}"
        assert layer.ffn_norm.shape == (
            config.emb_dim,
        ), f"Layer {layer_idx} ffn_norm: {layer.ffn_norm.shape}"
        assert layer.w_gate.shape == (
            config.emb_dim,
            config.hidden_dim,
        ), f"Layer {layer_idx} w_gate: {layer.w_gate.shape}"
        assert layer.w_up.shape == (
            config.emb_dim,
            config.hidden_dim,
        ), f"Layer {layer_idx} w_up: {layer.w_up.shape}"
        assert layer.w_down.shape == (
            config.hidden_dim,
            config.emb_dim,
        ), f"Layer {layer_idx} w_down: {layer.w_down.shape}"

        layers.append(layer)

    # --- final RMSNorm ---
    norm_key = "model.norm.weight"
    if norm_key not in key_to_file:
        raise KeyError(f"Missing weight: {norm_key}")
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, dtype)
    assert final_norm.shape == (
        config.emb_dim,
    ), f"final_norm shape mismatch: {final_norm.shape}"

    # --- lm_head (tied for Qwen2.5-1.5B) ---
    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, dtype)
        assert lm_head.shape == (
            config.vocab_size,
            config.emb_dim,
        ), f"lm_head shape mismatch: {lm_head.shape}"
    else:
        print(
            "Note: lm_head not found in safetensors -> tying to embed_table "
            "(expected for Qwen2.5-1.5B)"
        )
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# ---------------------------------------------------------------------------
# RoPE LUT  (identical to llama3 — only theta differs, taken from config)
# ---------------------------------------------------------------------------


def generate_rope_lut(
    config: Optional[LlamaConfig] = None,
    seq_len: int = 2048,
    dtype=bfloat16,
) -> np.ndarray:
    """Pre-computed RoPE table in [cos..., sin...] half-split layout.

    For Qwen2.5-1.5B, theta = 1,000,000 (set in config.rope_base).
    """
    if config is None:
        config = LlamaConfig()

    head_dim = config.head_dim
    half = head_dim // 2
    theta = config.rope_base

    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))

    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    lut = np.empty((seq_len, head_dim), dtype=np.float64)
    lut[:, :half] = cos_vals
    lut[:, half:] = sin_vals

    return lut.astype(dtype)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load Qwen2.5-1.5B weights and print shapes",
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="Qwen/Qwen2.5-1.5B",
        help="Local dir or HF model ID (default: Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--rope-seq-len", type=int, default=2048)
    args = parser.parse_args()

    dtype = bfloat16 if args.dtype == "bfloat16" else np.float32
    config = LlamaConfig()
    print(f"Loading weights from: {args.model_path}")
    print(f"Config: {config}")
    print()

    weights = load_weights(args.model_path, dtype=dtype, config=config)

    print("=== Global weights ===")
    print(
        f"  embed_table : {weights.embed_table.shape}  dtype={weights.embed_table.dtype}"
    )
    print(f"  final_norm  : {weights.final_norm.shape}")
    print(
        f"  lm_head     : {weights.lm_head.shape}  tied={weights.lm_head is weights.embed_table}"
    )
    print()

    print(f"=== Per-layer weights ({config.n_layers} layers; showing layer 0) ===")
    layer = weights.layers[0]
    print(f"  attn_norm : {layer.attn_norm.shape}")
    print(f"  wq        : {layer.wq.shape}     bq: {layer.bq.shape}")
    print(f"  wk        : {layer.wk.shape}      bk: {layer.bk.shape}")
    print(f"  wv        : {layer.wv.shape}      bv: {layer.bv.shape}")
    print(f"  wo        : {layer.wo.shape}     (no bias)")
    print(f"  ffn_norm  : {layer.ffn_norm.shape}")
    print(f"  w_gate    : {layer.w_gate.shape}")
    print(f"  w_up      : {layer.w_up.shape}")
    print(f"  w_down    : {layer.w_down.shape}")
    print()

    print("=== RoPE LUT ===")
    rope_lut = generate_rope_lut(config, seq_len=args.rope_seq_len, dtype=dtype)
    print(f"  rope_lut : {rope_lut.shape}  dtype={rope_lut.dtype}")
    print(f"  rope_lut[0, :8] = {rope_lut[0, :8].astype(np.float32)}")
    print(f"  rope_lut[1, :8] = {rope_lut[1, :8].astype(np.float32)}")
    print()
    print("All weights loaded successfully.")

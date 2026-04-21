# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-1.7B Weight Loader.

Loads Qwen3-1.7B (Qwen3ForCausalLM) weights from HuggingFace safetensors
into numpy arrays usable by MLIR-AIR kernels.

Differences from the qwen25 (Qwen2) loader:
  - **Q/K Norm** (NEW): per-layer `self_attn.q_norm.weight` and
    `self_attn.k_norm.weight`, each shape `(head_dim,)`. Applied as
    per-head RMSNorm BEFORE RoPE.
  - **No QKV bias**: Qwen3 dense models use `attention_bias=False`.
    No q_proj.bias / k_proj.bias / v_proj.bias to load.
  - 28 layers, head_dim=128, GQA group=2 (16 Q / 8 KV heads).
  - vocab=151936, rope_base=1e6.
  - tie_word_embeddings=True → lm_head_safetensors-key may or may not
    exist (Qwen3 stores it explicitly even when tied).

Weight convention (matches llama3/smollm2/qwen25):
    HuggingFace stores linear weights as (out_features, in_features).
    Our GEMM convention is y = x @ W → transpose to (in_features, out_features).
    Norm weights are 1-D (dim,) and not transposed.
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
    """Qwen3-1.7B model hyperparameters.

    Class kept named `LlamaConfig` so existing helpers (`llama3_inference`,
    `_llm_shared` infra) accept it unchanged. Qwen3-specific extensions:
      - `qk_norm` (always True for Qwen3 family)
      - `qkv_bias` (always False for Qwen3 dense; Qwen2 had True)
    """

    n_layers: int = 28
    emb_dim: int = 2048  # Qwen3-1.7B (was 1024 in 0.6B)
    n_heads: int = 16
    head_dim: int = 128
    n_kv_heads: int = 8  # GQA: 2 Q heads per KV head
    hidden_dim: int = 6144  # Qwen3-1.7B (was 3072 in 0.6B)
    vocab_size: int = 151936
    rope_base: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False  # Qwen3 has NO QKV bias
    qk_norm: bool = True  # Qwen3 NEW: per-layer Q/K RMSNorm before RoPE
    dtype: np.dtype = bfloat16


# ---------------------------------------------------------------------------
# Per-layer weight container
# ---------------------------------------------------------------------------


@dataclass
class LayerWeights:
    """Weight matrices for a single Qwen3 transformer layer.

    Q/K Norm weights (NEW vs qwen25 / llama3):
      - q_norm: (head_dim,)  RMSNorm weight applied per-head to Q before RoPE
      - k_norm: (head_dim,)  RMSNorm weight applied per-head to K before RoPE

    No QKV bias fields (Qwen3 dense has attention_bias=False).
    """

    attn_norm: np.ndarray  # (emb_dim,)  input layernorm
    wq: np.ndarray  # (emb_dim, n_heads*head_dim)
    wk: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    wv: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    q_norm: np.ndarray  # (head_dim,)  NEW
    k_norm: np.ndarray  # (head_dim,)  NEW
    wo: np.ndarray  # (emb_dim, emb_dim)
    ffn_norm: np.ndarray  # (emb_dim,)
    w_gate: np.ndarray  # (emb_dim, hidden_dim)
    w_up: np.ndarray  # (emb_dim, hidden_dim)
    w_down: np.ndarray  # (hidden_dim, emb_dim)


# ---------------------------------------------------------------------------
# Full model weight container
# ---------------------------------------------------------------------------


@dataclass
class LlamaWeights:
    """All weights for a Qwen3-1.7B model."""

    embed_table: np.ndarray  # (vocab, emb_dim)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (emb_dim,)
    lm_head: np.ndarray = None  # (vocab, emb_dim) — tied or explicit


# ---------------------------------------------------------------------------
# HuggingFace name mapping
# ---------------------------------------------------------------------------

# Standard Qwen2/Qwen3/Llama parameter naming.
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

# Qwen3 NEW: Q/K Norm weights (1-D per-head).
_HF_QK_NORM_MAP = {
    "self_attn.q_norm.weight": "q_norm",
    "self_attn.k_norm.weight": "k_norm",
}


# ---------------------------------------------------------------------------
# Safetensors helpers (mirrored from qwen25_weights.py)
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
        model_path, allow_patterns=["*.safetensors", "*.json"]
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
    """Load Qwen3-1.7B weights from safetensors into numpy arrays."""
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

    # --- per-layer weights ---
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

        # Q/K Norm (Qwen3-specific, 1-D per-head).
        for hf_suffix, field_name in _HF_QK_NORM_MAP.items():
            hf_key = f"model.layers.{layer_idx}.{hf_suffix}"
            if hf_key not in key_to_file:
                if config.qk_norm:
                    raise KeyError(
                        f"Layer {layer_idx} missing expected Q/K Norm: {hf_key}"
                    )
                # If qk_norm is disabled, populate with ones (no-op RMSNorm)
                layer_tensors[field_name] = np.ones((config.head_dim,), dtype=dtype)
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
        assert layer.q_norm.shape == (
            config.head_dim,
        ), f"Layer {layer_idx} q_norm: {layer.q_norm.shape}"
        assert layer.k_norm.shape == (
            config.head_dim,
        ), f"Layer {layer_idx} k_norm: {layer.k_norm.shape}"
        assert layer.wo.shape == (
            config.n_heads * config.head_dim,
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

    # --- lm_head ---
    # Qwen3 stores lm_head.weight explicitly even when tie_word_embeddings=True.
    # Use it if present; else fall back to tied embeddings.
    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, dtype)
        assert lm_head.shape == (
            config.vocab_size,
            config.emb_dim,
        ), f"lm_head shape mismatch: {lm_head.shape}"
    else:
        print("Note: lm_head not in safetensors -> tying to embed_table")
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# ---------------------------------------------------------------------------
# RoPE LUT (identical to qwen25 — only theta differs, taken from config)
# ---------------------------------------------------------------------------


def generate_rope_lut(
    config: Optional[LlamaConfig] = None,
    seq_len: int = 2048,
    dtype=bfloat16,
) -> np.ndarray:
    """Pre-computed RoPE table in [cos..., sin...] half-split layout."""
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
        description="Load Qwen3-1.7B weights and print shapes"
    )
    parser.add_argument("model_path", type=str, nargs="?", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16"
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
    print(f"  wq        : {layer.wq.shape}")
    print(f"  wk        : {layer.wk.shape}")
    print(f"  wv        : {layer.wv.shape}")
    print(f"  q_norm    : {layer.q_norm.shape}  (NEW vs Qwen2)")
    print(f"  k_norm    : {layer.k_norm.shape}  (NEW vs Qwen2)")
    print(f"  wo        : {layer.wo.shape}")
    print(f"  ffn_norm  : {layer.ffn_norm.shape}")
    print(f"  w_gate    : {layer.w_gate.shape}")
    print(f"  w_up      : {layer.w_up.shape}")
    print(f"  w_down    : {layer.w_down.shape}")
    print()

    print("=== RoPE LUT ===")
    rope_lut = generate_rope_lut(config, seq_len=args.rope_seq_len, dtype=dtype)
    print(f"  rope_lut : {rope_lut.shape}  dtype={rope_lut.dtype}")
    print()
    print("All weights loaded successfully.")

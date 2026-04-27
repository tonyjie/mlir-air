# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-135M Weight Loader

Loads SmolLM2-135M weights from HuggingFace safetensors format and provides
them as numpy arrays suitable for MLIR-AIR kernel invocations.

SmolLM2-135M is a LlamaForCausalLM model. Differences vs Llama-3.2-1B (the
reference deployment) handled via config:

  - 30 layers (deepest deployment so far)
  - emb_dim=576, n_heads=9 (odd, non-power-of-2), n_kv_heads=3 (GQA g=3)
  - head_dim=64, ffn_hidden=1536 (smallest hidden of any deployment)
  - vocab_size=49152 (same as smollm2_1_7b)
  - rope_base=100000 (new — we have 130k, 500k, 1M)
  - tied embeddings (lm_head shares embed_tokens.weight) — same fallback path
    as Llama-3.2-1B and SmolLM2-1.7B

Weight convention:
    HuggingFace stores linear weights as (out_features, in_features).
    Our GEMM convention is y = x @ W, so we need W as (in_features, out_features).
    All projection weights are transposed during loading.

Usage:
    from smollm2_135m_weights import load_weights, LlamaConfig

    config = LlamaConfig()
    weights = load_weights("HuggingFaceTB/SmolLM2-135M")
    print(weights.layers[0].wq.shape)  # (576, 576)
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
    """SmolLM2-135M model hyperparameters (LlamaForCausalLM architecture)."""

    n_layers: int = 30
    emb_dim: int = 576
    n_heads: int = 9
    head_dim: int = 64
    n_kv_heads: int = 3  # GQA g=3 (group_size = n_heads // n_kv_heads = 3)
    hidden_dim: int = 1536
    vocab_size: int = 49152
    rope_base: float = 100000.0
    dtype: np.dtype = bfloat16


# ---------------------------------------------------------------------------
# Per-layer weight container
# ---------------------------------------------------------------------------


@dataclass
class LayerWeights:
    """Weight matrices for a single transformer layer.

    All shapes follow the convention W such that y = x @ W, i.e.
    (in_features, out_features).

    Attributes:
        attn_norm:  (emb_dim,)              RMSNorm weight for attention
        wq:         (emb_dim, n_heads*head_dim)     Q projection (576,576)
        wk:         (emb_dim, n_kv_heads*head_dim)  K projection (576,192)
        wv:         (emb_dim, n_kv_heads*head_dim)  V projection (576,192)
        wo:         (emb_dim, emb_dim)              O projection (576,576)
        ffn_norm:   (emb_dim,)              RMSNorm weight for FFN
        w_gate:     (emb_dim, hidden_dim)   Gate projection (SwiGLU)
        w_up:       (emb_dim, hidden_dim)   Up projection (SwiGLU)
        w_down:     (hidden_dim, emb_dim)   Down projection
    """

    attn_norm: np.ndarray  # (576,)
    wq: np.ndarray  # (576, 576)
    wk: np.ndarray  # (576, 192)  — GQA: 9/3 = 3x smaller than wq
    wv: np.ndarray  # (576, 192)  — GQA: 9/3 = 3x smaller than wq
    wo: np.ndarray  # (576, 576)
    ffn_norm: np.ndarray  # (576,)
    w_gate: np.ndarray  # (576, 1536)
    w_up: np.ndarray  # (576, 1536)
    w_down: np.ndarray  # (1536, 576)


# ---------------------------------------------------------------------------
# Full model weight container
# ---------------------------------------------------------------------------


@dataclass
class LlamaWeights:
    """All weights for a SmolLM2-135M model.

    Attributes:
        embed_table:  (vocab_size, emb_dim)  Token embeddings (49152, 576)
        layers:       list of 30 LayerWeights
        final_norm:   (emb_dim,)             Final RMSNorm weight
        lm_head:      (vocab_size, emb_dim)  Output projection (tied to embed_table)
    """

    embed_table: np.ndarray  # (49152, 576)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (576,)
    lm_head: np.ndarray = None  # (49152, 576) — tied: same object as embed_table


# ---------------------------------------------------------------------------
# HuggingFace name mapping
# ---------------------------------------------------------------------------

# Map from HuggingFace parameter names to our field names.
# Weights marked with transpose=True are stored as (out, in) in HF and need
# to be transposed to (in, out) for our y = x @ W convention.
# SmolLM2-135M uses the standard LlamaForCausalLM names — identical to Llama-3.2-1B.

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


# ---------------------------------------------------------------------------
# Safetensors loading helpers
# ---------------------------------------------------------------------------


def _resolve_safetensor_files(model_path: str) -> List[str]:
    """Find all safetensor shard files for a model.

    Args:
        model_path: Either a local directory path or a HuggingFace model ID
                    (e.g. "HuggingFaceTB/SmolLM2-135M").

    Returns:
        List of absolute paths to .safetensors files.
    """
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
    """Load a single tensor from an open safetensors file handle."""
    tensor = file_handle.get_tensor(key)
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.astype(dtype)


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------


def load_weights(
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[LlamaConfig] = None,
) -> LlamaWeights:
    """Load SmolLM2-135M weights from safetensors into numpy arrays."""
    from safetensors import safe_open

    if config is None:
        config = LlamaConfig()

    safetensor_files = _resolve_safetensor_files(model_name_or_path)

    key_to_file = {}
    for filepath in safetensor_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    # --- Load embedding table ---
    embed_key = "model.embed_tokens.weight"
    if embed_key not in key_to_file:
        raise KeyError(f"Missing weight: {embed_key}")
    with safe_open(key_to_file[embed_key], framework="numpy") as f:
        embed_table = _load_tensor(f, embed_key, dtype)
    assert embed_table.shape == (config.vocab_size, config.emb_dim), (
        f"embed_table shape mismatch: expected "
        f"({config.vocab_size}, {config.emb_dim}), got {embed_table.shape}"
    )

    # --- Load per-layer weights ---
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

        layer = LayerWeights(**layer_tensors)

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

    # --- Load final RMSNorm ---
    norm_key = "model.norm.weight"
    if norm_key not in key_to_file:
        raise KeyError(f"Missing weight: {norm_key}")
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, dtype)
    assert final_norm.shape == (
        config.emb_dim,
    ), f"final_norm shape mismatch: {final_norm.shape}"

    # --- Load lm_head (or tie to embeddings) ---
    # SmolLM2-135M has tie_word_embeddings=true (per HF config), so lm_head.weight
    # is absent from the safetensors and we tie to embed_table.
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
            "Note: lm_head not found in safetensors -> tying to embed_table (expected for SmolLM2-135M)"
        )
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# ---------------------------------------------------------------------------
# RoPE look-up table generation
# ---------------------------------------------------------------------------


def generate_rope_lut(
    config: Optional[LlamaConfig] = None,
    seq_len: int = 2048,
    dtype=bfloat16,
) -> np.ndarray:
    """Generate a pre-computed RoPE (Rotary Position Embedding) look-up table.

    The LUT uses concatenated layout: [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
    matching the half-split RoPE kernel (rope_halfsplit.cc) and HuggingFace Llama convention.

    For position *pos* and dimension index *i* (0-indexed, i < head_dim/2):
        freq_i           = 1.0 / (theta ^ (2*i / head_dim))
        angle            = pos * freq_i
        LUT[pos, i]              = cos(angle)
        LUT[pos, i + head_dim/2] = sin(angle)

    For SmolLM2-135M, theta=100000 (vs SmolLM2-1.7B's 130000 / Llama-3.2-1B's 500000).

    Args:
        config: Model config (uses rope_base and head_dim).
        seq_len: Maximum sequence length for the LUT.
        dtype: Output dtype. Default is bfloat16.

    Returns:
        np.ndarray of shape (seq_len, head_dim) with concatenated [cos..., sin...].
    """
    if config is None:
        config = LlamaConfig()

    head_dim = config.head_dim
    half = head_dim // 2
    theta = config.rope_base

    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))

    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)  # (seq_len, head_dim/2)

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    lut = np.empty((seq_len, head_dim), dtype=np.float64)
    lut[:, :half] = cos_vals
    lut[:, half:] = sin_vals

    return lut.astype(dtype)


# ---------------------------------------------------------------------------
# Main -- test loading and print shapes
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load SmolLM2-135M weights and print shapes",
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="HuggingFaceTB/SmolLM2-135M",
        help=(
            "Path to local model directory or HuggingFace model ID "
            "(default: HuggingFaceTB/SmolLM2-135M)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Data type for loaded weights",
    )
    parser.add_argument(
        "--rope-seq-len",
        type=int,
        default=2048,
        help="Sequence length for RoPE LUT generation",
    )
    args = parser.parse_args()

    dtype = bfloat16 if args.dtype == "bfloat16" else np.float32
    config = LlamaConfig()

    print(f"Loading weights from: {args.model_path}")
    print(f"Target dtype: {args.dtype}")
    print(f"Config: {config}")
    print()

    weights = load_weights(args.model_path, dtype=dtype, config=config)

    print("=== Global weights ===")
    print(
        f"  embed_table : {weights.embed_table.shape}  dtype={weights.embed_table.dtype}"
    )
    print(
        f"  final_norm  : {weights.final_norm.shape}  dtype={weights.final_norm.dtype}"
    )
    print(f"  lm_head     : {weights.lm_head.shape}  dtype={weights.lm_head.dtype}")
    tied = weights.lm_head is weights.embed_table
    print(f"  lm_head tied to embed_table: {tied}")
    print()

    print(f"=== Per-layer weights ({config.n_layers} layers) ===")
    for i, layer in enumerate(weights.layers):
        print(f"  Layer {i:2d}:")
        print(f"    attn_norm : {layer.attn_norm.shape}")
        print(f"    wq        : {layer.wq.shape}")
        print(f"    wk        : {layer.wk.shape}")
        print(f"    wv        : {layer.wv.shape}")
        print(f"    wo        : {layer.wo.shape}")
        print(f"    ffn_norm  : {layer.ffn_norm.shape}")
        print(f"    w_gate    : {layer.w_gate.shape}")
        print(f"    w_up      : {layer.w_up.shape}")
        print(f"    w_down    : {layer.w_down.shape}")
        if i == 0:
            continue
        if i == 1:
            print("    ... (remaining layers have identical shapes)")
            break
    print()

    print("=== RoPE LUT ===")
    rope_lut = generate_rope_lut(config, seq_len=args.rope_seq_len, dtype=dtype)
    print(f"  rope_lut    : {rope_lut.shape}  dtype={rope_lut.dtype}")
    print(f"  rope_lut[0, :8] = {rope_lut[0, :8].astype(np.float32)}")
    print(f"  rope_lut[1, :8] = {rope_lut[1, :8].astype(np.float32)}")
    print()

    print("All weights loaded successfully.")

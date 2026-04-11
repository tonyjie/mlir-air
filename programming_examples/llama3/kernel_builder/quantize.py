"""Q4_1 weight quantization utilities.

Simulates the accuracy impact of 4-bit asymmetric quantization (Q4_1 format)
by quantizing weights to 4-bit and dequantizing back to BF16. This lets us
measure accuracy loss without changing NPU kernels.

Q4_1 format (per block of 32 values):
  - scale = (max - min) / 15        (BF16)
  - q = clamp(round((val - min) / scale), 0, 15)   (4-bit unsigned)
  - dequant = min + q * scale       (BF16)

Matches llama.cpp Q4_1 and FastFlowLM's quantization scheme.
"""

import numpy as np
from ml_dtypes import bfloat16


def quantize_dequant_q4_1(weight, block_size=32):
    """Quantize a weight array to Q4_1 and dequantize back.

    Simulates the precision loss of 4-bit quantization without producing
    the packed format. Returns a BF16 array with quantization error baked in.

    Args:
        weight: numpy array (any shape, BF16 or F32). Quantization is applied
                along the last axis in blocks of block_size.
        block_size: Number of values per quantization block. Default 32.

    Returns:
        Dequantized array with same shape and dtype as input.
    """
    original_dtype = weight.dtype
    w = weight.astype(np.float32)
    shape = w.shape

    # Reshape so last axis is divisible by block_size
    last_dim = shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(
            f"Last dimension {last_dim} not divisible by block_size {block_size}"
        )

    # Reshape to (..., n_blocks, block_size)
    n_blocks = last_dim // block_size
    w_blocks = w.reshape(-1, n_blocks, block_size)

    # Per-block min/max
    block_min = w_blocks.min(axis=-1, keepdims=True)  # (..., n_blocks, 1)
    block_max = w_blocks.max(axis=-1, keepdims=True)

    # Scale: map [min, max] to [0, 15]
    scale = (block_max - block_min) / 15.0
    # Avoid division by zero for constant blocks
    scale = np.where(scale == 0, 1.0, scale)

    # Quantize: round to nearest 4-bit unsigned int
    q = np.clip(np.round((w_blocks - block_min) / scale), 0, 15).astype(np.uint8)

    # Dequantize
    w_dequant = block_min + q.astype(np.float32) * scale

    # Reshape back
    w_dequant = w_dequant.reshape(shape)
    return w_dequant.astype(original_dtype)


def quantize_all_weights(weights, config):
    """Apply Q4_1 quantize→dequant to all linear projection weights.

    Modifies weights in-place. Reports per-layer quantization error.

    Quantizes: wq, wk, wv, wo, w_gate, w_up, w_down (per layer) + lm_head
    Keeps BF16: attn_norm, ffn_norm, final_norm, embed_table
    """
    linear_fields = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]

    print("Applying Q4_1 quantization (block_size=32)...")
    total_params = 0
    total_mse = 0.0

    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        layer_mse = 0.0
        layer_params = 0

        for field in linear_fields:
            w_orig = getattr(lw, field)
            w_quant = quantize_dequant_q4_1(w_orig)
            setattr(lw, field, w_quant)

            # Error metrics
            diff = (w_orig.astype(np.float32) - w_quant.astype(np.float32)).flatten()
            mse = float(np.mean(diff**2))
            layer_mse += mse * diff.size
            layer_params += diff.size

        avg_mse = layer_mse / layer_params if layer_params > 0 else 0
        total_mse += layer_mse
        total_params += layer_params
        print(f"  Layer {layer_idx:2d}: avg MSE={avg_mse:.2e}, params={layer_params:,}")

    # Quantize lm_head (if not tied to embed_table)
    if weights.lm_head is not weights.embed_table:
        w_orig = weights.lm_head
        weights.lm_head = quantize_dequant_q4_1(w_orig)
        diff = (
            w_orig.astype(np.float32) - weights.lm_head.astype(np.float32)
        ).flatten()
        lm_mse = float(np.mean(diff**2))
        total_mse += lm_mse * diff.size
        total_params += diff.size
        print(f"  LM Head:  avg MSE={lm_mse:.2e}, params={diff.size:,}")
    else:
        # lm_head is tied to embed_table — quantize embed_table
        w_orig = weights.embed_table
        weights.embed_table = quantize_dequant_q4_1(w_orig)
        weights.lm_head = weights.embed_table  # maintain tie
        diff = (
            w_orig.astype(np.float32) - weights.embed_table.astype(np.float32)
        ).flatten()
        emb_mse = float(np.mean(diff**2))
        total_mse += emb_mse * diff.size
        total_params += diff.size
        print(f"  Embed/LM: avg MSE={emb_mse:.2e}, params={diff.size:,}")

    overall_mse = total_mse / total_params if total_params > 0 else 0
    size_bf16_mb = total_params * 2 / 1024 / 1024
    size_q4_mb = total_params * 0.5 / 1024 / 1024  # 4 bits per param
    print(
        f"  Total: {total_params:,} params quantized, "
        f"overall MSE={overall_mse:.2e}, "
        f"BF16={size_bf16_mb:.0f}MB → Q4={size_q4_mb:.0f}MB ({size_q4_mb/size_bf16_mb*100:.0f}%)"
    )

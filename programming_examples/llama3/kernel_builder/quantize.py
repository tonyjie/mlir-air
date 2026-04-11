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


def pack_q4_for_npu(weight, block_size=32):
    """Pack a BF16 weight matrix to Q4 format for NPU BO storage.

    Returns three arrays matching the mv_q4.cc kernel's expected layout:
      - packed: uint8 array, 2 Q4 values per byte (low nibble first)
      - scales: BF16 array, one scale per block
      - mins: BF16 array, one min per block

    The kernel dequantizes as: val = min + q * scale

    Args:
        weight: (M, K) BF16 or F32 array.
        block_size: Values per quantization block. Default 32.

    Returns:
        (packed_uint8, scales_bf16, mins_bf16)
    """
    w = weight.astype(np.float32)
    M, K = w.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    n_blocks_per_row = K // block_size

    # Reshape to (M, n_blocks, block_size)
    w_blocks = w.reshape(M, n_blocks_per_row, block_size)

    block_min = w_blocks.min(axis=-1)  # (M, n_blocks)
    block_max = w_blocks.max(axis=-1)
    scale = (block_max - block_min) / 15.0
    scale = np.where(scale == 0, 1.0, scale)

    # Quantize
    q = np.clip(
        np.round((w_blocks - block_min[:, :, None]) / scale[:, :, None]),
        0,
        15,
    ).astype(
        np.uint8
    )  # (M, n_blocks, block_size)

    # Pack: 2 values per byte, low nibble first
    q_flat = q.reshape(M, -1)  # (M, K)
    packed = np.zeros((M, K // 2), dtype=np.uint8)
    packed = (q_flat[:, 0::2] & 0x0F) | ((q_flat[:, 1::2] & 0x0F) << 4)

    scales_bf16 = scale.astype(bfloat16)  # (M, n_blocks)
    mins_bf16 = block_min.astype(bfloat16)  # (M, n_blocks)

    return packed, scales_bf16, mins_bf16


def q4_dequant_reference(packed, scales, mins, M, K, block_size=32):
    """CPU reference dequantization matching mv_q4.cc behavior.

    Args:
        packed: (M, K//2) uint8, 2 values per byte
        scales: (M, K//block_size) bfloat16
        mins: (M, K//block_size) bfloat16

    Returns:
        (M, K) float32 dequantized weight
    """
    n_blocks = K // block_size
    result = np.zeros((M, K), dtype=np.float32)

    for row in range(M):
        for blk in range(n_blocks):
            s = float(scales[row, blk])
            mn = float(mins[row, blk])
            for i in range(block_size // 2):
                byte_idx = blk * (block_size // 2) + i
                byte = packed[row, byte_idx]
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                result[row, blk * block_size + 2 * i] = mn + lo * s
                result[row, blk * block_size + 2 * i + 1] = mn + hi * s

    return result


# ---------------------------------------------------------------------------
# Interleaved Q4 format (for NPU kernel mv_q4.cc)
# ---------------------------------------------------------------------------

_Q4_BLOCK_SIZE = 32
_Q4_PACKED_PER_BLOCK = 16  # bytes (32 values / 2)
_Q4_BLOCK_BYTES = 20  # 16 packed + 2 scale + 2 min


def q4_row_bytes(k, block_size=_Q4_BLOCK_SIZE):
    """Bytes per row in interleaved Q4 format."""
    return (k // block_size) * _Q4_BLOCK_BYTES


def pack_q4_interleaved(weight, block_size=_Q4_BLOCK_SIZE):
    """Pack BF16 weight to interleaved Q4 format for NPU mv_q4.cc.

    Each block of 32 values → 20 bytes: [16B packed | 2B scale | 2B min].
    Returns a single (M, row_bytes) uint8 array.
    """
    w = weight.astype(np.float32)
    M, K = w.shape
    assert K % block_size == 0
    n_blocks = K // block_size
    rb = n_blocks * _Q4_BLOCK_BYTES

    result = np.zeros((M, rb), dtype=np.uint8)

    for row in range(M):
        for blk in range(n_blocks):
            blk_vals = w[row, blk * block_size : (blk + 1) * block_size]
            mn = float(blk_vals.min())
            mx = float(blk_vals.max())
            scale = (mx - mn) / 15.0 if mx != mn else 1.0

            # Quantize
            q = np.clip(np.round((blk_vals - mn) / scale), 0, 15).astype(np.uint8)

            # Pack 2 values per byte
            packed = (q[0::2] & 0x0F) | ((q[1::2] & 0x0F) << 4)

            # Write block: [packed(16B) | scale(2B) | min(2B)]
            offset = blk * _Q4_BLOCK_BYTES
            result[row, offset : offset + _Q4_PACKED_PER_BLOCK] = packed
            scale_bf16 = np.array([scale], dtype=np.float32).astype(bfloat16)
            min_bf16 = np.array([mn], dtype=np.float32).astype(bfloat16)
            result[
                row, offset + _Q4_PACKED_PER_BLOCK : offset + _Q4_PACKED_PER_BLOCK + 2
            ] = scale_bf16.view(np.uint8)
            result[
                row, offset + _Q4_PACKED_PER_BLOCK + 2 : offset + _Q4_BLOCK_BYTES
            ] = min_bf16.view(np.uint8)

    return result


def q4_interleaved_dequant_reference(packed_buf, M, K, block_size=_Q4_BLOCK_SIZE):
    """CPU reference dequant for interleaved Q4 format (matching mv_q4.cc)."""
    n_blocks = K // block_size
    result = np.zeros((M, K), dtype=np.float32)

    for row in range(M):
        for blk in range(n_blocks):
            offset = blk * _Q4_BLOCK_BYTES
            packed = packed_buf[row, offset : offset + _Q4_PACKED_PER_BLOCK]
            scale = np.frombuffer(
                packed_buf[
                    row,
                    offset + _Q4_PACKED_PER_BLOCK : offset + _Q4_PACKED_PER_BLOCK + 2,
                ],
                dtype=bfloat16,
            )[0]
            mn = np.frombuffer(
                packed_buf[
                    row, offset + _Q4_PACKED_PER_BLOCK + 2 : offset + _Q4_BLOCK_BYTES
                ],
                dtype=bfloat16,
            )[0]
            s = float(scale)
            m = float(mn)

            for i in range(_Q4_PACKED_PER_BLOCK):
                byte = packed[i]
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                result[row, blk * block_size + 2 * i] = m + lo * s
                result[row, blk * block_size + 2 * i + 1] = m + hi * s

    return result


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

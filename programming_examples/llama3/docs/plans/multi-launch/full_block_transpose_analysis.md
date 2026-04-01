# Phase 3: Full Block Transpose Analysis

## Finding: GEMM→RoPE requires REAL data transpose

The transition from Q/K GEMM output to RoPE input requires a **physical data movement**, not just a view/reshape.

### Memory layouts

**GEMM Q output**: `memref<2048x2048xbf16>` — row-major
```
DDR layout: [row0_head0, row0_head1, ..., row0_head31, row1_head0, ...]
Shape interpretation: (seq_len=2048, n_heads*head_dim=2048)
= (seq_len, n_heads, head_dim) = (2048, 32, 64)
```

**RoPE Q input**: `memref<65536x64xbf16>` — heads-first
```
DDR layout: [head0_row0, head0_row1, ..., head0_row2047, head1_row0, ...]
Shape interpretation: (n_heads*seq_len, head_dim) = (65536, 64)
= (n_heads, seq_len, head_dim) = (32, 2048, 64) reshaped to 2D
```

### Verification

```python
q = np.random.rand(2048, 32, 64)  # GEMM output as 3D
q.transpose(1, 0, 2)              # → (32, 2048, 64) — NOT contiguous
q.transpose(1, 0, 2).reshape(65536, 64)  # FORCES a copy (data movement)
```

`transpose(1, 0, 2)` creates a non-contiguous view. The subsequent `reshape` must copy 8MB of data into a new contiguous layout.

### Impact

This means Phase 3 (full-block multi-launch) cannot simply chain GEMM→RoPE→FlashAttn in sequence without addressing the data layout transition.

### Options

1. **Transpose kernel launch**: Add a dedicated `air.launch` that transposes the data in DDR. Simple but adds 2 extra launches (Q and K transposes).

2. **Modify RoPE kernel**: Change RoPE to accept `(seq_len, n_heads, head_dim)` layout and handle the head iteration internally with strided DMA. The RoPE computation per-element is independent, so the memory access pattern change is feasible.

3. **Modify GEMM output layout**: Have the GEMM kernel write in transposed (heads-first) layout. Would require changes to the GEMM's output DMA patterns.

4. **Use strided memref**: In MLIR, use `memref<32x2048x64xbf16, strided<[64, 2048*64, 1]>>` to represent a transposed view without copying. Requires compiler support for non-standard strides in DMA.

### Recommendation

**Option 2 (modify RoPE)** is the cleanest — the RoPE computation is element-wise per head, so changing the memory access pattern doesn't affect the algorithm. The DMA can load rows with appropriate strides regardless of whether data is stored heads-first or rows-first.

### Same issue applies to:
- K GEMM → RoPE K (same transpose)
- RoPE Q/K → FlashAttention (similar but may be compatible — FlashAttn expects `(n_heads, seq_len, head_dim)` which is the same as RoPE output `(n_heads * seq_len, head_dim)` reshaped to 3D — this IS a free reshape since data is already heads-first)
- FlashAttention → O GEMM (reverse transpose: `(n_heads, seq_len, head_dim)` → `(seq_len, n_heads * head_dim)`)

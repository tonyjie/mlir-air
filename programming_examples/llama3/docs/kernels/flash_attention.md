# Flash Attention Kernel — Status & Analysis

## Role in LLAMA Pipeline

Step 7 of each transformer block: Multi-head attention with GQA and causal masking.

| Parameter | LLAMA-3.2-1B Value |
|-----------|-------------------|
| Sequence length (LQ=LK) | 2048 |
| Head dimension (DK=DV) | 64 |
| Number of Q heads | 32 |
| Number of KV heads | 8 (GQA ratio 4:1) |
| Causal masking | Yes |
| Tile sizes | LQP=256, LKP=64 |

---

## Current Status (2026-03-26): Integrated into LLAMA Pipeline

### FlashAttention is Working — Integrated and Verified

**Standalone test** (`make run`): corr=0.9976 for LLAMA causal config (32Q/8KV, seq_len=2048).

**LLAMA pipeline integration** (16 layers, all on NPU):

| Metric | Value |
|--------|-------|
| Top-1 prediction | " Paris" (correct) |
| Logits correlation vs CPU F32 | **0.993** |
| Per-step correlation | >0.999 (all 240 invocations) |
| NPU kernel time (flash_attn) | 0.46s total (29ms avg per layer) |
| NPU kernel time (all kernels) | 3.60s total |

### Integration Notes

- Kernel expects **unscaled Q** — scaling by 1/sqrt(dk) is handled internally by the kernel
- Kernel takes **4 args** (Q, K, V, Output) — no mask buffer (causal masking is internal)
- `test_precision.py` is out of date (passes 5 args for old kernel interface) — use `make run` instead

### Performance

| Metric | AIR FlashAttention | IRON MHA | AIR vs IRON |
|--------|-------------------|----------|-------------|
| Latency (standalone) | **15,022 µs** (15ms) | 30,989 µs (31ms) | **2.06× faster** |
| Latency (in pipeline) | **29,000 µs** (29ms) | — | — |
| GFLOP/s | **2,281** | ~1,100 | **2.1× faster** |
| corr (standalone) | 0.9976 | 0.9976 | **Matched** |

### Tile Config Analysis (LLAMA causal, 32h/8kv, 2048 seq)

Only **LQP=256, LKP=64** works for causal mode. Swept all valid combinations:

| LQP | LKP | Status | Constraint |
|-----|-----|--------|-----------|
| * | 32 | FAIL | Causal doesn't support DV tiling (LKP < DK) |
| 64 | 64 | FAIL | Causal requires `tile_size_q == LKP` (16 ≠ 64) |
| 128 | 64 | FAIL | Same (32 ≠ 64) |
| **256** | **64** | **PASS** | **tile_size_q = 256/4 = 64 == LKP ✓** |
| 512 | 64 | FAIL | Same constraint (128 ≠ 64) |
| 1024 | 64 | FAIL | Same (256 ≠ 64) |
| * | 128+ | FAIL | DK (64) must be divisible by LKP |

Three constraints intersect at exactly one point:
1. **LKP ≤ DK** → LKP ∈ {32, 64}
2. **LKP ≥ DK** (no DV tiling for causal) → LKP ≥ 64
3. **tile_size_q == LKP** (causal requirement) → LQP/4 = 64 → LQP = 256

---

## Reproducible Commands

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Correctness (LLAMA causal config)
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
    NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# Performance profiling
make profile LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
    NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# Precision analysis (correlation-based)
python3 test_precision.py --num-heads 32 --num-kv-heads 8 --causal
```

---

## Seq-First Layout Investigation (2026-04)

### Motivation

The LLAMA prefill pipeline has 4 remaining numpy transposes per layer, all at the FlashAttention boundary:

```
QKV GEMM output: (seq, emb_dim)      ← seq-first
    ↓ TRANSPOSE Q: (seq,32,64) → (32,seq,64)     16 MB
    ↓ TRANSPOSE K: (seq,8,64) → (8,seq,64)       4 MB
    ↓ TRANSPOSE V: (seq,8,64) → (8,seq,64)       4 MB
FlashAttention:  (n_h, seq, 64)       ← head-first
    ↓ TRANSPOSE output: (32,seq,64) → (seq,32,64) 16 MB
O GEMM input:   (seq, emb_dim)       ← seq-first
```

These are pure data movement (no compute), costing ~40ms across 16 layers. If FlashAttention accepted seq-first layout directly, all 4 transposes would be eliminated.

### What seq-first means for FlashAttention

**Head-first** `[num_heads, seq, head_dim]`: each head's data is a contiguous `(seq, 64)` block. The DMA reads one big contiguous chunk per head.

```
Memory: [h0_pos0_d0..d63 | h0_pos1_d0..d63 | ... | h0_pos2047 | h1_pos0 | h1_pos1 | ...]
         ←── head 0: contiguous 2048×64 block ──→  ←── head 1 ──→
```

**Seq-first** `[seq, num_heads * head_dim]`: heads are interleaved at each position. To read head 0's data, the DMA must read 64 elements, skip `(num_heads-1) * 64 = 1984` elements, read next 64, etc.

```
Memory: [pos0_h0 | pos0_h1 | ... | pos0_h31 | pos1_h0 | pos1_h1 | ... | pos1_h31 | ...]
         ←64B→    skip 1984B                   ←64B→    skip 1984B
```

This is a **strided DMA pattern** with row stride = `num_heads * head_dim = 2048` between positions for a given head.

### What we changed in `attn_npu2_seqfirst.py`

Created a copy of `attn_npu2.py` with these modifications:

1. **L3 memref types**: `[num_heads, lq, dk]` → `[lq, num_heads * dk]` (seq-first 2D)
2. **Head offset maps**: `head * lq * dk` → `head * dk` (column offset, not row offset)
3. **Launch offset maps**: `lx * lqp * dk` → `lx * lqp * num_heads * dk` (row offset in wider array)
4. **Q DMA strides**: `[tile_size_q * dk, ...]` → `[tile_size_q * num_heads * dk, ...]` (row stride = emb_dim)
5. **K DMA strides**: same pattern with `num_kv_heads * dk`
6. **V DMA strides**: same pattern with `num_kv_heads * dv`
7. **Output DMA**: contiguous `[lqp * dv_tile]` → strided `[lqp, dv_tile]` with stride `num_heads * dv`

### Result: compiler assertion failure

```
aircc: Assertion `willBeValidAffineMap(dimCount, symbolCount, {result})' failed.
```

The crash occurs in the `air-to-aie` lowering pipeline when the compiler tries to construct an AffineMap from the combined dynamic offset + strided DMA pattern. Specifically:

- The offset `q_combined = head_idx * 64 + launch_iter * 524288` is a valid affine expression
- The strides `[tile_size_q * 2048, dk_tile, 2048, 1]` are valid constants
- But when the compiler combines these during channel-to-DMA lowering, the resulting expression fails affine map validation

**This is a compiler limitation, not a hardware limitation.** The NPU hardware supports 4D buffer descriptors with arbitrary strides that could express this pattern. The `air-to-aie` pass's affine map construction doesn't handle the combination of dynamic offsets with large strided access patterns.

### Why head-first works

Head-first strides are `[tile_size_q * 64, dk_tile, 64, 1]` — the row stride (64) equals `head_dim`, which means the DMA reads a contiguous 2D block. The compiler can express this as a simple base+offset BD.

Seq-first strides are `[tile_size_q * 2048, dk_tile, 2048, 1]` — the row stride (2048) equals `num_heads * head_dim`, meaning the DMA must skip over other heads' data between rows. This strided non-contiguous access requires more complex BD configuration that the compiler can't construct from its affine map framework.

### What would fix this

1. **Compiler fix**: teach `air-to-aie` to handle strided DMA patterns where the offset and stride are both affine functions of loop variables and constants. The hardware BD supports this — the gap is in the compiler's affine map construction.

2. **L2 staging workaround**: use L2 (MemTile) as an intermediate — DMA the full seq-first row to L2, then extract per-head data from L2 to L1 using MemTile's more flexible DMA. This adds a hop but avoids the problematic L3 strided access.

3. **Host-side transpose (current)**: keep the `np.transpose` in Python between RoPE and FlashAttention. Cost: ~40ms across 16 layers.

### Files

| File | Purpose |
|------|---------|
| `attn_npu2_seqfirst.py` | Experimental seq-first variant (blocked by compiler) |
| `test_seqfirst.py` | Host-side transpose validation test |

---

## Historical Investigation (resolved)

### Previous Bug: corr=0.13-0.34 (before 2026-03-26 fix)

The kernel produced output with **no correlation** to standard attention for all configs at LQ=2048. `make run PASS` was a false positive — element-wise tolerance (`atol=0.15, rtol=0.04`) couldn't detect wrong attention patterns because output values were in the correct range (~1.5) but mathematically incorrect.

Fixed in upstream update (2026-03-26). Correlation now 0.996-0.998 for all configs.

### Earlier Bug: corr=0.31 (before PR #1438)

Original flash attention kernel with missing causal masking and BD exhaustion. Fixed by PR #1438 (causal BD exhaustion) and subsequent upstream fixes.

---

## Related Documents

- `perf_opt_prefill.md` — Overall LLAMA optimization roadmap
- `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md` — IRON MHA profiling
- `kernels/gemm.md` — GEMM optimization (similar precision analysis approach)

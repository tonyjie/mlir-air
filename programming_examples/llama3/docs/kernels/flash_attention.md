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

The LLAMA prefill pipeline has 4 numpy transposes per layer at the FlashAttention boundary, converting between seq-first (used by GEMMs, RoPE) and head-first (used by FlashAttention):

```
RoPE output:     (seq, emb_dim)       ← seq-first
    ↓ TRANSPOSE Q: (seq,32,64) → (32,seq,64)     16 MB
    ↓ TRANSPOSE K: (seq,8,64) → (8,seq,64)       4 MB
    ↓ TRANSPOSE V: (seq,8,64) → (8,seq,64)       4 MB
FlashAttention:  (n_h, seq, 64)       ← head-first
    ↓ TRANSPOSE output: (32,seq,64) → (seq,32,64) 16 MB
O GEMM input:   (seq, emb_dim)       ← seq-first
```

Total: ~40MB data movement × 16 layers = ~2.5ms/layer host CPU time.

### Fundamental insight: the transpose is unavoidable

FlashAttention computes `Q[h] @ K[h]^T` per head — each head's data must be gathered together at the compute level. In seq-first layout, per-head data is interleaved across positions. The layout conversion **must happen somewhere**:

| Level | Method | Cost |
|-------|--------|------|
| **Host CPU** | `np.transpose` in Python | ~2.5ms/layer |
| **Shim DMA (L3→L2)** | Strided read per head | ~1.5ms/layer extra vs head-first |
| **MemTile DMA (L2→L1)** | Contiguous L3→L2, strided L2→L1 extract | ~0ms (on-chip SRAM) |

The MemTile approach would be ideal (contiguous DDR access + free on-chip extraction), but requires significant kernel restructuring. The shim DMA approach works today.

### Implementation: `attn_npu2_seqfirst.py`

Created a seq-first variant with these changes:

1. **L3 memref types**: `[num_heads, lq, dk]` → `[lq, num_heads * dk]`
2. **Head offset maps**: `head * lq * dk` → `head * dk` (column offset)
3. **Launch offset maps**: row-only `lx * lqp` (not flat offset)
4. **DMA offsets**: 2D `[row, col]` instead of 1D flat (critical for compiler)
5. **DMA strides**: row stride = `num_heads * dk` (strided per-head access)

**Key fix**: using proper 2D offsets `[row_offset, col_offset]` for the 2D seq-first memref. Initial attempt with flattened 1D offsets crashed the compiler (`willBeValidAffineMap` assertion). The 2D offset approach compiles and runs correctly.

### Profiling: head-first vs seq-first kernel (C++ harness, LLAMA shape)

| Metric | Head-First | Seq-First | Diff |
|--------|-----------|----------|------|
| **Correctness** | corr=0.9976 | corr=0.9976 | Identical |
| **Min latency** | **15188 us** | 16647 us | +9.6% |
| **Avg latency** | **15196 us** | 16796 us | +10.5% |
| **GFLOPS** | **2261** | 2046 | -9.5% |

Profiled with correct data layouts: head-first harness generates `[32, 2048, 64]`, seq-first harness generates `[2048, 2048]`. Both with 10 warmup + 20 measured iterations.

### Why seq-first kernel is 10% slower

The overhead comes from **strided DDR access** at the L3→L2 (shim DMA) level:

**Head-first**: each head's Q data is a contiguous `[tile_size_q, dk]` = 8KB block. DMA reads one large sequential burst. DDR prefetcher works perfectly.

**Seq-first**: each head's Q data is scattered — 64 elements per position, then skip 1984 elements (other heads), then 64 more. The DMA issues 64 separate 128-byte reads, each 4KB apart. This causes:
- 64 separate DMA requests vs 1 large burst
- DDR prefetcher can't predict the strided pattern
- NoC per-request overhead

### Net effect in LLAMA pipeline

| Factor | Impact |
|--------|--------|
| Kernel overhead (strided DMA) | +1.6ms/layer × 16 = **+25.6ms** |
| Host transpose eliminated | -2.5ms/layer × 16 = **-40ms** |
| **Net savings** | **~14ms** |

### Future: MemTile extraction approach

To eliminate the DMA overhead entirely:
1. L3→L2: read full contiguous rows `[tile_size_q, emb_dim]` to MemTile
2. L2→L1: extract per-head `[tile_size_q, dk]` from wider L2 buffer (on-chip, fast)

This makes DDR access identical to head-first. Challenge: L2 capacity — `tile_size_q * emb_dim * 2 = 256KB` fills one MemTile. Would need to tile the Q rows into smaller batches.

### Files

| File | Purpose |
|------|---------|
| `attn_npu2_seqfirst.py` | Seq-first variant (working, 10% slower) |
| `test_seqfirst.py` | Host-side validation test |
| `test_elf_npu2_seqfirst.cpp` | C++ profiling harness with seq-first data |

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

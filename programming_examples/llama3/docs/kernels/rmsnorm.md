# Weighted RMSNorm Kernel — Multi-Tile Investigation

## Role in LLAMA Pipeline

Steps 1 and 10 of each transformer block: weighted RMS normalization.

```
y = x * rsqrt(mean(x², axis=-1) + eps) * weight
```

| Parameter | Value |
|-----------|-------|
| Input | [2048, 2048] BF16 (seq_len × emb_dim) |
| Weight | [2048] BF16 (per-column) |
| Output | [2048, 2048] BF16 |
| Invocations | 2 per layer × 16 layers + 1 final = 33 total |

---

## Current Status

| Config | Herd | Time | Corr | Status |
|--------|------|------|------|--------|
| **Current (single tile)** | **[1,1]** | **6.0ms** | 0.9999 | **Working** |
| IRON reference | 16 tiles (8 cols × 2 channels) | 4.3ms | 0.9999 | Working |

**Gap: 1.4× slower than IRON** (6.0ms vs 4.3ms).

---

## Multi-Tile Investigation (2026-03-31)

### Why parallelization is simple in theory

Each row's RMSNorm is **independent** — no cross-row dependencies. With 8 tiles, each processes 256 rows (2048/8). This is the same pattern as eltwise_add (which works at [8,1]).

### What we tried

| Attempt | Approach | herd=[1,1] | herd=[2,1]+ | Issue |
|---------|----------|-----------|-------------|-------|
| 1 | Single herd, weight DMA before loop | ✅ OK | ❌ FAIL | aiecc dominance error |
| 2 | 2-herd pipeline (norm + mul, like IRON) | ✅ OK | ❌ FAIL | Same issue in mul_herd |
| 3 | Weight DMA inside row loop | ❌ FAIL | ❌ FAIL | BD allocator exhaustion (2048 BDs needed, 16 available) |
| 4 | Unweighted only (no weight) | ✅ OK | ✅ OK | Works at all herd sizes |

### Root cause

**aiecc compiler bug** with one-shot broadcast DMA to multiple tiles.

The failing pattern:
```mlir
air.herd tile(%tx, %ty) in (8, 1) args(%h_weight = %weight) : memref<2048xbf16> {
    dma_memcpy_nd(%l1_weight, %h_weight)   // ← one-shot DMA, no offsets
    scf.for ... {
        // per-row computation using l1_weight
    }
}
```

When `air-to-aie` expands this herd to 8 cores, each core gets a replicated weight DMA. The resulting AIE IR has SSA dominance violations — the replicated DMAs reference values that don't dominate across core boundaries.

**Why unweighted works**: All DMAs have tile-dependent offsets (`src_offsets=[row, 0]`), which the compiler correctly replicates per-core. The one-shot weight DMA (no offsets, full copy) is the specific pattern that breaks.

**Why IRON doesn't have this issue**: IRON uses ObjectFIFO (from mlir-aie) which generates different DMA patterns — a shared FIFO with multiple consumers, not replicated point-to-point DMAs.

### Workaround: Unweighted multi-tile + separate weight multiply

Split weighted RMSNorm into two separate kernels:
1. **Unweighted RMSNorm** on [8,1] herd: `y = x * rsqrt(mean(x²) + eps)` — verified to compile at all herd sizes
2. **Elementwise weight multiply** on [8,1] herd: `z = y * weight` — same pattern as eltwise_add (proven to work)

**Implementation options:**
- Multi-launch stitching (like FFN): stitch both kernels into one ELF with shared intermediate buffer
- Two separate kernel invocations: simpler but adds host overhead
- 2-herd in same func: requires separate intermediate buffer argument (4th func arg); initial attempt had sequencing issues (in-place output read/write between herds produced garbage)

**Status**: Not yet implemented. The total gap is ~50ms across 33 invocations (6ms vs 4.3ms × 33). Lower priority than FFN optimization which saved 100ms/layer.

### IRON's approach

IRON uses the same 2-stage pipeline but at the kernel level:
- `core_body_norm`: RMSNorm without weight (8 Workers)
- `core_body_mul`: Elementwise multiply by weight (8 Workers)
- Weight is broadcast via a single shared ObjectFIFO (`of_in2s`)
- Total: 16 Workers = 16 compute tiles

See `design_weighted.py` in IRON: `/home/jiajli/apps/IRON/iron/operators/rms_norm/design_weighted.py`

---

## Files

| File | Purpose |
|------|---------|
| `weighted_rms_norm/weighted_rms_norm.py` | Multi-tile weighted RMSNorm (2-herd approach) |
| `weighted_rms_norm/Makefile` | Build targets |

## Commands

```bash
cd programming_examples/weighted_rms_norm

# Single tile (baseline)
make run M=2048 N=2048 HERD_X=1

# Multi-tile
make run M=2048 N=2048 HERD_X=8

# Profile
make profile M=2048 N=2048 HERD_X=8
```

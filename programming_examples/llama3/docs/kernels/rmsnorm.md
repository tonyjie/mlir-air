# Weighted RMSNorm Kernel — Performance Analysis

## Role in LLAMA Pipeline

Steps 1 and 10 of each transformer block: weighted RMS normalization.

```
y = x * rsqrt(mean(x^2, axis=-1) + eps) * weight
```

| Parameter | Value |
|-----------|-------|
| Input | [2048, 2048] BF16 (seq_len x emb_dim) for prefill; [1, 2048] for decode |
| Weight | [2048] BF16 (per-column, broadcast to all tiles) |
| Output | [2048, 2048] BF16 for prefill; [1, 2048] for decode |
| Invocations | 2 per layer x 16 layers + 1 final = 33 total (prefill) |

---

## Current Status: 8-Tile with Broadcast Weight DMA

| Config | Herd | Time | vs IRON | Status |
|--------|------|------|---------|--------|
| Single tile (old) | [1,1] | 6.0ms | 1.4x slower | Working |
| **8-tile (current)** | **[8,1]** | **0.9ms** | **4.8x faster** | **Working** |
| IRON reference | 16 tiles | 4.3ms | baseline | Working |

**AIR is now 4.8x faster than IRON** for standalone RMSNorm.

In the LLAMA prefill pipeline:
- `rms_attn_gemms`: 14ms -> **9ms** (with 8-tile RMSNorm launch)
- `ffn_full`: 57ms -> **52ms** (with 8-tile RMSNorm launch)
- `rmsnorm` standalone: 6ms -> **3ms**

---

## Architecture

The 8-tile RMSNorm uses a single herd with broadcast weight DMA:

```
herd [8, 1]:
  1. Broadcast weight vector to all 8 tiles (one DMA, no tile-dependent offset)
  2. Each tile processes M/8 rows:
     for row in tile's partition:
       DMA row from L3 -> L1 (tile-dependent offset)
       Compute: sum(x^2), rsqrt, scale by weight
       DMA result from L1 -> L3 (tile-dependent offset)
```

This requires the broadcast DMA pattern (same data to all tiles without tile-dependent offset), which was previously blocked by a compiler bug (stride=0 BD rejected by `aie.dma_bd` verifier). The bug was fixed in upstream MLIR-AIR (2026-04).

### Why it works now

The `air-to-aie` lowering now correctly handles DMAs with no tile-dependent offsets. The generated BD uses a repeat pattern that the AIE hardware supports natively. See `docs/issues/github_issue_weight_broadcast_dma.md` for the full investigation and resolution.

### Decode: stays at [1,1]

Decode RMSNorm uses M=1 (single token). Multi-tile requires `M % herd_x == 0`, so herd_x > 1 is impossible for M=1. Decode RMSNorm stays at [1,1], which runs in ~0.3ms.

---

## History

The original investigation (2026-03-31) tried several approaches before finding the root cause:

| Attempt | Approach | Result | Root Cause |
|---------|----------|--------|------------|
| 1 | Single herd, weight DMA before loop | FAIL at herd > [1,1] | Broadcast DMA compiler bug |
| 2 | 2-herd (norm + mul) | FAIL | Same bug in mul_herd weight DMA |
| 3 | Weight DMA inside loop | FAIL | BD exhaustion (2048 BDs needed) |
| 4 | Unweighted only (no weight) | PASS at all herds | Confirms bug is broadcast-specific |
| **5** | **Single herd after compiler fix** | **PASS** | **Bug fixed upstream** |

---

## Files

| File | Purpose |
|------|---------|
| `weighted_rms_norm/weighted_rms_norm.py` | Kernel builder (herd_x=1 and herd_x>1 paths) |
| `weighted_rms_norm/Makefile` | Build targets |

## Commands

```bash
cd programming_examples/weighted_rms_norm

# Single tile
python3 weighted_rms_norm.py --M 2048 --N 2048 --herd-x 1 --profile

# 8-tile (current default in LLAMA pipeline)
python3 weighted_rms_norm.py --M 2048 --N 2048 --herd-x 8 --profile
```

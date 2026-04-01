# GitHub Issue: Broadcast DMA generates `stride=0` in `aie.dma_bd`, rejected by AIE backend

## Title

`air-to-aie`: broadcast DMA (one-shot copy to multi-tile herd) generates `stride=0` BD dimension, rejected by `aie.dma_bd` verifier

## Labels

bug, air-to-aie, dma, broadcast, multi-tile

---

## Summary

When a multi-tile herd (`sizes=[N,1]` where N>1) has a DMA that copies the **same data** to all tiles (no tile-dependent offsets), the `air-to-aie` lowering generates a shim DMA BD with `stride=0` in the repeat dimension. The `aie.dma_bd` verifier rejects this: `Stride 1 must be a positive integer`.

This blocks multi-tile parallelization of any kernel that broadcasts shared data (e.g. weight vectors in RMSNorm, shared constants, lookup tables).

## Reproducer

Using the `weighted_rms_norm` example at `programming_examples/weighted_rms_norm/`:

```bash
cd programming_examples/weighted_rms_norm

# Single tile — PASSES
python3 weighted_rms_norm.py --output-format xclbin --M 128 --N 128 --herd-x 1
# PASS!

# Two tiles — FAILS
python3 weighted_rms_norm.py --output-format xclbin --M 128 --N 128 --herd-x 2
# error: 'aie.dma_bd' op Stride 1 must be a positive integer.
```

The relevant AIR pattern:

```mlir
// Weighted RMSNorm: weight vector is shared across all tiles (no tile-dependent offset)
air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c1)
    args(%h_in=%input, %h_weight=%weight, %h_out=%output)
    : memref<128x128xbf16>, memref<128xbf16>, memref<128x128xbf16> {

    %l1_weight = memref.alloc() : memref<128xbf16, 2 : i32>
    // One-shot DMA: same weight vector to ALL tiles (no offsets, no tile-index dependency)
    air.dma_memcpy_nd (%l1_weight, %h_weight) : (memref<128xbf16, 2 : i32>, memref<128xbf16>)

    scf.for %row = ... {
        // Per-tile rows (tile-dependent offsets — these work fine)
        air.dma_memcpy_nd (%l1_in, %h_in[%row_offset, %c0] [...] [...])
        // compute using l1_weight and l1_in
    }
}
```

## Error Details

After `air-to-aie` lowering and shim DMA BD optimization, the generated IR contains:

```mlir
// From aiecc failure IR (saved automatically):
aie.dma_bd(%arg1 : memref<128xbf16>, 0, 4096,
    [<size = 32, stride = 0>,    // ← REJECTED: stride must be > 0
     <size = 128, stride = 1>])
```

The `<size = 32, stride = 0>` dimension represents repeating the 128-element weight vector 32 times (for broadcast to 2 tiles across multiple loop iterations). The AIE DMA hardware DOES support repeat with stride=0 (repeat_count in BD), but the `aie.dma_bd` verifier rejects it.

Full error:
```
air_project/npu.air.mlir:212:9: error: 'aie.dma_bd' op Stride 1 must be a positive integer.
```

## Analysis

**Why single-tile works:** With `herd=[1,1]`, the weight DMA is point-to-point (one source, one destination). No broadcast pattern is generated.

**Why multi-tile fails:** With `herd=[N,1]` where N>1, the weight DMA needs to send the same data to N tiles. The `air-to-aie` pass converts this into a shim DMA BD with a repeat dimension (`size=N*iterations, stride=0`). The `aie.dma_bd` verifier rejects stride=0.

**Why tile-dependent DMAs work:** DMAs with tile-dependent offsets (e.g. `src_offsets=[row_offset]` where `row_offset` depends on `%tx`) generate non-zero strides in the BD and pass validation.

**Isolation tests:**

| Test | Herd | Weight DMA | Result |
|------|------|-----------|--------|
| Weighted RMSNorm | [1,1] | One-shot (no offsets) | PASS |
| Weighted RMSNorm | [2,1] | One-shot (no offsets) | **FAIL** (stride=0) |
| Weighted RMSNorm | [8,1] | One-shot (no offsets) | **FAIL** (stride=0) |
| Unweighted RMSNorm | [8,1] | No weight DMA | PASS |
| Eltwise Add | [8,1] | All DMAs tile-dependent | PASS |

## Suggested Fix Options

1. **Fix `aie.dma_bd` verifier**: Allow stride=0 for repeat/broadcast patterns. The AIE hardware supports this via `repeat_count` in BD configuration.

2. **Fix `air-to-aie` broadcast lowering**: Generate a different pattern for broadcast DMAs — e.g., use `repeat_count` in the BD directly instead of encoding it as a `<size=N, stride=0>` dimension.

3. **Use ObjectFIFO for broadcast**: IRON avoids this issue by using ObjectFIFO (shared FIFO with multiple consumers) instead of replicated point-to-point DMAs.

## Impact

Blocks multi-tile parallelization for any kernel with shared/broadcast data:
- **Weighted RMSNorm**: Cannot use herd > [1,1] (6ms vs IRON's 4.3ms with 16 tiles)
- **Any kernel with shared LUTs, constants, or weights**: Same pattern

## Workaround

Use single-tile herd (`[1,1]`) for kernels with broadcast data. For RMSNorm specifically, an alternative is to split into unweighted RMSNorm (works multi-tile) + separate weight multiply.

## Environment

- mlir-air: built from source (current HEAD)
- mlir-aie: installed from `my_install/mlir-aie/`
- Target: NPU2 (AIE2P, Strix)
- Reproducer: `programming_examples/weighted_rms_norm/weighted_rms_norm.py --herd-x 2`

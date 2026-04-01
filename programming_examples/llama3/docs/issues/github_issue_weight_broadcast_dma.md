# GitHub Issue: Broadcast DMA generates `stride=0` in `aie.dma_bd`, rejected by AIE backend

## Title

`air-to-aie`: broadcast DMA (one-shot copy to multi-tile herd) causes SSA dominance violation or invalid `stride=0` BD

## Labels

bug, air-to-aie, dma, broadcast, multi-tile

---

## Summary

When a multi-tile herd (`sizes=[N,1]` where N>1) has a DMA that copies the **same data** to all tiles (no tile-dependent offsets), the `air-to-aie` lowering fails. Depending on the exact pattern, the error is either:

1. **SSA dominance violation**: `operand #N does not dominate this use` — when herd expansion creates references across core boundaries
2. **Invalid BD stride**: `'aie.dma_bd' op Stride 1 must be a positive integer` — when the broadcast is encoded as `<size=N, stride=0>`

This blocks multi-tile parallelization of any kernel that broadcasts shared data (e.g. weight vectors in RMSNorm, shared constants, lookup tables).

## Reproducer

Self-contained minimal script — a multi-tile herd with one DMA that has no tile-dependent offsets (broadcast pattern):

```python
#!/usr/bin/env python3
"""Reproducer: broadcast DMA to multi-tile herd generates stride=0 BD.

herd=[1,1] → PASS
herd=[2,1] → FAIL: 'aie.dma_bd' op Stride 1 must be a positive integer.
"""
from ml_dtypes import bfloat16
from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import type_mapper

range_ = for_

def build_module(M, N, herd_x):
    """Eltwise scale: output[i] = input[i] * weight[i % N].

    Weight is broadcast (same data to all tiles — no tile-dependent offset).
    """
    @module_builder
    def mod():
        xrt = type_mapper(bfloat16)
        in_ty = MemRefType.get([M * N], xrt)
        wt_ty = MemRefType.get([N], xrt)
        out_ty = MemRefType.get([M * N], xrt)
        l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_row = MemRefType.get([N], xrt, memory_space=l1)
        l1_wt = MemRefType.get([N], xrt, memory_space=l1)
        vec_ty = VectorType.get([16], xrt)
        imap = AffineMapAttr.get(AffineMap.get_identity(1))
        rows_per_tile = M // herd_x

        @FuncOp.from_py_func(in_ty, wt_ty, out_ty)
        def scale(inp, weight, out):
            @launch(operands=[inp, weight, out])
            def launch_body(l_in, l_wt, l_out):
                @segment(name="seg", operands=[l_in, l_wt, l_out])
                def seg_body(s_in, s_wt, s_out):
                    @herd(name="herd", sizes=[herd_x, 1], operands=[s_in, s_wt, s_out])
                    def herd_body(tx, ty, sx, sy, h_in, h_wt, h_out):
                        l1_w = AllocOp(l1_wt, [], [])
                        l1_r = AllocOp(l1_row, [], [])
                        l1_o = AllocOp(l1_row, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst = arith.ConstantOp(xrt, 0.0)

                        # BROADCAST DMA: weight → all tiles (NO tile-dependent offset)
                        dma_memcpy_nd(l1_w, h_wt)

                        for row in range_(0, rows_per_tile, 1):
                            # Tile-dependent DMA (works fine)
                            off_map = AffineMap.get(0, 2, [
                                AffineExpr.get_mul(
                                    AffineExpr.get_add(
                                        AffineSymbolExpr.get(0),
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(1),
                                            AffineConstantExpr.get(rows_per_tile))),
                                    AffineConstantExpr.get(N))])
                            from air.dialects.affine import apply as affine_apply
                            off = affine_apply(off_map, [row, tx])
                            dma_memcpy_nd(l1_r, h_in, src_offsets=[off], src_sizes=[N], src_strides=[1])
                            for j in range_(0, N, 16):
                                sr = subview(l1_r.result, [j], [16], [1])
                                sw = subview(l1_w.result, [j], [16], [1])
                                so = subview(l1_o.result, [j], [16], [1])
                                vr = transfer_read(vec_ty, sr, [c0], imap, cst, [True])
                                vw = transfer_read(vec_ty, sw, [c0], imap, cst, [True])
                                transfer_write(None, arith.mulf(vr, vw), so, [c0], imap, [True])
                                yield_([])
                            dma_memcpy_nd(h_out, l1_o, dst_offsets=[off], dst_sizes=[N], dst_strides=[1])
                            yield_([])
                        DeallocOp(l1_w); DeallocOp(l1_r); DeallocOp(l1_o)
    return mod()

def compile_test(name, module):
    backend = XRTBackend(verbose=False, omit_while_true_loop=False, instance_name=name)
    try:
        backend.compile(module); backend.unload(); return True, "OK"
    except Exception as e:
        backend.unload()
        lines = [l for l in str(e).split("\n") if "error:" in l.lower()]
        return False, lines[0][:150] if lines else str(e)[:150]

print("Test 1: herd=[1,1] — single tile (should PASS)")
ok, msg = compile_test("t1", build_module(128, 64, herd_x=1))
print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

print("Test 2: herd=[2,1] — multi-tile with broadcast weight DMA (should FAIL)")
ok, msg = compile_test("t2", build_module(128, 64, herd_x=2))
print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
```

**Expected output:**
```
Test 1: herd=[1,1] — single tile (should PASS)
  PASS: OK
Test 2: herd=[2,1] — multi-tile with broadcast weight DMA (should FAIL)
  FAIL: error: operand #1 does not dominate this use
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

Two failure modes depending on the exact IR pattern:

**Failure mode 1: SSA dominance violation** (from the self-contained reproducer above)

When `air-to-aie` expands the herd to N cores, the broadcast DMA's SSA value (written once for all tiles) gets replicated but references values that don't dominate across the replicated core regions:

```
error: operand #1 does not dominate this use
```

**Failure mode 2: Invalid BD stride** (from `weighted_rms_norm.py --herd-x 2`, which has a slightly different DMA pattern)

After `air-to-aie` lowering and shim DMA BD optimization, the broadcast is encoded as a repeat dimension with stride=0:

```mlir
aie.dma_bd(%arg1 : memref<128xbf16>, 0, 4096,
    [<size = 32, stride = 0>,    // ← REJECTED: stride must be > 0
     <size = 128, stride = 1>])
```

```
error: 'aie.dma_bd' op Stride 1 must be a positive integer.
```

Both failures have the same root cause: the compiler doesn't properly handle DMAs with no tile-dependent offsets (broadcast pattern) when expanding herds to multiple tiles.

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

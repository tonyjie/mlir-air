# GitHub Issue: `airrt-to-npu` fails for multi-launch ELF with bare `air.herd` (no `air.segment`)

> **Compiler issue record.** Documents a specific compiler bug or limitation
> encountered during development. May be resolved in newer compiler versions.


## Title

`airrt-to-npu`: `identifyLaunchRegions` only handles `SegmentLoadOp`, silently drops bare `HerdLoadOp` launches in multi-launch ELF

## Labels

bug, airrt-to-npu, multi-launch, elf

---

## Summary

When a multi-launch ELF module contains a launch with a bare `air.herd` (no `air.segment` wrapper) alongside launches that DO have `air.segment`, the `airrt-to-npu` pass fails with:

```
error: failed to legalize operation 'airrt.dma_memcpy_nd' that was explicitly marked illegal
```

## Reproducer

Self-contained Python script — two identical eltwise-add launches. The only difference is whether each has an `air.segment` wrapper:

```python
#!/usr/bin/env python3
"""Reproducer: airrt-to-npu fails when mixing segment/bare-herd launches.

Test 1: Both have air.segment → PASS
Test 2: One has segment, one has bare herd → FAIL
"""
from ml_dtypes import bfloat16
from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import type_mapper

range_ = for_

def _build_add_launch(xrt_dtype, n, tile_n, herd_x, vector_size, use_segment, name="add"):
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    imap = AffineMapAttr.get(AffineMap.get_identity(1))
    chunk = n // herd_x
    omap = AffineMap.get(0, 2, [AffineExpr.get_add(
        AffineSymbolExpr.get(0),
        AffineExpr.get_mul(AffineSymbolExpr.get(1), AffineConstantExpr.get(tile_n)))])

    def _herd_body(ha, hb, ho, tx, ty):
        l1a = AllocOp(l1_ty, [], [])
        l1b = AllocOp(l1_ty, [], [])
        l1o = AllocOp(l1_ty, [], [])
        c0 = arith.ConstantOp.create_index(0)
        cst = arith.ConstantOp(xrt_dtype, 0.0)
        for iv in range_(0, chunk, tile_n):
            off = affine_apply(omap, [iv, tx])
            dma_memcpy_nd(l1a, ha, src_offsets=[off], src_sizes=[tile_n], src_strides=[1])
            dma_memcpy_nd(l1b, hb, src_offsets=[off], src_sizes=[tile_n], src_strides=[1])
            for j in range_(0, tile_n, vector_size):
                sa = subview(l1a.result, [j], [vector_size], [1])
                sb = subview(l1b.result, [j], [vector_size], [1])
                so = subview(l1o.result, [j], [vector_size], [1])
                va = transfer_read(vec_ty, sa, [c0], imap, cst, [True])
                vb = transfer_read(vec_ty, sb, [c0], imap, cst, [True])
                transfer_write(None, arith.addf(va, vb), so, [c0], imap, [True])
                yield_([])
            dma_memcpy_nd(ho, l1o, dst_offsets=[off], dst_sizes=[tile_n], dst_strides=[1])
            yield_([])
        DeallocOp(l1a); DeallocOp(l1b); DeallocOp(l1o)

    def builder(a, b, out):
        @launch(operands=[a, b, out])
        def the_launch(la, lb, lo):
            if use_segment:
                @segment(name=f"{name}_seg", operands=[la, lb, lo])
                def seg(sa, sb, so):
                    @herd(name=f"{name}_herd", sizes=[herd_x, 1], operands=[sa, sb, so])
                    def h(tx, ty, sx, sy, ha, hb, ho): _herd_body(ha, hb, ho, tx, ty)
            else:
                @herd(name=f"{name}_herd", sizes=[herd_x, 1], operands=[la, lb, lo])
                def h(tx, ty, sx, sy, ha, hb, ho): _herd_body(ha, hb, ho, tx, ty)
    return builder

def build_two_launches(n, use_segment_1, use_segment_2):
    @module_builder
    def mod():
        xrt = type_mapper(bfloat16)
        l3_ty = MemRefType.get([n], xrt)
        b1 = _build_add_launch(xrt, n, 128, 8, 16, use_segment_1, name="add1")
        b2 = _build_add_launch(xrt, n, 128, 8, 16, use_segment_2, name="add2")
        @FuncOp.from_py_func(l3_ty, l3_ty, l3_ty, l3_ty, l3_ty, l3_ty)
        def test(a1, b1_arg, out1, a2, b2_arg, out2):
            b1(a1, b1_arg, out1); b2(a2, b2_arg, out2)
    return mod()

def compile_test(name, module):
    backend = XRTBackend(verbose=False, omit_while_true_loop=False,
                         output_format="elf", instance_name=name)
    try:
        backend.compile(module); backend.unload(); return True, "OK"
    except Exception as e:
        backend.unload()
        lines = [l for l in str(e).split("\n") if "error:" in l.lower()]
        return False, lines[0][:150] if lines else str(e)[:150]

N = 1024
print("Test 1: Both with air.segment (should PASS)")
ok, msg = compile_test("both_seg", build_two_launches(N, True, True))
print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

print("Test 2: Mixed — launch 1 has segment, launch 2 has bare herd (should FAIL)")
ok2, msg2 = compile_test("mixed", build_two_launches(N, True, False))
print(f"  {'PASS' if ok2 else 'FAIL'}: {msg2}")
```

**Expected output:**
```
Test 1: Both with air.segment (should PASS)
  PASS: OK
Test 2: Mixed — launch 1 has segment, launch 2 has bare herd (should FAIL)
  FAIL: failed to legalize operation 'airrt.dma_memcpy_nd' that was explicitly marked illegal
```

## Root Cause

In `AIRRtToNpuPass.cpp`, `identifyLaunchRegions()` (~line 1005) only walks for `airrt::SegmentLoadOp`:

```cpp
forOp.walk([&](airrt::SegmentLoadOp segLoadOp) {
    StringRef deviceName = segLoadOp.getSymName();
    AIE::DeviceOp device = getDeviceByName(module, deviceName);
    if (device) {
        regions.push_back({forOp, deviceName, device});
    }
});
```

Launches with bare `air.herd` (no `air.segment`) lower to `airrt::HerdLoadOp`, which is **not matched**. There IS a fallback path (~line 1553) that handles both:

```cpp
if (regions.empty()) {
    funcOp.walk([&](Operation *o) {
        if (isa<airrt::SegmentLoadOp, airrt::HerdLoadOp>(o)) { ... }
    });
}
```

But this only runs when `regions.empty()`. In the mixed case, segment-based launches populate `regions`, so the fallback is skipped. The bare-herd launch's DMA ops are orphaned outside any `aie.device`, and `DmaToNpuPattern` fails because `op->getParentOfType<AIE::DeviceOp>()` returns null.

## Suggested Fix

In `identifyLaunchRegions()`, also handle `HerdLoadOp`:

```cpp
forOp.walk([&](Operation *op) {
    StringRef deviceName;
    if (auto segLoad = dyn_cast<airrt::SegmentLoadOp>(op))
        deviceName = segLoad.getSymName();
    else if (auto herdLoad = dyn_cast<airrt::HerdLoadOp>(op))
        deviceName = herdLoad.getSymName();
    else
        return;
    AIE::DeviceOp device = getDeviceByName(module, deviceName);
    if (device)
        regions.push_back({forOp, deviceName, device});
});
```

## Workaround

Wrap all bare `air.herd` ops in `air.segment` before combining into multi-launch modules.

## Environment

- mlir-air: built from source
- Target: NPU2 (AIE2P, Strix)
- Output format: ELF (`--output-format elf`)

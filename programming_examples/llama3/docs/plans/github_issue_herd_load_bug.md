# GitHub Issue Draft: `airrt-to-npu` fails for multi-launch ELF with bare `air.herd` (no `air.segment`)

---

## Title

`airrt-to-npu`: bare `air.herd` launches silently dropped in multi-launch ELF, causing "failed to legalize airrt.dma_memcpy_nd"

## Labels

bug, airrt-to-npu, multi-launch, elf

## Description

### Summary

When a multi-launch ELF module contains a launch with a bare `air.herd` (no `air.segment` wrapper) alongside launches that DO have `air.segment`, the `airrt-to-npu` pass fails with:

```
error: failed to legalize operation 'airrt.dma_memcpy_nd' that was explicitly marked illegal
```

The root cause is that `identifyLaunchRegions()` in `AIRRtToNpuPass.cpp` only looks for `airrt::SegmentLoadOp` to associate launch regions with `aie.device` ops. Launches that lower to `airrt::HerdLoadOp` (from bare `air.herd`) are silently skipped. Their DMA ops are left orphaned outside any `aie.device`, and `DmaToNpuPattern::matchAndRewrite` fails because `op->getParentOfType<AIE::DeviceOp>()` returns null.

### Reproducer

Two identical eltwise-add launches in a single function. The only difference is whether each launch wraps its herd in an `air.segment`:

```mlir
// WORKS: Both launches have air.segment
func.func @test(%arg0: ..., %arg5: ...) {
    air.launch args(...) {
        air.segment @add1_seg args(...) {      // <-- segment present
            air.herd @add1_herd tile(...) { ... }
        }
    }
    air.launch args(...) {
        air.segment @add2_seg args(...) {      // <-- segment present
            air.herd @add2_herd tile(...) { ... }
        }
    }
    return
}

// FAILS: Second launch has bare herd (no segment)
func.func @test(%arg0: ..., %arg5: ...) {
    air.launch args(...) {
        air.segment @add1_seg args(...) {      // <-- segment present
            air.herd @add1_herd tile(...) { ... }
        }
    }
    air.launch args(...) {
        air.herd @add2_herd tile(...) { ... }  // <-- NO segment wrapper
    }
    return
}
```

Self-contained Python reproducer (requires `air` Python bindings): [`repro_herd_load_bug.py`](https://github.com/.../repro_herd_load_bug.py)

```bash
python3 repro_herd_load_bug.py
# Test 1: Both launches have air.segment (should PASS) → PASS: OK
# Test 2: Launch 1 has segment, Launch 2 has bare herd (should FAIL) → FAIL: failed to legalize
```

### Root Cause Analysis

In `AIRRtToNpuPass.cpp`, `identifyLaunchRegions()` at ~line 1005:

```cpp
forOp.walk([&](airrt::SegmentLoadOp segLoadOp) {
    StringRef deviceName = segLoadOp.getSymName();
    AIE::DeviceOp device = getDeviceByName(module, deviceName);
    if (device) {
        regions.push_back({forOp, deviceName, device});
    }
});
```

This only walks for `SegmentLoadOp`. Launches with bare herds lower to `airrt.herd_load` (not `airrt.segment_load`), so they are never added to `regions`.

There IS a fallback path (~line 1551) that handles both `SegmentLoadOp` and `HerdLoadOp`:

```cpp
if (regions.empty()) {
    // Fallback: no launch boundaries found, use old behavior
    funcOp.walk([&](Operation *o) {
        if (isa<airrt::SegmentLoadOp, airrt::HerdLoadOp>(o)) {
            ...
        }
    });
}
```

But this fallback only runs when `regions.empty()`. In the mixed case (some segment, some herd), `regions` is non-empty, so the fallback is skipped.

Later, `moveFuncOpToEndOfDeviceOp` splits the function into per-device functions (one per identified region). The unidentified bare-herd region's DMA ops are lost — they were in the original func which is erased (line 1643: `funcOp.erase()`).

### Expected Behavior

All launch regions should be correctly identified and associated with their `aie.device`, regardless of whether they use `air.segment` or bare `air.herd`.

### Suggested Fix

In `identifyLaunchRegions()`, also walk for `airrt::HerdLoadOp`:

```cpp
// Handle both segment-based and herd-based launches
forOp.walk([&](Operation *op) {
    StringRef deviceName;
    if (auto segLoad = dyn_cast<airrt::SegmentLoadOp>(op))
        deviceName = segLoad.getSymName();
    else if (auto herdLoad = dyn_cast<airrt::HerdLoadOp>(op))
        deviceName = herdLoad.getSymName();
    else
        return;

    AIE::DeviceOp device = getDeviceByName(module, deviceName);
    if (device) {
        regions.push_back({forOp, deviceName, device});
    }
});
```

### Workaround

Wrap all bare `air.herd` ops in `air.segment` before combining into a multi-launch module. This ensures `airrt.segment_load` is generated for every launch.

### Impact

This blocks multi-launch ELF compilation for any module that mixes segment-wrapped launches (e.g., GEMM with tiled data movement) with simpler bare-herd launches (e.g., element-wise operations, normalization kernels). Our use case: combining RMSNorm (bare herd) + FFN GEMM (with segment) + eltwise add in a single 6-launch ELF for LLAMA inference.

### Environment

- mlir-air: built from source (current HEAD)
- mlir-aie: installed from `my_install/mlir-aie/`
- Target: NPU2 (AIE2P, Strix)
- Output format: ELF (`--output-format elf`)

# Multi-Launch ELF Compilation Failure — Root Cause Analysis

> **Compiler issue record.** Documents a specific compiler bug or limitation
> encountered during development. May be resolved in newer compiler versions.


**Purpose**: Document the `airrt-to-npu` legalization failure when combining 6+ `air.launch` ops in a single ELF. Intended for compiler engineers working on the AIR/AIE toolchain.

**Environment**: MLIR-AIR on NPU2 (AIE2P, Strix), ELF output format.

---

## Summary

Combining 4 `air.launch` operations in one function works correctly (the FFN multi-launch kernel). However, combining 6 launches fails during `airrt-to-npu` conversion with `failed to legalize operation 'airrt.dma_memcpy_nd'`. The 56-pass aircc pipeline completes successfully; the failure is in the aiecc backend.

---

## Reproduction

### Working case: 4 launches (FFN block)

```bash
cd programming_examples/llama3/ffn_swiglu
make run   # Compiles and runs successfully
```

Module structure:
```mlir
func.func @ffn_block(%arg0..%arg7 : all 2D bf16 memrefs) {
    air.launch 1: Gate GEMM   [8,4] herd, 3 herds inside
    air.launch 2: Up GEMM     [8,4] herd, 3 herds inside
    air.launch 3: SwiGLU      [8,1] herd, collapse_shape 2D→1D inside launch
    air.launch 4: Down GEMM   [8,4] herd, 3 herds inside
    return
}
```

**Key**: The SwiGLU launch uses `memref.collapse_shape` inside the launch body to convert 2D func args to 1D for the herd. This pattern is fully resolved during the aircc pass pipeline — zero `collapse_shape` references remain at the airrt level.

### Failing case: 6 launches (RMSNorm + FFN + Residual Add)

```bash
cd programming_examples/llama3
python3 ffn_full_multi.py -p   # Prints valid 665-line MLIR
# Compilation fails during aiecc stage
```

Module structure:
```mlir
func.func @ffn_full(%arg0..%arg10 : mixed 2D/1D bf16 memrefs) {
    air.launch 1: RMSNorm     [1,1] herd
    air.launch 2: Gate GEMM   [8,4] herd, 3 herds inside
    air.launch 3: Up GEMM     [8,4] herd, 3 herds inside
    air.launch 4: SwiGLU      [8,1] herd, collapse_shape 2D→1D inside launch
    air.launch 5: Down GEMM   [8,4] herd, 3 herds inside
    air.launch 6: Eltwise Add [8,1] herd, collapse_shape 2D→1D inside launch
    return
}
```

### Error output

```
loc("-":24555:12): error: failed to legalize operation 'airrt.dma_memcpy_nd' that was explicitly marked illegal
Error: pass failed: airrt-to-npu{ trace-size=0 trace-offset=0 output-elf=true}
```

---

## Analysis

### Pipeline stage that succeeds vs fails

| Stage | 4-launch (works) | 6-launch (fails) |
|-------|-------------------|-------------------|
| AIR module generation | ✅ 543 lines | ✅ 665 lines |
| aircc pass pipeline (56 passes) | ✅ All pass | ✅ All pass |
| airrt-level IR generation | ✅ | ✅ (803 airrt.dma_memcpy_nd ops) |
| **aiecc: airrt-to-npu** | **✅** | **❌ legalization failure** |

### What airrt-to-npu does

For each `airrt.dma_memcpy_nd` operation, the pass:
1. Traces the memref operand back to a **function argument** to determine the XRT BO index
2. Maps the DMA to a physical ShimDMA tile
3. Generates NPU instruction sequences (BD configuration)

### What goes wrong in the 6-launch case

In the final airrt IR (pass_053_after_cse.mlir), some DMA operations reference `%collapse_shape` results instead of function arguments:

```mlir
// From 6-launch lowered IR — the problematic pattern:
%collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<2048x2048xbf16> into memref<4194304xbf16>

// These DMAs reference %collapse_shape, not a func arg:
%1 = airrt.dma_memcpy_nd(..., %collapse_shape[0, 0, 0, 0], [1, 1, 1, 524288], [0, 0, 0, 1])
%2 = airrt.dma_memcpy_nd(..., %collapse_shape[0, 0, 0, 524288], [1, 1, 1, 524288], [0, 0, 0, 1])
```

The `airrt-to-npu` pass cannot trace `%collapse_shape` → `%arg0` → BO #0, so it fails to legalize these DMA ops.

### Why the same pattern works in the 4-launch FFN

In the 4-launch FFN, the SwiGLU's `collapse_shape` has the **identical** AIR-level structure:

```mlir
// AIR level (pre-lowering):
air.launch args(%arg8=%arg2) : memref<2048x8192xbf16> {
    %cs = memref.collapse_shape %arg8 [[0, 1]] : memref<2048x8192xbf16> into memref<16777216xbf16>
    air.segment args(%arg11=%cs) : memref<16777216xbf16> {
        air.herd args(%arg13=%arg11) {
            air.dma_memcpy_nd(%l1, %arg13[...])
        }
    }
}
```

After the 56-pass aircc pipeline, this is **fully resolved**: zero `collapse_shape` references remain at the airrt level. All DMA operands trace cleanly to function arguments.

### Isolation tests

| Test | Launches | Has collapse_shape | Compiles | Notes |
|------|----------|-------------------|----------|-------|
| FFN only (Gate+Up+SwiGLU+Down) | 4 | SwiGLU: yes | ✅ | collapse_shape fully resolved at airrt level |
| Standalone 2D eltwise_add | 1 | Yes | ✅ | collapse_shape fully resolved |
| FFN + RMSNorm + Add | 6 | SwiGLU: yes, Add: yes | ❌ | collapse_shape NOT resolved for Add |
| Need to test: FFN + Add (no RMSNorm) | 5 | Both | ? | Would help determine threshold |

### Updated Finding: Structural Bug, NOT Resource Exhaustion

**Disproved**: The resource exhaustion hypothesis was tested by compiling the same 6-launch module at tiny scale (M=128, emb=128, hidden=512). It fails with the **identical error**, proving the issue is structural, not resource-dependent.

**Systematic isolation** (all tests at full LLAMA scale unless noted):

| Test | Launches | collapse_shape in | Result |
|------|----------|-------------------|--------|
| FFN (Gate+Up+SwiGLU+Down) | 4 | SwiGLU only | ✅ OK |
| FFN + RMSNorm | 5 | SwiGLU only | ✅ OK |
| FFN + 1D Add (no cs) | 5 | SwiGLU only | ✅ OK |
| RMS + FFN + 1D Add (no cs) | 6 | SwiGLU only | ✅ OK |
| Standalone Add_2D | 1 | Add only | ✅ OK |
| 2-launch (simple + Add_2D) | 2 | Add only | ✅ OK |
| **RMS + FFN + Add_2D** | **6** | **SwiGLU + Add** | **❌ FAIL** |
| **RMS + FFN + Add_2D (tiny)** | **6** | **SwiGLU + Add** | **❌ FAIL** |

**CORRECTED Root cause**: See below. The original hypothesis about `memref.collapse_shape` hoisting was incorrect.

### Definitive Root Cause: Missing `air.segment` Wrapper

**Date identified**: 2026-03-31

The actual root cause is that the RMSNorm kernel generates a **bare `air.herd` without an `air.segment` wrapper** inside its `air.launch`. After lowering, this produces `airrt.herd_load` instead of `airrt.segment_load`. The `airrt-to-npu` pass's `identifyLaunchRegions()` function **only looks for `airrt::SegmentLoadOp`** to associate launch regions with `aie.device` ops:

```cpp
// AIRRtToNpuPass.cpp, identifyLaunchRegions():
forOp.walk([&](airrt::SegmentLoadOp segLoadOp) {
    StringRef deviceName = segLoadOp.getSymName();
    AIE::DeviceOp device = getDeviceByName(module, deviceName);
    if (device) {
        regions.push_back({forOp, deviceName, device});
    }
});
// NOTE: Does NOT handle airrt::HerdLoadOp!
```

When other segment-based launches exist in the same function, `regions.empty()` is false, so the fallback path (which does handle `HerdLoadOp`) is never reached. The unmatched RMSNorm launch region's DMA ops are left in a function that gets split across devices — but the RMSNorm DMAs never get moved into their device. When `DmaToNpuPattern::matchAndRewrite` calls `op->getParentOfType<AIE::DeviceOp>()`, it returns null, and the pattern returns `failure()`.

**Evidence**:
- The failing DMA references `%arg1 : memref<128xbf16>` (RMSNorm weight) with `metadata = @air_channel_0`
- `@air_channel_0`'s `shim_dma_allocation` is inside `aie.device @r_herd_0`
- The DMA is inside `func.func @ffn_full` at the module level, NOT inside any `aie.device`
- `identifyLaunchRegions` returns 5 regions (4 FFN + 1 Add, all with `segment_load`) but misses the RMSNorm (which has `herd_load`)

**Fix applied (user-side workaround)**: Wrap bare `air.herd` in both `air.launch` AND `air.segment`. This ensures `airrt.segment_load` is generated, and `identifyLaunchRegions` correctly associates the region with its device. After this fix, the 6-launch module compiles successfully.

**Upstream fix needed**: `identifyLaunchRegions()` in `AIRRtToNpuPass.cpp` should also handle `airrt::HerdLoadOp`, not just `airrt::SegmentLoadOp`. The fallback path (line ~1553) already handles both op types, but it's only reached when `regions.empty()`.

### Previous (incorrect) Hypothesis: collapse_shape Hoisting

The earlier analysis pointed to `memref.collapse_shape` being hoisted out of `air.launch` bodies by `air-dma-to-channel`. While this hoisting does occur, it was a red herring — the `collapse_shape` ops in the SwiGLU and Add launches are fully resolved by canonicalize before reaching `airrt-to-npu`. The actual failure was the RMSNorm's bare herd causing its launch region to be missed during device association.

---

## Related Blocker: Weight Broadcast DMA

Separately, weighted RMSNorm with herd > [1,1] fails with a different error (`operand #N does not dominate this use`). This is caused by one-shot DMA of a 1D weight vector to multiple tiles — the `air-to-aie` herd expansion creates SSA dominance violations. This is an orthogonal issue.

---

## Files for Reproduction

| File | Description |
|------|-------------|
| `programming_examples/llama3/ffn_swiglu/run.py` | Working 4-launch FFN (reference) |
| `programming_examples/llama3/ffn_full_multi.py` | Failing 6-launch module |
| `programming_examples/llama3/ffn_swiglu/silu_and_mul.py` | SwiGLU with working collapse_shape pattern |

To reproduce:
```bash
cd programming_examples/llama3

# Working: 4-launch FFN
cd ffn_swiglu && make run  # PASS

# Failing: 6-launch FFN Full
python3 ffn_full_multi.py  # Compilation fails at airrt-to-npu
```

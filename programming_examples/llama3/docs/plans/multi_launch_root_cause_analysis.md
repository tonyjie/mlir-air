# Multi-Launch ELF Compilation Failure — Root Cause Analysis

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

**Root cause**: The `airrt-to-npu` pass fails when a module has **two or more launches containing `memref.collapse_shape`** combined with GEMM launches that have complex multi-herd structure. A single collapse_shape launch (SwiGLU in 4-launch FFN) is correctly resolved. Adding a second collapse_shape launch (Add) triggers the bug regardless of data size.

The pass appears to correctly trace the first collapse_shape through the lowered DMA chain but fails on the second, leaving `airrt.dma_memcpy_nd` ops referencing unresolved `%collapse_shape` SSA values.

---

## Related Blocker: Weight Broadcast DMA

Separately, weighted RMSNorm with herd > [1,1] fails with a different error (`operand #N does not dominate this use`). This is caused by one-shot DMA of a 1D weight vector to multiple tiles — the `air-to-aie` herd expansion creates SSA dominance violations. This is an orthogonal issue to the multi-launch resource problem.

---

## Concrete Questions for Compiler Engineers

1. **Is there a maximum number of launches per ELF?** If so, is it a hard limit or resource-dependent?

2. **Why does `memref.collapse_shape` resolve to func args in the 4-launch case but not in the 6-launch case?** Is there a pass that handles this resolution, and does it have resource-dependent behavior?

3. **Can `airrt-to-npu` be extended to trace through `memref.collapse_shape` operations?** The semantic is simple — `collapse_shape` doesn't change the base pointer, just the view. The BO index should be the same as the source operand.

4. **Is there a way to pre-resolve `collapse_shape` before `airrt-to-npu`?** A canonicalization pass that replaces `collapse_shape` results in DMA operands with the source operand + adjusted offsets/strides.

5. **For the weight broadcast DMA issue**: Can `air-to-aie` handle one-shot DMA (`dma_memcpy_nd` with no tile-dependent offsets) to multiple tiles without SSA dominance violations?

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

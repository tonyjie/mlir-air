# Multi-Launch Integration — Compiler Blockers

## Blocker 1: Weight Broadcast DMA (RMSNorm multi-tile)

**Affects**: Weighted RMSNorm with herd > [1,1]

**Error**: `operand #N does not dominate this use` in aiecc

**Root cause**: One-shot DMA of weight vector (`dma_memcpy_nd(l1_weight, l3_weight)`) outside the row loop. When `air-to-aie` replicates the herd to multiple cores, the weight DMA SSA value loses dominance across replicated core regions.

**Workaround**: Use single-tile [1,1] for weighted RMSNorm (6ms vs IRON's 4.3ms)

**Status**: Documented in `kernels/rmsnorm.md`. Not yet fixed upstream.

---

## Blocker 2: `memref.collapse_shape` at airrt level (Merge C)

**Affects**: Combining kernels with different memref shapes (2D GEMM + 1D eltwise_add) in one multi-launch ELF

**Error**: `failed to legalize operation 'airrt.dma_memcpy_nd'` in `airrt-to-npu` pass

**Root cause**: When `memref.collapse_shape` is placed **between launches** (at the func body level, outside any `air.launch`), it survives lowering to the airrt dialect. The `airrt-to-npu` pass then encounters DMA operations referencing `%collapse_shape` results instead of direct func args. It cannot trace these back to XRT buffer objects.

**Working pattern** (FFN SwiGLU):
```mlir
// collapse_shape INSIDE air.launch → works
air.launch args(%arg8 = %func_arg : memref<2048x8192xbf16>) {
    %cs = memref.collapse_shape %arg8 [[0, 1]] : ... into memref<16777216xbf16>
    air.segment args(%seg_arg = %cs : memref<16777216xbf16>) {
        // DMA references %seg_arg (a segment arg, not collapse_shape result)
    }
}
```

**Failing pattern** (FFN Full):
```mlir
// collapse_shape BETWEEN launches → fails
%cs = memref.collapse_shape %func_arg [[0, 1]] : ... into memref<4194304xbf16>
air.launch args(%arg = %cs : memref<4194304xbf16>) {
    // After lowering, airrt.dma_memcpy_nd references %cs directly
    // airrt-to-npu can't trace %cs back to a func arg → FAILS
}
```

**Fix**: Move `collapse_shape` inside the eltwise_add's `air.launch` body (same pattern as SwiGLU). The launch should receive the 2D func arg, then collapse inside.

**Status**: Fix identified but not yet implemented. The `_wrap_ir_in_launch` utility in `ffn_full_multi.py` needs to handle this case — pass 2D args through launch, collapse inside.

---

## Impact on Multi-Launch Roadmap

| Merge | Blocker | Fix |
|-------|---------|-----|
| FFN multi-launch (DONE) | None | — |
| Attn GEMMs multi-launch (DONE) | None | — |
| RoPE Q+K (built, not integrated) | None (likely works) | — |
| **RMSNorm + FFN + Add** | **Blocker 2** (collapse_shape) | Move collapse inside launch |
| **O GEMM + Add** | **Blocker 2** (same pattern) | Same fix |
| **Multi-tile RMSNorm** | **Blocker 1** (weight broadcast) | Upstream fix needed |

The collapse_shape fix (Blocker 2) is implementable — it requires modifying how the eltwise_add kernel is stitched to receive 2D args and collapse internally (same pattern that already works for SwiGLU in the FFN).

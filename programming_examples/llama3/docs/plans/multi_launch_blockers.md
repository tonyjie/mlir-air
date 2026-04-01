# Multi-Launch Integration — Compiler Blockers

## Blocker 1: Weight Broadcast DMA (RMSNorm multi-tile)

**Affects**: Weighted RMSNorm with herd > [1,1]

**Error**: `operand #N does not dominate this use` in aiecc

**Root cause**: One-shot DMA of weight vector (`dma_memcpy_nd(l1_weight, l3_weight)`) outside the row loop. When `air-to-aie` replicates the herd to multiple cores, the weight DMA SSA value loses dominance across replicated core regions.

**Workaround**: Use single-tile [1,1] for weighted RMSNorm (6ms vs IRON's 4.3ms)

**Status**: Documented in `kernels/rmsnorm.md`. Not yet fixed upstream.

---

## Blocker 2: Missing `air.segment` wrapper on bare herd launches (Merge C) — FIXED

**Affects**: Multi-launch ELF containing a mix of segment-wrapped and bare-herd launches

**Error**: `failed to legalize operation 'airrt.dma_memcpy_nd'` in `airrt-to-npu` pass

**Root cause**: The `airrt-to-npu` pass's `identifyLaunchRegions()` only looks for `airrt::SegmentLoadOp` to associate launch regions with `aie.device` ops. If a launch contains only a bare `air.herd` (no `air.segment` wrapper), it lowers to `airrt.herd_load` instead of `airrt.segment_load`. When other launches in the same function DO have segment_load, `regions.empty()` is false, so the fallback path (which handles both op types) is never reached. The bare-herd launch's DMA ops are silently dropped during per-device function splitting, and `DmaToNpuPattern` fails because the DMAs aren't inside any `aie.device`.

**Failing pattern**:
```mlir
// Bare air.herd inside air.launch (no air.segment) → FAILS in multi-launch
air.launch args(%arg3=%arg0, ...) {
    air.herd @herd_0 tile (...) args(...) { ... }
}
```

**Working pattern** (after fix):
```mlir
// air.herd wrapped in air.segment inside air.launch → WORKS
air.launch args(%arg3=%arg0, ...) {
    air.segment @seg_name args(%arg6=%arg3, ...) {
        air.herd @herd_0 tile (...) args(...) { ... }
    }
}
```

**Fix applied**: Modified `_wrap_ir_in_launch()` in `ffn_full_multi.py` to add both `air.launch` AND `air.segment` wrappers around bare herds. This ensures `airrt.segment_load` is generated.

**Upstream fix recommended**: `identifyLaunchRegions()` in `AIRRtToNpuPass.cpp` should also handle `airrt::HerdLoadOp`, not just `airrt::SegmentLoadOp`.

**Status**: User-side workaround applied. 6-launch module (RMS+FFN+Add) now compiles and runs successfully.

---

## Impact on Multi-Launch Roadmap

| Merge | Status | Notes |
|-------|--------|-------|
| FFN multi-launch | **DONE** | Now part of ffn_full (Merge C) |
| Attn GEMMs multi-launch | **DONE** | Now part of rms_attn_gemms (Plan A) |
| RoPE Q+K (Merge A) | **DONE** | 2 herds, integrated |
| O GEMM + Add (Merge B) | **DONE** | 2 launches, integrated |
| RMSNorm + FFN + Add (Merge C) | **DONE** | 6 launches, integrated |
| RMSNorm + Attn GEMMs (Plan A) | **DONE** | 4 launches, integrated |
| DMA Transpose (Plan B) | **DEFERRED** | BF16 DMA stride hardware limitation — see below |
| Multi-tile RMSNorm | **BLOCKED** | Blocker 1 — upstream fix needed |

All merges that were blocked by Blocker 2 (missing air.segment) are resolved.

---

## Blocker 3: BF16 DMA Stride Limitation (Plan B)

**Affects**: NPU-side data transpose for BF16 between kernels with different data layouts

**Error**: `'aie.dma_bd' op For <32b width datatypes, inner-most dim stride must be 1`

**Root cause**: AIE DMA hardware requires the innermost dimension stride to be 1 for sub-32b datatypes (BF16 is 16-bit). Strided DMA writes (needed for transpose) only work for 32b+ types (uint32, float32).

**Evidence**:
- `data_transfer_transpose/dma/transpose.py` — works with uint32 (strided DMA write)
- `data_transfer_transpose/dma_bf16/transpose_bf16.py` — requires C++ kernel (cannot use strided DMA)
- Our `dma_transpose.py` attempt — fails at `air-to-aie` pass with stride error

**Impact**: Cannot merge QKV GEMMs + transpose + RoPE or FlashAttn + transpose + O GEMM into single ELFs using pure DMA. A tiled C++ transpose kernel would be needed.

**Workaround**: CPU-side transpose between kernel invocations (current approach, ~2-3ms overhead).

**Status**: Documented as future opportunity. Current 5-invocation pipeline at 113ms/layer is 26% faster than IRON.

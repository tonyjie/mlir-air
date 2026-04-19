# Multi-Launch Compiler Scaling Analysis

## Summary

Multi-launch ELF compilation time scales super-linearly with launch count. Beyond 8 launches, compilation becomes impractically slow (>10 minutes). The bottleneck is in the `aircc` pipeline's routing and instruction generation phases, not in per-tile ELF compilation.

## Compilation Time Data

| Launches | Herd Types | Args | Time | Module | Status |
|----------|------------|------|------|--------|--------|
| 1 | FlashAttn | 4 | ~46s | flash_attn | OK (standalone) |
| 2 | GEMM+Add | 5 | ~10s | o_proj_add | OK |
| **3** | **FlashAttn+GEMM+Add** | **8** | **>10min** | **flash_o_add** | **Too slow** |
| 4 | RMS+3×GEMM | 9 | ~24s | rms_attn_gemms | OK |
| 6 | RMS+3×GEMM+2×RoPE | 13 | ~33s | rms_gemms_rope | OK, **in production** |
| 8 | GEMM+Add+RMS+3×GEMM+SiLU+GEMM+Add | 15 | ~50s | o_ffn | OK, **in production** |
| 8 | 8×GEMM (identical) | 17 | ~108s | lm_head | OK |
| 8 | 8×GEMV (identical) | 17 | ~16s | lm_head_gemv | OK, **decode LM Head** |
| 8 | GEMV+Add+RMS+3×GEMV+SiLU+GEMV+Add | 15 | ~7s | o_gemv_ffn | OK, **decode FFN** |
| 9 | RMS+3×GEMM+2×RoPE+FlashAttn+GEMM+Add | 17 | >10min | attn_half | **Too slow** |
| 15 | Full block (all types) | 27 | >1hr | transformer_block | **Too slow** |

**Key finding**: The bottleneck is NOT launch count — it's **FlashAttention's channel/cascade architecture**. Even 3 launches (FlashAttn + GEMM + Add) exceed 10 minutes. All non-FlashAttn modules with 6-8 launches compile in <2 minutes.

## Key Observations

### 1. Stack Overflow in Dependency Analysis (Fixed)
- **Pass**: `AIRDependencyCanonicalize` (`fillAllDependencyListsFromTR`)
- **Cause**: Recursive dependency traversal with deep launch chains
- **Fix**: `ulimit -s unlimited` before compilation
- **Affects**: 9+ launches

### 2. Compilation Time Scaling Cliff (8→9 launches)
- 8 identical GEMM launches: 108s (manageable)
- 9 heterogeneous launches: >10 minutes (impractical)
- The scaling is NOT simply O(n²) — it's worse for heterogeneous launch types
- Hypothesis: heterogeneous launches create more complex routing constraints than identical ones

### 3. Bottleneck: `air-opt-shim-dma-bds` Pass (Pass 047)

Using `--debug-ir` to dump per-pass IR, the exact bottleneck was identified:

- **Passes 001-046 complete** within ~1 minute (AIR transforms, dependency analysis, placement, AIR→AIE lowering, device merging)
- **Pass 047 (`func.func(air-opt-shim-dma-bds)`)** — shim DMA buffer descriptor optimization — runs at 100% CPU indefinitely
- Input to this pass: **45,831 lines** of AIE MLIR (after `air-merge-unrolled-devices`)
- NPU2 has 8×4 = 32 compute tiles. Each launch generates per-tile code, time-multiplexed on the same physical tiles. With 9 launches, the merged AIE module contains code for all tiles across all launches.

**Update**: Further testing revealed that the bottleneck is specifically **FlashAttention's channel/cascade architecture**, not launch count. Even a 3-launch module (FlashAttn + GEMM + Add) exceeds 10 minutes, while 6-8 launch modules without FlashAttention compile in <2 minutes. FlashAttention uses 20 `air.channel` declarations with broadcast shapes and cascade patterns. When merged with other launches, the AIE routing/lock assignment in the aiecc phase becomes intractable.

### 4. FlashAttention Cannot Be Merged (Root Cause)

FlashAttention's 20 channel declarations create complex routing constraints:
- 4 QKIn channels with broadcast shapes `[2, 1, 4]`
- 4 VIn channels with broadcast shapes
- 4 QK2L1 and 4 V2L1 relay channels
- 3 cascade channels for softmax partial results
- 2 output channels (Gp2L2, GpOut)

When merged with even simple GEMM/Add launches, the AIE router must solve a much harder constraint satisfaction problem. The compilation passes through `air-opt-shim-dma-bds` (pass 047) but then stalls in the subsequent aiecc phase (AIE vector lowering, routing, lock assignment, tile ELF generation).

**Conclusion**: FlashAttention must remain a separate XRT invocation. The practical multi-launch boundary is: any combination of GEMM/RMSNorm/RoPE/eltwise launches (up to ~8) compiles fine, but FlashAttention cannot be combined with other launches.

### 4. Channel Declarations Required for FlashAttention
- FlashAttention uses `air.channel` for L2↔L1 communication
- Channel declarations (`air.channel @name [dims]`) must be at module level
- In multi-launch stitching, channel names are renamed with prefix (`@QKIn_0` → `@fa_QKIn_0`)
- Both declarations AND put/get operations must be renamed consistently

### 5. Affine Integer Sets (#set) Required for FlashAttention
- FlashAttention uses `affine.if #set[...]` for conditional tile dispatch
- `#set` declarations must be extracted, renamed, and included alongside `#map` declarations
- Standard `_extract_affine_maps()` misses these — need `_extract_affine_sets()`

## What Works

- **6 launches** (`rms_gemms_rope`): 33s compile, production-ready
- **8 launches** (`lm_head`): 108s compile, acceptable for one-time compilation
- Text stitching handles all kernel types correctly (GEMM, RMSNorm, RoPE, FlashAttention, SwiGLU, eltwise_add)
- `collapse_shape` inside launch bodies for 2D↔1D type aliasing works perfectly
- Channel and #set renaming works correctly

## Current Production Pipeline (3 invocations/layer)

```
1. rms_gemms_rope (6 launches)  ~11ms   33s compile
2. flash_attn    (1 launch)     ~22ms   46s compile
3. o_ffn         (8 launches)   ~58ms   50s compile
```

Total: 3 XRT invocations/layer, ~91ms kernel time, **1.65s prefill** (40% faster than IRON)

This is the theoretical minimum — FlashAttention cannot be merged with other launches.

## Future Directions

1. **Decode multi-launch merging**: Decode uses CPU attention (no FlashAttention), so the
   channel/cascade routing blocker doesn't apply. Can merge to 2 XRT calls/block:
   - Merge A: RMSNorm+QKV GEMV+RoPE Q+K → 6 launches
   - Merge B: O GEMV+Add+RMSNorm+Gate+Up GEMV+SiLU+Down GEMV+Add → 8 launches
   See `decode_multi_launch.md` for the full plan.
2. **Compiler optimization**: The `AIRDependencyCanonicalize` and routing passes could be
   optimized for multi-launch modules (e.g., batch processing independent launches).
3. **FlashAttention isolation**: FlashAttention must remain a separate invocation until
   the AIE router can handle channel/cascade patterns in multi-launch modules.

## Files

| File | Launches | Status |
|------|----------|--------|
| `rms_gemms_rope_multi.py` | 6 | **Production** |
| `attn_half_multi.py` | 9 | Parses OK, compile too slow |
| `transformer_block_multi.py` | 15 | Parses OK, compile too slow |

# GEMV Decode Investigation

## Finding: ffn_decode.py cannot scale to LLAMA dimensions

The existing `ffn_swiglu/decode/ffn_decode.py` loads the **entire weight partition** (`dim_m × K`) into L1 in one shot:
- At dim=128, num_cols=4: weight partition = 32 × 128 × 2 = 8KB → fits in 64KB L1
- At dim=256, num_cols=4: weight partition = 64 × 256 × 2 = 32KB → fits
- At dim=512, num_cols=4: weight partition = 128 × 512 × 2 = 128KB → **EXCEEDS 64KB L1**

For LLAMA: gate GEMV (8192, 2048) with 4 cols → weight per col = 2048 × 2048 × 2 = 8MB → **impossible**.

**Root cause**: No K-dimension tiling in L1. The kernel call processes all K elements at once.

## Comparison of approaches

| Approach | K tiling? | Multi-col? | L1 weight per call | LLAMA feasible? |
|---|---|---|---|---|
| `ffn_decode.py` | No (full K in L1) | Yes [1,num_cols] | dim_m × K × 2B | No (>64KB at LLAMA scale) |
| `matrix_vector_multiplication/bf16` | Yes (tile_m_l2 rows via L2) | No (single core) | tile_m_l2 × K × 2B | Yes (configurable tile) |
| IRON GEMV | Yes (m_input rows, full K) | Yes (8 columns) | m_input × K × 2B | Yes (m_input=1: 4-16KB) |

## Architecture needed for LLAMA decode GEMV

Follow IRON's approach:
1. **Full K vector** loaded into L1 once (K=2048: 4KB, K=8192: 16KB — fits)
2. **Stream M rows** one at a time (m_input=1): each row is K elements = 4-16KB
3. **Accumulate output** in L1 buffer (m_output elements)
4. **Multi-column**: [1, num_cols] herd, each column handles M/num_cols rows

L1 budget per core with IRON approach:
- B vector: K × 2B = 4-16KB
- A row (m_input=1): K × 2B = 4-16KB
- C output buffer: m_output × 2B = 256B-1KB
- **Total**: ~8-33KB — fits easily in 64KB

## Multi-Column GEMV Architecture (to build)

Follow the `matvec.py` tiling but add column parallelism:

```
Architecture: [1, num_cols] herd, each column handles M/num_cols rows

Per-column L1 budget (K=2048, m_input=1):
  B vector:      K × 2B = 4KB     (broadcast to all cols)
  A row:         m_input × K × 2B = 4KB  (per-col partition)
  C output tile: m_output × 2B = ~512B
  Total: ~8.5KB  (fits easily in 64KB)

Per-column L1 budget (K=8192, m_input=1):
  B vector:      K × 2B = 16KB
  A row:         m_input × K × 2B = 16KB
  C output tile: m_output × 2B = ~1KB
  Total: ~33KB  (fits in 64KB)

Data flow:
  1. B vector: L3 → (broadcast to all cols L1)
  2. For each output tile (m_output rows per col):
     a. Zero-fill C output in L1
     b. For each m_input rows:
        - A row: L3 → L1 (col-partitioned offset)
        - Call matvec kernel: C += A_row × B
     c. Write C tile back: L1 → L3 (col-partitioned offset)

Launch grid: [M // (num_cols * m_output), 1]
  - Each launch handles num_cols * m_output output rows
  - With M=8192, num_cols=4, m_output=512: 4 launches

Key difference from current single-core matvec.py:
  - Replace herd(sizes=[1,1]) with herd(sizes=[1,num_cols])
  - Each col reads A rows from col-partitioned offset
  - Each col writes C to col-partitioned offset
  - B is shared (broadcast) across all cols
```

## Implementation approach

Build `llama3/decode_kernels/gemv_multi_col.py`:
1. Start from `matvec.py` architecture (L3→L2→L1 tiling)
2. Add `num_cols` parameter, `herd(sizes=[1, num_cols])`
3. Each column uses `_ty` to compute its M offset
4. B vector broadcast to all columns
5. Use existing `mv.cc` / `mv.o` C++ kernel (same as matvec.py)
6. Test at all 5 LLAMA shapes
7. Profile vs IRON (target: match 8-column performance)

## Next steps

1. Build `gemv_multi_col.py` following above architecture
2. Test at all 5 LLAMA GEMV shapes with num_cols=4
3. Profile and compare with single-core results
4. If 4 cols hits row limit, try adjusting `row-offset` in air-to-aie pass

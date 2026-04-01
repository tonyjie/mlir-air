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

## Next steps

1. Adapt `matrix_vector_multiplication/bf16/matvec.py` to support multi-column herd
2. OR build a new GEMV kernel from scratch following IRON's 3-level tiling
3. Test at all 5 LLAMA GEMV shapes

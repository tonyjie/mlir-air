# GEMV Kernel — Decode Performance Analysis

## Role in LLAMA Decode Pipeline

Every linear projection in the decode phase is a GEMV (matrix-vector multiply): `C[M] = A[M,K] @ B[K]`. This is 60% of standalone decode kernel time.

**IRON reference (8-column, standalone kernel benchmark):**

| Shape (M × K) | Role | Calls/token | IRON latency | IRON BW |
|---|---|---|---|---|
| 2048 × 2048 | Q/O projection | 32 | 214µs | 39.2 GB/s |
| 512 × 2048 | K/V projection | 32 | 98µs | 21.5 GB/s |
| 8192 × 2048 | FFN gate/up | 32 | 657µs | 51.1 GB/s |
| 2048 × 8192 | FFN down | 16 | 660µs | 50.9 GB/s |
| 128256 × 2048 | Final vocab | 1 | 9443µs | 55.7 GB/s |

---

## AIR Implementation

**File**: `programming_examples/matrix_vector_multiplication/bf16/matvec.py`

**Architecture**: Multi-column herd with L2 staging.
- `herd_m` AIE columns process independent row chunks in parallel
- B vector: L3 → L2 (segment-level shared alloc) → L1 (per-tile broadcast via L2)
- A rows: L3 → L2 → L1, streamed `m_input` rows per kernel call
- C output: accumulated in L1, written back via L2 → L3
- External kernel: `mv.o` (`matvec_vectorized_bf16_bf16`)

**Key parameters:**

| Parameter | What it controls |
|---|---|
| `herd_m` | Number of parallel AIE columns (1-8) |
| `tile_m` | Output rows per column per launch instance |
| `m_input` | Rows loaded into L1 per kernel call |
| `num_launches` | `M / (herd_m × tile_m)` — derived, sequential over time |

**L2 constraint**: `herd_m × tile_m × K × 2 bytes ≤ 512KB`

**L1 per core**: `m_input × K × 2` (A) + `K × 2` (B) + `tile_m × 2` (C) ≤ 64KB. With ping-pong, double the A and B buffers.

---

## Profiling Methodology

All kernel times measured with **C++ test harness** (`test.exe`):
- 10 warmup iterations + 20 timed iterations, reporting average
- **Timer scope**: `kernel(opcode, bo_instr, len, bo_a, bo_b, bo_c)` + `run.wait()` only
- **NOT included**: `bo.sync(TO_DEVICE)` (input write, done once before loop), `bo.sync(FROM_DEVICE)` (output read, done after timer stop)
- This is **kernel-only time** — same scope as IRON's standalone kernel benchmarks
- Python `invoker()` adds ~2ms overhead and must NOT be used for µs-scale GEMV profiling

```bash
# Compile + profile flow:
cd programming_examples/matrix_vector_multiplication/bf16/build_peano

# Step 1: Compile with specific flags via Python
python3 -c "
from matvec import build_module
from air.backend.xrt import XRTBackend
from ml_dtypes import bfloat16
module = build_module(M, K, TILE_M, M_INPUT, HERD_M, bfloat16, bfloat16)
XRTBackend(verbose=False, omit_while_true_loop=False,
    omit_pingpong='',                      # '' = ping-pong ON, 'all' = OFF
    runtime_loop_tiling_sizes=[16, 16],
    use_lock_race_condition_fix=False,
).compile(module)
"

# Step 2: Profile with C++ harness
./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $M -K $K
```

---

## Aircc Backend Flags Study

Fixed config: M=2048, K=2048, tile_m=8, m_input=4, herd_m=8.

| omit_pingpong | lock_fix | tile_sizes | Avg (µs) | vs baseline |
|---|---|---|---|---|
| all (pp OFF) | True | [4,4] | 309 | baseline |
| all (pp OFF) | False | [4,4] | 306 | -1% |
| "" (pp ON) | True | [4,4] | 290 | -6% |
| **"" (pp ON)** | **False** | **[4,4]** | **277** | **-10%** |
| all (pp OFF) | True | [2,2] | 373 | +21% |
| "" (pp ON) | True | [2,2] | 341 | +10% |
| all (pp OFF) | True | [8,8] | 265 | -14% |
| "" (pp ON) | True | [8,8] | 256 | -17% |
| "" (pp ON) | False | [8,8] | 246 | -21% |
| **"" (pp ON)** | **False** | **[16,16]** | **228** | **-26%** |

**Findings:**
- **`runtime_loop_tiling_sizes`**: Largest impact. [16,16] is 26% faster than [4,4]. Larger inner tile = fewer runtime loop iterations = less overhead between BD chains.
- **Ping-pong (double-buffering)**: ~6-10% improvement. Overlaps DMA load of next data with compute on current data.
- **Lock race fix**: ~1-2% improvement when disabled. Extra lock synchronization has small overhead.
- **[32,32]** fails (BD overflow — inner tile too large for BD allocator).

**Flag explanations:**

- **`omit_while_true_loop=False`**: Keep AIE cores alive in `while(true)` loop between launches. Required for multi-launch ELFs.
- **`omit_pingpong=''`**: Enable double-buffering. L1 has two copies of A and B buffers; DMA fills one while core computes on the other. Uses 2x L1 but overlaps data transfer with compute.
- **`runtime_loop_tiling_sizes=[N,N]`**: Tiles the shim DMA loop nest. The inner tile (size N) is unrolled into BDs (must fit ≤16 per channel). The outer loop runs at NPU firmware runtime, re-executing the same BD chain. Without this flag, the compiler tries to unroll the entire launch loop into BDs, exceeding the 16-BD hardware limit → compilation hangs.
- **`use_lock_race_condition_fix=False`**: Skip extra lock acquire/release pairs for DMA/compute ordering. Saves ~2% BD overhead.

---

## Tile Config Sweep

Best flags: `omit_pingpong=''`, `lock_fix=False`, `tile_sizes=[16,16]`.

**M=2048, K=2048 (herd_m=8):**

| tile_m | m_input | Avg (µs) | Notes |
|---|---|---|---|
| 2 | 1 | 282 | |
| 4 | 2 | 256 | |
| 4 | 4 | 250 | |
| **8** | **4** | **233** | **Best** |
| 4 | 1 | FAIL | BD overflow with tile=[16,16] |
| 8 | 1 | FAIL | BD overflow |

**M=512, K=2048 (herd_m=8):**

Best: tile_m=8, m_input=4 → **81µs**

**M=8192, K=2048 (herd_m=8):**

Best: tile_m=8, m_input=4 → **837µs**

**M=2048, K=8192 (herd_m=8):**

Ping-pong FAILS for K=8192 (L1 overflow — double-buffering K=8192 vector needs 2×16KB=32KB for B alone). Must use `omit_pingpong='all'`.

Best: tile_m=2, m_input=1, pp=OFF, tile=[16,16] → **946µs**

---

## Final Optimal Results

| Shape | tile_m | m_input | Flags | AIR (µs) | IRON (µs) | Gap | BW (GB/s) |
|---|---|---|---|---|---|---|---|
| 2048 × 2048 | 8 | 4 | pp=ON, tile=[16,16] | **233** | 214 | 1.1x | 36 |
| 512 × 2048 | 8 | 4 | pp=ON, tile=[16,16] | **81** | 98 | **0.8x** | 26 |
| 8192 × 2048 | 8 | 4 | pp=ON, tile=[16,16] | **837** | 657 | 1.3x | 40 |
| 2048 × 8192 | 2 | 1 | pp=OFF, tile=[16,16] | **946** | 660 | 1.4x | 35 |

All configs use `herd_m=8`, `use_lock_race_condition_fix=False`.

---

## Per-Token Decode Time Estimate

| Kernel | AIR (µs) | Calls/token | Total (ms) | IRON total (ms) |
|---|---|---|---|---|
| GEMV Q/O (2048×2048) | 233 | 32 | 7.5 | 6.8 |
| GEMV K/V (512×2048) | 81 | 32 | 2.6 | 3.1 |
| GEMV gate/up (8192×2048) | 837 | 32 | 26.8 | 21.0 |
| GEMV down (2048×8192) | 946 | 16 | 15.1 | 10.6 |
| **GEMV subtotal** | | | **52.0** | **41.5** |

---

## AIR vs IRON Detailed Comparison

### Kernel Computation: Identical

| Aspect | AIR | IRON |
|---|---|---|
| Accumulator type | `accfloat` | `accfloat` |
| Vector width | r=64 | r=64 |
| MAC operation | `aie::mac()` | `aie::mac()` |
| Rounding mode | `conv_even` | `conv_even` |
| BFP16 emulation | Not used | Not used |
| Loop pragma | Removed (no-op for Peano) | `AIE_LOOP_MIN_ITERATION_COUNT(2)` (no-op for Peano) |

The compute kernels (`mv.cc`) are functionally identical. Both use `accfloat` accumulator with r=64 vector width. BFP16 emulation is irrelevant for GEMV — it only affects `aie::mmul` (matrix multiply unit used by GEMM), not `aie::mac` (vector MAC used by GEMV).

### Data Flow: The Key Difference

| Aspect | AIR | IRON |
|---|---|---|
| **A (weight matrix)** | DDR → L2 → L1 | DDR → L1 (direct via ObjectFIFO) |
| **B (input vector)** | DDR → L1 (direct) | DDR → L1 (direct via ObjectFIFO) |
| **C (output)** | L1 → L2 → DDR | L1 → DDR (direct via ObjectFIFO) |
| **L2 usage** | Yes (MemTile stages A and C) | None |
| **BD pattern** | Individual BDs per DMA | Repeating BD with auto-increment (2 per FIFO) |
| **Ping-pong** | Via aircc flag | Via ObjectFIFO depth=2 |

IRON's ObjectFIFO generates BD-efficient streaming (2 BDs per FIFO with hardware auto-increment), enabling direct DDR↔L1 paths. AIR's `dma_memcpy_nd` generates individual BDs per transfer, requiring L2 aggregation to stay within the 16-BD shim DMA limit.

### Root Cause of Remaining Gap

GEMV is bandwidth-bound. AIR achieves 26-40 GB/s vs IRON's 21-51 GB/s:

- **AIR path**: DDR → Shim DMA → MemTile (L2) → Compute Tile (L1) — two NoC hops
- **IRON path**: DDR → Shim DMA → Compute Tile (L1) — one hop, no L2

L2 doesn't provide data reuse for GEMV (each weight row is used exactly once). The extra L2 hop is pure overhead.

### L2 Bypass Investigation (`matvec_no_l2.py`)

All data goes L3→L1 directly inside the herd (no MemTile staging). Compiles and runs correctly (corr=1.0).

**IMPORTANT**: Must profile with C++ test harness. Python invoker adds ~2ms overhead that previously gave misleading 7-14x slowdown numbers.

| Shape | With-L2 (best) | No-L2 | IRON | No-L2 vs L2 |
|---|---|---|---|---|
| 2048×2048 | 233µs | 349µs | 214µs | 1.5x slower |
| 512×2048 | 81µs | 116µs | 98µs | 1.4x slower |
| 8192×2048 | 837µs | 1271µs | 657µs | 1.5x slower |
| 2048×8192 | 946µs | 1243µs | 660µs | 1.3x slower |

No-L2 is **1.3-1.5x slower than with-L2** (not 7-14x as previously reported with Python invoker).

**Why with-L2 wins despite the extra hop**: No-L2 can only compile with `tile=4,4` (larger tiles cause BD overflow or compilation timeout). With-L2 can use `tile=16,16` + ping-pong, which reduce runtime loop overhead and overlap DMA with compute. These optimizations more than compensate for the L2 staging cost.

**Why no-L2 has BD limits**: Without L2 aggregation, each of 8 tiles issues its own shim DMA requests. The shim DMA has 16 BDs per channel — with 8 tiles × multiple iterations per tile, larger tile sizes overflow the BD allocator.

---

## Historical Notes

### Broadcast DMA Bug (Fixed 2026-03-31)

Multi-column GEMV was initially blocked by a broadcast DMA compiler bug. The B vector (same data to all columns) caused `operand does not dominate this use` errors. Fixed in mlir-air rebuild via `air-specialize-dma-broadcast` for 3D channel patterns.

The GEMV routes B through L2 (segment-level shared alloc), then each tile copies from L2→L1 independently — avoiding the stride=0 pattern that remains broken for direct L3→L1 broadcast.

---

## Takeaway

**AIR GEMV is 1.0-1.4x of IRON** at optimal configs — close enough to build a functional decode pipeline. The remaining gap is architectural (L2 staging vs IRON's direct L3↔L1 via ObjectFIFO) and requires compiler-level changes to close.

**Best configs for LLAMA decode** (all use herd_m=8, lock_fix=OFF):

| Shape | tile_m | m_input | omit_pp | tile_sizes | Time (µs) |
|---|---|---|---|---|---|
| 2048×2048 | 8 | 4 | OFF (pp=ON) | [16,16] | 233 |
| 512×2048 | 8 | 4 | OFF (pp=ON) | [16,16] | 81 |
| 8192×2048 | 8 | 4 | OFF (pp=ON) | [16,16] | 837 |
| 2048×8192 | 2 | 1 | ON (pp=OFF) | [16,16] | 946 |

**What we learned:**
- Kernel computation is identical to IRON (same accumulator, MAC, vector width)
- `runtime_loop_tiling_sizes` has the largest performance impact (26% between [4,4] and [16,16])
- Ping-pong helps ~10% for K=2048 but fails for K=8192 (L1 too tight)
- L2 bypass is slower (1.3-1.5x) because it can't use large tile sizes or ping-pong
- Python invoker adds ~2ms overhead — always use C++ test harness for GEMV profiling

# Flash Attention — Kernel Fusion Based

This directory implements Flash Attention on AMD NPU (AIE2P) using a kernel-fusion approach where all attention operations (matmul, softmax, accumulation) are fused into a single herd of compute tiles.

## Parameters and Matrix Shapes

### The Attention Equation

```
Output = softmax(Q @ K + M) @ V
```

### Per-Head Matrix Dimensions

```
         DK                         LK                           DV
    ┌──────────┐            ┌──────────────────┐          ┌──────────┐
    │          │            │                  │          │          │
LQ  │    Q     │   @   DK  │        K         │   =  LQ  │  Scores  │
    │          │            │                  │          │          │
    │ [LQ, DK] │            │    [DK, LK]      │          │ [LQ, LK] │
    └──────────┘            └──────────────────┘          └──────────┘

         LK                          DV                          DV
    ┌──────────────────┐      ┌──────────┐              ┌──────────┐
    │                  │      │          │              │          │
LQ  │ softmax(Scores)  │  @   │    V     │   =     LQ  │  Output  │
    │                  │  LK  │          │              │          │
    │    [LQ, LK]      │      │ [LK, DV] │              │ [LQ, DV] │
    └──────────────────┘      └──────────┘              └──────────┘
```

- **LQ** = number of query tokens (rows of Q). "How many tokens am I computing attention for?"
- **LK** = number of key/value tokens (columns of K, rows of V). "How many tokens am I attending to?"
- **DK** = head dimension for keys. Inner dimension of Q @ K.
- **DV** = head dimension for values. Column count of output.

### Multi-Head: NUM_HEADS and NUM_KV_HEADS

```
NUM_HEADS = 32 (Q heads)          NUM_KV_HEADS = 8 (KV heads)

Q: [32, LQ, DK]                  K: [8, DK, LK]         V: [8, LK, DV]
┌────┐                            ┌────┐                  ┌────┐
│ h0 │ ─── uses ──────────────→   │kv0 │                  │kv0 │
│ h1 │ ─── uses ──────────────→   │    │                  │    │
│ h2 │ ─── uses ──────────────→   │    │                  │    │
│ h3 │ ─── uses ──────────────→   │    │                  │    │
├────┤                            ├────┤                  ├────┤
│ h4 │ ─── uses ──────────────→   │kv1 │                  │kv1 │
│ h5 │ ─── uses ──────────────→   │    │                  │    │
│ h6 │ ─── uses ──────────────→   │    │                  │    │
│ h7 │ ─── uses ──────────────→   │    │                  │    │
├────┤                            ├────┤                  ├────┤
│... │                            │... │                  │... │
├────┤                            ├────┤                  ├────┤
│h28 │ ─── uses ──────────────→   │kv7 │                  │kv7 │
│h29 │ ─── uses ──────────────→   │    │                  │    │
│h30 │ ─── uses ──────────────→   │    │                  │    │
│h31 │ ─── uses ──────────────→   │    │                  │    │
└────┘                            └────┘                  └────┘

Output: [32, LQ, DV]       gqa_group_size = 32/8 = 4
```

Each Q head computes its own independent output, but groups of 4 Q heads read the **same** K/V head (GQA). When `NUM_KV_HEADS == NUM_HEADS`, every Q head has its own K/V (standard MHA).

### Tiling: LQP and LKP

The full matrices are too large for L1/L2 memory, so they're processed in tiles:

```
Q matrix [LQ, DK] = [2048, 64]
─────────────────────────────────
│  LQP = 256   │  ← processed per launch iteration
│───────────────│
│  LQP = 256   │
│───────────────│
│  LQP = 256   │     LQ / LQP = 2048 / 256 = 8 iterations
│───────────────│
│  ...          │
│───────────────│
│  LQP = 256   │
─────────────────────────────────

Each LQP chunk is further split into 4 Q-tiles (one per herd row):

  LQP = 256
  ┌─────────────┐
  │ tile 0: 64  │  tile_size_q = LQP / 4 = 64
  │ tile 1: 64  │
  │ tile 2: 64  │  ← 4 rows of the herd process these in parallel
  │ tile 3: 64  │
  └─────────────┘


K matrix [DK, LK] = [64, 2048]
──────────────────────────────────────────────────────────────
│  LKP  │  LKP  │  LKP  │  LKP  │  LKP  │ ... │  LKP      │
│  =64  │  =64  │  =64  │  =64  │  =64  │     │  =64      │
──────────────────────────────────────────────────────────────
   ↓       ↓       ↓       ↓
 stage0  stage1  stage2  stage3  stage0  ...    (interleaved)

LK / LKP = 2048 / 64 = 32 total chunks
32 chunks / 4 stages = 8 chunks per stage
```

### Per-Core Tiled Computation

For one Q-tile (64 rows) at one cascade stage, processing one K/V chunk:

```
    Q tile          K chunk          Score chunk
    [64, 64]   @    [64, 64]    =    [64, 64]
   ┌────────┐      ┌────────┐      ┌────────┐
   │        │      │        │      │        │
   │  DK=64 │  @   │ LKP=64 │  =   │ LKP=64 │
   │        │      │        │      │        │
   └────────┘      └────────┘      └────────┘
  tile_size_q        DK             tile_size_q
     = 64                              = 64

                                        ↓ softmax

  Softmax chunk      V chunk         Output accum
    [64, 64]    @    [64, 64]    +=    [64, 64]
   ┌────────┐      ┌────────┐      ┌────────┐
   │        │      │        │      │        │
   │ LKP=64 │  @   │  DV=64 │  +=  │  DV=64 │   (Gp accumulator)
   │        │      │        │      │        │
   └────────┘      └────────┘      └────────┘
  tile_size_q       LKP            tile_size_q
     = 64                             = 64
```

### The Complete Tiling Grid (4x4 Herd)

```
                    K/V sequence (LK = 2048)
           ┌──────────┬──────────┬──────────┬──────────┐
           │ Stage 0  │ Stage 1  │ Stage 2  │ Stage 3  │
           │ 8 chunks │ 8 chunks │ 8 chunks │ 8 chunks │
    ┌──────┼──────────┼──────────┼──────────┼──────────┤
    │tile 0│ Core 0,0 │ Core 0,1 │ Core 0,2 │ Core 0,3 │  Q rows 0-63
    │  64  │          │          │          │          │
Q   ├──────┼──────────┼──────────┼──────────┼──────────┤
    │tile 1│ Core 1,0 │ Core 1,1 │ Core 1,2 │ Core 1,3 │  Q rows 64-127
chunk│  64  │          │          │          │          │
LQP ├──────┼──────────┼──────────┼──────────┼──────────┤
=256│tile 2│ Core 2,0 │ Core 2,1 │ Core 2,2 │ Core 2,3 │  Q rows 128-191
    │  64  │          │          │          │          │
    ├──────┼──────────┼──────────┼──────────┼──────────┤
    │tile 3│ Core 3,0 │ Core 3,1 │ Core 3,2 │ Core 3,3 │  Q rows 192-255
    │  64  │          │          │          │          │
    └──────┴──────────┴──────────┴──────────┴──────────┘

    Each core processes:
    - Its Q tile (64 rows) x all K/V chunks assigned to its stage (8 chunks)
    - Then cascade merges right-to-left across stages
    - Stage 0 produces the final output for its Q tile rows
```

### Parameter Quick Reference

| Param | Example | What It Controls |
|-------|---------|-----------------|
| **LQ** | 2048 | Total Q rows. Outer loop: `LQ / LQP` iterations |
| **LK** | 2048 | Total K/V columns. Split across 4 cascade stages |
| **LQP** | 256 | Q rows per launch iteration. Split into 4 tiles of `LQP/4` each |
| **LKP** | 64 | K/V columns per chunk. Each stage processes `LK / (LKP*4)` chunks |
| **DK** | 64 | Inner dimension of Q @ K. When `LKP == DK`, shared buffers enabled |
| **DV** | 64 | Inner dimension of softmax @ V. Width of output |
| **NUM_HEADS** | 32 | Q head count. Processed 2 at a time (16 segment iterations) |
| **NUM_KV_HEADS** | 8 | K/V head count. `NUM_HEADS / NUM_KV_HEADS` Q heads share each K/V head |

## Files

| File | Purpose |
|------|---------|
| `attn.py` | Python IR generator — produces AIR MLIR, compiles via XRTRunner, validates against NumPy reference |
| `attn.cc` | C++ AIE kernel — vectorized matmul, softmax, reduction ops compiled for AIE2P cores |
| `test_elf.cpp` | C++ profiling harness — loads compiled ELF via XRT, benchmarks NPU execution time and GFLOPS |
| `Makefile` | Build system — `make run` (correctness), `make profile` (benchmark), `make compile-kernel` |

## Algorithm

Flash Attention computes `softmax(Q @ K + M) @ V` in a tiled, numerically-stable way. Instead of materializing the full `[lq, lk]` attention matrix, it processes K/V in small chunks while maintaining running statistics:

```
Init: Gp = 0, sp = 0, up = -inf

for each K/V chunk:
    G  = Q @ K_chunk + M_chunk        # score
    u  = rowmax(G)                     # chunk max
    u_new = max(up, u)                 # running max
    G  = exp(G - u_new)                # stable softmax numerator
    s  = rowsum(G)                     # chunk sum
    r  = exp(up - u_new)               # rescale factor
    Gp = Gp * r + G @ V_chunk          # accumulate weighted values
    sp = sp * r + s                    # accumulate denominator
    up = u_new

Output = Gp / sp
```

The key insight: `r = exp(old_max - new_max)` rescales previous partial results whenever a new maximum is found, maintaining numerical stability without needing the global max upfront.

## Hardware Mapping

### Tile Grid: 4x4 Herd

The computation maps to a `[4, 4]` grid of AIE compute tiles within a single `air.herd`:

```
                  Cascade Stage 0   Stage 1   Stage 2   Stage 3
Q Tile 0         [Core 0,0]       [Core 0,1] [Core 0,2] [Core 0,3]
Q Tile 1         [Core 1,0]       [Core 1,1] [Core 1,2] [Core 1,3]
Q Tile 2         [Core 2,0]       [Core 2,1] [Core 2,2] [Core 2,3]
Q Tile 3         [Core 3,0]       [Core 3,1] [Core 3,2] [Core 3,3]
```

- **Rows (Q tiles)**: Each row processes a different `tile_size_q`-row slice of Q in parallel
- **Columns (cascade stages)**: Each column handles 1/4 of the K/V sequence length, then merges results via cascade

### AIR Execution Hierarchy

```
air.launch (num_heads/2 iterations — iterates over head groups)
  └─ air.segment (unrolled 2x — processes 2 heads simultaneously)
       └─ air.herd [4x4] (4 Q-tiles x 4 cascade stages)
            └─ scf.for (K/V chunks within each stage)
                 └─ func.call @matmul_*, @exp_*, @max_*, ... (C++ kernel functions)
```

- **2 heads per segment**: Hardware constraint — the segment is unrolled to process 2 Q heads at a time
- **num_heads/2 launch iterations**: Covers all heads (e.g., 16 iterations for 32 heads)

### Memory Hierarchy

```
L3 (DDR)         Q[num_heads, lq, dk], K[num_kv_heads, dk, lk], V[num_kv_heads, lk, dv]
    |  air.channel L3ToL2 (per head, per cascade stage)
L2 (MemTile)     Q_buf[tile_size_q, dk], K_buf[dk, lkp], V_buf[lkp, dv]
    |  air.channel L2ToL1 (broadcast from 1 L2 buffer to 4 L1 tiles)
L1 (Core, 64KB)  Q_tile, K_tile, V_tile, Gp, up, sp, G_temp (~40KB per core)
```

Memory spaces: L3 = no annotation (DDR), L2 = memory_space 1 (MemTile, 256KB), L1 = memory_space 2 (per-core, 64KB)

## Cascade Merge

### Why Cascade?

The K/V sequence (e.g., 12288 elements) is too large for one core. It's split across 4 cascade stages, each processing 1/4 of K/V. Each stage produces partial results `(Gp, up, sp)` that must be merged to get the correct final answer.

### Data Partitioning: Interleaved

Chunks are assigned to stages in an **interleaved** (not contiguous) pattern:

```
K/V chunk index:   0  1  2  3  4  5  6  7  ...
Assigned stage:    0  1  2  3  0  1  2  3  ...
```

This is implemented via strided DMA: `strides=[lkp * num_cascade_stages, ...]` skips 3 chunks between reads.

### Merge Flow: Right-to-Left

After all stages finish their chunks, results flow stage 3 → 2 → 1 → 0 via `air.channel` with type `"cascade"`:

```
Stage 3 (last)                Stage 2                    Stage 1                    Stage 0 (first)
──────────────                ─────────                  ─────────                  ──────────────
Has: Gp3, up3, sp3           Has: Gp2, up2, sp2         Has: Gp1, up1, sp1         Has: Gp0, up0, sp0

  channel.put ──────────────►  channel.get
                               merge(Gp3, Gp2)
                               channel.put ──────────────► channel.get
                                                           merge(acc, Gp1)
                                                           channel.put ──────────────► channel.get
                                                                                       merge(acc, Gp0)
                                                                                       Output = Gp / sp
```

### Merge Algorithm

Same rescaling trick used within a stage, but applied across stages:

```
new_max = max(up_A, up_B)
r_A = exp(up_A - new_max)     # rescale factor for incoming partial
r_B = exp(up_B - new_max)     # rescale factor for local partial
Gp_merged = Gp_A * r_A + Gp_B * r_B
sp_merged = sp_A * r_A + sp_B * r_B
```

This is mathematically exact — produces the same result as single-core sequential processing.

### Implementation via affine.if

All 4 stages share the same herd body. `affine.if` on the stage index (`arg23`) dispatches different behavior:

- `arg23 == 3` (last stage): Only `channel.put` — sends its partial results left
- `1 <= arg23 <= 2` (middle stages): `channel.get` from right, merge, `channel.put` left
- `arg23 == 0` (first stage): `channel.get` from right, merge, normalize (`Gp/sp`), write output

The cascade channel is declared as `Channel("cascade", size=[num_q_tiles, num_cascade_stages - 1])` — 4 rows, 3 links between 4 stages.

## Grouped Query Attention (GQA)

When `NUM_KV_HEADS < NUM_HEADS`, multiple Q heads share the same K/V head. The mapping is:

```
kv_head_index = q_head_index // gqa_group_size
gqa_group_size = NUM_HEADS / NUM_KV_HEADS
```

Example with NUM_HEADS=32, NUM_KV_HEADS=8 (gqa_group_size=4):

```
Q heads:   0  1  2  3 | 4  5  6  7 | ... | 28 29 30 31
KV head:       0       |     1      | ... |      7
```

In the code, this is an `affine.apply` with `floor_div`. K/V data loading uses `kv_head_base` (not the Q head index) to index into the K/V tensors, so multiple Q head groups read the same K/V data.

Input tensor shapes reflect this:
- Q: `[num_heads, lq, dk]`
- K: `[num_kv_heads, dk, lk]` (smaller when GQA)
- V: `[num_kv_heads, lk, dv]` (smaller when GQA)
- M: `[num_heads, lq, lk]` (mask per Q head)
- Output: `[num_heads, lq, dv]`

## Parameters

| Parameter | Default | Description | Constraint |
|-----------|---------|-------------|------------|
| `LK` | 12288 | K/V sequence length | Must be divisible by `LKP * 4` |
| `LKP` | 96 | K/V chunk size per iteration | `LK % (LKP * 4) == 0` |
| `LQ` | 512 | Q sequence length | Must be divisible by `LQP` |
| `LQP` | 128 | Q chunk size per launch iter | Must be divisible by 4 |
| `DK` | 64 | Key dimension | |
| `DV` | 64 | Value dimension | |
| `NUM_HEADS` | 12 | Number of Q attention heads | Must be even |
| `NUM_KV_HEADS` | NUM_HEADS | Number of K/V heads | `NUM_HEADS % NUM_KV_HEADS == 0` |

Derived: `tile_size_q = LQP / 4`, `chunks_per_stage = LK / (LKP * 4)`, `gqa_group_size = NUM_HEADS / NUM_KV_HEADS`

When `LKP == DK`, shared buffers are enabled (Q and K share L1 memory). **This is critical for performance** — see Performance Notes below.

## Performance Notes

### Shared Buffers and the While-True Loop

The `enable_shared_buffers` flag (`LKP == DK`) has a major performance impact beyond just L1 memory sharing. It controls whether the NPU runs a tight internal loop or requires host round-trips:

| `LKP == DK` | `enable_shared_buffers` | `omit_while_true_loop` | Behavior |
|-------------|------------------------|----------------------|----------|
| Yes | True | False (loop kept) | NPU loops over all Q chunks internally — no host round-trips |
| No | False | True (loop removed) | Each Q chunk requires a separate host → NPU invocation |

The host round-trip overhead is substantial. In practice this causes a ~4x performance difference:

| Config | GFLOPS | Notes |
|--------|--------|-------|
| `LKP=128 LQP=128` (shared buffers OFF) | ~537 | 16 host round-trips (2048/128) |
| `LKP=64 LQP=256` (shared buffers ON) | ~2460 | NPU-internal loop, 8 iterations (2048/256) |

**Always prefer `LKP == DK`** (typically `LKP=64` when `DK=64`) for best performance.

### XRT Turbo Mode

The NPU clock frequency can be increased via `xrt-smi`:

```bash
# Enable turbo (max clocks, higher power)
sudo /opt/xilinx/xrt/bin/xrt-smi configure --pmode turbo

# Check current mode
xrt-smi examine -r platform
# Look for: Power Mode : Turbo

# Other modes: default, powersaver, balanced, performance, turbo
```

### LLaMA-3.2-1B Benchmark (seq_len=2048, Turbo Mode)

Best known config for LLaMA-3.2-1B style GQA attention (32 Q heads, 8 KV heads, head_dim=64):

```bash
make profile LQ=2048 LK=2048 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8
```

Result: **~2460 GFLOPS** (NPU Strix, Ryzen AI 9 HX 370, turbo mode)

## Channel Infrastructure

| Channel | Size | Purpose |
|---------|------|---------|
| `L3ToL2Chan1` | [2, 4] | K from DDR to L2 (per head, per stage) |
| `L3ToL2Chan2` | [2, 4] | V from DDR to L2 |
| `L2ToL1Chan1` | broadcast [4, 4] | Q from L2 to L1 (broadcast across Q tiles) |
| `L2ToL1Chan2` | broadcast [4, 4] | K from L2 to L1 |
| `L2ToL1Chan3` | broadcast [4, 4] | V from L2 to L1 |
| `L1ToL2Chan1` | [4, 1] | Results from L1 back to L2 |
| `L2ToL3Chan1` | [2] | Final output from L2 to DDR |
| `cascade` | [4, 3] | Inter-stage partial result transfer |

## C++ Kernel Functions (attn.cc → attn.o)

| Function | Operation |
|----------|-----------|
| `matmul_a_b_bf16` | Q @ K (score computation) |
| `matmul_g_b_bf16` | softmax(scores) @ V (value accumulation) |
| `max_g_bf16` | Row-wise maximum of score matrix |
| `maximum_up_u_bf16` | Element-wise max of two vectors (running max update) |
| `exp_g_minus_u` | exp(G - u) (stable softmax numerator) |
| `exp_up_minus_u` | exp(up - u) (rescale factor computation) |
| `sum_g` | Row-wise sum (softmax denominator) |
| `mul_r_gp` | Gp *= r (rescale accumulated values) |
| `accum_sp_r_s` | sp += s * r (accumulate denominator) |
| `add_gp_g` | Gp += G (add partial results) |
| `div_gp_sp` | Gp / sp (final normalization) |
| `zero_fill_*`, `neg_inf_fill_*` | Buffer initialization |
| `vector_copy_*` | Data movement helpers |

## Build & Run

```bash
# Correctness test (compile + run on NPU + validate vs NumPy)
make run LQ=1920 LK=1920 LKP=96 LQP=128 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8

# Profile (compile ELF + benchmark NPU execution time)
make profile LQ=1920 LK=1920 LKP=96 LQP=128 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8

# Print generated MLIR without compiling
make print LQ=1920 LK=1920 LKP=96 LQP=128 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8

# Debug: dump per-pass IR
make run DEBUG_AIRCC=1 ...
```

## Debugging

1. **`make print`**: Inspect the generated AIR MLIR without compilation
2. **`DEBUG_AIRCC=1`**: Dumps per-pass IR to `build_peano/air_project/debug_ir/`
3. **`-v` flag**: `attn.py -v` for verbose XRTRunner output
4. Key intermediate files in `build_peano/air_project/`:
   - `placed.air.mlir` — after placement (channels, herds assigned)
   - `aie.air.mlir` — after AIR-to-AIE lowering (physical tiles, locks)
   - `input_physical.mlir` — final physical mapping

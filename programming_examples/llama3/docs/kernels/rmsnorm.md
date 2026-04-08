# Weighted RMSNorm Kernel — Performance Analysis

## Role in LLAMA Pipeline

Steps 1 and 10 of each transformer block: weighted RMS normalization.

```
y = x * rsqrt(mean(x^2, axis=-1) + eps) * weight
```

| Parameter | Value |
|-----------|-------|
| Input | [2048, 2048] BF16 (seq_len x emb_dim) for prefill; [1, 2048] for decode |
| Weight | [2048] BF16 (per-column, broadcast to all tiles) |
| Output | [2048, 2048] BF16 for prefill; [1, 2048] for decode |

---

## LLAMA-3.2-1B Configurations

### Config summary

| Usage | Kernel | Shape (M × N) | herd | Standalone (C++) | In pipeline | Rationale |
|-------|--------|--------------|------|-----------------|-------------|-----------|
| **Prefill pre-attn** | `rms_attn_gemms` ELF (launch 1 of 4) | 2048 × 2048 | **[8,1]** | **798 us** | embedded in 9ms | 8 tiles, broadcast weight DMA |
| **Prefill pre-FFN** | `ffn_full` ELF (launch 1 of 6) | 2048 × 2048 | **[8,1]** | **798 us** | embedded in 52ms | Same kernel, different multi-launch |
| **Prefill final** | `rmsnorm` xclbin | 2048 × 2048 | **[8,1]** | **798 us** | **3ms** | Standalone, before LM Head |
| **Decode pre-attn** | `rmsnorm` xclbin | 1 × 2048 | **[1,1]** | **50 us** | **0.3ms** | M=1, can't row-parallel (see below) |
| **Decode pre-FFN** | `rmsnorm` xclbin | 1 × 2048 | **[1,1]** | **50 us** | **0.3ms** | Same kernel, called twice per block |

### Prefill: why [8,1]

Each of M=2048 rows is independent — distributes 256 rows per tile. Weight vector is broadcast to all tiles via a single DMA (no tile-dependent offset). This pattern was previously blocked by a compiler bug (`stride=0` BD rejected by `aie.dma_bd` verifier), fixed upstream in April 2026.

```python
# llama3_prefill.py, compile_all_kernels()
build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)    # standalone

# multi_launch_builder/rms_attn_gemms_multi.py
build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)    # embedded in 4-launch ELF

# multi_launch_builder/ffn_full_multi.py
build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)    # embedded in 6-launch ELF
```

**Profiling (C++ harness, `make profile-cpp`, 10 warmup + 20 measured):**

| Config | Herd | Standalone (min/avg) | Bandwidth | vs IRON |
|--------|------|---------------------|-----------|---------|
| Single tile (old) | [1,1] | 5962 / 5968 us | 2.81 GB/s | 1.4x slower |
| **8-tile (current)** | **[8,1]** | **798 / 799 us** | **21.0 GB/s** | **5.4x faster** |
| Decode (M=1) | [1,1] | 50 / 50 us | 0.25 GB/s | — |
| IRON reference | 16 tiles | 4300 us | — | baseline |

**In-pipeline impact** (from `llama3_prefill.py --profile`):
- `rms_attn_gemms`: 14ms → **9ms** (5ms saved from 8-tile RMSNorm launch)
- `ffn_full`: 57ms → **52ms** (5ms saved)
- `rmsnorm` standalone: 6ms → **3ms**

### Decode: why [1,1]

Decode RMSNorm uses M=1 (single token). The kernel distributes **rows** across tiles (`rows_per_tile = M // herd_x`). With M=1, `1 // 8 = 0` — can't split 1 row across 8 tiles.

A column-parallel alternative is architecturally possible (distribute N=2048 across tiles, with cross-tile L2 reduction for the sum-of-squares), but requires a new kernel design. See `LLAMA_inference.md` section 10A for analysis.

```python
# llama3_decode.py, compile_decode_kernels()
build_rms(1, emb_dim, bfloat16, 16)    # herd_x=1 default, M=1
```

Called **twice per block** (pre-attention + pre-FFN) × 16 blocks = 32 calls per token.
At 0.3ms each, total decode RMSNorm = ~10ms/token (~3% of steady-state).

**Optimization path**: The 50us kernel time is dwarfed by ~250us dispatch overhead per call. The improvement isn't faster RMSNorm — it's **fewer dispatches** by merging into adjacent kernels:
- Merge pre-attn RMSNorm + QKV GEMV → 1 ELF (saves 1 dispatch × 16 = ~5ms)
- Merge pre-FFN RMSNorm + Gate/Up GEMV → 1 ELF (saves 1 dispatch × 16 = ~5ms)

The `rms_qkv_gemv_multi.py` builder already supports this pattern (4-launch ELF with RMS+Q+K+V) but the decode runtime calls them separately. See `LLAMA_inference.md` section 10C.

### Compile-time parameters

| Parameter | Prefill (all 3) | Decode |
|-----------|----------------|--------|
| `M` (rows) | 2048 | 1 |
| `N` (cols) | 2048 | 2048 |
| `herd_x` | 8 | 1 |
| `vector_size` | 16 | 16 |
| Weight broadcast | Yes (single DMA, no tile offset) | N/A (1 tile) |
| Output format | ELF (embedded) / xclbin (standalone) | xclbin |
| Backend flags | `omit_while_true_loop=False` | same |

---

## Architecture

The 8-tile RMSNorm uses a single herd with broadcast weight DMA:

```
herd [8, 1]:
  1. Broadcast weight vector to all 8 tiles (one DMA, no tile-dependent offset)
  2. Each tile processes M/8 = 256 rows:
     for row in tile's partition:
       DMA row from L3 -> L1 (tile-dependent offset)
       Compute: sum(x^2), rsqrt, scale by weight
       DMA result from L1 -> L3 (tile-dependent offset)
```

The broadcast DMA pattern requires the compiler to generate a BD with stride=0, which was fixed upstream in April 2026. See `docs/issues/github_issue_weight_broadcast_dma.md` for the full investigation and resolution.

---

## History

| Date | Change | Impact |
|------|--------|--------|
| 2026-03 | Single tile [1,1] | 6.0ms standalone |
| 2026-03-31 | Multi-tile investigation | 5 attempts, blocked by broadcast DMA bug |
| 2026-04-02 | Broadcast DMA bug fixed upstream | Unblocked 8-tile |
| 2026-04-02 | 8-tile [8,1] integrated into prefill | 0.9ms standalone, total prefill -80ms |

### Investigation attempts

| Attempt | Approach | Result | Root Cause |
|---------|----------|--------|------------|
| 1 | Single herd, weight DMA before loop | FAIL at herd > [1,1] | Broadcast DMA compiler bug |
| 2 | 2-herd (norm + mul) | FAIL | Same bug in mul_herd weight DMA |
| 3 | Weight DMA inside loop | FAIL | BD exhaustion (2048 BDs needed) |
| 4 | Unweighted only (no weight) | PASS at all herds | Confirms bug is broadcast-specific |
| **5** | **Single herd after compiler fix** | **PASS** | **Bug fixed upstream** |

---

## Files

| File | Purpose |
|------|---------|
| `weighted_rms_norm/weighted_rms_norm.py` | Kernel builder (herd_x=1 and herd_x>1 unified path) |
| `weighted_rms_norm/Makefile` | Build targets |

## Commands

```bash
cd programming_examples/weighted_rms_norm

# Single tile
python3 weighted_rms_norm.py --M 2048 --N 2048 --herd-x 1 --profile

# 8-tile (current default in LLAMA pipeline)
python3 weighted_rms_norm.py --M 2048 --N 2048 --herd-x 8 --profile
```

# Flash Attention Issues — LLAMA-3.2-1B Prefill

## Current Status

The 16-layer LLAMA-3.2-1B prefill produces wrong output. **The flash attention kernel is the sole problematic kernel** — all other 14 kernels have correlation >0.999 against CPU reference. Two issues have been identified:

1. **Causal masking was missing** (now fixed): The kernel was compiled with `causal=False`. Fixed by adding `causal=True` to `build_attn()` — the BD exhaustion compilation issue was resolved upstream.

2. **NPU flash attention output doesn't match standard attention** (current blocker): Even with `causal=True`, the NPU kernel output has only **corr=0.31** against `attention_reference()` in `llama3_reference.py`. The kernel passes its own standalone validation (`make run PASS!`) but produces different results from standard causal attention. This difference compounds across 16 layers, resulting in wrong predictions.

## Investigation Timeline

### Phase 1: Identifying the Bad Kernel

- **Single-layer (n_layers=1)**: NPU top-1 matches CPU top-1 (`" is"`). Per-step correlations all >0.999.
- **16-layer (n_layers=16)**: NPU top-1 is `" }\n"` (nonsensical), CPU top-1 is `" the"`. Logits correlation: 0.205.
- Initial hypothesis was BF16 truncation — **disproved** by per-kernel diagnostic.

Built diagnostic infrastructure:
- `--diagnostic` flag in `llama3_prefill.py`: per-layer NPU vs CPU comparison
- `diagnose_layer.py`: per-kernel isolation test (same input, compare outputs)

**Per-kernel isolation test result** (layer 0):

```
 #  Step                   Corr          Max Err    Verdict
 1  RMSNorm(pre-attn)      0.99998600    0.2367     OK
 2  Q proj (GEMM)          0.99998525    0.0467     OK
 3  K proj (GEMM)          0.99997482    0.0498     OK
 4  V proj (GEMM)          0.99985036    0.0086     OK
 5  RoPE Q                 0.99999222    0.1250     OK
 6  RoPE K                 0.99999269    0.0725     OK
 7  FlashAttn GQA          0.34026316    0.6926     FAIL  <<<
 8  O proj (GEMM)          0.99930524    0.0062     OK
 9  Residual add(attn)     1.00000000    0.0000     OK
10  RMSNorm(pre-FFN)       0.99998911    0.1162     OK
11  Gate GEMM              0.99994419    0.0337     OK
12  Up GEMM                0.99992662    0.0167     OK
13  SwiGLU                 0.99997097    0.0681     OK
14  Down GEMM              0.99984624    0.0581     OK
15  Residual add(FFN)      1.00000000    0.0000     OK
```

### Phase 2: Causal Masking Fix

**Root cause**: The CPU reference (`attention_reference()`) applies a causal mask (upper triangular -inf). The NPU kernel was compiled with `causal=False`, performing bidirectional attention.

**Initial fix attempts (before upstream fix):**

1. **`causal=True` compilation** — Failed with BD exhaustion (`Allocator exhausted available buffer descriptor IDs`). Causal mode forces all Q-block iterations device-internal, generating ~144 BDs vs 48 MemTile limit.

2. **External mask input** — The non-causal kernel ignores `arg3` (mask not in launch operands). The mask is never DMA'd.

**Upstream fix**: The BD exhaustion issue was resolved in the flash attention kernel. `causal=True` now compiles successfully:

```bash
cd programming_examples/flash_attention/kernel_fusion_based
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
  NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"
# Result: PASS!
```

### Phase 3: Causal Kernel Integrated — Still Doesn't Match

After integrating `causal=True` into `llama3_prefill.py`:

**Single-layer result**:
- Top-1: `" is"` — matches CPU reference
- Logits correlation against CPU: 0.276 (low, but top-1 correct due to dominant logit)

**16-layer result**:
- Top-1: `','` (wrong). CPU: `' the'`
- Logits correlation: 0.162

**Per-kernel diagnostic with causal kernel** (same-input comparison against `attention_reference()`):

```
 7  FlashAttn GQA           0.30821540    0.7535     FAIL
```

Still corr=0.31 — the causal kernel output differs significantly from standard causal attention (`attention_reference()`), even with identical inputs.

### Phase 4: Understanding the Discrepancy

Key finding: **The standalone `make run` test passes because it validates against its own reference (`flash_attn_per_stage()` in `attn.py`), not against standard attention.**

The two reference implementations:
- `attention_reference()` in `llama3_reference.py`: Standard causal attention in F32. `softmax(Q @ K^T / sqrt(dk) + causal_mask) @ V`.
- `flash_attn_per_stage()` in `attn.py`: Tiled flash attention with cascade merge, BF16 numerics, `bf16_lowest` instead of `-inf`, block-level causal mask skip.

These are mathematically equivalent in exact arithmetic but may differ due to:
1. **Numerical differences**: Flash attention uses online softmax with running max/sum rescaling. The cascade merge across 4 stages introduces additional rescaling steps. With BF16 inputs, these intermediate computations lose precision differently than standard F32 softmax.
2. **`bf16_lowest` vs `-inf`**: The kernel uses `0xFF7F ≈ -3.39e38` instead of `-inf` for masked positions. After `exp()`, this gives a tiny but non-zero value instead of exactly 0, which can shift results.
3. **Block-level causal skip**: The kernel skips entire blocks above the diagonal without computing them. This is correct but changes which partial sums contribute to the cascade merge.
4. **Input/output layout or invocation differences**: There may be subtle differences in how we invoke the kernel from `llama3_prefill.py` vs how the standalone test invokes it.

### Phase 5: Systematic Elimination — Root Cause is the Kernel Itself

Verified step by step:

1. **Two CPU references agree**: `attention_reference()` vs `flash_attn_per_stage()` have **corr=0.99999847** on real LLAMA data. The references are mathematically equivalent.

2. **Input preparation is identical**: Byte-for-byte comparison of Q/K/V arrays between `llama3_prefill.py` and standalone test — **exact match**. Input layout is correct.

3. **Kernel binary is identical**: MD5 hash match between cached and standalone ELF. Same binary, same function name.

4. **Invocation method doesn't matter**: `cache.load_and_run()` vs direct `XRTBackend.load()` produce **identical NPU output**.

5. **Random data also fails**: Tested with the standalone test's own random data (uniform [0.5, 1.0]) — **corr=0.086 against `attention_reference()`**. The kernel doesn't match standard attention for ANY data.

6. **Standalone `PASS!` is meaningless**: The standalone test uses `atol=0.5, rtol=0.2` with V values in [0.5, 1.0]. The tolerance threshold is `0.5 + 0.2 * 1.0 = 0.7`, meaning any output between 0 and ~1.5 passes. The kernel could output constant 0.75 everywhere and still "PASS". **The test cannot distinguish correct from incorrect output.**

## Root Cause: Causal Flash Attention Kernel Correctness Bug

The `causal=True` flash attention kernel does not compute standard causal attention. The NPU output has **corr=0.08-0.31** against both `attention_reference()` and `flash_attn_per_stage()` (which agree with each other at corr=0.9999).

The standalone `make run PASS!` is a false positive due to:
- Test data in narrow range [0.5, 1.0] (all positive, similar magnitude)
- Tolerances `atol=0.5, rtol=0.2` are too loose to detect wrong computation
- Any output in [0, 1.5] passes, regardless of correctness

**This is a kernel-level correctness bug in the causal=True path**, not an invocation or integration issue. The non-causal kernel may have similar issues (it also showed corr=0.34 against standard attention when tested without causal masking).

### Suggested Fix for Standalone Test

The standalone test should use tighter tolerances and/or data with a wider dynamic range:
```python
# Current (too loose):
runner.run_test(..., atol=0.5, rtol=0.2)

# Suggested:
runner.run_test(..., atol=0.05, rtol=0.05)
# Or use data with wider range:
input_q = rng.standard_normal((num_heads, lq, dk)).astype(bfloat16) * 0.1
```

### Next Steps

1. File upstream issue on the causal kernel correctness bug with reproduction steps
2. Check if the **non-causal** kernel also has a correctness bug (it showed corr=0.34 too)
3. As a workaround, consider CPU attention fallback for functional correctness while the kernel is fixed

## Systematic Configuration Study

Tested 10 configurations at LLAMA-3.2-1B shapes (seq_len=2048) using `test_flash_attn_configs.py`. Only 3 out of 10 compiled and passed correctness validation.

**Note**: This study was done before the upstream causal fix. Config #8 (causal baseline) now compiles successfully.

### Results

| # | LQP | LKP | Causal | SharedBuf | Compile | Correct | GFLOPS | Time | Failure Mode |
|---|-----|-----|--------|-----------|---------|---------|--------|------|-------------|
| 1 | 256 | 64  | No  | Yes | OK  | PASS | **2427** | 26s  | — |
| 2 | 128 | 64  | No  | Yes | FAIL | —   | —    | 91s  | Timeout (compilation hangs) |
| 3 | 512 | 64  | No  | Yes | FAIL | —   | —    | 4s   | Pass pipeline failure |
| 4 | 128 | 128 | No  | No  | OK  | PASS | **573** | 164s | — |
| 5 | 256 | 128 | No  | No  | FAIL | —   | —    | 17s  | Pass pipeline failure |
| 6 | 256 | 32  | No  | No  | OK  | PASS | **687** | 61s  | — |
| 7 | 128 | 32  | No  | No  | FAIL | —   | —    | 174s | Timeout (compilation hangs) |
| 8 | 256 | 64  | Yes | Yes | ~~FAIL~~ **NOW OK** | **PASS** | — | — | BD exhaustion (fixed upstream) |
| 9 | 128 | 32  | Yes | No  | FAIL | —   | —    | 600s | Timeout (10 min limit) |
| 10| 512 | 128 | Yes | No  | FAIL | —   | —    | 22s  | Pass pipeline failure |

### Key Observations

1. **Only 3/10 configs work** — the kernel is very fragile to parameter changes
2. **Best config** (#1, LQP=256/LKP=64) is **4.2x faster** than #4 and **3.5x faster** than #6
3. **LQP=128 with shared bufs (#2) hangs** — surprising since it's a simple change from working baseline
4. **LQP=512 (#3) fails immediately** — tile_size_q=128 too large for pass pipeline
5. **Pattern**: LQP=128 configs tend to timeout; LQP=512/LKP=128 configs hit pass pipeline failures

### Failure Categories

- **BD exhaustion**: Causal mode generates too many DMA descriptors — fixed upstream for config #8
- **Pass pipeline failure** (configs #3, #5, #10): Immediate MLIR pass failure — tile sizes likely exceed L1/L2 memory constraints
- **Timeout/hang** (configs #2, #7, #9): Compilation never completes — possible O(n^2) or worse algorithmic issue in a compilation pass

## Code Changes Made

### `llama3_prefill.py`
1. **`causal=True` in `build_attn()`**: Flash attention kernel now compiled with causal masking enabled.
2. **F32 residual path fix**: `run_transformer_block()` carries both `x_bf16` and `x_f32` through residual adds (steps 9, 15).
3. **`--diagnostic` flag**: Runs NPU and CPU per-layer, prints correlation curve.
4. **`run_full_model_diagnostic()`**: Compares NPU vs CPU at each layer boundary.

### New Files
- `diagnose_layer.py`: Per-kernel NPU vs CPU isolation diagnostic.
- `test_flash_attn_configs.py`: Systematic configuration sweep script.
- `flash_attn_study_results.json`: Sweep results data.
- `flash_attn_github_issue.md`: GitHub issue template for upstream.

### Key Diagnostic Commands
```bash
# Per-layer degradation curve (NPU vs CPU)
cd build_peano && python3 ../llama3_prefill.py --run-only --n-layers 16 --diagnostic

# Per-kernel isolation test (identifies the bad kernel)
cd build_peano && python3 ../diagnose_layer.py --layer 0

# Verify single layer
cd build_peano && python3 ../llama3_prefill.py --run-only --n-layers 1 --verify

# Full 16-layer verify (currently produces wrong output)
cd build_peano && python3 ../llama3_prefill.py --run-only --n-layers 16 --verify

# Flash attention standalone causal test (PASS — but uses its own reference)
cd programming_examples/flash_attention/kernel_fusion_based
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
  NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# Configuration sweep
cd build_peano && python3 ../test_flash_attn_configs.py
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `llama3_prefill.py` | NPU orchestrator — builds and runs all kernels |
| `llama3_reference.py` | CPU F32 reference (standard causal attention) |
| `diagnose_layer.py` | Per-kernel NPU vs CPU diagnostic |
| `test_flash_attn_configs.py` | Flash attention configuration sweep |
| `flash_attention/kernel_fusion_based/attn.py` | Flash attention kernel IR generator + its own NumPy reference |
| `flash_attention/kernel_fusion_based/attn.cc` | Flash attention C++ kernel (has `apply_causal_mask()`) |

## Technical Details

### Flash Attention Parameters (LLAMA-3.2-1B at seq_len=2048)
- LQ=2048, LK=2048, LQP=256, LKP=64 (=DK, shared buffers enabled)
- DK=64, DV=64, num_heads=32, num_kv_heads=8
- 4x4 herd: 4 Q-tiles x 4 cascade stages
- tile_size_q = LQP / 4 = 64
- chunks_per_stage = LK / (LKP * 4) = 2048 / 256 = 8

### Hardware BD Limits (AIE2P / NPU2)
- Core tiles: 16 BDs max
- MemTile tiles: 48 BDs max

### Two Attention Reference Implementations

| | `attention_reference()` | `flash_attn_per_stage()` |
|---|---|---|
| File | `llama3_reference.py` | `attn.py` |
| Algorithm | Standard attention | Flash attention (tiled, online softmax) |
| Precision | F32 throughout | BF16 inputs, F32 accumulators |
| Causal mask | `-inf` upper triangular | `bf16_lowest` (≈-3.39e38), block-level skip |
| Softmax | Global softmax per row | Online softmax with running max/sum |
| Multi-stage | N/A | Cascade merge across 4 stages |
| Used by | LLAMA prefill CPU reference | Standalone `make run` validation |

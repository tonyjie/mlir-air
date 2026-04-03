# LLAMA-3.2-1B BF16 Inference on MLIR-AIR (NPU2)

## Progress Report — April 2026

---

## 1. Background

### LLAMA-3.2-1B Architecture

LLAMA-3.2-1B is a 1-billion parameter decoder-only transformer with 16 layers:

| Parameter | Value |
|-----------|-------|
| Layers | 16 |
| Embedding dim | 2048 |
| Attention heads | 32 (Q) / 8 (KV) — Grouped Query Attention |
| Head dim | 64 |
| FFN hidden dim | 8192 |
| Vocab size | 128,256 |
| Data type | BF16 |

Each transformer layer runs the following operations:

```
x -> RMSNorm -> Q/K/V GEMMs -> RoPE -> Attention -> O GEMM + Residual Add
  -> RMSNorm -> Gate/Up GEMMs -> SiLU x mul -> Down GEMM + Residual Add -> output
```

**Prefill** processes the full prompt (seq_len=2048) with GEMM (matrix-matrix multiply).
**Decode** generates tokens one at a time with GEMV (matrix-vector multiply).

### MLIR-AIR Starting Point

MLIR-AIR compiles AI workloads onto AMD NPU2 (AIE2P, Strix) via the AIR dialect pipeline:

```
Python IR gen -> AIR MLIR -> air-opt -> aircc.py -> aiecc -> xclbin/ELF
```

At project start, MLIR-AIR had:
- Individual operator examples (GEMM, softmax, RMSNorm, etc.) at small sizes
- No end-to-end model inference
- No multi-launch ELF support for operator fusion
- No decode (GEMV) pipeline

**IRON** (AMD's reference NPU framework) had a working LLAMA-3.2-1B with both prefill (2.744s) and decode (370ms/token). Our goal: match or exceed IRON's performance using MLIR-AIR.

---

## 2. What We Built

### End-to-End Results

| Phase | AIR | IRON | Improvement |
|-------|-----|------|-------------|
| **Prefill** (seq_len=2048) | **1.92s** | 2.744s | **30% faster** |
| **Decode** (steady-state) | **351ms/token** | 370ms/token | **5% faster** |

Correctness: Top-1 = "Paris" for "The capital of France is" (logits correlation 0.993 vs CPU F32 reference).

### Prefill Pipeline (5 XRT invocations/layer)

| # | Operation | Kernel | Launches | Time |
|---|-----------|--------|----------|------|
| 1 | RMSNorm + Q/K/V GEMMs | `rms_attn_gemms` | 4 | 9ms |
| 2 | RoPE Q+K | `rope_qk` | 2 herds | 11ms |
| 3 | Flash Attention GQA | `flash_attn` | 1 | 20ms |
| 4 | O GEMM + Residual Add | `o_proj_add` | 2 | 6ms |
| 5 | RMSNorm + FFN + Add | `ffn_full` | 6 | 52ms |
| | **Per-layer total** | | **5 calls** | **~100ms** |
| | LM Head | `lm_head` | 8 | 171ms |

### Decode Pipeline (10 XRT invocations/block)

| # | Operation | Kernel | Herd | Time |
|---|-----------|--------|------|------|
| 1 | RMSNorm | `rmsnorm` | [1,1] | 0.3ms |
| 2 | Q+K+V GEMV | `qkv_gemv` ELF | [8,1]x3 | 1.0ms |
| 3-4 | RoPE Q, K | `rope_q/k` | [1,1] | 0.5ms |
| | CPU attention | KV cache + GQA | CPU | ~2ms |
| 5 | O GEMV + Add | `o_gemv_add` ELF | [8,1]+[8,1] | 0.6ms |
| 6 | RMSNorm | `rmsnorm` | [1,1] | 0.3ms |
| 7 | Gate+Up GEMV | `gate_up_gemv` ELF | [8,1]x2 | 2.5ms |
| 8 | SiLU x mul | `silu_mul` | [8,1] | 0.3ms |
| 9 | Down GEMV | `gemv_down` | [8,1] | 2.1ms |
| 10 | Add | `add` | [8,1] | 0.3ms |
| | **Per-block total** | | **10 calls** | **~8ms** |

---

## 3. Prefill: The Optimization Journey

### From 18.67s to 1.92s (10x improvement)

Starting from a naive Python-driven pipeline with scalar F32 computation, we systematically optimized across kernel, host, and dispatch levels:

```
18.67s                                                    1.92s
  |=====================================================>|
  Baseline  BF16  XRT  BO  GEMM  FA  FFN  Read  QKV  Merges  LM  RMS8
  (F32)     Add  Reuse Reuse Tune  NPU Multi Opt  Multi  All   Head Tile
```

| Step | What | Per-Layer | Total | vs IRON |
|------|------|-----------|-------|---------|
| Baseline | F32 scalar, no opt | 1.17s | 18.67s | 7.8x slower |
| +BF16 vectorization | BF16 eltwise, vec=16 | 0.84s | 13.40s | 5.6x |
| +XRT context reuse | Cache XRT contexts | 0.55s | 8.77s | 3.7x |
| +BO pre-allocation | Reuse buffer objects | 0.41s | 6.49s | 2.7x |
| +GEMM/SwiGLU tuning | Tile size optimization | 0.22s | 3.57s | 1.5x |
| +FlashAttention NPU | CPU -> NPU attention | 0.24s | 3.88s | 1.6x |
| +FFN multi-launch | 4 ops in 1 ELF | 0.19s | 3.25s | 1.25x |
| +Read optimization | Skip unnecessary bo.read | 0.16s | 2.65s | 1.05x |
| +QKV multi-launch | 3 GEMMs in 1 ELF | 0.14s | 2.45s | 0.92x |
| +All merges | 5 inv/layer | 0.11s | 1.81s | 0.74x |
| +NPU LM Head + bo.map | 8-launch ELF, zero-copy | 0.107s | 2.05s | 0.75x |
| **+8-tile RMSNorm** | **Broadcast DMA fixed** | **~0.10s** | **1.92s** | **0.70x** |

### Key Optimization Techniques

**Multi-Launch ELF Fusion** — The single biggest architectural decision. Adjacent operators (e.g., RMSNorm + Q/K/V GEMMs) are stitched into one MLIR module containing multiple `air.launch` ops. One `xrt.run()` executes all launches sequentially, eliminating XRT dispatch overhead (~1-2ms per call). This reduced invocations from 10 to 5 per layer.

The stitching is done at the MLIR text level: each sub-kernel is compiled independently, then their IR bodies are extracted, renamed (to avoid symbol conflicts), and reassembled into a combined `func.func` with shared arguments.

**bo.map() Zero-Copy** — IRON uses `bo.map()` to get a memory-mapped view of XRT buffer objects, avoiding memcpy. We adopted the same approach, replacing AIR's default `bo.read()` (which allocates + copies) with `np.frombuffer(bo.map())`. This saved ~30ms/layer for large FFN buffers.

**8-Tile RMSNorm with Broadcast DMA** — RMSNorm's weight vector is shared across all rows. With 8 tiles, we broadcast the weight via a single DMA to all tiles (no tile-dependent offset). This was blocked by a compiler bug (`stride=0` in `aie.dma_bd` rejected by the verifier) until an upstream fix landed. After the fix: standalone RMSNorm went from 6.0ms to 0.9ms (6.7x faster, 4.8x faster than IRON).

**NPU LM Head** — The vocabulary projection (2048 x 128,256) is too large for a single NPU invocation. We partition it into 8 GEMMs (each 2048 x 2048 x 16,032), stitched as an 8-launch ELF. Static weight BOs are pre-loaded once. Result: 171ms (vs IRON's 217ms, vs CPU's 1,526ms).

---

## 4. Decode: From Scratch to Faster than IRON

### Building the Decode Pipeline

Decode required a new kernel type: GEMV (matrix-vector multiply, `C[M] = A[M,K] @ B[K]`), which didn't exist in MLIR-AIR. We built it using the existing `matvec` kernel with 8 AIE columns (`herd_m=8`), each processing M/8 output rows.

The decode pipeline also introduced:
- **KV cache**: CPU-managed numpy arrays, populated from CPU prefill, updated per decode token
- **CPU attention**: GQA (4 Q heads share 1 KV head) with growing sequence length
- **Static weight BO caching**: Per-layer BO isolation via `bo_key` naming (128 BO sets, 8 XRT contexts). Weights written once on the first token, skipped on subsequent tokens

### Three Optimization Phases

| Phase | Time/token | Key Change |
|-------|-----------|------------|
| 1. First working pipeline | ~500ms | 15 NPU calls/block, CPU SiLU, no BO caching |
| 2. Static BOs + bo.map() | ~340ms | Weight write-once, zero-copy reads |
| **3. Multi-launch + NPU SiLU** | **~351ms** | Merge Q+K+V, O+Add, Gate+Up; NPU SiLU x mul |

The biggest single optimization was static weight BOs (Phase 2), which eliminated 1GB+ of redundant weight memcpy per token, saving ~160ms.

### GEMV Kernel Performance vs IRON

| Shape (M x K) | Role | AIR | IRON | Ratio |
|---------------|------|-----|------|-------|
| 2048 x 2048 | Q/O projection | 233us | 214us | 1.1x |
| 512 x 2048 | K/V projection | 81us | 98us | **0.8x (faster)** |
| 8192 x 2048 | FFN gate/up | 837us | 657us | 1.3x |
| 2048 x 8192 | FFN down | 946us | 660us | 1.4x |

The 1.3-1.4x gap on large shapes is architectural: AIR routes data through L2 (MemTile staging), while IRON uses ObjectFIFO for direct DDR-to-L1 transfer. This is a compiler-level difference, not a kernel issue.

---

## 5. AIR vs IRON: Architecture Comparison

| Aspect | AIR | IRON |
|--------|-----|------|
| **Kernel format** | Multi-launch ELF (2-8 `air.launch` per ELF) | Runlist (multiple xclbin entries) |
| **Operator fusion** | Text-based MLIR stitching | Built-in runlist support |
| **Data path** | DDR -> L2 -> L1 (MemTile staging) | DDR -> L1 direct (ObjectFIFO) |
| **Buffer I/O** | `bo.map()` zero-copy (adopted from IRON) | `bo.map()` zero-copy |
| **Weight management** | Static BOs per layer (`bo_key`) | Static BOs |
| **FFN (decode)** | 4 separate NPU calls | 1 fused SwiGLU (5 runlist entries) |
| **LM Head (prefill)** | 8-launch ELF (171ms) | Single op (217ms) |
| **RMSNorm** | 8-tile broadcast (0.9ms) | 16-tile ObjectFIFO (4.3ms) |
| **Prefill dispatches/layer** | 5 | ~12 |
| **Decode dispatches/block** | 10 | ~12 |

---

## 6. Compiler Bugs Encountered & Resolved

| Bug | Impact | Status |
|-----|--------|--------|
| **Broadcast DMA stride=0** | Multi-tile kernels with shared weights rejected | **FIXED** upstream |
| **Bare herd in multi-launch** | Herd without `air.segment` silently dropped from ELF | **Workaround**: `_wrap_ir_in_launch()` adds segment wrapper |
| **BF16 DMA stride limitation** | Cannot do DMA-only transpose for BF16 data | **Hardware limitation** (stride=1 required for sub-32b types) |
| **linalg_fill type mismatch** | Different GEMV tile_m produces incompatible func signatures | **Open**: blocks FFN full merge for decode |

---

## 7. Methodology: Working with Claude Code

### Documentation-Driven Development

We maintained a structured set of documents throughout the project to keep progress organized across multiple working sessions:

| Document | Purpose |
|----------|---------|
| `LLAMA_PLAN.md` | Single-page overview: status, kernel tables, remaining work |
| `LLAMA_progress.md` | Session log with per-phase milestones and kernel breakdowns |
| `perf_opt_prefill.md` | Prefill optimization history with timing progression table |
| `decode/DECODE_PROGRESS.md` | Decode optimization phases and per-kernel breakdown |
| `decode/DECODE_EXPLANATION.md` | Detailed code walkthrough with execution trace |
| `kernels/*.md` | Per-kernel analysis (GEMV flags, RMSNorm investigation, etc.) |
| `issues/*.md` | Compiler bug reports with reproducers |

### Roadmap Approach

The project followed a phased approach, documented in `LLAMA_PLAN.md`:

1. **Infrastructure** — weight loading, CPU reference, kernel cache system
2. **Kernel validation** — each kernel shape verified against CPU F32 reference
3. **Single-layer integration** — one transformer block end-to-end
4. **Full model** — 16 layers, correctness check (Top-1 = "Paris")
5. **Performance optimization** — systematic profiling and kernel-by-kernel improvement
6. **Decode phase** — new GEMV kernel, KV cache, decode-specific optimizations

Each phase had clear entry/exit criteria: correctness verification before moving to performance work, and profiling data to guide which optimization to pursue next.

### Claude Code Workflow

Claude Code was used throughout with several patterns:

- **Plan mode** for non-trivial changes: explore codebase, design approach, get approval before writing code
- **Subagents** for parallel exploration: research IRON's approach, search for existing utilities, investigate compiler bugs
- **Task tracking** for multi-step work: create tasks, mark in-progress/completed, track blockers
- **Auto-memory** for cross-session continuity: key decisions, user preferences, and project context persisted between sessions
- **Iterative profiling loop**: change -> compile -> run -> profile -> compare -> decide next step

The documentation served as both progress tracking and context restoration: each new session reads `LLAMA_PLAN.md` to understand current state and remaining work.

---

## 8. Remaining Work

| Priority | Task | Expected Impact |
|----------|------|-----------------|
| 1 | NPU LM Head for decode | ~40ms/token faster decode |
| 2 | FFN full merge for decode | ~10ms/token (blocked by type mismatch) |
| 3 | NPU prefill for KV cache | 16s -> 2s init time |
| 4 | Unified prefill + decode script | User-facing convenience |
| 5 | DMA transpose (prefill) | 5 -> 3 invocations/layer (blocked by BF16 stride) |

---

## 9. How to Run

```bash
cd programming_examples/llama3/build_peano

# Prefill (compile once, ~3 min)
python3 ../llama3_prefill.py --compile-only --profile
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --profile

# Decode (compile once, ~10s)
python3 ../llama3_decode.py --compile-only
python3 ../llama3_decode.py --run-only --n-tokens 100 --profile
```

---

## 10. Files

```
programming_examples/llama3/
  llama3_prefill.py              Main prefill pipeline
  llama3_decode.py               Main decode pipeline
  llama3_weights.py              Weight loading + RoPE LUT
  llama3_reference.py            CPU F32 reference
  multi_launch_builder/          Multi-launch ELF builders (9 files)
  ffn_swiglu/                    SiLU x mul kernel
  docs/
    LLAMA_PLAN.md                Status overview
    LLAMA_progress.md            Session log
    perf_opt_prefill.md          Prefill optimization history
    decode/                      Decode documentation (5 files)
    kernels/                     Per-kernel analysis (6 files)
    issues/                      Compiler bug reports (4 files)
    report/                      This report
```

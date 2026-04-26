# Performance Profile: LLAMA-3.2-1B BF16 on NPU2

**Model**: LLAMA-3.2-1B, BF16, 16 layers, emb_dim=2048, hidden_dim=8192, vocab=128256

## Performance Summary

| Phase | AIR (NPU2) | IRON | Speedup |
|-------|------------|------|---------|
| **Prefill** (seq_len=2048) | **1.15s kernel / 1.264s wall** | 2.744s | **2.4x (kernel), 2.17x (wall)** |
| **Decode** (steady-state) | **92ms/token (10.8 tok/s)** | 370ms/token (2.7 tok/s) | **4.0x** |

- **Kernel time**: Sum of all `load_and_run()` durations (BO Write + NPU Run + BO Read)
- **Wall time**: End-to-end from embedding to LM Head argmax (includes Python host overhead)
- **2026-04-25 / 2026-04-26 update — two back-to-back optimizations**:
  1. **LM Head GEMM → GEMV** (commit 5de80750): prefill LM Head
     refactored from full-sequence GEMM (171 ms) to single-position GEMV
     (~14 ms) by reusing the decode `lm_head_gemv.elf`; the full-seq NPU
     `rmsnorm` (3 ms) was likewise replaced by CPU RMSNorm on the
     1×emb_dim last row. Wall: 1.528 → 1.374 s (−154 ms).
  2. **Heap-churn fix** (commit a2ad5aa5): the per-call
     `cached_args[3] = x_bf16.reshape(...).astype(bfloat16)` in the
     o_ffn argument prep allocated a fresh ~8 MB array per layer
     (16 layers × ~10 ms = ~160 ms / prefill of glibc growth + page
     faults on cold heap). Passing `copy=False` makes the .astype a
     no-op when source is already bf16. Wall: 1.374 → **1.264 s**
     (−110 ms more, 5-trial mean σ ≈ 5 ms).
  3. **Dead code removal** (commit b7d8c065): post-(1), the
     `preload_lm_head_weights` call in `prepare_runtime` is dead
     (the prefill GEMM ELF it warmed up is no longer invoked). Removed.
     Side benefit: setup phase ~5 s shorter.

  Headline vs IRON 2.744 s wall: **1.78× → 2.17×**.

---

## End-to-End Inference Workflow

The unified pipeline (`llama3_inference.py`) runs through these phases:

```
Phase 1: Compilation  (one-time, ~4 min, cached to disk)
  ┌──────────────────────────────────────────────────────────────┐
  │  compile_all_external_kernels()                              │
  │    silu_and_mul.o  ← ffn_swiglu/silu_and_mul.cc              │
  │    rope.o          ← aie_kernels/rope.cc                     │
  │    attn_npu2.o     ← flash_attention/attn_npu2.cc            │
  │    mv.o            ← matrix_vector_multiplication/mv.cc      │
  │    mv_k8192.o      ← mv.cc with -D renamed symbols          │
  │                                                              │
  │  compile_all_kernels() → prefill_kernel_cache/               │
  │    rms_gemms_rope.elf   (6 launches, 33s)                    │
  │    flash_attn.elf       (1 launch,  46s)                     │
  │    o_ffn.elf            (8 launches, 50s)                    │
  │    lm_head.elf          (8 launches, 108s)                   │
  │    rmsnorm.xclbin       (1 launch,  1s)                      │
  │                                                              │
  │  compile_decode_kernels() → decode_kernel_cache/             │
  │    rms_gemv_rope.elf    (6 launches, 3s)                     │
  │    o_gemv_ffn.elf       (8 launches, 7s)                     │
  │    lm_head_gemv.elf     (8 launches, 12s)                    │
  └──────────────────────────────────────────────────────────────┘

Phase 2: Prepare Runtime  (one-time, before inference)
  ┌──────────────────────────────────────────────────────────────┐
  │  Load model weights from safetensors                   ~3s   │
  │  Pre-transpose decode GEMV weights                     ~2s   │
  │  Pre-load prefill weights into per-layer BOs           ~5s   │
  │    16 layers × (wq, wk, wv, wo, w_gate, w_up, w_down)       │
  │    + LM Head: 8 partitions × 64MB = 512MB                   │
  │  Pre-load decode weights into per-layer BOs            ~8s   │
  │    16 layers × (wq_t, wk_t, wv_t, wo_t, wgate_t, etc.)     │
  │    + LM Head GEMV: 8 partitions × 64MB = 512MB              │
  └──────────────────────────────────────────────────────────────┘

  ══════════════════ PROFILED SCOPE ════════════════════════

Phase 3: Inference
  ┌──────────────────────────────────────────────────────────────┐
  │  PREFILL: 16 layers + CPU final RMSNorm (1 row) + LM Head   │
  │    Kernel time: 1.15s  (BO Write + NPU Run + BO Read)       │
  │    Wall time:   1.264s (includes Python host overhead)      │
  │    LM Head: NPU GEMV at last position only (reuses decode   │
  │             lm_head_gemv.elf — same path as decode loop)    │
  │  DECODE:  per token (16 layers + LM Head GEMV)    → 92ms    │
  └──────────────────────────────────────────────────────────────┘

  ═════════════════════════════════════════════════════════════
```

**Profiled scope matches IRON**: Both frameworks pre-load weights before timing.
IRON reports wall time (end-to-end timed section). AIR kernel time (1.15s) is
the fair comparison to IRON's 2.744s — both measure BO syncs + NPU execution.
AIR wall time (1.264s) includes minimal Python host overhead (KV cache
extraction, embedding, numpy views, 1-row CPU RMSNorm) that IRON's C++
runtime avoids.

Key differences favoring AIR:
- AIR skips intermediate BO syncs (`intermediate_indices`) — IRON syncs ALL BOs
- AIR only reads `output_indices` — IRON reads ALL BOs back after each kernel

---

## Prefill Breakdown (seq_len=2048, 16 layers)

### Wall Time Breakdown: 1.264s

| Component | Time | Notes |
|-----------|------|-------|
| **Kernel time** (sum of `load_and_run`) | 1.15s | BO Write + NPU Run + BO Read (49 kernel calls: 16L × 3 + 1 LM Head GEMV) |
| **Python host overhead** | 0.11s | Everything outside kernel execution; halved by heap-churn fix (`copy=False`) |
| **Total wall time** | **1.264s** | mean over 5 trials, σ ≈ 5 ms |

The 0.22s Python host overhead includes:
- KV cache extraction (reshape + transpose per layer): ~20ms
- Embedding lookup + bf16 conversion: ~15ms
- CPU RMSNorm at last position (1 vector): <1ms
- LM Head argmax (vocab_size=128256 vector): ~1ms
- Other (Python loop, numpy views, filelock): ~180ms

Overhead reduced from 0.67s → 0.22s by:
- Suppressing print I/O in non-profile mode (4 prints × 16 layers)
- Removing dead `x_f32` dual-precision code and `output_f32` conversion
- LM Head: GEMM → GEMV refactor (2026-04-25) — full-sequence projection
  was wasteful since only the last row's logits are used; reuse decode
  `lm_head_gemv.elf` to project just the last position
- Skipping `bf16→f32` conversion on full logits array
- Skipping intermediate dict storage when not verifying
- Removing redundant `.astype(bfloat16)` on already-bf16 kernel results

### Per-Kernel Timing

| Kernel | Launches | Per-call | x Calls | Total | % |
|--------|----------|----------|---------|-------|---|
| **o_ffn** | 8 | 41ms | 16 | **656ms** | **57%** |
| **flash_attn** | 1 | 22ms | 16 | **352ms** | **31%** |
| **rms_gemms_rope** | 6 | 8ms | 16 | **128ms** | **11%** |
| **lm_head_gemv** (prefill end) | 8 | 14ms | 1 | **14ms** | **1%** |

The "lm_head_gemv" row above is the **same ELF used by the decode loop**
(`decode_kernel_cache/lm_head_gemv.elf`) — invoked once at prefill end on
the last real-token's hidden state. Replaced two previously-separate
prefill calls: NPU `rmsnorm` (3 ms, full-sequence) and NPU `lm_head` GEMM
(171 ms, full-sequence).

### Host vs NPU Breakdown (kernel time only)

| | BO Write | NPU Run | BO Read | Total |
|---|----------|---------|---------|-------|
| **Sum** | ~45ms | ~1095ms | ~10ms | ~1150ms |
| **%** | **4%** | **95%** | **1%** | 100% |

### Per-Layer Data Flow

```
Layer input: x_bf16 (2048x2048, 8MB)

┌─ KERNEL 1: rms_gemms_rope (8ms/layer) ─────────────────────────┐
│                                                                 │
│  WRITE: x_in (8MB)              ← activation, changes/layer    │
│  SKIP:  norm_w, wq, wk, wv     ← STATIC (per-layer BO)        │
│  SKIP:  lut_q, lut_k           ← STATIC (same across layers)  │
│  SKIP:  normed, q, k, v,       ← INTERMEDIATE                  │
│         q_roped, k_roped                                        │
│                                                                 │
│  NPU (6 launches):                                              │
│    RMSNorm [8,1] → Q GEMM [8,4] → K GEMM [8,4] →              │
│    V GEMM [8,4] → RoPE Q [8,1] → RoPE K [8,1]                 │
│    (intermediates stay in DDR, no CPU round-trip)              │
│                                                                 │
│  READ: v (2MB), q_roped (8MB), k_roped (2MB)                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 2: flash_attn (22ms/layer) ────────────────────────────┐
│                                                                 │
│  WRITE: q_roped (8MB), k_roped (2MB), v (2MB)                  │
│  SKIP:  attn_out                ← INTERMEDIATE                  │
│                                                                 │
│  NPU (1 launch):                                                │
│    FlashAttention GQA (32Q/8KV heads, causal, seq-first)        │
│                                                                 │
│  READ: attn_out (8MB)                                           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 3: o_ffn (41ms/layer) ─────────────────────────────────┐
│                                                                 │
│  WRITE: attn_out (8MB),         ← from kernel 2                │
│         x_residual (8MB)        ← skip connection from input    │
│  SKIP:  wo, ffn_norm_w,        ← STATIC (per-layer BO)         │
│         w_gate, w_up, w_down                                    │
│  SKIP:  proj, res1, normed2,   ← INTERMEDIATE                   │
│         gate, up, swiglu,                                       │
│         down, output                                            │
│                                                                 │
│  NPU (8 launches):                                              │
│    O GEMM [8,4] → Add [8,1] → RMSNorm [8,1] →                 │
│    Gate GEMM [8,4] → Up GEMM [8,4] → SiLU×mul [8,1] →         │
│    Down GEMM [8,4] → Add [8,1]                                 │
│    (all intermediates stay in DDR, no CPU round-trip)                            │
│                                                                 │
│  READ: output (8MB) → next layer's x_in                        │
└─────────────────────────────────────────────────────────────────┘

× 16 layers, then:
  CPU RMSNorm (<1ms): final layer normalization on last row only (1×2048)
  lm_head_gemv (14ms): 8-partition GEMV → vocab logits → argmax → first token
                       (reuses decode_kernel_cache/lm_head_gemv.elf —
                        same ELF the decode loop calls per token)
```

---

## Decode Breakdown (per token): 92ms

### Per-Component Timing

| Component | Per-token | % |
|-----------|-----------|---|
| **o_gemv_ffn** (NPU, 3.6ms × 16 layers) | **58ms** | **63%** |
| **rms_gemv_rope** (NPU, 0.9ms × 16 layers) | **14ms** | **15%** |
| **lm_head_gemv** (NPU, 1 call) | **14ms** | **15%** |
| CPU (attention + RMSNorm + host) | **5ms** | **5%** |
| BO overhead | **<1ms** | **<1%** |

### Per-Layer Data Flow

```
Token input: x_bf16 (2048 elements, 4KB)

┌─ KERNEL 1: rms_gemv_rope (0.9ms/layer) ────────────────────────┐
│                                                                  │
│  WRITE: x_in (4KB), lut_q (4KB), lut_k (1KB)                    │
│  SKIP:  norm_w, wq, wk, wv        ← STATIC (per-layer BO)      │
│  SKIP:  normed, q, k, v,          ← INTERMEDIATE                │
│         q_roped, k_roped                                         │
│                                                                  │
│  NPU (6 launches):                                               │
│    RMSNorm [1,1] → Q GEMV [8,1] → K GEMV [8,1] →               │
│    V GEMV [8,1] → RoPE Q [1,1] → RoPE K [1,1]                  │
│                                                                  │
│  READ: v (1KB), q_roped (4KB), k_roped (1KB)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌─ CPU ATTENTION (0.3ms/layer) ──────────────────────────────────┐
│  GQA: 32 Q heads attend to KV cache (8 KV heads)               │
│  Update KV cache at current_pos                                 │
│  Output: attn_out (4KB)                                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 2: o_gemv_ffn (3.6ms/layer) ───────────────────────────┐
│                                                                  │
│  WRITE: attn_out (4KB), x_residual (4KB)                         │
│  SKIP:  wo, ffn_norm_w, wgate,     ← STATIC (per-layer BO)      │
│         wup, wdown                                               │
│  SKIP:  proj, res1, normed2,       ← INTERMEDIATE                │
│         gate, up, swiglu,                                        │
│         down, output                                             │
│                                                                  │
│  NPU (8 launches):                                               │
│    O GEMV [8,1] → Add [8,1] → RMSNorm [1,1] →                  │
│    Gate GEMV [8,1] → Up GEMV [8,1] → SiLU×mul [8,1] →           │
│    Down GEMV [8,1] → Add [8,1]                                   │
│                                                                  │
│  READ: output (4KB)                                              │
└──────────────────────────────────────────────────────────────────┘

× 16 layers, then:
  CPU RMSNorm (0.01ms): normalize 2048-element vector
  lm_head_gemv (13.5ms): 8-partition GEMV → vocab logits → argmax → next token
```

### 100-Token Profile

```
Token  1:  92ms  ← steady from first token (NPU prefill keeps NPU warm)
Token  2:  91ms  ┐
Token  3:  91ms  │
...              ├ steady state: 92ms ± 2ms
Token 99:  92ms  │
Token100:  92ms  ┘

Total: 9.21s for 100 tokens = 10.86 tok/s
```

---

## Multi-Launch Memory Model

Each `air.launch` within a multi-launch ELF reads inputs from DDR and writes outputs
back to DDR. Intermediates do **not** stay in L1/L2 between launches — they go through
DDR (L3 memory). What multi-launch saves is the **CPU round-trip**, not the DDR access:

```
SEPARATE XRT CALLS (before multi-launch merging):
  Launch 1 output → DDR
    → bo.sync(FROM_DEVICE)        CPU pulls data from DDR into host cache
    → numpy array in host memory  CPU processes/reshapes
    → bo.map() + memcpy           CPU writes to new BO's mapped memory
    → bo.sync(TO_DEVICE)          CPU pushes data back to DDR
  Launch 2 input ← DDR

MULTI-LAUNCH ELF (current):
  Launch 1 output → DDR
    (no sync, no memcpy, no CPU involvement)
  Launch 2 input ← DDR            NPU DMA reads same DDR buffer directly
```

The DDR is shared physical memory accessible by both CPU and NPU. The difference is:
- **Separate calls**: CPU orchestrates data movement (cache sync + memcpy + cache sync)
- **Multi-launch**: NPU DMA engines handle DDR reads/writes autonomously within
  one `xrt.run()` invocation. The CPU only writes actual input activations and reads
  final outputs.

This is why `intermediate_indices` (SKIP) is effective: these DDR buffers are written
by one launch and read by a subsequent launch — the CPU never needs to see them.

---

## BO Write Categories

Every BO argument falls into one of three categories, controlling whether data is
synced to device on each kernel invocation:

```python
for i, array in enumerate(inputs):
    if i in static_input_indices and not first_call:
        continue    # STATIC: weight pre-loaded, skip
    if i in intermediate_indices and not first_call:
        continue    # INTERMEDIATE: kernel overwrites, skip
    # WRITE: activation data that changes each call
    bo.map() → memcpy → bo.sync(TO_DEVICE)
```

| Category | When Written | Examples |
|----------|-------------|---------|
| **WRITE** | Every call | x_in, attn_out, x_residual, LUTs |
| **STATIC** | First call only | wq, wk, wv, wo, w_gate, w_up, w_down, norm_w |
| **INTERMEDIATE** | First call only | normed, q, k, v, proj, gate, up, swiglu, down |

**Per-layer BOs** (`bo_key=f"kernel_L{layer_idx}"`): Each of 16 layers gets its own
BO set. Weights written once during pre-load, reused forever.

---

## NPU Power Management

The AMD NPU enters low-power state after ~10 seconds of inactivity:

| Idle Duration | Penalty on Next Kernel |
|--------------|----------------------|
| 0-5 seconds | None |
| 10+ seconds | +150ms |

In the unified pipeline (`llama3_inference.py`), this is not an issue: the NPU
prefill (~1.264s of continuous NPU activity) keeps the hardware warm right
up until decode starts. No explicit warmup pass is needed.

In the decode-only script (`llama3_decode.py`), the CPU prefill takes ~17s (NPU idle),
so an explicit warmup pass runs before the timed decode loop.

---

## Key Optimizations

| Optimization | How it Works | Impact |
|-------------|-------------|--------|
| Multi-launch ELF | Stitch multiple kernels into one air.launch func via text-based MLIR stitching | 10→3 prefill calls/layer, 10→3 decode calls/block |
| Per-layer weight BOs | Each layer has dedicated BOs; weights written once, reused | -240ms prefill (14%→4% BO overhead) |
| `intermediate_indices` | Skip host→device sync for buffers the kernel overwrites | -150ms prefill, <1% decode overhead |
| NPU LM Head GEMV | 8-partition GEMV replaces CPU matmul for decode LM Head | 258ms→13.5ms per token |
| **Decode-GEMV reuse for prefill end** | Last-position-only LM Head: drop full-seq NPU GEMM (171ms) + NPU rmsnorm (3ms), reuse decode `lm_head_gemv.elf` on the last row + 1-row CPU RMSNorm | **−154ms wall (1.528 → 1.374s, 5-trial mean), 1.78× → 2.00× vs IRON** |
| **Heap-churn fix in o_ffn arg prep** | `cached_args[3] = x_bf16.reshape(...).astype(bfloat16)` defaulted to `copy=True` even when source was already bf16 — fresh ~8 MB alloc per layer × 16 = ~130 MB heap churn per prefill. Pass `copy=False` so `.astype` is a no-op | **−110ms wall (1.374 → 1.264s, 5-trial mean σ ≈ 5ms), 2.00× → 2.17× vs IRON** |
| External kernel rename | Compile mv.cc with `-D` defines for renamed symbols | Enables K=2048+K=8192 GEMV in one ELF |
| Seq-first layout | RoPE + FlashAttention natively accept (seq, heads×dim) | Zero host transposes in prefill |
| `collapse_shape`/`expand_shape` | 2D↔1D type aliasing inside launch bodies | Enables shape-incompatible kernel merging |
| All .o from source | `compile_all_external_kernels()` builds all C++ kernels fresh | No stale pre-compiled artifacts |

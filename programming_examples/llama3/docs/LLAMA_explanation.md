# LLAMA-3.2-1B Code Explanation

A guide to understand the code, run it, and navigate the file structure.

---

## 1. Big Picture

We run LLAMA-3.2-1B BF16 prefill inference on AMD NPU2 (AIE2P). The model has 16 transformer layers. Instead of running each operator separately, we **merge adjacent operators into multi-launch ELF binaries** — each ELF contains multiple `air.launch` ops that execute sequentially in a single XRT invocation.

```
Python host (llama3_prefill.py)
  ├── Load weights from HuggingFace safetensors
  ├── Compile 7 unique kernel ELFs (one-time, cached)
  ├── For each of 16 layers (5 XRT invocations per layer):
  │     1. rms_attn_gemms   → RMSNorm + Q/K/V GEMMs    (4 launches, 9ms)
  │     2. rope_qk          → RoPE on Q and K           (2 herds [8,1], 4ms))
  │     3. flash_attn       → Flash Attention GQA        (1 launch, 20ms)
  │     4. o_proj_add       → O GEMM + Residual Add      (2 launches, 6ms)
  │     5. ffn_full         → RMSNorm + FFN + Residual   (6 launches, 52ms)
  ├── Final RMSNorm (1 invocation, 3ms)
  ├── LM Head on NPU (8-launch ELF, 171ms)
  └── Output: next token prediction (" Paris")
```

**Performance**: 1.84s total prefill, **33% faster than IRON** (2.744s). RMSNorm uses 8-tile herd with broadcast weight DMA.

---

## 2. How to Run

All commands from `programming_examples/llama3/build_peano/`:

```bash
# First time: compile all kernels (takes ~4 min, cached for reuse)
python3 ../llama3_prefill.py --compile-only

# Run 16-layer inference with profiling
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile

# Run 1 layer with per-step verification against CPU reference
python3 ../llama3_prefill.py --run-only --n-layers 1 --verify

# Run with CPU attention fallback (for debugging)
python3 ../llama3_prefill.py --run-only --n-layers 16 --cpu-attn

# Recompile and run (if code changes)
python3 ../llama3_prefill.py --n-layers 16 --profile
```

**What `--compile-only` does**: Builds 7 MLIR modules (one per kernel), compiles each through `aircc` → `aiecc` to produce ELF binaries, and caches them in `kernel_cache/`. External C++ kernels (SiLU×mul, RoPE, FlashAttention) are compiled from `.cc` source with the Peano compiler.

**What `--run-only` does**: Loads cached ELFs, loads LLAMA weights from HuggingFace, runs the 16-layer pipeline, and reports top-5 predictions and timing.

**What `--verify` does**: After each step in the transformer block, compares NPU output against CPU F32 reference and reports correlation. All steps should show `[OK]` with corr > 0.99.

---

## 3. File Structure

```
llama3/
├── llama3_prefill.py              # Main pipeline orchestrator
│                                    KernelCache: compile, cache, load, run
│                                    compile_all_kernels(): builds all 7 ELFs
│                                    run_transformer_block(): 5 invocations/layer
│                                    run_full_model(): 16 layers + LM Head
│
├── llama3_weights.py              # Weight loading from HuggingFace safetensors
│                                    LlamaConfig, LayerWeights, LlamaWeights
│                                    generate_rope_lut(): RoPE cos/sin LUT
│
├── llama3_reference.py            # CPU F32 reference (for verification)
│                                    rms_norm(), apply_rope(), attention_reference()
│                                    transformer_block(), forward()
│
├── multi_launch_builder/          # Multi-launch ELF builders
│   ├── ffn_full_multi.py            6 launches: RMS + Gate + Up + SiLU + Down + Add
│   │                                 Also contains MLIR text stitching utilities:
│   │                                 _rename_all(), _fix_launch_func_args(), _wrap_ir_in_launch()
│   ├── rms_attn_gemms_multi.py      4 launches: RMSNorm + Q + K + V GEMMs
│   ├── rope_qk_multi.py             2 herds: RoPE Q + RoPE K
│   ├── o_proj_add_multi.py          2 launches: O GEMM + Residual Add
│   ├── lm_head_multi.py             8 launches: LM Head (vocab partitioned)
│   └── attn_gemms_multi.py          3 launches: Q + K + V (standalone, superseded)
│
├── ffn_swiglu/                    # SiLU×mul kernel
│   ├── silu_and_mul.py              Kernel builder (1D and 2D variants)
│   ├── silu_and_mul.cc              C++ kernel for AIE (uses aie::tanh)
│   └── run.py                       Standalone 4-launch FFN test
│
└── docs/                          # Documentation
    ├── LLAMA_PLAN.md                Plan & status overview
    ├── LLAMA_progress.md            Session log
    ├── LLAMA_explanation.md         This file
    ├── perf_opt_prefill.md          Prefill performance results & history
    ├── host_optimization.md         BO/host overhead analysis
    ├── kernels/                     Per-kernel analysis (6 files)
    ├── plans/multi-launch/          Multi-launch optimization plans
    ├── issues/                      Compiler bugs & reproducers
    └── decode/                      Future decode phase planning
```

---

## 4. How Multi-Launch ELF Works

The key optimization technique. Instead of running each operator as a separate XRT call (10 dispatches × 16 layers = 160 dispatches), we **stitch multiple operators into one ELF**:

```
Old: 10 separate XRT calls per layer
  rmsnorm → gemm_q → gemm_k → gemm_v → rope_q → rope_k → flash_attn → gemm_o → add → rmsnorm → ffn... → add

New: 5 XRT calls per layer (each is a multi-launch ELF)
  [rms+q+k+v] → [rope_q+k] → [flash_attn] → [o+add] → [rms+ffn+add]
```

**How stitching works** (in `multi_launch_builder/ffn_full_multi.py`):

1. Build each sub-kernel independently as a standalone MLIR module
2. Serialize each to text (`str(module)`)
3. Rename SSA values with unique prefixes (`_rename_all("g"` for gate, `"u"` for up, etc.)
4. Remap func-arg references to the combined function's args (`_fix_launch_func_args`)
5. Assemble into one MLIR text with a single `func.func` containing multiple `air.launch` ops
6. Parse back into a Module and compile to ELF

**Important**: Bare `air.herd` ops (without `air.segment` wrapper) cause compilation failures in multi-launch ELF. Always wrap with `_wrap_ir_in_launch()` which adds both `air.launch` and `air.segment`. See `docs/issues/github_issue_herd_load_bug.md`.

---

## 5. Data Flow Through One Transformer Block

Using concrete shapes for seq_len=2048:

```
INPUT: x (2048, 2048) bf16

STEP 1-4: rms_attn_gemms [4-launch ELF, single XRT call]
  Launch 1: RMSNorm (1×1 herd)
    x (2048, 2048) + weight (2048,) → normed (2048, 2048)
  Launch 2: Q GEMM (8×4 herd)
    normed @ wq (2048, 2048) → q (2048, 2048)
  Launch 3: K GEMM (8×4 herd)
    normed @ wk (2048, 512) → k (2048, 512)
  Launch 4: V GEMM (8×4 herd)
    normed @ wv (2048, 512) → v (2048, 512)

  --- CPU TRANSPOSE ---
  q (2048, 2048) → reshape (2048, 32, 64) → transpose (32, 2048, 64) → flatten (65536, 64)
  k (2048, 512) → reshape (2048, 8, 64) → transpose (8, 2048, 64) → flatten (16384, 64)

STEP 5-6: rope_qk [2-herd ELF, single XRT call]
  Herd 1: RoPE Q (1×1 herd)
    q_flat (65536, 64) + lut (65536, 64) → q_roped (65536, 64)
  Herd 2: RoPE K (1×1 herd)
    k_flat (16384, 64) + lut (16384, 64) → k_roped (16384, 64)

  --- CPU TRANSPOSE (redundant but kept for clarity) ---
  q_roped → (32, 2048, 64), k_roped → (8, 2048, 64), v → (8, 2048, 64)

STEP 7: flash_attn [single ELF]
  Q (32, 2048, 64) + K (8, 2048, 64) + V (8, 2048, 64) → attn_out (32, 2048, 64)
  GQA: each KV head serves 4 Q heads. Causal masking built-in.

  --- CPU TRANSPOSE ---
  attn_out (32, 2048, 64) → transpose → reshape → (2048, 2048)

STEP 8-9: o_proj_add [2-launch ELF, single XRT call]
  Launch 1: O GEMM (8×4 herd)
    attn_out (2048, 2048) @ wo (2048, 2048) → proj (2048, 2048)
  Launch 2: Residual Add (8×1 herd)
    proj + x_residual → res1 (collapse_shape 2D→1D inside launch)

STEP 10-15: ffn_full [6-launch ELF, single XRT call]
  Launch 1: RMSNorm (1×1 herd, wrapped in air.segment)
    res1 + ffn_norm_weight → normed2
  Launch 2: Gate GEMM (8×4 herd)
    normed2 @ w_gate → gate_buf (2048, 8192)
  Launch 3: Up GEMM (8×4 herd)
    normed2 @ w_up → up_buf (2048, 8192)
  Launch 4: SiLU×mul (8×1 herd, external C++ kernel)
    SiLU(gate_buf) × up_buf → swiglu_buf (2048, 8192)
  Launch 5: Down GEMM (8×4 herd)
    swiglu_buf @ w_down → down_out (2048, 2048)
  Launch 6: Residual Add (8×1 herd)
    res1 + down_out → output (collapse_shape 2D→1D inside launch)

OUTPUT: (2048, 2048) bf16
```

---

## 6. LM Head (After 16 Layers)

The vocabulary projection `x @ lm_head.T` is `(2048, 2048) × (2048, 128256)` — too large for one kernel. We partition N=128256 into 8 chunks of N=16384:

- **8-launch multi-launch ELF**: All 8 GEMM partitions in one `func.func`, one XRT call
- **Static weight BOs**: Weight partitions pre-transposed and pre-loaded into BOs at init time (outside profiling scope)
- **`bo.map()` zero-copy**: All kernels read/write via `bo.map()` instead of `bo.read()`/`bo.write()` — eliminates memcpy for output reads

**Result**: 173ms (vs IRON 217ms, vs CPU 1526ms).

---

## 7. Key Patterns

### `bo.map()` zero-copy (matching IRON's approach)
```python
# Write: map BO memory, copy data directly into it
mv = bo.map()
np.copyto(np.frombuffer(mv, dtype=np.uint8, count=len(src)), src, casting="no")
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)  # Cache flush

# Read: map BO memory, get numpy view (no copy!)
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)  # Cache invalidate
result = np.frombuffer(bo.map(), dtype=bfloat16, count=size)  # Zero-copy view
```

### Static weight BO pre-loading
```python
# First call: write weight to BO (included in timing)
# Subsequent calls: skip writing (static_input_indices)
results = cache.load_and_run("lm_head", backend_kwargs, *inputs,
    static_input_indices={1, 3, 5, 7, 9, 11, 13, 15},  # Weight BO indices
    output_indices=[2, 4, 6, 8, 10, 12, 14, 16],        # Output BO indices
)
```

### `_wrap_ir_in_launch()` — segment wrapper for bare herds
```python
# Bare air.herd FAILS in multi-launch ELF (compiler bug)
# Must wrap in both air.launch AND air.segment:
rms_ir = _wrap_ir_in_launch(str(build_rms_module(...)))
# This transforms: func { air.herd { ... } }
# Into:           func { air.launch { air.segment { air.herd { ... } } } }
```

### Non-contiguous array pitfall
**Any array passed to NPU must be C-contiguous.** `.T` creates a non-contiguous view. Always use `np.ascontiguousarray()`:
```python
# WRONG: .T is a view, DMA will read wrong data
B = weight_matrix.T

# RIGHT: creates a contiguous copy
B = np.ascontiguousarray(weight_matrix.T)
```

---

## 8. Document Index

| Doc | Purpose |
|-----|---------|
| `LLAMA_PLAN.md` | Plan, phase status, file tree |
| `LLAMA_progress.md` | Session log, current status |
| `LLAMA_explanation.md` | This file — code walkthrough |
| `perf_opt_prefill.md` | Prefill performance results, progression, future work |
| `host_optimization.md` | BO/host overhead analysis |
| `kernels/*.md` | Per-kernel analysis (eltwise_add, gemm, silu_and_mul, flash_attention, ffn_swiglu, rmsnorm) |
| `plans/multi-launch/*.md` | Multi-launch merge plans and analysis |
| `issues/*.md` | Compiler bugs, reproducers, GitHub issue drafts |
| `decode/*.md` | Future decode phase planning |

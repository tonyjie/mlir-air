# LLAMA-3.2-1B Code Explanation

A guide to understand the code, from high-level architecture to line-level details.

---

## 1. Big Picture: What Are We Building?

We run LLAMA-3.2-1B inference on AMD NPU2 hardware. The model has 16 transformer layers, each with 15 operations. Instead of compiling the whole model as one monolithic kernel, we compile **each operation as a separate kernel** and invoke them sequentially from Python.

```
Python host code
  |
  +-- Load weights from disk (safetensors)
  +-- For each of 16 layers:
  |     +-- Compile & run RMSNorm kernel on NPU
  |     +-- Compile & run GEMM kernel on NPU (Q projection)
  |     +-- Compile & run GEMM kernel on NPU (K projection)
  |     +-- ... (15 total kernel invocations)
  |     +-- Get result back on CPU
  +-- Run LM Head on CPU (too large for NPU)
  +-- Output: next token prediction
```

The NPU kernels are **reused** across layers -- e.g., the same GEMM kernel config works for all Q projections in all 16 layers. Only ~10 unique kernel configurations are needed.

---

## 2. File-by-File Explanation

### `llama3_weights.py` -- Weight Loading

**What it does**: Reads the model weights from HuggingFace's `.safetensors` format and organizes them into Python dataclasses.

**Key data structures**:
```python
@dataclass
class LlamaConfig:
    n_layers: int = 16
    emb_dim: int = 2048
    n_heads: int = 32        # Q heads
    head_dim: int = 64       # emb_dim / n_heads
    n_kv_heads: int = 8      # K/V heads (GQA: 4 Q heads share 1 KV head)
    hidden_dim: int = 8192   # FFN intermediate size
    vocab_size: int = 128256
    rope_base: float = 500000.0

@dataclass
class LayerWeights:
    attn_norm: np.ndarray   # (2048,)       RMSNorm weight
    wq: np.ndarray          # (2048, 2048)  Q projection: y = x @ wq
    wk: np.ndarray          # (2048, 512)   K projection
    wv: np.ndarray          # (2048, 512)   V projection
    wo: np.ndarray          # (2048, 2048)  O projection
    ffn_norm: np.ndarray    # (2048,)       FFN RMSNorm weight
    w_gate: np.ndarray      # (2048, 8192)  Gate projection
    w_up: np.ndarray        # (2048, 8192)  Up projection
    w_down: np.ndarray      # (8192, 2048)  Down projection

@dataclass
class LlamaWeights:
    embed_table: np.ndarray  # (128256, 2048) token embeddings
    layers: List[LayerWeights]  # 16 layers
    final_norm: np.ndarray   # (2048,)
    lm_head: np.ndarray      # (128256, 2048) -- tied to embed_table in LLAMA-3.2-1B
```

**Key detail -- the transpose**: HuggingFace stores linear layer weights as `(out_features, in_features)`. Our GEMM convention is `y = x @ W`, so W must be `(in_features, out_features)`. We transpose during loading:
```python
if needs_transpose:
    tensor = np.ascontiguousarray(tensor.T)  # Must be contiguous for NPU DMA!
```

**`generate_rope_lut()`**: Pre-computes cos/sin values for Rotary Position Embeddings. Returns `(seq_len, 64)` array with interleaved `[cos, sin, cos, sin, ...]` layout matching the RoPE kernel's expected input format.

---

### `llama3_reference.py` -- CPU Reference

**What it does**: Implements the complete LLAMA forward pass in pure NumPy F32. Used to verify NPU results.

**Key functions** (each mirrors one NPU kernel):
```python
def rms_norm(x, weight, eps=1e-5):
    """x / sqrt(mean(x^2) + eps) * weight"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def apply_rope(x, lut):
    """Rotate pairs: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)"""
    cos_vals, sin_vals = lut[:, 0::2], lut[:, 1::2]
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out_even = x_even * cos_vals - x_odd * sin_vals
    out_odd = x_even * sin_vals + x_odd * cos_vals
    # interleave back...

def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Scaled dot-product attention with GQA and causal mask."""
    # For each Q head, find its KV head: kv_idx = h // group_size
    # scores = q[h] @ k[kv_idx].T / sqrt(head_dim)
    # Apply causal mask (upper triangle = -inf)
    # probs = softmax(scores)
    # out = probs @ v[kv_idx]

def transformer_block(x, layer_weights, rope_lut, config):
    """Returns (output, dict_of_all_15_intermediates)"""
    # Steps 1-15 matching the NPU pipeline

def forward(token_ids, weights, config, rope_lut=None):
    """Full model: embedding -> 16 blocks -> final norm -> lm_head"""
```

The `transformer_block()` returns a dictionary of intermediates (`{"attn_norm": ..., "q": ..., "k": ..., ...}`) which we use for per-step verification against NPU output.

---

### `llama3_prefill.py` -- NPU Integration (The Main File)

This is the most complex file. It has three layers:

#### Layer 1: `KernelCompiler` class

Each method builds an MLIR module and returns a callable that runs on the NPU.

```python
class KernelCompiler:
    def compile_gemm(self, m, k, n, ...):
        """Returns: run_fn(a_bf16, b_bf16) -> c_bf16"""

        # Step 1: Generate AIR MLIR from Python
        mlir_module = build_gemm(m, k, n, ..., arch="aie2p", direct_codegen=True)

        # Step 2: Apply vectorization transform (GEMM-specific)
        transform_ir = Module.parse(GEMM_TRANSFORM_IR, context=mlir_module.context)
        run_transform(transform_ir, mlir_module)

        # Step 3: Return a closure that compiles and runs
        def run_fn(a, b):
            a = np.asarray(a, dtype=bfloat16).reshape(m, k)
            b = np.asarray(b, dtype=bfloat16).reshape(k, n)
            c = np.zeros((m, n), dtype=bfloat16)

            prepare_air_project()  # Clean stale artifacts
            backend = XRTBackend(...)
            compiled = backend.compile(mlir_module)  # aircc -> xclbin
            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)      # Load onto NPU
                results = invoker(a, b, c)            # Execute
            backend.unload()
            return results[-1].reshape(m, n)          # Output is last array

        return run_fn
```

Other `compile_*` methods follow the same pattern but are simpler (no transform IR needed):
- `compile_rms_norm(m, n)` -- calls `weighted_rms_norm.build_module()`
- `compile_rope(seq_len, embed_dim)` -- calls `rope_lut.build_module()`
- `compile_eltwise_add(n)` -- calls `eltwise_add.build_module()`
- `compile_swiglu(n)` -- calls `swiglu_activation.build_module()`
- `compile_flash_attention(lq, lk, ...)` -- calls `attn.build_module()`

#### Layer 2: `run_transformer_block()`

Orchestrates the 15 kernel invocations and handles data reshaping between them:

```python
def run_transformer_block(x_bf16, layer_weights, rope_lut_bf16, config, compiler, ...):
    # Step 1: RMSNorm
    run_rms = compiler.compile_rms_norm(seq_len, emb_dim)
    normed = run_rms(x_bf16, layer_weights.attn_norm)

    # Step 2: Q = normed @ wq
    run_gemm_qo = compiler.compile_gemm(seq_len, emb_dim, emb_dim, ...)
    q = run_gemm_qo(normed, layer_weights.wq)

    # Steps 3-4: K, V projections (similar)

    # Step 5: RoPE on Q
    # KEY RESHAPE: (2048, 2048) -> (2048, 32, 64) -> (32, 2048, 64) -> (65536, 64)
    q_heads = q.reshape(seq_len, n_heads, head_dim)           # Split into heads
    q_flat = q_heads.transpose(1, 0, 2).reshape(n_heads * seq_len, head_dim)  # Flatten
    rope_lut_q = np.tile(rope_lut[:seq_len], (n_heads, 1))    # Repeat LUT for each head
    run_rope = compiler.compile_rope(n_heads * seq_len, head_dim)
    q_roped_flat = run_rope(q_flat, rope_lut_q)
    # Reshape back: (65536, 64) -> (32, 2048, 64) -> (2048, 32, 64) -> (2048, 2048)
    q_roped = q_roped_flat.reshape(n_heads, seq_len, head_dim) \
                          .transpose(1, 0, 2) \
                          .reshape(seq_len, n_heads * head_dim)

    # Step 7: Flash Attention
    # Reshape Q: (2048, 2048) -> (32, 2048, 64), scale by 1/sqrt(64)
    # Reshape K: (2048, 512) -> (8, 2048, 64) -> (8, 64, 2048) [transposed!]
    # Reshape V: (2048, 512) -> (8, 2048, 64)
    # Output: (32, 2048, 64) -> (2048, 2048)

    # Steps 8-15: O projection, residual, FFN (RMSNorm, Gate, Up, SwiGLU, Down, residual)
```

#### Layer 3: `run_full_model()`

```python
def run_full_model(token_ids, weights, config, compiler, rope_lut_bf16, verify=False):
    x = weights.embed_table[token_ids].astype(bfloat16)  # CPU embedding lookup

    for i in range(config.n_layers):  # 16 layers
        x, _ = run_transformer_block(x, weights.layers[i], ...)

    # Final RMSNorm on NPU
    run_rms = compiler.compile_rms_norm(seq_len, config.emb_dim)
    x_normed = run_rms(x, weights.final_norm)

    # LM Head on CPU (128256-dim output is too large for NPU)
    logits = x_normed.astype(np.float32) @ weights.lm_head.T
    return logits
```

---

### `swiglu_activation.py` -- SwiGLU AIR Kernel

**What it does**: Generates AIR MLIR for element-wise SwiGLU: `output[i] = SiLU(gate[i]) * up[i]`.

The Python code constructs the MLIR IR using the `@module_builder` decorator pattern:

```python
@module_builder
def build_module(n, tile_n, np_dtype_in):
    # Declare external C++ function (compiled separately as swiglu_activation.o)
    swiglu_func = FuncOp("swiglu_bf16", ([l1MemrefTy, l1MemrefTy, l1MemrefTy, T.i32()], []))
    swiglu_func.attributes["link_with"] = StringAttr.get("swiglu_activation.o")

    @FuncOp.from_py_func(l3MemrefTy, l3MemrefTy, l3MemrefTy)
    def swiglu_activation(gate, up, output):
        # 1x2 herd: 2 AIE tiles process data in parallel
        @herd(name="herd_0", sizes=[1, 2], operands=[gate, up, output])
        def herd_body(_tx, _ty, ...):
            # Allocate L1 buffers (on-chip SRAM, per tile)
            l1_gate = AllocOp(l1MemrefTy)
            l1_up = AllocOp(l1MemrefTy)
            l1_out = AllocOp(l1MemrefTy)

            # Loop over data tiles
            for loop_iv in range_(0, n, tile_n * 2):
                offset = loop_iv + _ty * tile_n  # Each tile handles its portion

                # DMA: copy tile from DDR (L3) to on-chip L1
                dma_memcpy_nd(l1_gate, gate, src_offsets=[offset], src_sizes=[tile_n])
                dma_memcpy_nd(l1_up, up, src_offsets=[offset], src_sizes=[tile_n])

                # Call C++ kernel (runs on AIE core)
                CallOp(swiglu_func, [l1_gate, l1_up, l1_out, tile_n_i32])

                # DMA: copy result from L1 back to DDR
                dma_memcpy_nd(output, l1_out, dst_offsets=[offset], dst_sizes=[tile_n])
```

The corresponding C++ kernel (`swiglu_activation.cc`) uses AIE vector intrinsics:
```cpp
void swiglu_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
    for (int i = 0; i < n; i += 8) {
        auto g = aie::load_v<8>(gate + i);  // Load 8 BF16 values
        auto u = aie::load_v<8>(up + i);
        // SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
        auto tanh_val = aie::tanh(g_half);
        auto silu = g * 0.5 * (1 + tanh_val);
        auto result = silu * u;
        aie::store_v(out + i, result);
    }
}
```

---

### `swiglu_activation.cc` -- C++ AIE Kernel

Extracted from `ffn_swiglu/prefill/ffn_kernels.cc`. Implements SiLU(x) * y using AIE vector intrinsics (`aie::tanh`, `aie::mul`, `aie::load_v`, `aie::store_v`). Compiled with Peano compiler targeting `aie2p-none-unknown-elf`.

---

## 3. Makefile Commands Explained

Each Makefile target expands to specific Python commands. Here's what actually runs:

### `make compile-external-kernels`

Compiles three C++ kernels with Peano for AIE2P target:

```bash
# 1. SwiGLU kernel
$PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  -I $AIEOPT_DIR/include \
  -c swiglu_activation.cc -o build_peano/swiglu_activation.o

# 2. RoPE kernel (from rope_lut example)
$PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  -I $AIEOPT_DIR/include \
  -c ../rope_lut/rope.cc -o build_peano/rope.o

# 3. Flash Attention kernel (with preprocessor defines for tile sizes)
$PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  -I $AIEOPT_DIR/include -DBIT_WIDTH=8 \
  -Dlqp=64 -Dlkp=64 -Ddk=64 -Ddv=64 \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
  -c ../flash_attention/kernel_fusion_based/attn.cc -o build_peano/attn.o

# Copy .o files where aircc can find them
cp build_peano/*.o build_peano/air_project/
```

### `make run-swiglu`

```bash
# Step 1: Compile swiglu_activation.cc -> swiglu_activation.o
(same as above, just the swiglu part)

# Step 2: Run the Python test
cd build_peano && python3 ../swiglu_activation.py --n 65536 --tile-n 1024 --output-format xclbin
```

The Python script:
1. Calls `build_module(65536, 1024, bfloat16)` -> generates AIR MLIR
2. Creates random test data: `gate = random(-4, 4)`, `up = random(-4, 4)`
3. Computes CPU reference: `expected = SiLU(gate) * up`
4. `XRTRunner.run_test(mlir_module, inputs=[gate, up], expected_outputs=[expected])` -> compiles with aircc, loads xclbin, runs on NPU, compares output

### `make run-prefill-1layer`

```bash
cd build_peano && python3 ../llama3_prefill.py \
  --model /home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/... \
  --seq-len 2048 --n-layers 1 --verify
```

The Python script:
1. Loads LLAMA-3.2-1B weights via `load_weights()` (safetensors -> numpy)
2. Tokenizes prompt "The capital of France is" -> `[128000, 791, 6864, 315, 9822, 374]`
3. Pads to 2048 tokens with EOS
4. Creates `KernelCompiler()` instance
5. Runs `run_full_model()` which calls `run_transformer_block()` once (n_layers=1)
6. Each of the 15 steps: `compile_*()` generates MLIR, `backend.compile()` -> aircc -> xclbin, `backend.load()` -> NPU, `invoker()` -> execute, compare with CPU ref
7. Prints top-5 next-token predictions

### `make run-prefill`

Same as above but `--n-layers 16` and no `--verify` (full model, no per-step comparison).

---

## 4. Data Flow Through One Transformer Block

Using concrete shapes for seq_len=2048:

```
INPUT: x (2048, 2048) bf16

STEP 1: RMSNorm
  kernel: weighted_rms_norm (1x1 herd, single AIE tile)
  input:  x (2048, 2048) + weight (2048,)
  output: normed (2048, 2048)
  how:    For each of 2048 rows: normalize by sqrt(mean(x^2)) then scale by weight

STEP 2: Q Projection
  kernel: matrix_multiplication/bf16 (4x4 herd, 16 AIE tiles)
  input:  normed (2048, 2048) @ wq (2048, 2048)
  output: q (2048, 2048)
  how:    Tiled GEMM with tile_m=32, tile_k_l2=64, tile_k_l1=32, tile_n=32
          Each tile: DMA in -> vectorized matmul (8x8x8 BF16) -> DMA out
          Accumulation in F32, output truncated to BF16

STEPS 3-4: K, V Projections
  Same kernel but n=512 (8 KV heads * 64 dim instead of 32 * 64)

STEP 5: RoPE on Q
  DATA RESHAPE (on CPU, between kernel calls):
    q (2048, 2048)
      -> reshape to (2048, 32, 64)          # Split into 32 heads of dim 64
      -> transpose to (32, 2048, 64)        # Group by head
      -> reshape to (65536, 64)             # Flatten for RoPE kernel
    LUT (2048, 64) -> tile 32x -> (65536, 64)  # Same LUT repeated per head

  kernel: rope_lut (1x1 herd, single AIE tile)
  input:  q_flat (65536, 64) as 1D + lut (65536, 64) as 1D
  output: q_roped_flat (65536, 64) as 1D
  how:    For each row: rotate pairs using cos/sin from LUT

  RESHAPE BACK (on CPU):
    q_roped_flat (65536, 64)
      -> reshape to (32, 2048, 64)
      -> transpose to (2048, 32, 64)
      -> reshape to (2048, 2048)

STEP 7: Flash Attention GQA
  DATA RESHAPE (on CPU):
    q_roped (2048, 2048) -> (2048, 32, 64) -> transpose -> (32, 2048, 64)
    Scale by 1/sqrt(64) = 0.125
    k_roped (2048, 512) -> (2048, 8, 64) -> transpose -> (8, 64, 2048)  [K transposed!]
    v       (2048, 512) -> (2048, 8, 64) -> transpose -> (8, 2048, 64)

  kernel: flash_attention (4x4 herd, cascade pipeline)
  config: LQ=2048, LK=2048, LKP=64, LQP=256, 32Q/8KV heads
  input:  Q (32, 2048, 64) + K (8, 64, 2048) + V (8, 2048, 64) + mask (zeros)
  output: attn_out (32, 2048, 64)
  how:    Online softmax with cascade stages, processes K/V in chunks of LKP=64
          GQA: each KV head serves 4 Q heads (group_size = 32/8 = 4)

  RESHAPE BACK:
    attn_out (32, 2048, 64) -> transpose -> (2048, 32, 64) -> reshape -> (2048, 2048)

STEP 9: Residual Add
  kernel: eltwise_add (1x2 herd, 2 AIE tiles)
  input:  x_flat (4194304,) + proj_flat (4194304,) as float32
  output: res1_flat (4194304,) -> reshape to (2048, 2048) -> cast to bf16

STEPS 11-12: Gate/Up GEMMs
  kernel: matrix_multiplication/bf16 (4x4 herd)
  input:  normed2 (2048, 2048) @ w_gate (2048, 8192) -> (2048, 8192)
  Same for w_up

STEP 13: SwiGLU
  kernel: swiglu_activation (1x2 herd)
  input:  gate_flat (16777216,) + up_flat (16777216,)
  output: swiglu_flat (16777216,) -> reshape to (2048, 8192)

STEP 14: Down GEMM
  kernel: matrix_multiplication/bf16 (4x4 herd)
  input:  swiglu (2048, 8192) @ w_down (8192, 2048) -> (2048, 2048)

STEP 15: Residual Add
  Same as step 9: res1 + down -> output

OUTPUT: (2048, 2048) bf16
```

---

## 5. How the Existing Kernels Work (Brief)

### GEMM (`matrix_multiplication/bf16/run.py`)

The most complex kernel. Uses `direct_codegen=True` which means the vectorized matmul code is generated directly from MLIR (no external `.o` file).

The `build_module()` function creates a 3-level tiled GEMM:
- **Launch level**: Partitions the output matrix into blocks assigned to different NPU segments
- **Segment level**: Allocates L2 (MemTile) buffers, orchestrates DMA between L3 and L2
- **Herd level (4x4)**: Each of 16 AIE tiles handles one sub-tile. DMA from L2 to L1, then a `block_matmul` linalg operation

The transform IR then vectorizes the linalg ops into `vector.contract` operations that map to AIE2P's 8x8x8 BF16 matrix multiply unit.

### Weighted RMSNorm (`weighted_rms_norm.py`)

Single AIE tile (1x1 herd). For each row:
1. DMA row from L3 to L1
2. Vectorized sum of squares (vector.transfer_read, arith.mulf, arith.addf)
3. Horizontal reduction to scalar
4. rsqrt in F32 (for precision)
5. Vectorized multiply: x * rstd * weight
6. DMA result back to L3

### RoPE LUT (`rope_lut.py`)

Single AIE tile (1x1 herd). Calls external `rope()` C++ function:
1. DMA one row (64 elements) from L3 to L1
2. DMA corresponding LUT row to L1
3. Call `rope(input, lut, output, dims)` -- rotates pairs using cos/sin
4. DMA result back to L3
5. Loop over all rows

### Flash Attention (`attn.py`)

Most complex kernel. Uses cascade pipeline (4 stages) across 4x4 tile grid:
- Each cascade stage processes a chunk of K/V (LKP=64 columns at a time)
- Online softmax: maintains running max and sum across chunks
- Stages communicate via cascade channels (hardware FIFO between adjacent tiles)
- Final stage normalizes by sum and outputs result
- GQA handled by segment unrolling: 2 Q heads processed per segment iteration

### Eltwise Add (`eltwise_add.py`)

Simple 1x2 herd. Each tile processes a portion of the data:
- DMA tile from L3 to L1
- Scalar load/add/store loop
- DMA result back to L3

---

## 6. Important Patterns and Pitfalls

### Pattern: `@module_builder` decorator
Every kernel uses this pattern to generate MLIR:
```python
@module_builder
def build_module(params...):
    @FuncOp.from_py_func(input_types...)
    def kernel_name(args...):
        @herd(name="herd_0", sizes=[rows, cols], operands=[args...])
        def herd_body(tx, ty, sx, sy, *data):
            # DMA in, compute, DMA out
```
The decorator creates an MLIR `Module` and returns it.

### Pattern: XRT execution lifecycle
```python
backend = XRTBackend(...)
compiled = backend.compile(mlir_module)       # aircc -> xclbin
invoker = backend.load(compiled)               # Load onto NPU
results = invoker(input1, input2, output)      # Execute
backend.unload()                               # Release
# results is a tuple of ALL arrays (inputs + outputs), output is last
```

### Pitfall: Non-contiguous arrays
**Any array passed to NPU must be C-contiguous.** NumPy operations like `.T`, slicing, or transpose create views with non-standard strides. The XRT DMA copies raw bytes without respecting strides, so non-contiguous data gets scrambled. Always use `np.ascontiguousarray()` before passing to NPU.

### Pitfall: air_project/ directory
aircc uses a hardcoded `air_project/` working directory. When compiling multiple kernels sequentially, stale files from previous compilations can interfere. The `prepare_air_project()` function wipes and recreates this directory before each compilation.

### Pitfall: XRT returns flat arrays
`invoker()` returns all input and output arrays as flat 1D arrays, losing shape information. Always `.reshape()` the output to the expected shape.

---

## Document References

- `LLAMA_PLAN.md` -- High-level plan (phases, architecture decisions)
- `LLAMA_progress.md` -- Session log and current status
- `LLAMA_verification.md` -- Test results, commands, bugs
- `LLAMA_explanation.md` -- This file (code walkthrough)

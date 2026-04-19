# programming_examples/ — End-to-End Operator Examples

Self-contained examples demonstrating AI/ML operators compiled and executed on AMD NPU hardware. Each example generates AIR MLIR from Python, compiles to xclbin, and validates on hardware.

## Directory Structure

37+ example directories organized by operator type. Each contains:

| File | Purpose |
|------|---------|
| `<name>.py` | Python IR generator using `@module_builder` pattern |
| `Makefile` | Build targets: `run`, `print`, `clean` (some: `profile`, `compile-kernel`) |
| `*.lit` | LIT test specs for CI (peano/chess, npu1/npu2 variants) |
| `*.cc` | Optional C++ kernels for external functions |
| `test.cpp` | Optional C++ XRT profiling harness |

### Example categories
- **Linear algebra**: `matrix_multiplication/`, `vector_matrix_multiplication/`
- **Activations**: `relu/`, `silu/`, `gelu/`, `leaky_relu/`
- **Normalization**: `softmax/`, `rms_norm/`, `layer_norm/`, `weighted_rms_norm/`
- **Transformer blocks**: `flash_attention/`, `llama2_mha/`, `llama2_rope/`, `ffn_swiglu/`, `swiglu/`
- **Element-wise**: `eltwise_add/`, `eltwise_add_with_l2/`
- **Data transfer**: `passthrough/`, `data_transfer_transpose/`, `shim_dma_2d/`
- **Primitives**: `primitives/scalar_examples/`, `primitives/vector_examples/`

## Python IR Generation Pattern

### Standard structure
```python
@module_builder
def build_module(size, tile_size, dtype):
    xrt_dtype = type_mapper(dtype)
    memrefTy = MemRefType.get([size], xrt_dtype)

    @FuncOp.from_py_func(memrefTy, memrefTy, memrefTy)
    def func(arg0, arg1, arg2):
        @launch(operands=[arg0, arg1, arg2])
        def launch_body(a, b, c):
            @segment(name="seg0")
            def segment_body():
                @herd(name="herd0", sizes=[1, 1])
                def herd_body(tx, ty):
                    # L1 alloc, DMA in, compute, DMA out
                    yield_([])
```

### Key imports
```python
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
```

### Memory hierarchy in Python
```python
# L2 (MemTile, shared)
l2_mem = IntegerAttr.get(T.i32(), MemorySpace.L2)  # value 1
# L1 (Local, per-core)
l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)  # value 2
# DDR (host) — no memory_space attr
```

### DMA transfers
```python
dma_memcpy_nd(dst, src,
    src_offsets=[...], src_sizes=[...], src_strides=[1],
    dst_offsets=[...], dst_sizes=[...], dst_strides=[1])
```

### Vectorized computation
```python
vecTy = VectorType.get([vector_size], xrt_dtype)
for j in range_(0, size, vector_size):
    sub = subview(buffer, [j], [vector_size], [1])
    v = transfer_read(vecTy, sub, [c0], identity_map, cst0, [True])
    result = arith.mulf(v_a, v_b)
    transfer_write(None, result, sub, [c0], identity_map, [True])
    yield_([])
```

## Execution Pattern

### Command-line arguments (all examples)
```python
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-p", "--print-module-only", action="store_true")
parser.add_argument("--compile-mode", choices=["compile-only", "compile-and-run"], default="compile-and-run")
parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
```

### Validation via XRTRunner
```python
runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=True,
                   output_format=args.output_format, instance_name="example_name")
exit(runner.run_test(mlir_module, inputs=[A, B], expected_outputs=[C],
                     rtol=1e-3, atol=1e-5))
```

### Compile-only mode
```python
backend = XRTBackend(verbose=args.verbose, omit_while_true_loop=True,
                     output_format=args.output_format)
module_function = backend.compile(mlir_module)
backend.unload()
```

## Build & Run

```bash
cd programming_examples/<example_name>
make run                    # compile and validate on NPU
make print                  # print generated MLIR only (no compilation)
make clean                  # remove build artifacts
```

### Complex examples (matrix_multiplication, softmax)
```bash
make profile                # profile with C++ test executable
make run4x4                 # specific herd configuration
make sweep4x4               # benchmarking sweep
make compile-kernel         # compile external C++ kernel only
make run PEANO_INSTALL_DIR=/path/to/peano  # explicit compiler path
```

### Key Makefile variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `PEANO_INSTALL_DIR` | auto | Peano compiler (determines `build_peano/` vs `build_chess/`) |
| `OUTPUT_FORMAT` | `xclbin` | Output: `xclbin` (hardware) or `elf` (direct codegen) |
| `AIE_TARGET` | example-specific | Architecture: `aie2` (NPU1) or `aie2p` (NPU2) |

### Build directory convention
```
build_peano/    # when PEANO_INSTALL_DIR is set
build_chess/    # when using Chess compiler
```

## LIT Test Integration

Tests use LLVM Integrated Tester. Files named `run_[variant]_[compiler].lit`.

```
// REQUIRES: ryzen_ai, peano
// RUN: mkdir -p test_peano
// RUN: cd test_peano && make -f %S/Makefile clean && make -f %S/Makefile run PEANO_INSTALL_DIR=%PEANO_INSTALL_DIR
// CHECK: PASS!
```

### REQUIRES tags
- `ryzen_ai` — any NPU hardware
- `ryzen_ai_npu1` — NPU1 (Phoenix) only
- `ryzen_ai_npu2` — NPU2 (Strix) only
- `peano` / `chess` — compiler backend

### Running LIT tests
```bash
# From build directory (after cmake)
cmake --build . --target check-programming-examples-peano
cmake --build . --target check-programming-examples        # all backends
```

### Operator Dashboard
Auto-generated from LIT files: `python3 programming_examples/generate_readme.py`

## Shared Infrastructure

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | LIT test suite discovery, defines `check-programming-examples-*` targets |
| `lit.cfg.py` | LIT config: detects hardware, sets PYTHONPATH, tool substitutions |
| `makefile-common` | Shared Vitis/AIE build variables and compiler flags |
| `generate_readme.py` | Auto-generates README.md operator dashboard from LIT files |

## Data Types

| NumPy | MLIR | Vector width (AIE2P) | Import |
|-------|------|---------------------|--------|
| `bfloat16` | `BF16Type.get()` | 16 lanes | `from ml_dtypes import bfloat16` |
| `np.float32` | `F32Type.get()` | 8 lanes | stdlib |
| `np.int32` | `T.i32()` | 8 lanes | stdlib |
| `np.int8` | `T.i8()` | 32 lanes | stdlib |

Type mapping: `xrt_dtype = type_mapper(np_dtype)` (from `air.backend.xrt_runner`)

## Complexity Tiers

**Simple** (single file, 1x1 herd): `passthrough_dma`, `eltwise_add`, `relu`
- One `.py` file generates full IR, no external kernels

**Medium** (external kernels or transform scripts): `softmax`, `silu`, `sine_cosine`
- Python IR + `.cc` kernel compiled with Peano/Chess
- Optional transform script for tiling

**Complex** (multi-variant, parameterized): `matrix_multiplication`, `flash_attention`, `ffn_swiglu`
- Multiple Python scripts per architecture variant
- Configurable tile sizes, herd shapes, sweep infrastructure
- C++ profiling harness

## LLM deployments on NPU2

Three end-to-end validated LlamaForCausalLM-class deployments using the
`deploy-new-llm` skill chain (`.claude/skills/deploy-new-llm/`):

| Model | Layers | head_dim | Per-layer rate (decode) | NPU prefill (warm) |
|---|---|---|---|---|
| `llama3/`         (Llama-3.2-1B)  | 16 | 64  | 5.75 ms/layer (10.8 tok/s) | 1.30 s |
| `smollm2_1_7b/`   (SmolLM2-1.7B)  | 24 | 64  | 5.7 ms/layer (7.3 tok/s)   | 1.88 s |
| `llama32_3b/`     (Llama-3.2-3B)  | 28 | **128** | 7.7 ms/layer (4.7 tok/s)  | **3.2 s** (NPU FA via Option C) |

Quick start with the original llama3 reference deployment:
```bash
cd programming_examples/llama3
make compile && make run
```

For deploying a NEW LlamaForCausalLM-class model on NPU2:
- **Invoke `/deploy-new-llm <hf_model_id>` from Claude Code** — the skill walks
  the 7-phase deployment workflow (bootstrap → per-kernel shapes → single block
  → full model → prefill perf → decode perf → finalize) with explicit gates.
- Shared infrastructure (kernel builders, host helpers, head-first FA wrapper
  for head_dim ≥ 128) lives in `_llm_shared/`. See
  [`_llm_shared/README.md`](_llm_shared/README.md) for the helper map.
- Reference deployments to copy from: `llama32_3b/` exercises the full helper
  API (head_dim=128 + Option C FA wrapper); `smollm2_1_7b/` is the simpler
  drop-in case (head_dim=64 + standard seq-first FA).

See `<model>/CLAUDE.md` in each deployment dir for architecture, file map,
design patterns, and tile configs.

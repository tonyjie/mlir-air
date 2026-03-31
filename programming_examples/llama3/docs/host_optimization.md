# Host-Side Optimization Analysis

## Shared-Memory Architecture (Ryzen AI)

CPU and NPU share the same DDR — there is no discrete device memory. XRT buffer objects (BOs) are DDR allocations accessible by both. The `bo.sync()` calls are **CPU cache coherency operations**, not DMA transfers:

- `bo.sync(TO_DEVICE)` = flush CPU cache → ensures NPU sees latest CPU writes
- `bo.sync(FROM_DEVICE)` = invalidate CPU cache → ensures CPU sees latest NPU writes

All data stays in DDR. The overhead comes from **memcpy between different DDR regions** (numpy arrays ↔ BO memory), not from transfers across a bus.

---

## XRT Buffer Operations Compared

### Write path (host → BO)

**AIR approach** (`KernelCache.load_and_run`):
```python
bos[i].write(numpy_array, 0)      # memcpy: numpy DDR addr A → BO DDR addr B
bos[i].sync(TO_DEVICE)             # flush CPU cache
```

**IRON approach** (`aie_base.py`):
```python
mv = bo.map()                      # memory-mapped view of BO DDR addr B
dst = np.frombuffer(mv, ...)       # numpy wrapping addr B (no copy)
np.copyto(dst, src_bytes)          # memcpy: numpy DDR addr A → addr B
# later: bo.sync(TO_DEVICE)        # flush CPU cache
```

**Both do exactly 1 memcpy** of the same size. `bo.write()` is C++ XRT internally, `np.copyto()` is numpy. No meaningful difference.

### Read path (BO → host)

**AIR approach**:
```python
bos[i].sync(FROM_DEVICE)           # invalidate CPU cache
result = bos[i].read(size, 0)      # ALLOCATES new numpy array + memcpy: BO → new array
```

**IRON approach**:
```python
# bo.sync(FROM_DEVICE) in run_runlist
mv = bo.map()                      # memory-mapped view of BO memory
arr = np.frombuffer(mv, ...)       # numpy view pointing at BO memory — NO COPY
return arr                          # zero-copy: numpy wraps BO memory directly
```

**Key difference**: `bo.read()` allocates a new array and copies. `bo.map()` + `np.frombuffer()` returns a zero-copy view — numpy points directly at the BO's DDR allocation. For large buffers this saves both allocation overhead and memcpy time.

### Summary

| Operation | AIR (`bo.write/read`) | IRON (`bo.map`) | Difference |
|-----------|----------------------|-----------------|-----------|
| Write | 1 memcpy | 1 memcpy | **Same** |
| Read | 1 alloc + 1 memcpy | **Zero-copy view** | **IRON wins** |
| Sync | Same cache ops | Same cache ops | **Same** |

---

## Optimizations Applied

### 1. Read-only-output (2026-03-31)

**Problem**: `load_and_run` read back ALL buffers after kernel execution, including inputs, weights, and intermediates that the host doesn't need.

For the FFN multi-launch (8 buffers, 218MB total), reading all buffers took **46ms** — more than the kernel itself (36ms).

**Fix**: Only sync + read the last buffer (the output). Inputs, weights, and intermediates stay in BO memory untouched.

**Impact**: FFN multi-launch 83ms → 52ms (**37% faster**). Per-layer 190ms → 160ms. Total kernel time 2.71s → 2.10s.

### 2. Multi-launch eliminates inter-kernel memcpy (2026-03-30)

**Problem**: With 4 separate FFN kernels, each kernel's output was read back to host (`bo.read` → numpy), reshaped, then written to the next kernel's input BO (`bo.write`). This caused 3 unnecessary DDR→DDR round-trips for intermediates:

```
Gate GEMM: bo.write(input) → kernel → bo.read(gate_out)     # 33MB read
           numpy reshape
Up GEMM:   bo.write(input) → kernel → bo.read(up_out)       # 33MB read
           numpy reshape
SiLU×mul:  bo.write(gate_out, up_out) → kernel → bo.read(swiglu_out)  # 33MB+33MB write, 33MB read
           numpy reshape
Down GEMM: bo.write(swiglu_out) → kernel → bo.read(output)  # 33MB write, 8MB read
```

Total unnecessary memcpy: ~200MB of DDR→DDR copies.

**Fix**: Multi-launch puts all 4 kernels in one ELF with shared DDR buffer arguments. The NPU writes gate_out/up_out/swiglu_out to DDR in one launch and reads from the same DDR addresses in the next launch. No host involvement between launches.

**Impact**: FFN block 149ms → 83ms → 52ms (after combining with read-only-output).

### 3. Skip writing intermediate buffers (potential)

**Status**: Not yet implemented.

We still `bo.write()` zeros to intermediate buffers (gate_buf, up_buf, swiglu_buf, output) even though the kernel will overwrite them. This wastes ~109MB of writes (~4ms).

The fix would be to only write buffers that contain actual input data (input, w_gate, w_up, w_down) and skip the intermediate/output buffers.

---

## Current Overhead Breakdown (FFN Multi-Launch)

After read-only-output optimization:

| Phase | Time | Data | Notes |
|-------|------|------|-------|
| BO write (input + 3 weights) | ~5ms | 109 MB needed | Changes each layer |
| BO write (4 intermediates) | ~4ms | 109 MB zeros | **Unnecessary** — kernel overwrites |
| BO sync (TO_DEVICE) | ~1ms | cache flush | |
| **Kernel (4 PDI launches)** | **~36ms** | | Gate + Up + SiLU×mul + Down |
| BO sync (FROM_DEVICE) | ~0.1ms | cache invalidate | Output only |
| BO read (output only) | ~1ms | 8.4 MB | |
| **Total** | **~52ms** | | |

Eliminating unnecessary intermediate writes would save ~4ms → ~48ms total.

---

## How IRON Handles Intermediate Buffers

IRON's `AIESwiGLUPrefill` allocates intermediate BOs once during `set_up_runtime()`:

```python
self.add_buffer("left", padded_size)           # gate output
self.add_buffer("left_swished", padded_size)   # SiLU output
self.add_buffer("right", padded_size)          # up output
self.add_buffer("intermediate", padded_size)   # SiLU×mul output
```

These BOs are allocated once and never re-written by the host. Each `run_runlist()` call just dispatches kernels that read/write these BOs. The host only writes the **input** buffer and reads the **output** buffer.

In `_execute_aie_operation()`:
```python
self.write_buffer("input", x_flat)    # Only write the actual input
self.run_runlist()                     # Kernels fill intermediates internally
result = self.read_buffer("output", ...)  # Only read the final output
```

Intermediates persist across `run_runlist()` calls within the same forward pass but are overwritten by the kernel each time. No host memcpy for intermediates.

---

## Performance Progression

| Stage | FFN (ms) | Per-layer (ms) | 16-layer (s) | Gap to IRON |
|-------|---------|---------------|-------------|-------------|
| 4 separate kernels | 149 | 243 | 3.88 | 1.59× |
| + Multi-launch | 83 | 190 | 3.25 | 1.25× |
| + Read-only-output | **52** | **~160** | **2.65** | **~1.08×** |
| IRON reference | 57.4 | 152 | 2.44 | 1.0× |

---

## Remaining Optimization Opportunities

| Priority | Action | Savings | Complexity |
|----------|--------|---------|-----------|
| 1 | Skip writing intermediate/output buffer zeros | ~4ms/layer | Low — check buffer index |
| 2 | Use `bo.map()` for zero-copy reads | ~1ms/layer | Low — replace `bo.read` with `np.frombuffer(bo.map())` |
| 3 | Pre-load static weights via `bo.map()` | ~2ms/layer | Medium — cache weight BOs across layers |
| 4 | Attention-path multi-launch | ~15ms/layer | High — same stitching approach as FFN |

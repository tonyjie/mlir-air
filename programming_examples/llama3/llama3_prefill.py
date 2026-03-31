# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Prefill on MLIR-AIR (NPU2)

Orchestrates sequential NPU kernel invocations for a LLAMA-3.2-1B prefill
(seq_len=2048). Supports kernel caching (compile once, run many) and profiling.

Architecture:
  1. Compile phase: Build MLIR for each unique kernel config, compile via
     aircc, save binaries to kernel_cache/ directory.
  2. Run phase: Construct XRTCompileArtifact from cached binary paths,
     load via XRTBackend.load(), execute on NPU.

There are only 10 unique kernel configs across 16 layers (240 invocations):
  - 1 RMSNorm config
  - 4 GEMM configs (Q/O, K/V, Gate/Up, Down)
  - 2 RoPE configs (Q heads, K heads)
  - 1 Flash Attention config
  - 1 SwiGLU config
  - 1 Eltwise Add config

Usage:
  python3 llama3_prefill.py --model ... --seq-len 2048 --n-layers 16
    --compile-only    # Just compile kernels to cache
    --run-only        # Run using cached kernels (skip compilation)
    --profile         # Enable timing instrumentation
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

# Add parent directory to path for kernel imports
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)

from llama3_weights import LlamaConfig, load_weights, generate_rope_lut
from llama3_reference import (
    rms_norm as rms_norm_ref,
    apply_rope as apply_rope_ref,
    swiglu as swiglu_ref,
    attention_reference,
    forward as forward_ref,
    transformer_block as transformer_block_ref,
    softmax,
)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def prepare_air_project():
    """Clean and prepare the air_project/ directory for a fresh compilation.

    aircc defaults to 'air_project/' as its working directory. Sequential
    compilations leave stale artifacts that corrupt subsequent kernels.
    This method wipes the directory and re-copies external .o files.
    """
    air_proj = Path("air_project")
    if air_proj.exists():
        shutil.rmtree(air_proj)
    air_proj.mkdir(parents=True, exist_ok=True)

    # Copy external kernel .o files if they exist in the current build dir
    for obj_name in ["silu_and_mul.o", "rope.o", "attn.o"]:
        src = Path(obj_name)
        if src.exists():
            shutil.copy2(src, air_proj / obj_name)


# ---------------------------------------------------------------------------
# GEMM transform IR (shared by compilation)
# ---------------------------------------------------------------------------

GEMM_TRANSFORM_IR = """
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func0 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

        %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %inner_most_matmul, %vec_loops:3 =
          transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
        transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

        %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

        %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op

        %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
        %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
        %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

        %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
        %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

        %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
        %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %all_extf_loop = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_1, %extf_bf16_2, %extf_bf16_3, %extf_bf16_4 = transform.split_handle %all_extf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %truncf_1, %truncf_2, %truncf_3, %truncf_4 = transform.split_handle %all_truncf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extf_bf16_1, %truncf_1, %innermost_for1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

        %all_extf_loop_2 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_2 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %truncf_2_1, %truncf_2_2, %truncf_2_3 = transform.split_handle %all_truncf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extf_bf16_2_new, %truncf_2_1, %for1_1_hoisted_1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

        %all_extf_loop_3 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_3 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
        %extf_bf16_3_new, %e3_7 = transform.split_handle %all_extf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %truncf_3_1, %truncf_3_2 = transform.split_handle %all_truncf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extf_bf16_3_new, %truncf_3_1, %for1_1_hoisted_2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

        %all_extf_loop_4 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
        %all_truncf_loop_4 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
        %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extf_loop_4, %all_truncf_loop_4, %for1_1_hoisted_3 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    }
"""


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class Profiler:
    """Tracks per-kernel and per-layer execution times."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.compile_times = {}  # name -> seconds
        self.kernel_times = {}  # name -> list of seconds
        self.layer_times = []  # list of (layer_idx, seconds)

    def record_compile(self, name, duration):
        if self.enabled:
            self.compile_times[name] = duration

    def record_kernel(self, name, duration):
        if self.enabled:
            self.kernel_times.setdefault(name, []).append(duration)

    def start_layer(self):
        if self.enabled:
            return time.time()
        return None

    def end_layer(self, layer_idx, t0):
        if self.enabled and t0 is not None:
            self.layer_times.append((layer_idx, time.time() - t0))

    def report(self):
        if not self.enabled:
            return

        print(f"\n{'='*60}")
        print("PROFILING REPORT")
        print(f"{'='*60}")

        if self.compile_times:
            print(f"\n--- Compilation Phase ---")
            total_compile = 0
            for name, t in sorted(self.compile_times.items()):
                print(f"  {name:40s} {t:8.1f}s")
                total_compile += t
            print(
                f"  {'Total compilation':40s} {total_compile:8.1f}s ({len(self.compile_times)} kernels)"
            )

        if self.layer_times:
            print(f"\n--- Per-Layer Execution ---")
            for idx, t in self.layer_times:
                print(f"  Layer {idx:3d}: {t:8.2f}s")
            total_layers = sum(t for _, t in self.layer_times)
            print(f"  {'Total prefill':40s} {total_layers:8.2f}s")

        if self.kernel_times:
            print(f"\n--- Kernel Breakdown (avg per invocation) ---")
            total_avg = 0
            for name, times in sorted(self.kernel_times.items()):
                avg = sum(times) / len(times)
                total_avg += avg * len(times)
                mn = min(times)
                mx = max(times)
                count = len(times)
                print(
                    f"  {name:40s} avg={avg:6.3f}s  "
                    f"min={mn:6.3f}s  max={mx:6.3f}s  (x{count})"
                )
            if self.layer_times:
                n_layers = len(self.layer_times)
                print(f"  {'Total kernel time':40s} {total_avg:8.2f}s")
                print(
                    f"  {'Avg per layer (kernel time)':40s} {total_avg/n_layers:8.2f}s"
                )


# ---------------------------------------------------------------------------
# Kernel Cache
# ---------------------------------------------------------------------------


class KernelCache:
    """Pre-compiles unique kernel binaries and caches them for reuse.

    The key insight: XRTCompileArtifact is a simple dataclass with
    (output_binary, kernel, insts) paths. backend.load(artifact) reads from
    these paths -- no compilation context needed. So we compile each unique
    kernel once, save the binary to a cache directory, and construct artifacts
    from saved paths at runtime.
    """

    # Manifest file stores artifact metadata for --run-only mode
    MANIFEST_FILE = "manifest.json"

    def __init__(self, cache_dir=None, verbose=False, profiler=None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "kernel_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.profiler = profiler or Profiler()
        self.artifacts = {}  # name -> XRTCompileArtifact
        self._loaded = {}  # name -> (backend, invoker) for XRT context reuse
        self._cached_bos = {}  # name -> list of xrt.bo for BO reuse

    def _log(self, msg):
        if self.verbose:
            print(f"  [KernelCache] {msg}")

    def _save_manifest(self):
        """Save artifact metadata so --run-only can reconstruct artifacts."""
        manifest = {}
        for name, art in self.artifacts.items():
            manifest[name] = {
                "output_binary": str(art.output_binary),
                "kernel": art.kernel,
                "insts": str(art.insts) if art.insts else None,
            }
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self._log(f"Saved manifest with {len(manifest)} entries")

    def load_manifest(self):
        """Load artifact metadata from a previous compilation.

        Returns True if manifest was loaded successfully, False otherwise.
        """
        from air.backend.xrt import XRTCompileArtifact

        manifest_path = self.cache_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        for name, info in manifest.items():
            binary_path = info["output_binary"]
            if not Path(binary_path).exists():
                print(f"  WARNING: cached binary not found: {binary_path}")
                return False
            self.artifacts[name] = XRTCompileArtifact(
                binary_path, info["kernel"], info["insts"]
            )

        self._log(f"Loaded manifest with {len(self.artifacts)} entries")
        return True

    def compile_and_cache(
        self, name, mlir_module, backend_kwargs, output_binary_name="air"
    ):
        """Compile kernel and save binary to cache.

        Args:
            name: Unique name for this kernel config
            mlir_module: MLIR module to compile
            backend_kwargs: Dict of kwargs for XRTBackend constructor
            output_binary_name: Base name for output binary
        """
        from air.backend.xrt import XRTBackend

        self._log(f"Compiling {name}...")
        prepare_air_project()
        backend = XRTBackend(**backend_kwargs)

        t0 = time.time()
        artifact = backend.compile(mlir_module, output_binary_name=output_binary_name)
        compile_time = time.time() - t0
        self.profiler.record_compile(name, compile_time)

        # Copy binary to cache with unique name
        src_binary = Path(artifact.output_binary)
        ext = src_binary.suffix  # .xclbin, .elf, or .txn
        cached_binary = self.cache_dir / f"{name}{ext}"
        shutil.copy2(src_binary, cached_binary)

        # Copy instructions file if present (xclbin mode)
        cached_insts = None
        if artifact.insts and Path(artifact.insts).exists():
            cached_insts = str(self.cache_dir / f"{name}.insts.bin")
            shutil.copy2(artifact.insts, cached_insts)

        from air.backend.xrt import XRTCompileArtifact

        self.artifacts[name] = XRTCompileArtifact(
            str(cached_binary), artifact.kernel, cached_insts
        )
        backend.unload()

        print(f"  Compiled {name}: {compile_time:.1f}s -> {cached_binary.name}")

    def load_and_run(self, name, backend_kwargs, *inputs):
        """Load cached kernel and execute with BO reuse.

        Three levels of caching to minimize per-invocation overhead:
        1. XRT context (device, xclbin, kernel) — cached per kernel name
        2. Buffer Objects — cached per kernel name, reused across calls
        3. Instruction BO sync — done once on first call

        Args:
            name: Kernel name (must have been compiled first)
            backend_kwargs: Dict of kwargs for XRTBackend constructor
            *inputs: numpy arrays to pass to the kernel

        Returns:
            Tuple of numpy arrays (all kernel outputs)
        """
        import pyxrt as xrt
        from air.backend.xrt import XRTBackend

        if name not in self.artifacts:
            raise RuntimeError(
                f"Kernel '{name}' not found in cache. "
                f"Available: {list(self.artifacts.keys())}"
            )

        # Level 1: Load backend on first call (XRT context reuse)
        if name not in self._loaded:
            artifact = self.artifacts[name]
            backend = XRTBackend(**backend_kwargs)
            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(artifact)
            self._loaded[name] = (backend, invoker)
            self._log(f"Loaded {name} (XRT context cached)")

        backend, _ = self._loaded[name]

        # Level 2: Allocate BOs on first call, reuse on subsequent calls
        sizes_in_bytes = [a.size * a.itemsize for a in inputs]
        is_elf = self.artifacts[name].output_binary.endswith(".elf")

        if name not in self._cached_bos:
            if is_elf:
                bos = [xrt.ext.bo(backend.device, s) for s in sizes_in_bytes]
            else:
                bos = [
                    xrt.bo(
                        backend.device,
                        s,
                        xrt.bo.host_only,
                        backend.kernel.group_id(i + 3),
                    )
                    for i, s in enumerate(sizes_in_bytes)
                ]
            self._cached_bos[name] = bos
            # Sync instruction BO once (only needed for xclbin mode)
            if not is_elf and backend.bo_instr is not None:
                backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._log(f"Allocated {len(bos)} BOs for {name}")

        bos = self._cached_bos[name]

        # Write input data to cached BOs
        t0 = time.time()
        with filelock.FileLock("/tmp/npu.lock"):
            for i, a in enumerate(inputs):
                if a.dtype == bfloat16:
                    a = a.view(np.int16)
                bos[i].write(a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Launch kernel
            if is_elf:
                run = xrt.run(backend.kernel)
                for i, bo in enumerate(bos):
                    run.set_arg(i, bo)
                run.start()
                run.wait2()
            else:
                h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
                h.wait()

            # Read results back
            for i in range(len(inputs)):
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            results = tuple(
                bos[i].read(s, 0).view(inputs[i].dtype)
                for i, s in enumerate(sizes_in_bytes)
            )

        duration = time.time() - t0
        self.profiler.record_kernel(name, duration)
        return results

    def unload_all(self):
        """Unload all cached XRT backends and BOs. Call at program end."""
        self._cached_bos.clear()
        for name, (backend, _) in self._loaded.items():
            backend.unload()
            self._log(f"Unloaded {name}")
        self._loaded.clear()

    @property
    def kernel_names(self):
        return list(self.artifacts.keys())


# ---------------------------------------------------------------------------
# Kernel compilation definitions
# ---------------------------------------------------------------------------

# Each kernel config is defined as a dict with:
#   build_fn: callable that returns an MLIR module
#   backend_kwargs: dict for XRTBackend constructor
#   output_binary_name: base name for output (optional)


def _build_gemm_module(
    m, k, n, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=64, herd_m=8, herd_n=4
):
    """Build and transform a GEMM MLIR module.

    Uses BF16 output (rounding mode fix landed upstream 2026-03-19).
    Default tile config optimized for NPU2 8×4 herd.
    """
    from matrix_multiplication.bf16.run import build_module as build_gemm
    from air.ir import Module
    from air.dialects.air import run_transform

    mlir_module = build_gemm(
        m,
        k,
        n,
        tile_m,
        tile_k_l2,
        tile_k_l1,
        tile_n,
        herd_m,
        herd_n,
        bfloat16,
        bfloat16,  # BF16 output (rounding fix applied)
        arch="aie2p",
        direct_codegen=True,
    )

    transform_ir = Module.parse(GEMM_TRANSFORM_IR, context=mlir_module.context)
    run_transform(transform_ir, mlir_module)
    return mlir_module


def compile_all_kernels(cache, config, seq_len, cpu_attn=True):
    """Pre-compile all unique kernel configs to cache.

    Args:
        cache: KernelCache instance
        config: LlamaConfig
        seq_len: Sequence length (e.g. 2048)
        cpu_attn: If True, skip flash attention compilation (use CPU fallback)
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim
    n_total = seq_len * emb_dim
    n_hidden_total = seq_len * hidden_dim

    print(f"\n{'='*60}")
    print(f"Compiling {10} unique kernels (seq_len={seq_len})...")
    print(f"{'='*60}\n")

    # 1. RMSNorm: (seq_len, emb_dim)
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    cache.compile_and_cache(
        "rmsnorm",
        build_rms(seq_len, emb_dim, bfloat16, 16),
        {"verbose": cache.verbose, "omit_while_true_loop": False},
    )

    # 2. GEMM Q/O: (seq_len, emb_dim, emb_dim) — large K, tile_k_l2=256
    cache.compile_and_cache(
        "gemm_qo",
        _build_gemm_module(
            seq_len, emb_dim, emb_dim, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64
        ),
        {
            "verbose": cache.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        },
    )

    # 3. GEMM K/V: (seq_len, emb_dim, kv_dim) — small N, tile_n=128
    cache.compile_and_cache(
        "gemm_kv",
        _build_gemm_module(
            seq_len, emb_dim, kv_dim, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128
        ),
        {
            "verbose": cache.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        },
    )

    # 4. GEMM Gate/Up: (seq_len, emb_dim, hidden_dim) — large N, tile_n=128
    cache.compile_and_cache(
        "gemm_gate_up",
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            tile_m=64,
            tile_k_l2=64,
            tile_k_l1=32,
            tile_n=128,
        ),
        {
            "verbose": cache.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        },
    )

    # 5. GEMM Down: (seq_len, hidden_dim, emb_dim) — large K, tile_k_l2=256
    cache.compile_and_cache(
        "gemm_down",
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            tile_m=64,
            tile_k_l2=256,
            tile_k_l1=32,
            tile_n=64,
        ),
        {
            "verbose": cache.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        },
    )

    # 6. RoPE Q: (n_heads * seq_len, head_dim)
    from rope_lut.rope_lut import build_module as build_rope

    cache.compile_and_cache(
        "rope_q",
        build_rope(n_heads * seq_len, head_dim, bfloat16),
        {"verbose": cache.verbose, "omit_while_true_loop": False},
    )

    # 7. RoPE K: (n_kv_heads * seq_len, head_dim)
    cache.compile_and_cache(
        "rope_k",
        build_rope(n_kv_heads * seq_len, head_dim, bfloat16),
        {"verbose": cache.verbose, "omit_while_true_loop": False},
    )

    # 8. Flash Attention GQA (skip if using CPU attention fallback)
    if not cpu_attn:
        from flash_attention.kernel_fusion_based.attn import (
            build_module as build_attn,
        )

        lkp = head_dim  # 64
        lqp = 256
        enable_shared_buffers = lkp == head_dim
        cache.compile_and_cache(
            "flash_attn",
            build_attn(
                lk=seq_len,
                lkp=lkp,
                lq=seq_len,
                lqp=lqp,
                dk=head_dim,
                dv=head_dim,
                num_q_tiles=4,
                num_cascade_stages=4,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                causal=True,
            ),
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": not enable_shared_buffers,
                "omit_pingpong": "all",
                "runtime_loop_tiling_sizes": [1, 1],
                "output_format": "elf",
                "instance_name": "attention_bf16",
            },
        )
    else:
        print("  Skipping flash_attn compilation (using CPU attention fallback)")

    # 9. SwiGLU: n = seq_len * hidden_dim
    from llama3.ffn_swiglu.silu_and_mul import build_module as build_swiglu

    cache.compile_and_cache(
        "swiglu",
        build_swiglu(n_hidden_total, 4096, bfloat16, herd_x=8, herd_y=1),
        {"verbose": cache.verbose, "omit_while_true_loop": False},
    )

    # 10. Eltwise Add: n = seq_len * emb_dim
    from eltwise_add.eltwise_add import build_module as build_add

    cache.compile_and_cache(
        "add",
        build_add(n_total, 2048, bfloat16, vector_size=16, herd_x=8, herd_y=1),
        {"verbose": cache.verbose, "omit_while_true_loop": False},
    )

    # Save manifest for --run-only
    cache._save_manifest()

    print(
        f"\nAll {len(cache.artifacts)} kernels compiled and cached to {cache.cache_dir}/"
    )
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Transformer block execution
# ---------------------------------------------------------------------------


def _run_cached(cache, name, backend_kwargs, *inputs):
    """Run a cached kernel. Thin wrapper around cache.load_and_run."""
    return cache.load_and_run(name, backend_kwargs, *inputs)


# Backend kwarg presets for each kernel type
_GEMM_BACKEND = {
    "omit_while_true_loop": False,
    "runtime_loop_tiling_sizes": [2, 2],
}
_SIMPLE_BACKEND = {"omit_while_true_loop": False}


def _attn_backend_kwargs(head_dim):
    lkp = head_dim
    enable_shared_buffers = lkp == head_dim
    return {
        "omit_while_true_loop": not enable_shared_buffers,
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def run_transformer_block(
    x_bf16,
    x_f32,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    verify=False,
    cpu_attn=True,
):
    """Execute a single transformer block on NPU using cached kernels.

    Carries both BF16 and F32 copies of the residual state to avoid
    compounding truncation errors across layers. The F32 copy is used
    as input to residual adds (steps 9 and 15), while BF16 copies feed
    kernels that require BF16 input (RMSNorm).

    Args:
        x_bf16: (seq_len, emb_dim) bfloat16 input
        x_f32: (seq_len, emb_dim) float32 input (F32 residual state)
        layer_weights: LayerWeights for this layer
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        config: LlamaConfig
        cache: KernelCache instance (kernels must be pre-compiled)
        layer_idx: Layer index for logging
        verify: If True, compare each intermediate against CPU reference
        cpu_attn: If True, use CPU attention fallback instead of NPU kernel

    Returns:
        (output_bf16, output_f32, npu_intermediates_dict)
    """
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    n_total = seq_len * emb_dim

    intermediates = {}

    def _compare(name, npu_result, cpu_ref=None):
        """Compare NPU result against a per-step CPU reference."""
        intermediates[name] = npu_result
        if cpu_ref is not None:
            npu_f32 = npu_result.astype(np.float32).flatten()
            ref_f32 = np.asarray(cpu_ref, dtype=np.float32).flatten()
            if npu_f32.shape == ref_f32.shape:
                abs_err = np.max(np.abs(npu_f32 - ref_f32))
                denom = np.maximum(np.abs(ref_f32), 1e-6)
                rel_err = np.mean(np.abs(npu_f32 - ref_f32) / denom)
                corr = np.corrcoef(npu_f32, ref_f32)[0, 1] if len(npu_f32) > 1 else 1.0
                status = "OK" if corr > 0.99 else "WARN"
                print(
                    f"    [{status}] {name}: max_err={abs_err:.4f}, "
                    f"mean_rel={rel_err:.4f}, corr={corr:.6f}"
                )

    print(f"  Layer {layer_idx}: Running transformer block...")

    # 1. Pre-attention RMSNorm
    print(f"    Step 1: RMSNorm (pre-attn)")
    x_in = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)
    weight_in = np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim)
    y_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "rmsnorm", _SIMPLE_BACKEND, x_in, weight_in, y_out)
    normed = results[-1].reshape(seq_len, emb_dim)
    if verify:
        ref = rms_norm_ref(x_bf16.astype(np.float32), layer_weights.attn_norm)
        _compare("attn_norm", normed, ref)
    else:
        _compare("attn_norm", normed)

    # 2. Q Projection
    print(f"    Step 2: Q projection ({seq_len}x{emb_dim}x{emb_dim})")
    a = np.asarray(normed, dtype=bfloat16).reshape(seq_len, emb_dim)
    b = np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim)
    c = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_qo", _GEMM_BACKEND, a, b, c)
    q = results[-1].reshape(seq_len, emb_dim).astype(bfloat16)
    if verify:
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wq, dtype=np.float32)
        _compare("q", q, ref)
    else:
        _compare("q", q)

    # 3. K Projection
    kv_dim = n_kv_heads * head_dim  # 512
    print(f"    Step 3: K projection ({seq_len}x{emb_dim}x{kv_dim})")
    b_kv = np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim)
    c_kv = np.zeros((seq_len, kv_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_kv", _GEMM_BACKEND, a, b_kv, c_kv)
    k = results[-1].reshape(seq_len, kv_dim).astype(bfloat16)
    if verify:
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wk, dtype=np.float32)
        _compare("k", k, ref)
    else:
        _compare("k", k)

    # 4. V Projection (same shape as K)
    print(f"    Step 4: V projection ({seq_len}x{emb_dim}x{kv_dim})")
    b_v = np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim)
    c_v = np.zeros((seq_len, kv_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_kv", _GEMM_BACKEND, a, b_v, c_v)
    v = results[-1].reshape(seq_len, kv_dim).astype(bfloat16)
    if verify:
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wv, dtype=np.float32)
        _compare("v", v, ref)
    else:
        _compare("v", v)

    # 5. RoPE on Q
    print(f"    Step 5: RoPE on Q")
    q_heads = q.reshape(seq_len, n_heads, head_dim)
    q_flat = q_heads.transpose(1, 0, 2).reshape(n_heads * seq_len, head_dim)
    rope_lut_q = np.tile(rope_lut_bf16[:seq_len], (n_heads, 1))
    total_q = n_heads * seq_len * head_dim
    q_in = np.asarray(q_flat, dtype=bfloat16).flatten()
    lut_in = np.asarray(rope_lut_q, dtype=bfloat16).flatten()
    q_out = np.zeros(total_q, dtype=bfloat16)
    results = _run_cached(cache, "rope_q", _SIMPLE_BACKEND, q_in, lut_in, q_out)
    q_roped_flat = results[-1].reshape(n_heads, seq_len, head_dim)
    q_roped = q_roped_flat.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
    if verify:
        q_heads_f32 = q.astype(np.float32).reshape(seq_len, n_heads, head_dim)
        ref_q_roped = np.empty_like(q_heads_f32)
        lut_f32 = rope_lut_bf16[:seq_len].astype(np.float32)
        for h in range(n_heads):
            ref_q_roped[:, h, :] = apply_rope_ref(q_heads_f32[:, h, :], lut_f32)
        ref = ref_q_roped.reshape(seq_len, n_heads * head_dim)
        _compare("q_roped", q_roped, ref)
    else:
        _compare("q_roped", q_roped)

    # 6. RoPE on K
    print(f"    Step 6: RoPE on K")
    k_heads = k.reshape(seq_len, n_kv_heads, head_dim)
    k_flat = k_heads.transpose(1, 0, 2).reshape(n_kv_heads * seq_len, head_dim)
    rope_lut_k = np.tile(rope_lut_bf16[:seq_len], (n_kv_heads, 1))
    total_k = n_kv_heads * seq_len * head_dim
    k_in = np.asarray(k_flat, dtype=bfloat16).flatten()
    lut_k_in = np.asarray(rope_lut_k, dtype=bfloat16).flatten()
    k_out = np.zeros(total_k, dtype=bfloat16)
    results = _run_cached(cache, "rope_k", _SIMPLE_BACKEND, k_in, lut_k_in, k_out)
    k_roped_flat = results[-1].reshape(n_kv_heads, seq_len, head_dim)
    k_roped = k_roped_flat.transpose(1, 0, 2).reshape(seq_len, n_kv_heads * head_dim)
    if verify:
        k_heads_f32 = k.astype(np.float32).reshape(seq_len, n_kv_heads, head_dim)
        ref_k_roped = np.empty_like(k_heads_f32)
        for h in range(n_kv_heads):
            ref_k_roped[:, h, :] = apply_rope_ref(k_heads_f32[:, h, :], lut_f32)
        ref = ref_k_roped.reshape(seq_len, n_kv_heads * head_dim)
        _compare("k_roped", k_roped, ref)
    else:
        _compare("k_roped", k_roped)

    # 7. Flash Attention GQA
    if cpu_attn:
        # CPU fallback for debugging/comparison
        print(
            f"    Step 7: Attention GQA [CPU fallback] ({n_heads}Q/{n_kv_heads}KV heads)"
        )
        attn_out = attention_reference(
            q_roped.astype(np.float32),
            k_roped.astype(np.float32),
            v.astype(np.float32),
            n_heads,
            n_kv_heads,
        ).astype(bfloat16)
    else:
        print(
            f"    Step 7: Flash Attention GQA [NPU] ({n_heads}Q/{n_kv_heads}KV heads)"
        )
        # Reshape to (heads, seq, dim) — kernel expects unscaled Q
        # (scaling by 1/sqrt(dk) is handled internally by the kernel)
        q_attn = np.ascontiguousarray(
            q_roped.reshape(seq_len, n_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        k_attn = np.ascontiguousarray(
            k_roped.reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        v_attn = np.ascontiguousarray(
            v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2).astype(bfloat16)
        )
        # Causal masking is applied internally by the kernel (causal=True).
        # No external mask argument — kernel handles causal masking.
        attn_output = np.zeros((n_heads, seq_len, head_dim), dtype=bfloat16)
        attn_bk = _attn_backend_kwargs(head_dim)
        results = _run_cached(
            cache,
            "flash_attn",
            attn_bk,
            q_attn,
            k_attn,
            v_attn,
            attn_output,
        )
        attn_out_heads = results[-1].reshape(n_heads, seq_len, head_dim)
        attn_out = attn_out_heads.transpose(1, 0, 2).reshape(
            seq_len, n_heads * head_dim
        )
    _compare("attn_out", attn_out)

    # 8. O Projection
    print(f"    Step 8: O projection")
    a_o = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
    b_o = np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim)
    c_o = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_qo", _GEMM_BACKEND, a_o, b_o, c_o)
    proj = results[-1].reshape(seq_len, emb_dim).astype(bfloat16)
    if verify:
        ref = attn_out.astype(np.float32) @ np.asarray(
            layer_weights.wo, dtype=np.float32
        )
        _compare("proj", proj, ref)
    else:
        _compare("proj", proj)

    # 9. Residual Add (BF16 vectorized on NPU)
    print(f"    Step 9: Residual add")
    a_add = x_bf16.flatten().astype(bfloat16)
    b_add = proj.flatten().astype(bfloat16)
    c_add = np.zeros(n_total, dtype=bfloat16)
    results = _run_cached(cache, "add", _SIMPLE_BACKEND, a_add, b_add, c_add)
    res1_bf16 = results[-1].reshape(seq_len, emb_dim).astype(bfloat16)
    res1_f32 = res1_bf16.astype(np.float32)
    if verify:
        ref = x_bf16.astype(np.float32) + proj.astype(np.float32)
        _compare("res1", res1_bf16, ref)
    else:
        _compare("res1", res1_bf16)

    # 10. Pre-FFN RMSNorm (uses BF16 copy since RMSNorm kernel requires BF16)
    print(f"    Step 10: RMSNorm (pre-FFN)")
    x_in2 = np.asarray(res1_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)
    w_in2 = np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim)
    y_out2 = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "rmsnorm", _SIMPLE_BACKEND, x_in2, w_in2, y_out2)
    normed2 = results[-1].reshape(seq_len, emb_dim)
    if verify:
        ref = rms_norm_ref(res1_bf16.astype(np.float32), layer_weights.ffn_norm)
        _compare("ffn_norm", normed2, ref)
    else:
        _compare("ffn_norm", normed2)

    # 11. Gate GEMM
    print(f"    Step 11: Gate GEMM ({seq_len}x{emb_dim}x{hidden_dim})")
    a_g = np.asarray(normed2, dtype=bfloat16).reshape(seq_len, emb_dim)
    b_gate = np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
        emb_dim, hidden_dim
    )
    c_gate = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_gate_up", _GEMM_BACKEND, a_g, b_gate, c_gate)
    gate = results[-1].reshape(seq_len, hidden_dim).astype(bfloat16)
    if verify:
        ref = normed2.astype(np.float32) @ np.asarray(
            layer_weights.w_gate, dtype=np.float32
        )
        _compare("gate", gate, ref)
    else:
        _compare("gate", gate)

    # 12. Up GEMM (same shape as gate)
    print(f"    Step 12: Up GEMM ({seq_len}x{emb_dim}x{hidden_dim})")
    b_up = np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim)
    c_up = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_gate_up", _GEMM_BACKEND, a_g, b_up, c_up)
    up = results[-1].reshape(seq_len, hidden_dim).astype(bfloat16)
    if verify:
        ref = normed2.astype(np.float32) @ np.asarray(
            layer_weights.w_up, dtype=np.float32
        )
        _compare("up", up, ref)
    else:
        _compare("up", up)

    # 13. SwiGLU activation
    n_hidden_total = seq_len * hidden_dim
    print(f"    Step 13: SwiGLU activation (n={n_hidden_total})")
    gate_flat = np.asarray(gate, dtype=bfloat16).flatten()
    up_flat = np.asarray(up, dtype=bfloat16).flatten()
    swiglu_out_buf = np.zeros(n_hidden_total, dtype=bfloat16)
    results = _run_cached(
        cache, "swiglu", _SIMPLE_BACKEND, gate_flat, up_flat, swiglu_out_buf
    )
    swiglu_out = results[-1].reshape(seq_len, hidden_dim)
    if verify:
        ref = swiglu_ref(gate.astype(np.float32), up.astype(np.float32))
        _compare("swiglu", swiglu_out, ref)
    else:
        _compare("swiglu", swiglu_out)

    # 14. Down GEMM
    print(f"    Step 14: Down GEMM ({seq_len}x{hidden_dim}x{emb_dim})")
    a_d = np.asarray(swiglu_out, dtype=bfloat16).reshape(seq_len, hidden_dim)
    b_d = np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim)
    c_d = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "gemm_down", _GEMM_BACKEND, a_d, b_d, c_d)
    down = results[-1].reshape(seq_len, emb_dim).astype(bfloat16)
    if verify:
        ref = swiglu_out.astype(np.float32) @ np.asarray(
            layer_weights.w_down, dtype=np.float32
        )
        _compare("down", down, ref)
    else:
        _compare("down", down)

    # 15. Residual Add (BF16 vectorized on NPU)
    print(f"    Step 15: Residual add")
    a_add2 = res1_bf16.flatten().astype(bfloat16)
    b_add2 = down.flatten().astype(bfloat16)
    c_add2 = np.zeros(n_total, dtype=bfloat16)
    results = _run_cached(cache, "add", _SIMPLE_BACKEND, a_add2, b_add2, c_add2)
    output_bf16 = results[-1].reshape(seq_len, emb_dim).astype(bfloat16)
    output_f32 = output_bf16.astype(np.float32)
    if verify:
        ref = res1_bf16.astype(np.float32) + down.astype(np.float32)
        _compare("output", output_bf16, ref)
    else:
        _compare("output", output_bf16)

    return output_bf16, output_f32, intermediates


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------


def run_full_model(
    token_ids, weights, config, cache, rope_lut_bf16, verify=False, cpu_attn=True
):
    """Run the full LLAMA-3.2-1B forward pass using cached kernels.

    Args:
        token_ids: (seq_len,) int array
        weights: LlamaWeights
        config: LlamaConfig
        cache: KernelCache instance
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        verify: If True, compare each intermediate against CPU reference
        cpu_attn: If True, use CPU attention fallback instead of NPU kernel

    Returns:
        logits: (seq_len, vocab_size) float32
    """
    seq_len = len(token_ids)
    emb_dim = config.emb_dim

    # 1. Token embedding (CPU) — initialize both BF16 and F32 copies
    embed_f32 = weights.embed_table[token_ids].astype(np.float32)  # (seq_len, emb_dim)
    x_bf16 = embed_f32.astype(bfloat16)
    x_f32 = embed_f32.copy()
    print(f"Embedding: {x_bf16.shape}")

    # 2. Transformer blocks (carry dual-precision state)
    for i in range(config.n_layers):
        layer_t0 = cache.profiler.start_layer()

        x_bf16, x_f32, _ = run_transformer_block(
            x_bf16,
            x_f32,
            weights.layers[i],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=i,
            verify=verify,
            cpu_attn=cpu_attn,
        )

        cache.profiler.end_layer(i, layer_t0)

    # 3. Final RMSNorm (NPU) — uses BF16 copy
    print("Final RMSNorm...")
    x_in = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)
    w_in = np.asarray(weights.final_norm, dtype=bfloat16).reshape(emb_dim)
    y_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "rmsnorm", _SIMPLE_BACKEND, x_in, w_in, y_out)
    x_normed = results[-1].reshape(seq_len, emb_dim)

    # 4. LM Head (CPU - too large for NPU initially)
    print("LM Head (CPU)...")
    lm_head = weights.lm_head.astype(np.float32)
    logits = x_normed.astype(np.float32) @ lm_head.T  # (seq_len, vocab_size)

    return logits


def run_full_model_diagnostic(
    token_ids, weights, config, cache, rope_lut_bf16, rope_lut_f32, cpu_attn=True
):
    """Run NPU and CPU in parallel per layer, comparing outputs at each boundary.

    This shows the degradation curve across layers, helping identify where
    precision loss accumulates.

    Args:
        token_ids: (seq_len,) int array
        weights: LlamaWeights
        config: LlamaConfig
        cache: KernelCache instance
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        rope_lut_f32: (seq_len, head_dim) float32 RoPE LUT
    """
    seq_len = len(token_ids)
    emb_dim = config.emb_dim

    # Initialize both pipelines from embedding
    embed_f32 = weights.embed_table[token_ids].astype(np.float32)
    npu_bf16 = embed_f32.astype(bfloat16)
    npu_f32 = embed_f32.copy()
    cpu_x = embed_f32.copy()

    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Per-layer NPU vs CPU comparison")
    print(f"{'='*70}")
    print(
        f"{'Layer':>6s}  {'Corr':>12s}  {'Cos Sim':>12s}  {'Max Err':>12s}  {'Mean Rel':>12s}"
    )
    print(f"{'-'*6:>6s}  {'-'*12:>12s}  {'-'*12:>12s}  {'-'*12:>12s}  {'-'*12:>12s}")

    for i in range(config.n_layers):
        # NPU path
        npu_bf16, npu_f32, _ = run_transformer_block(
            npu_bf16,
            npu_f32,
            weights.layers[i],
            rope_lut_bf16,
            config,
            cache,
            layer_idx=i,
            verify=False,
            cpu_attn=cpu_attn,
        )

        # CPU reference path (pure F32)
        cpu_x, _ = transformer_block_ref(cpu_x, weights.layers[i], rope_lut_f32, config)

        # Compare NPU output (F32 residual) vs CPU output
        npu_flat = npu_f32.flatten()
        cpu_flat = cpu_x.flatten()

        corr = np.corrcoef(npu_flat, cpu_flat)[0, 1]
        cos_sim = np.dot(npu_flat, cpu_flat) / (
            np.linalg.norm(npu_flat) * np.linalg.norm(cpu_flat) + 1e-12
        )
        max_err = np.max(np.abs(npu_flat - cpu_flat))
        denom = np.maximum(np.abs(cpu_flat), 1e-6)
        mean_rel = np.mean(np.abs(npu_flat - cpu_flat) / denom)

        status = "OK" if corr > 0.99 else ("WARN" if corr > 0.95 else "FAIL")
        print(
            f"  {i:4d}  {corr:12.8f}  {cos_sim:12.8f}  "
            f"{max_err:12.4f}  {mean_rel:12.6f}  [{status}]"
        )

    # Final logits comparison
    print(f"\n--- Final logits comparison ---")

    # NPU final norm + lm_head
    x_in = np.asarray(npu_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)
    w_in = np.asarray(weights.final_norm, dtype=bfloat16).reshape(emb_dim)
    y_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    results = _run_cached(cache, "rmsnorm", _SIMPLE_BACKEND, x_in, w_in, y_out)
    x_normed = results[-1].reshape(seq_len, emb_dim)
    lm_head = weights.lm_head.astype(np.float32)
    npu_logits = x_normed.astype(np.float32) @ lm_head.T

    # CPU final norm + lm_head
    cpu_logits = forward_ref(token_ids, weights, config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_len = len(tokenizer.encode("The capital of France is"))
    pred_pos = min(prompt_len - 1, seq_len - 1)

    npu_next = npu_logits[pred_pos]
    cpu_next = cpu_logits[pred_pos]
    corr = np.corrcoef(npu_next, cpu_next)[0, 1]
    npu_top1 = np.argmax(npu_next)
    cpu_top1 = np.argmax(cpu_next)

    print(f"  Logits correlation at pos {pred_pos}: {corr:.8f}")
    print(f"  NPU top-1: '{tokenizer.decode([npu_top1])}' (id={npu_top1})")
    print(f"  CPU top-1: '{tokenizer.decode([cpu_top1])}' (id={cpu_top1})")
    print(f"  Top-1 match: {'YES' if npu_top1 == cpu_top1 else 'NO'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLAMA-3.2-1B prefill on MLIR-AIR (NPU2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of layers to run (default: all 16). Use 1 for single-block test.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare intermediates against CPU reference",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output from kernel compilation",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile kernels to cache, do not run inference",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Skip compilation, use cached kernels from a previous --compile-only run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable timing instrumentation (per-kernel, per-layer, total)",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run NPU and CPU in parallel per layer, printing correlation at each boundary",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached kernel binaries (default: kernel_cache/)",
    )
    parser.add_argument(
        "--cpu-attn",
        action="store_true",
        default=False,
        help="Use CPU attention fallback instead of NPU flash attention "
        "(for debugging/comparison)",
    )
    parser.add_argument(
        "--npu-attn",
        action="store_true",
        help="Use NPU flash attention kernel (default; kept for backward compat)",
    )
    args = parser.parse_args()

    # --cpu-attn explicitly requests CPU fallback; --npu-attn overrides it back
    if args.npu_attn:
        args.cpu_attn = False

    if args.compile_only and args.run_only:
        print("ERROR: --compile-only and --run-only are mutually exclusive")
        sys.exit(1)

    config = LlamaConfig()
    if args.n_layers is not None:
        config.n_layers = args.n_layers

    # Create profiler and cache
    profiler = Profiler(enabled=args.profile or args.compile_only)
    cache = KernelCache(
        cache_dir=args.cache_dir, verbose=args.verbose, profiler=profiler
    )

    # --- Compilation phase ---
    if not args.run_only:
        compile_all_kernels(cache, config, args.seq_len, cpu_attn=args.cpu_attn)
        if args.compile_only:
            profiler.report()
            print(f"\nKernels cached to {cache.cache_dir}/")
            print("Re-run with --run-only to skip recompilation.")
            sys.exit(0)
    else:
        # Load from manifest
        if not cache.load_manifest():
            print(
                "ERROR: No cached kernels found. Run with --compile-only first, "
                f"or check cache dir: {cache.cache_dir}"
            )
            sys.exit(1)
        print(f"Loaded {len(cache.artifacts)} cached kernels from {cache.cache_dir}/")

    # --- Run phase ---
    # Load weights
    print(f"\nLoading weights from {args.model}...")
    weights = load_weights(args.model, config=config)

    # Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: '{args.prompt}'")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")

    # Pad/truncate
    if len(token_ids) > args.seq_len:
        token_ids = token_ids[: args.seq_len]
    elif len(token_ids) < args.seq_len:
        pad_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_ids = token_ids + [pad_token] * (args.seq_len - len(token_ids))
    token_ids = np.array(token_ids, dtype=np.int64)

    # Generate RoPE LUT
    rope_lut_bf16 = generate_rope_lut(config, args.seq_len, dtype=bfloat16)

    # Diagnostic mode: run NPU vs CPU per-layer comparison
    if args.diagnostic:
        rope_lut_f32 = generate_rope_lut(config, args.seq_len, dtype=np.float32)
        run_full_model_diagnostic(
            token_ids,
            weights,
            config,
            cache,
            rope_lut_bf16,
            rope_lut_f32,
            cpu_attn=args.cpu_attn,
        )
        profiler.report()
        sys.exit(0)

    # Run forward pass
    print(f"\n{'='*60}")
    attn_mode = "CPU fallback" if args.cpu_attn else "NPU flash attention"
    print(
        f"Running LLAMA-3.2-1B prefill ({config.n_layers} layers, seq_len={args.seq_len})"
    )
    print(f"Attention mode: {attn_mode}")
    print(f"{'='*60}\n")

    t_prefill_start = time.time()
    logits = run_full_model(
        token_ids,
        weights,
        config,
        cache,
        rope_lut_bf16,
        verify=args.verify,
        cpu_attn=args.cpu_attn,
    )
    t_prefill_total = time.time() - t_prefill_start

    # Get prediction at last real token position
    prompt_len = len(tokenizer.encode(args.prompt))
    pred_pos = min(prompt_len - 1, args.seq_len - 1)

    # Top-5 predictions
    next_token_logits = logits[pred_pos]
    top5_indices = np.argsort(next_token_logits)[-5:][::-1]
    top5_probs = softmax(next_token_logits)

    print(f"\n{'='*60}")
    print(f"Top-5 predicted next tokens (position {pred_pos}):")
    print(f"{'='*60}")
    for rank, idx in enumerate(top5_indices):
        token_str = tokenizer.decode([idx])
        prob = top5_probs[idx]
        print(
            f"  {rank+1}. '{token_str}' (id={idx}, logit={next_token_logits[idx]:.4f}, "
            f"prob={prob:.4f})"
        )

    print(f"\nTotal prefill wall time: {t_prefill_total:.2f}s")

    # Compare against CPU reference if verifying
    if args.verify:
        print(f"\n{'='*60}")
        print("Comparing against CPU reference...")
        ref_logits = forward_ref(token_ids, weights, config)

        our_next = logits[pred_pos]
        ref_next = ref_logits[pred_pos]
        abs_err = np.max(np.abs(our_next - ref_next))
        corr = np.corrcoef(our_next, ref_next)[0, 1]
        our_top1 = np.argmax(our_next)
        ref_top1 = np.argmax(ref_next)

        print(f"  Max abs error: {abs_err:.4f}")
        print(f"  Logits correlation: {corr:.8f}")
        print(f"  Top-1 match: {'YES' if our_top1 == ref_top1 else 'NO'}")
        print(f"    NPU: '{tokenizer.decode([our_top1])}' (id={our_top1})")
        print(f"    CPU: '{tokenizer.decode([ref_top1])}' (id={ref_top1})")

    # Print profiling report
    profiler.report()

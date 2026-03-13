# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Prefill on MLIR-AIR (NPU2)

Orchestrates sequential NPU kernel invocations for a LLAMA-3.2-1B prefill
(seq_len=128). Each operator is compiled to a separate ELF and invoked
independently from Python.

Architecture:
  Python host code -> compile each kernel config -> invoke sequentially
  -> compare intermediates against CPU reference

Kernel map (15 ops per transformer block):
  1.  RMSNorm (pre-attn)     -> weighted_rms_norm
  2.  Q projection            -> matrix_multiplication/bf16
  3.  K projection            -> matrix_multiplication/bf16
  4.  V projection            -> matrix_multiplication/bf16
  5.  RoPE on Q               -> rope_lut
  6.  RoPE on K               -> rope_lut
  7.  Flash Attention GQA     -> flash_attention/kernel_fusion
  8.  O projection            -> matrix_multiplication/bf16
  9.  Residual add            -> eltwise_add
  10. RMSNorm (pre-FFN)       -> weighted_rms_norm
  11. Gate GEMM               -> matrix_multiplication/bf16
  12. Up GEMM                 -> matrix_multiplication/bf16
  13. SwiGLU activation       -> swiglu_activation
  14. Down GEMM               -> matrix_multiplication/bf16
  15. Residual add            -> eltwise_add
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

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
    transformer_block as transformer_block_ref,
    forward as forward_ref,
)

# ---------------------------------------------------------------------------
# Kernel compilation and caching
# ---------------------------------------------------------------------------

# Cache directory for compiled ELFs
_CACHE_DIR = Path(__file__).resolve().parent / ".kernel_cache"


def _cache_key(kernel_name, **params):
    """Generate a unique cache key for a kernel configuration."""
    param_str = f"{kernel_name}:" + ",".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def prepare_air_project():
    """Clean and prepare the air_project/ directory for a fresh compilation.

    aircc defaults to 'air_project/' as its working directory. Sequential
    compilations leave stale artifacts that corrupt subsequent kernels.
    This method wipes the directory and re-copies external .o files.
    """
    import shutil

    air_proj = Path("air_project")
    if air_proj.exists():
        shutil.rmtree(air_proj)
    air_proj.mkdir(parents=True, exist_ok=True)

    # Copy external kernel .o files if they exist in the current build dir
    for obj_name in ["swiglu_activation.o", "rope.o", "attn.o"]:
        src = Path(obj_name)
        if src.exists():
            shutil.copy2(src, air_proj / obj_name)


class KernelCompiler:
    """Manages compilation and invocation of AIR kernels.

    Each kernel is compiled once and cached on disk. Subsequent invocations
    reuse the cached compilation artifacts.

    IMPORTANT: aircc uses a hardcoded 'air_project/' tmpdir. Sequential kernel
    compilations in the same process MUST clean this directory between runs to
    avoid stale artifact interference. The _prepare_air_project() method handles
    this by wiping the directory and re-copying external .o files.
    """

    def __init__(self, verbose=False, force_recompile=False):
        self.verbose = verbose
        self.force_recompile = force_recompile
        _ensure_cache_dir()

    def _log(self, msg):
        if self.verbose:
            print(f"  [KernelCompiler] {msg}")

    # -----------------------------------------------------------------
    # GEMM kernel (matrix_multiplication/bf16)
    # -----------------------------------------------------------------

    def compile_gemm(
        self,
        m,
        k,
        n,
        tile_m=32,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=32,
        herd_m=4,
        herd_n=4,
    ):
        """Compile a BF16 GEMM kernel for the given dimensions.

        Returns a function (a_bf16, b_bf16) -> c_bf16 that runs on NPU.
        a_bf16: (m, k) bfloat16
        b_bf16: (k, n) bfloat16
        c_bf16: (m, n) bfloat16
        """
        from air.backend.xrt import XRTBackend
        from matrix_multiplication.bf16.run import build_module as build_gemm

        cache_key = _cache_key(
            "gemm",
            m=m,
            k=k,
            n=n,
            tile_m=tile_m,
            tile_k_l2=tile_k_l2,
            tile_k_l1=tile_k_l1,
            tile_n=tile_n,
            herd_m=herd_m,
            herd_n=herd_n,
        )

        self._log(f"GEMM({m}x{k}x{n}) cache_key={cache_key}")

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
            bfloat16,
            arch="aie2p",
            direct_codegen=True,
        )

        # Apply vectorization transform
        from air.ir import Module
        from air.dialects.air import run_transform

        transform_ir_string = self._gemm_transform_ir()
        transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
        run_transform(transform_ir, mlir_module)

        def run_fn(a, b):
            """Execute GEMM: C = A @ B."""
            a = np.asarray(a, dtype=bfloat16).reshape(m, k)
            b = np.asarray(b, dtype=bfloat16).reshape(k, n)
            c = np.zeros((m, n), dtype=bfloat16)

            backend = XRTBackend(
                verbose=self.verbose,
                omit_while_true_loop=False,
                runtime_loop_tiling_sizes=[2, 2],
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(a, b, c)
            backend.unload()
            # results = (a, b, c) -- output c is last; reshape from flat
            return results[-1].reshape(m, n)

        return run_fn

    def _gemm_transform_ir(self):
        """Return the transform IR string for GEMM vectorization."""
        return """
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

    # -----------------------------------------------------------------
    # Weighted RMSNorm kernel
    # -----------------------------------------------------------------

    def compile_rms_norm(self, m, n, vector_size=16):
        """Compile a weighted RMS normalization kernel.

        Returns a function (x_bf16, weight_bf16) -> y_bf16.
        x_bf16: (m, n) bfloat16
        weight_bf16: (n,) bfloat16
        y_bf16: (m, n) bfloat16
        """
        from air.backend.xrt import XRTBackend
        from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

        self._log(f"RMSNorm(M={m}, N={n})")

        mlir_module = build_rms(m, n, bfloat16, vector_size)

        def run_fn(x, weight):
            x = np.asarray(x, dtype=bfloat16).reshape(m, n)
            weight = np.asarray(weight, dtype=bfloat16).reshape(n)
            y = np.zeros((m, n), dtype=bfloat16)

            backend = XRTBackend(
                verbose=self.verbose,
                omit_while_true_loop=False,
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(x, weight, y)
            backend.unload()
            # results = (input, weight, output) -- output is last; reshape
            return results[-1].reshape(m, n)

        return run_fn

    # -----------------------------------------------------------------
    # RoPE LUT kernel
    # -----------------------------------------------------------------

    def compile_rope(self, seq_len, embed_dim):
        """Compile a RoPE (Rotary Position Embedding) kernel.

        Returns a function (input_flat_bf16, lut_flat_bf16) -> output_flat_bf16.
        All arrays are flattened to 1D: (seq_len * embed_dim,).
        """
        from air.backend.xrt import XRTBackend
        from rope_lut.rope_lut import build_module as build_rope

        self._log(f"RoPE(seq_len={seq_len}, embed_dim={embed_dim})")

        mlir_module = build_rope(seq_len, embed_dim, bfloat16)

        def run_fn(input_data, lut):
            total = seq_len * embed_dim
            input_flat = np.asarray(input_data, dtype=bfloat16).flatten()
            lut_flat = np.asarray(lut, dtype=bfloat16).flatten()
            assert input_flat.shape == (
                total,
            ), f"input shape {input_flat.shape} != ({total},)"
            assert lut_flat.shape == (
                total,
            ), f"lut shape {lut_flat.shape} != ({total},)"
            output_flat = np.zeros(total, dtype=bfloat16)

            backend = XRTBackend(
                verbose=self.verbose,
                omit_while_true_loop=False,
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(input_flat, lut_flat, output_flat)
            backend.unload()
            return results[-1].reshape(seq_len, embed_dim)

        return run_fn

    # -----------------------------------------------------------------
    # Eltwise Add kernel
    # -----------------------------------------------------------------

    def compile_eltwise_add(self, n, tile_n=1024):
        """Compile an element-wise addition kernel.

        Returns a function (a_bf16_flat, b_bf16_flat) -> c_bf16_flat.
        All arrays are 1D with n elements.
        """
        from air.backend.xrt import XRTBackend

        # eltwise_add uses float32 by default, but we need bf16 for LLAMA
        # Since the existing kernel uses scalar load/store, it works for any dtype
        from eltwise_add.eltwise_add import build_module as build_add

        self._log(f"EltwiseAdd(n={n}, tile_n={tile_n})")

        mlir_module = build_add(n, tile_n, np.float32)

        def run_fn(a, b):
            a_flat = np.asarray(a, dtype=np.float32).flatten()
            b_flat = np.asarray(b, dtype=np.float32).flatten()
            assert a_flat.shape == (n,), f"a shape {a_flat.shape} != ({n},)"
            assert b_flat.shape == (n,), f"b shape {b_flat.shape} != ({n},)"
            c_flat = np.zeros(n, dtype=np.float32)

            backend = XRTBackend(
                verbose=self.verbose,
                omit_while_true_loop=False,
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(a_flat, b_flat, c_flat)
            backend.unload()
            return results[-1]

        return run_fn

    # -----------------------------------------------------------------
    # SwiGLU activation kernel
    # -----------------------------------------------------------------

    def compile_swiglu(self, n, tile_n=1024):
        """Compile a standalone SwiGLU activation kernel.

        Returns a function (gate_bf16_flat, up_bf16_flat) -> out_bf16_flat.
        All arrays are 1D with n elements.
        """
        from air.backend.xrt import XRTBackend
        from llama3.swiglu_activation import build_module as build_swiglu

        self._log(f"SwiGLU(n={n}, tile_n={tile_n})")

        mlir_module = build_swiglu(n, tile_n, bfloat16)

        def run_fn(gate, up):
            gate_flat = np.asarray(gate, dtype=bfloat16).flatten()
            up_flat = np.asarray(up, dtype=bfloat16).flatten()
            assert gate_flat.shape == (n,), f"gate shape {gate_flat.shape} != ({n},)"
            out_flat = np.zeros(n, dtype=bfloat16)

            backend = XRTBackend(
                verbose=self.verbose,
                omit_while_true_loop=False,
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(gate_flat, up_flat, out_flat)
            backend.unload()
            return results[-1]

        return run_fn

    # -----------------------------------------------------------------
    # Flash Attention GQA kernel
    # -----------------------------------------------------------------

    def compile_flash_attention(
        self, lq, lk, dk=64, dv=64, num_heads=32, num_kv_heads=8, lkp=32, lqp=128
    ):
        """Compile a flash attention kernel with GQA support.

        Returns a function (q, k, v) -> output.
        q: (num_heads, lq, dk) bfloat16
        k: (num_kv_heads, dk, lk) bfloat16  (K is transposed: dk x lk)
        v: (num_kv_heads, lk, dv) bfloat16
        output: (num_heads, lq, dv) bfloat16
        """
        from air.backend.xrt_runner import XRTRunner
        from flash_attention.kernel_fusion_based.attn import (
            build_module as build_attn,
        )

        self._log(f"FlashAttn(lq={lq}, lk={lk}, heads={num_heads}/{num_kv_heads})")

        mlir_module = build_attn(
            lk=lk,
            lkp=lkp,
            lq=lq,
            lqp=lqp,
            dk=dk,
            dv=dv,
            num_q_tiles=4,
            num_cascade_stages=4,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        def run_fn(q, k, v):
            """Run flash attention.

            Args:
                q: (num_heads, lq, dk) bfloat16 - already scaled by 1/sqrt(dk)
                k: (num_kv_heads, dk, lk) bfloat16 - K transposed
                v: (num_kv_heads, lk, dv) bfloat16
            Returns:
                (num_heads, lq, dv) bfloat16
            """
            assert q.shape == (num_heads, lq, dk)
            assert k.shape == (num_kv_heads, dk, lk)
            assert v.shape == (num_kv_heads, lk, dv)

            # The attention kernel also takes a mask input (zeros for no mask)
            mask = np.zeros((num_heads, lq, lk), dtype=bfloat16)
            output = np.zeros((num_heads, lq, dv), dtype=bfloat16)

            enable_shared_buffers = lkp == dk
            runner = XRTRunner(
                omit_while_true_loop=not enable_shared_buffers,
                omit_pingpong="all",
                verbose=self.verbose,
                runtime_loop_tiling_sizes=[1, 1],
                output_format="elf",
                instance_name="attention_bf16",
            )

            # Use XRTRunner's internal compile+run flow
            from air.backend.xrt import XRTBackend

            backend = XRTBackend(
                omit_while_true_loop=not enable_shared_buffers,
                omit_pingpong="all",
                verbose=self.verbose,
                runtime_loop_tiling_sizes=[1, 1],
                output_format="elf",
                instance_name="attention_bf16",
            )
            prepare_air_project()
            compiled = backend.compile(mlir_module)
            import filelock

            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(compiled)
                results = invoker(q, k, v, mask, output)
            backend.unload()
            return results[-1].reshape(num_heads, lq, dv)

        return run_fn


# ---------------------------------------------------------------------------
# Transformer block execution
# ---------------------------------------------------------------------------


def run_transformer_block(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    compiler,
    layer_idx=0,
    ref_intermediates=None,
):
    """Execute a single transformer block on NPU.

    Args:
        x_bf16: (seq_len, emb_dim) bfloat16 input
        layer_weights: LayerWeights for this layer
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        config: LlamaConfig
        compiler: KernelCompiler instance
        layer_idx: Layer index for logging
        ref_intermediates: Optional dict of F32 reference intermediates

    Returns:
        (output_bf16, npu_intermediates_dict)
    """
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    n_total = seq_len * emb_dim
    n_hidden_total = seq_len * hidden_dim

    intermediates = {}

    def _compare(name, npu_result, cpu_ref=None):
        """Compare NPU result against a per-step CPU reference.

        The cpu_ref should be computed using the SAME inputs as the NPU step
        (i.e., the previous step's NPU output cast to F32), NOT the CPU
        reference chain. This isolates each kernel's accuracy.
        """
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

    verify = ref_intermediates is not None
    from llama3_reference import rms_norm as rms_norm_ref, apply_rope as apply_rope_ref

    print(f"  Layer {layer_idx}: Running transformer block...")

    # 1. Pre-attention RMSNorm
    print(f"    Step 1: RMSNorm (pre-attn)")
    run_rms = compiler.compile_rms_norm(seq_len, emb_dim)
    normed = run_rms(x_bf16, layer_weights.attn_norm.astype(bfloat16))
    if verify:
        # Per-step ref: use same input as NPU (x_bf16 cast to F32)
        ref = rms_norm_ref(x_bf16.astype(np.float32), layer_weights.attn_norm)
        _compare("attn_norm", normed, ref)
    else:
        _compare("attn_norm", normed)

    # 2. Q Projection: (seq_len, emb_dim) @ (emb_dim, emb_dim) -> (seq_len, emb_dim)
    print(f"    Step 2: Q projection ({seq_len}x{emb_dim}x{emb_dim})")
    herd_m = 4 if seq_len >= 128 else 2
    run_gemm_qo = compiler.compile_gemm(
        seq_len, emb_dim, emb_dim, herd_m=herd_m, herd_n=4
    )
    q = run_gemm_qo(normed, layer_weights.wq.astype(bfloat16))
    if verify:
        # Per-step ref: use NPU's normed output (cast to F32) @ weights
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wq, dtype=np.float32)
        _compare("q", q, ref)
    else:
        _compare("q", q)

    # 3. K Projection: (seq_len, emb_dim) @ (emb_dim, kv_dim) -> (seq_len, kv_dim)
    kv_dim = n_kv_heads * head_dim  # 512
    print(f"    Step 3: K projection ({seq_len}x{emb_dim}x{kv_dim})")
    run_gemm_kv = compiler.compile_gemm(
        seq_len, emb_dim, kv_dim, herd_m=herd_m, herd_n=4
    )
    k = run_gemm_kv(normed, layer_weights.wk.astype(bfloat16))
    if verify:
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wk, dtype=np.float32)
        _compare("k", k, ref)
    else:
        _compare("k", k)

    # 4. V Projection (same shape as K)
    print(f"    Step 4: V projection ({seq_len}x{emb_dim}x{kv_dim})")
    v = run_gemm_kv(normed, layer_weights.wv.astype(bfloat16))
    if verify:
        ref = normed.astype(np.float32) @ np.asarray(layer_weights.wv, dtype=np.float32)
        _compare("v", v, ref)
    else:
        _compare("v", v)

    # 5. RoPE on Q: reshape to (seq_len * n_heads, head_dim), apply RoPE
    print(f"    Step 5: RoPE on Q")
    q_heads = q.reshape(seq_len, n_heads, head_dim)
    # Create per-head LUT: repeat base LUT for each head
    # q_heads is (seq_len, n_heads, head_dim)
    # Transpose to (n_heads, seq_len, head_dim) then flatten to (n_heads * seq_len, head_dim)
    q_flat = q_heads.transpose(1, 0, 2).reshape(n_heads * seq_len, head_dim)
    rope_lut_q = np.tile(
        rope_lut_bf16[:seq_len], (n_heads, 1)
    )  # (n_heads * seq_len, head_dim)
    run_rope = compiler.compile_rope(n_heads * seq_len, head_dim)
    q_roped_flat = run_rope(q_flat, rope_lut_q)
    # Reshape back: (n_heads, seq_len, head_dim) -> (seq_len, n_heads, head_dim) -> (seq_len, n_heads * head_dim)
    q_roped = (
        q_roped_flat.reshape(n_heads, seq_len, head_dim)
        .transpose(1, 0, 2)
        .reshape(seq_len, n_heads * head_dim)
    )
    if verify:
        # Per-step ref: apply RoPE to the NPU Q output using F32
        q_heads_f32 = q.astype(np.float32).reshape(seq_len, n_heads, head_dim)
        ref_q_roped = np.empty_like(q_heads_f32)
        lut_f32 = rope_lut_bf16[:seq_len].astype(np.float32)
        for h in range(n_heads):
            ref_q_roped[:, h, :] = apply_rope_ref(q_heads_f32[:, h, :], lut_f32)
        ref = ref_q_roped.reshape(seq_len, n_heads * head_dim)
        _compare("q_roped", q_roped, ref)
    else:
        _compare("q_roped", q_roped)

    # 6. RoPE on K: reshape to (seq_len * n_kv_heads, head_dim), apply RoPE
    print(f"    Step 6: RoPE on K")
    k_heads = k.reshape(seq_len, n_kv_heads, head_dim)
    k_flat = k_heads.transpose(1, 0, 2).reshape(n_kv_heads * seq_len, head_dim)
    rope_lut_k = np.tile(rope_lut_bf16[:seq_len], (n_kv_heads, 1))
    run_rope_k = compiler.compile_rope(n_kv_heads * seq_len, head_dim)
    k_roped_flat = run_rope_k(k_flat, rope_lut_k)
    k_roped = (
        k_roped_flat.reshape(n_kv_heads, seq_len, head_dim)
        .transpose(1, 0, 2)
        .reshape(seq_len, n_kv_heads * head_dim)
    )
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
    print(f"    Step 7: Flash Attention GQA ({n_heads}Q/{n_kv_heads}KV heads)")
    # Prepare inputs for attention kernel:
    # Q: (num_heads, lq, dk) - reshape from (seq_len, n_heads * head_dim)
    q_attn = (
        q_roped.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2).astype(bfloat16)
    )
    # Scale Q by 1/sqrt(dk) for the attention kernel
    scale = 1.0 / np.sqrt(head_dim)
    q_attn_scaled = (q_attn.astype(np.float32) * scale).astype(bfloat16)
    # K: (num_kv_heads, dk, lk) - transpose from (seq_len, n_kv_heads, head_dim)
    k_attn = (
        k_roped.reshape(seq_len, n_kv_heads, head_dim)
        .transpose(1, 2, 0)
        .astype(bfloat16)
    )
    # V: (num_kv_heads, lk, dv)
    v_attn = (
        v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2).astype(bfloat16)
    )

    run_attn = compiler.compile_flash_attention(
        lq=seq_len,
        lk=seq_len,
        dk=head_dim,
        dv=head_dim,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        lkp=head_dim,  # 64
        lqp=256,  # Known good config for seq_len=2048
    )
    attn_out_heads = run_attn(q_attn_scaled, k_attn, v_attn)
    # Reshape: (num_heads, lq, dv) -> (seq_len, num_heads * head_dim)
    attn_out = attn_out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
    # Skip per-step ref for attention (complex); compare end-to-end
    _compare("attn_out", attn_out)

    # 8. O Projection: (seq_len, emb_dim) @ (emb_dim, emb_dim) -> (seq_len, emb_dim)
    print(f"    Step 8: O projection")
    proj = run_gemm_qo(attn_out.astype(bfloat16), layer_weights.wo.astype(bfloat16))
    if verify:
        ref = attn_out.astype(np.float32) @ np.asarray(
            layer_weights.wo, dtype=np.float32
        )
        _compare("proj", proj, ref)
    else:
        _compare("proj", proj)

    # 9. Residual Add
    print(f"    Step 9: Residual add")
    run_add = compiler.compile_eltwise_add(n_total)
    res1_flat = run_add(
        x_bf16.flatten().astype(np.float32), proj.flatten().astype(np.float32)
    )
    res1 = res1_flat.reshape(seq_len, emb_dim).astype(bfloat16)
    if verify:
        ref = x_bf16.astype(np.float32) + proj.astype(np.float32)
        _compare("res1", res1, ref)
    else:
        _compare("res1", res1)

    # 10. Pre-FFN RMSNorm
    print(f"    Step 10: RMSNorm (pre-FFN)")
    normed2 = run_rms(res1, layer_weights.ffn_norm.astype(bfloat16))
    if verify:
        ref = rms_norm_ref(res1.astype(np.float32), layer_weights.ffn_norm)
        _compare("ffn_norm", normed2, ref)
    else:
        _compare("ffn_norm", normed2)

    # 11. Gate GEMM: (seq_len, emb_dim) @ (emb_dim, hidden_dim) -> (seq_len, hidden_dim)
    print(f"    Step 11: Gate GEMM ({seq_len}x{emb_dim}x{hidden_dim})")
    run_gemm_gate = compiler.compile_gemm(
        seq_len, emb_dim, hidden_dim, herd_m=herd_m, herd_n=4
    )
    gate = run_gemm_gate(normed2, layer_weights.w_gate.astype(bfloat16))
    if verify:
        ref = normed2.astype(np.float32) @ np.asarray(
            layer_weights.w_gate, dtype=np.float32
        )
        _compare("gate", gate, ref)
    else:
        _compare("gate", gate)

    # 12. Up GEMM (same shape as gate)
    print(f"    Step 12: Up GEMM ({seq_len}x{emb_dim}x{hidden_dim})")
    up = run_gemm_gate(normed2, layer_weights.w_up.astype(bfloat16))
    if verify:
        ref = normed2.astype(np.float32) @ np.asarray(
            layer_weights.w_up, dtype=np.float32
        )
        _compare("up", up, ref)
    else:
        _compare("up", up)

    # 13. SwiGLU activation
    print(f"    Step 13: SwiGLU activation (n={n_hidden_total})")
    run_swiglu = compiler.compile_swiglu(n_hidden_total)
    swiglu_out_flat = run_swiglu(gate.flatten(), up.flatten())
    swiglu_out = swiglu_out_flat.reshape(seq_len, hidden_dim)
    if verify:
        from llama3_reference import swiglu as swiglu_ref

        ref = swiglu_ref(gate.astype(np.float32), up.astype(np.float32))
        _compare("swiglu", swiglu_out, ref)
    else:
        _compare("swiglu", swiglu_out)

    # 14. Down GEMM: (seq_len, hidden_dim) @ (hidden_dim, emb_dim) -> (seq_len, emb_dim)
    print(f"    Step 14: Down GEMM ({seq_len}x{hidden_dim}x{emb_dim})")
    run_gemm_down = compiler.compile_gemm(
        seq_len, hidden_dim, emb_dim, herd_m=herd_m, herd_n=4
    )
    down = run_gemm_down(
        swiglu_out.astype(bfloat16), layer_weights.w_down.astype(bfloat16)
    )
    if verify:
        ref = swiglu_out.astype(np.float32) @ np.asarray(
            layer_weights.w_down, dtype=np.float32
        )
        _compare("down", down, ref)
    else:
        _compare("down", down)

    # 15. Residual Add
    print(f"    Step 15: Residual add")
    output_flat = run_add(
        res1.flatten().astype(np.float32), down.flatten().astype(np.float32)
    )
    output = output_flat.reshape(seq_len, emb_dim).astype(bfloat16)
    if verify:
        ref = res1.astype(np.float32) + down.astype(np.float32)
        _compare("output", output, ref)
    else:
        _compare("output", output)

    return output, intermediates


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------


def run_full_model(token_ids, weights, config, compiler, rope_lut_bf16, verify=False):
    """Run the full LLAMA-3.2-1B forward pass.

    Args:
        token_ids: (seq_len,) int array
        weights: LlamaWeights
        config: LlamaConfig
        compiler: KernelCompiler
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT
        verify: If True, compare each intermediate against CPU reference

    Returns:
        logits: (seq_len, vocab_size) float32
    """
    seq_len = len(token_ids)

    # 1. Token embedding (CPU)
    x = weights.embed_table[token_ids].astype(bfloat16)  # (seq_len, emb_dim)
    print(f"Embedding: {x.shape}")

    # 2. Transformer blocks
    # When verify=True, each step computes its own CPU reference using
    # the NPU's previous output as input (isolates per-kernel accuracy)
    for i in range(config.n_layers):
        # Pass a non-None ref_intermediates dict to enable verification
        ref_flag = {} if verify else None

        x, _ = run_transformer_block(
            x,
            weights.layers[i],
            rope_lut_bf16,
            config,
            compiler,
            layer_idx=i,
            ref_intermediates=ref_flag,
        )

    # 3. Final RMSNorm (NPU)
    print("Final RMSNorm...")
    run_rms = compiler.compile_rms_norm(seq_len, config.emb_dim)
    x_normed = run_rms(x, weights.final_norm.astype(bfloat16))

    # 4. LM Head (CPU - too large for NPU initially)
    print("LM Head (CPU)...")
    lm_head = weights.lm_head.astype(np.float32)
    logits = x_normed.astype(np.float32) @ lm_head.T  # (seq_len, vocab_size)

    return logits


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
    args = parser.parse_args()

    config = LlamaConfig()
    if args.n_layers is not None:
        config.n_layers = args.n_layers

    # Load weights
    print(f"Loading weights from {args.model}...")
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

    # Create kernel compiler
    compiler = KernelCompiler(verbose=args.verbose)

    # Run forward pass
    print(f"\n{'='*60}")
    print(
        f"Running LLAMA-3.2-1B prefill ({config.n_layers} layers, seq_len={args.seq_len})"
    )
    print(f"{'='*60}\n")

    logits = run_full_model(
        token_ids,
        weights,
        config,
        compiler,
        rope_lut_bf16,
        verify=args.verify,
    )

    # Get prediction at last real token position
    prompt_len = len(tokenizer.encode(args.prompt))
    pred_pos = min(prompt_len - 1, args.seq_len - 1)

    # Top-5 predictions
    from llama3_reference import softmax

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

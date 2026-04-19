# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Option C: head-first FlashAttention via host transposes (LESSON 3).

Background (full writeup in
`programming_examples/llama32_3b/docs/development_progress/LESSONS.md` Lesson 3):
the seq-first FA kernel (`attn_npu2_seqfirst.py`) used by llama3 production
hangs at runtime in its `dk_chunks > 1` shim-DMA path. That path was never
lit-tested upstream and is the only L1-feasible config at head_dim ≥ 128.

Workaround: use the head-first FA kernel (`attn_npu2.py`, exercised by the
llama3-8b lit test) and wrap with host transposes between seq-first and
head-first I/O at the FA call boundary. Cost: a few ms/layer host transpose;
gain: NPU FA actually runs (4.2× warm prefill speedup vs CPU-attn on
llama32_3b).

Public API:
- HEAD_DIM_128_FA_CONFIG       : standard L1-feasible tile params
- install_headfirst_fa_wrapper : install the runtime monkey-patches +
                                 register transpose params for one model
- compile_headfirst_fa_kernel  : compile attn_npu2.o + flash_attn.elf with
                                 the right per-tile flags (LESSON 3 fix)

Usage from per-model phase test (call ONCE before any FA invocation):
    from _llm_shared.phase_helpers.headfirst_fa import (
        HEAD_DIM_128_FA_CONFIG,
        install_headfirst_fa_wrapper,
        compile_headfirst_fa_kernel,
    )
    install_headfirst_fa_wrapper()  # idempotent monkey-patches
    # ... in compile_block_kernels:
    compile_headfirst_fa_kernel(cache, seq_len, n_heads, n_kv_heads, head_dim)
"""

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# L1-feasible tile config for head_dim=128 (proven by llama3-8b lit test).
# tile_size_q = lqp / num_q_tiles = 64; per-tile L1 ≈ 50 KB (fits 64 KB budget).
HEAD_DIM_128_FA_CONFIG = {"lqp": 256, "lkp": 64, "dk": 128, "dv": 128}

# Set by `install_headfirst_fa_wrapper` so the _run_cached interceptor
# knows how to transpose seq-first <-> head-first arrays.
# Keys: n_heads, n_kv_heads, dk, dv, lkp.
_HEADFIRST_FA_PARAMS = {}

_INSTALLED = False


def install_headfirst_fa_wrapper():
    """Monkey-patch llama3_prefill so head_dim=128 routes to head-first FA.

    Two patches:
    1. `_attn_backend_kwargs(head_dim)` returns head-first kwargs at hd=128
       (omit_while_true_loop=False, runtime_loop_tiling_sizes=[1,1,1] for
       dv_chunks=2, target_device='npu2').
    2. `_run_cached(cache, 'flash_attn', ...)` intercepts FA calls when
       `_HEADFIRST_FA_PARAMS` is populated: transpose seq-first inputs to
       head-first, call the head-first ELF, transpose output back.

    Idempotent — safe to call from multiple per-model scripts in the same
    process. The second patch is a no-op until the per-model script populates
    `_HEADFIRST_FA_PARAMS` via `compile_headfirst_fa_kernel(...)`.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import llama3_prefill as _lp

    _orig_attn_bk = _lp._attn_backend_kwargs
    _orig_run_cached = _lp._run_cached

    def _patched_attn_bk(head_dim):
        if head_dim == 128:
            return {
                "omit_while_true_loop": False,
                "omit_pingpong": "all",
                "runtime_loop_tiling_sizes": [1, 1, 1],
                "output_format": "elf",
                "instance_name": "attention_bf16",
                "target_device": "npu2",
            }
        return _orig_attn_bk(head_dim)

    def _patched_run_cached(cache, name, backend_kwargs, *inputs, **kwargs):
        # Skip interception if not flash_attn or no params registered yet.
        if name != "flash_attn" or not _HEADFIRST_FA_PARAMS:
            return _orig_run_cached(cache, name, backend_kwargs, *inputs, **kwargs)

        n_heads = _HEADFIRST_FA_PARAMS["n_heads"]
        n_kv_heads = _HEADFIRST_FA_PARAMS["n_kv_heads"]
        dk = _HEADFIRST_FA_PARAMS["dk"]
        dv = _HEADFIRST_FA_PARAMS["dv"]
        lkp = _HEADFIRST_FA_PARAMS["lkp"]
        dv_chunks = dv // lkp

        # Caller passes (q_seq, k_seq, v_seq, attn_output_seq) per llama3_prefill.
        q_seq, k_seq, v_seq, _attn_out_seq = inputs
        lq = q_seq.shape[0]
        lk = k_seq.shape[0]

        # Seq-first -> head-first transposes (host-side; ~few ms total at this size).
        q_hf = np.ascontiguousarray(q_seq.reshape(lq, n_heads, dk).transpose(1, 0, 2))
        k_hf = np.ascontiguousarray(
            k_seq.reshape(lk, n_kv_heads, dk).transpose(1, 0, 2)
        )
        # V is re-tiled along dv: split dv into dv_chunks of lkp; head-first kernel
        # expects [n_kv_heads * dv_chunks, lk, lkp] in head-then-chunk order
        # (matches attn_npu2.py:1280 reference impl).
        v_hf = np.ascontiguousarray(
            v_seq.reshape(lk, n_kv_heads, dv_chunks, lkp)
            .transpose(1, 2, 0, 3)
            .reshape(n_kv_heads * dv_chunks, lk, lkp)
        )
        out_hf = np.zeros((n_heads * dv_chunks, lq, lkp), dtype=bfloat16)

        # Optional debug knobs for FA bisect work (see debug-fa-runtime-failure skill).
        _dbg = os.environ.get("HEADFIRST_FA_DEBUG", "")
        if _dbg:
            _q = q_hf.astype(np.float32)
            print(
                f"  [HF-FA pre]  q.shape={q_hf.shape} min={_q.min():.4f} "
                f"max={_q.max():.4f} any_nan={bool(np.any(np.isnan(_q)))}",
                flush=True,
            )
        if os.environ.get("HEADFIRST_FA_SCALE_QK"):
            scale = float(os.environ["HEADFIRST_FA_SCALE_QK"])
            q_hf = (q_hf.astype(np.float32) * scale).astype(bfloat16)
            k_hf = (k_hf.astype(np.float32) * scale).astype(bfloat16)
        if os.environ.get("HEADFIRST_FA_SUBSTITUTE_INPUTS"):
            rng = np.random.default_rng(42)
            q_hf = rng.uniform(0, 4.0, q_hf.shape).astype(bfloat16)
            k_hf = rng.uniform(0, 4.0, k_hf.shape).astype(bfloat16)
            v_hf = rng.uniform(0, 4.0, v_hf.shape).astype(bfloat16)

        results_hf = cache.load_and_run(
            "flash_attn", backend_kwargs, q_hf, k_hf, v_hf, out_hf
        )

        if _dbg:
            _r = results_hf[-1].astype(np.float32)
            print(
                f"  [HF-FA post] results[-1].shape={results_hf[-1].shape} "
                f"min={_r.min():.4f} max={_r.max():.4f} "
                f"any_nan={bool(np.any(np.isnan(_r)))}",
                flush=True,
            )

        # Head-first output -> seq-first.
        out_packed = results_hf[-1].reshape(n_heads, dv_chunks, lq, lkp)
        out_seq = np.ascontiguousarray(
            out_packed.transpose(2, 0, 1, 3).reshape(lq, n_heads * dv)
        )

        # Caller does `results[-1].reshape(seq_len, n_heads * head_dim)`.
        # Return a list whose last element is the seq-first output.
        return [None] * (len(inputs) - 1) + [out_seq]

    _lp._attn_backend_kwargs = _patched_attn_bk
    _lp._run_cached = _patched_run_cached
    _INSTALLED = True


def compile_headfirst_fa_kernel(
    cache, seq_len, n_heads, n_kv_heads, head_dim, fa_config=None, verbose=False
):
    """Compile head-first attn_npu2.o + flash_attn.elf for this model's shape.

    Registers the model's transpose params in `_HEADFIRST_FA_PARAMS` so the
    `_run_cached` interceptor (installed by `install_headfirst_fa_wrapper`)
    knows how to reshape seq-first arrays at call time.

    Args:
        cache: KernelCache instance
        seq_len: prefill seq_len
        n_heads, n_kv_heads, head_dim: from the model config
        fa_config: tile params (lqp, lkp, dk, dv) — defaults to
                   HEAD_DIM_128_FA_CONFIG when head_dim == 128
        verbose: pass-through to compile-time backend kwargs
    """
    from _llm_shared.kernel_builder.external_kernels import compile_attn_npu2_split
    from flash_attention.kernel_fusion_based.attn_npu2 import (
        build_module as build_attn_hf,
    )

    if fa_config is None:
        if head_dim == 128:
            fa_config = HEAD_DIM_128_FA_CONFIG
        else:
            fa_config = {
                "lqp": 256,
                "lkp": head_dim,
                "dk": head_dim,
                "dv": head_dim,
            }

    # Register transpose params for the runtime interceptor.
    _HEADFIRST_FA_PARAMS.update(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dk=fa_config["dk"],
        dv=fa_config["dv"],
        lkp=fa_config["lkp"],
    )

    # Recompile attn_npu2.o with the per-tile flag conventions (LESSON 3 fix).
    if head_dim == 128:
        Path("attn_npu2.o").unlink(missing_ok=True)
        Path("attn.o").unlink(missing_ok=True)
        compile_attn_npu2_split(**fa_config)

    if "flash_attn" in cache.artifacts:
        return  # already cached; transpose params have been re-registered above

    dv_chunks = fa_config["dv"] // fa_config["lkp"]
    runtime_tiling = [1, 1, 1] if dv_chunks > 1 else [1, 1]
    cache.compile_and_cache(
        "flash_attn",
        build_attn_hf(
            lk=seq_len,
            lkp=fa_config["lkp"],
            lq=seq_len,
            lqp=fa_config["lqp"],
            dk=fa_config["dk"],
            dv=fa_config["dv"],
            num_q_tiles=4,
            num_cascade_stages=4,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            causal=True,
        ),
        {
            "verbose": verbose,
            "omit_while_true_loop": False,
            "omit_pingpong": "all",
            "runtime_loop_tiling_sizes": runtime_tiling,
            "output_format": "elf",
            "instance_name": "attention_bf16",
            "target_device": "npu2",
        },
    )

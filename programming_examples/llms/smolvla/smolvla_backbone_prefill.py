# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Language-Backbone Single-Block Prefill on MLIR-AIR (NPU2)

Thin fork of llama32_1b/llama32_1b_prefill.py's kernel-cache + single-block
assembly, adapted to the SmolVLA backbone's GQA (15Q/5KV heads, head_dim=64,
emb_dim=960, hidden_dim=2560) and its prefix-LM **non-causal** attention
(bidirectional among prefix tokens, state token sees everything, padding
excluded) instead of llama's causal attention.

Two-step de-risking (see plan Task 2.1):
  Step A (this module's default, cpu_attn=True): RMSNorm+QKV+RoPE on NPU via
    the fused `rms_gemms_rope` ELF, attention on CPU via
    `noncausal_attention_reference` (with the prefix mask + frozen padding
    positions), O-proj+residual+FFN on NPU via the fused `o_ffn` ELF. This
    isolates the NPU GQA assembly from the NPU-attention risk.
  Step B (cpu_attn=False, NOT wired here yet): move attention onto NPU via
    per-head GEMM S=Q@K^T -> +mask -> masked_softmax -> P@V. Left for a
    follow-up once Step A's gate (layer-0 cosine > 0.99) is confirmed.

Only ONE block (layer 0) is exercised here; test_single_block.py drives it.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add parent directory to path for kernel imports (programming_examples/).
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
# Also add llms/ for sibling LLM packages (shared.infra, shared.builders).
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from smolvla_backbone_weights import SmolVLABackboneConfig
from smolvla_cpu_helpers import (
    noncausal_attention_reference,
    _rope_half_split,
    rms_norm,
)
from shared.infra.cache import KernelCache, Profiler  # noqa: F401 (re-exported)
from shared.infra.backend_presets import (  # noqa: F401 (re-exported for callers)
    SIMPLE_BACKEND,
    RMS_GEMMS_ROPE_BACKEND,
    O_FFN_BACKEND,
)

# ---------------------------------------------------------------------------
# Kernel compilation definitions
# ---------------------------------------------------------------------------

# Same registry-driven external-GEMM strategy as llama32_1b: every GEMM's
# method (fused-cast vs drain) + tile sizes come from the kernel_registry JSON
# per shape (gemm_registry_config). The rms_gemms_rope / o_ffn builders are
# GQA-generic (parametrized by kv_dim), so SmolVLA's 15/5 heads flow through
# unchanged -- no fork of those builders needed.


def _o_ffn_run_backend():
    from shared.infra.backend_presets import O_FFN_BACKEND as _base

    return {**_base, "runtime_loop_tiling_sizes": [2, 2]}


def _rms_gemms_rope_run_backend():
    from shared.infra.backend_presets import RMS_GEMMS_ROPE_BACKEND as _base

    return {**_base, "runtime_loop_tiling_sizes": [2, 2]}


def _rms_scratch_specs(seq_len, emb_dim, kv_dim):
    """Registry-driven f32 C-scratch args for the Q/K/V GEMMs of rms_gemms_rope,
    in builder order (Q, K, V). Returns (list_of_scratch_arrays, set_of_indices).

    Mirrors llama32_1b_prefill._rms_scratch_specs exactly (same builder, same
    per-shape gemm_registry_config lookup) -- kept as a local copy rather than
    importing the sibling module so this file has no llama32_1b dependency.
    """
    from shared.builders.gemm_builder import gemm_registry_config

    q_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    arrays, inter = [], set()
    nxt = 13
    for spec, cols in ((q_spec, emb_dim), (k_spec, kv_dim), (v_spec, kv_dim)):
        if spec["needs_f32_scratch"]:
            arrays.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return arrays, inter


def _offn_scratch_specs(seq_len, emb_dim, hidden_dim):
    """Registry-driven f32 C-scratch args for the O/Gate/Up/Down GEMMs of
    o_ffn, in builder order. Mirrors shared/builders/o_ffn_multi.py's own
    per-shape gemm_registry_config + alloc_gemm_scratch lookup (base arg
    count 15) so the args passed here match the ELF's declared scratch
    slots. At llama's seq_len=2048 all 4 GEMMs are large enough to resolve
    to fused-cast (4 scratch args); at SmolVLA's seq_len=256 they resolve to
    drain instead (0 scratch args) -- hardcoding "always 4" was the bug that
    broke o_ffn_multi.py's stitching at this shape (see build_o_ffn_module).
    """
    from shared.builders.gemm_builder import gemm_registry_config

    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
    arrays, inter = [], set()
    nxt = 15
    for spec, cols in (
        (o_spec, emb_dim),
        (g_spec, hidden_dim),
        (g_spec, hidden_dim),
        (d_spec, emb_dim),
    ):
        if spec["needs_f32_scratch"]:
            arrays.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return arrays, inter


def compile_all_kernels(cache, config, seq_len, cpu_attn=True):
    """Pre-compile all unique kernel configs to cache.

    Args:
        cache: KernelCache instance
        config: SmolVLABackboneConfig
        seq_len: Sequence length (padded, e.g. 256)
        cpu_attn: If True (Step A), no NPU attention kernel is compiled.
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling unique kernels (seq_len={seq_len})...")
    print(f"{'='*60}\n")

    # External-GEMM mm.o variants — compile FIRST (before any compile_and_cache,
    # so prepare_air_project copies them into air_project/ for every ELF that
    # links them). Discovered dynamically from the SAME per-shape
    # gemm_registry_config lookups the two ELF builders make (rather than
    # hardcoding tile_n=128 the way llama32_1b does): at SmolVLA's seq_len=256
    # the O/Down GEMMs resolve to tile_n=80 while Gate/Up resolve to tile_n=128
    # (both "drain" method) -- two DISTINCT compiled objects are required since
    # compile_gemm_mm bakes DIM_N into the object at compile time, and
    # o_ffn_multi.py's disambiguate_by_tile_n() gives them non-colliding
    # sym_suffix/obj names precisely so this loop compiles the right set.
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import (
        gemm_registry_config,
        disambiguate_by_tile_n,
    )

    q_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
    # rms_gemms_rope (Q,K,V) doesn't disambiguate internally (its own GEMMs
    # happen to share tile_n uniformly at every shape exercised so far); o_ffn
    # (O,Gate,Up,Down) does -- mirror each builder's own decision exactly.
    o_spec, g_spec, d_spec = disambiguate_by_tile_n([o_spec, g_spec, d_spec])

    _needed = {}
    for spec in (q_spec, k_spec, v_spec, o_spec, g_spec, d_spec):
        _needed[spec["sym_suffix"]] = spec
    for spec in _needed.values():
        compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )

    # 1. RMSNorm + QKV GEMMs + RoPE Q+K: one ELF (registry-driven per-GEMM method).
    from shared.builders.rms_gemms_rope_multi import build_rms_gemms_rope_module

    cache.compile_and_cache(
        "rms_gemms_rope",
        build_rms_gemms_rope_module(
            seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
        ),
        {"verbose": cache.verbose, **_rms_gemms_rope_run_backend()},
    )

    # 2. O GEMM + Residual Add + FFN (registry-driven fused-cast GEMMs).
    from shared.builders.o_ffn_multi import build_o_ffn_module

    o_ffn_backend = {
        "verbose": cache.verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }
    _o_ffn_mod = build_o_ffn_module(seq_len, emb_dim, hidden_dim)
    cache.compile_and_cache("o_ffn", _o_ffn_mod, o_ffn_backend)

    if not cpu_attn:
        _compile_npu_attention_kernels(cache, seq_len, head_dim, n_heads)
    else:
        print("  Skipping NPU-attention compilation (Step A: CPU attention fallback)")

    cache._save_manifest()

    print(
        f"\nAll {len(cache.artifacts)} kernels compiled and cached to {cache.cache_dir}/"
    )
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


def _attn_gemm_backend(instance_name="matmul_bf16"):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": instance_name,
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _masked_softmax_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "masked_softmax",
        "runtime_loop_tiling_sizes": [4, 4],
    }


# Attention GEMM tile configs (validated on NPU in validate_attn_gemms.py;
# recorded in kernel_registry GEMM_bf16_in_bf16_out for shapes 256x64x256 and
# 256x256x64). Drain method (tile_m=32), bf16-in/bf16-out.
_QKT_TILE_N = 64  # N=seq_len=256 -> tile_n=64, herd_n=4
_QKT_TILE_K = None  # = head_dim, filled at compile time
_PV_TILE_N = 16  # N=head_dim=64 -> tile_n=16, herd_n=4
_PV_TILE_K = 64  # K=seq_len chunk; tile_k_l1=64 divides 256


def _compile_npu_attention_kernels(cache, seq_len, head_dim, n_heads):
    """Compile the NPU attention ELFs (Step B, cpu_attn=False):
      - qkt : per-head S = Qh @ Khᵀ  (seq x head_dim x seq)
      - pv  : per-head O = P @ Vh     (seq x seq x head_dim)
      - masked_softmax : batched row-softmax over all heads' (seq x seq) scores.

    Each GEMM is a standalone ELF that bakes mm.o (DIM_M/N/K) at compile time,
    so the qkt (tile_n=64) and pv (tile_n=16) configs never collide -- each
    compile_and_cache's prepare_air_project rebuilds mm.o fresh. masked_softmax.o
    is compiled + staged into air_project/ for its ELF's link.
    """
    import shutil
    from pathlib import Path as _Path
    from shared.infra.external_kernels import compile_gemm_mm, compile_masked_softmax
    from matrix_multiplication.bf16_in_bf16_out.run import (
        build_module as _build_gemm,
    )
    from masked_softmax.masked_softmax import build_module as _build_masked_softmax

    print("  Compiling NPU attention kernels (Step B: qkt + pv GEMMs + masked_softmax)")

    # 1. QKᵀ GEMM: (seq, head_dim) @ (head_dim, seq) -> (seq, seq).
    compile_gemm_mm(
        tile_m=32,
        tile_n=_QKT_TILE_N,
        tile_k_l1=head_dim,
        sym_suffix="",
        out_name="mm.o",
    )
    qkt_mod = _build_gemm(
        seq_len,
        head_dim,
        seq_len,
        32,
        head_dim,
        head_dim,
        _QKT_TILE_N,
        8,
        4,
        bfloat16,
        bfloat16,
        arch="aie2p",
        emit_external_call=True,
    )
    cache.compile_and_cache("qkt", qkt_mod, _attn_gemm_backend())

    # 2. P@V GEMM: (seq, seq) @ (seq, head_dim) -> (seq, head_dim).
    compile_gemm_mm(
        tile_m=32,
        tile_n=_PV_TILE_N,
        tile_k_l1=_PV_TILE_K,
        sym_suffix="",
        out_name="mm.o",
    )
    pv_mod = _build_gemm(
        seq_len,
        seq_len,
        head_dim,
        32,
        _PV_TILE_K,
        _PV_TILE_K,
        _PV_TILE_N,
        8,
        4,
        bfloat16,
        bfloat16,
        arch="aie2p",
        emit_external_call=True,
    )
    cache.compile_and_cache("pv", pv_mod, _attn_gemm_backend())

    # 3. masked_softmax over ALL heads at once: n = n_heads * seq * seq, row
    #    width tile_n = seq (softmax reduction axis).
    compile_masked_softmax()
    ms_n = n_heads * seq_len * seq_len
    ms_mod = _build_masked_softmax(ms_n, seq_len, 4, bfloat16)
    cache.compile_and_cache("masked_softmax", ms_mod, _masked_softmax_backend())
    # prepare_air_project (inside compile_and_cache) now stages masked_softmax.o
    # because compile_masked_softmax() ran first (existence-guarded copy).


def _npu_attention(q_roped, k_roped, v, kernel_mask, config, cache, seq_len, layer_idx):
    """Non-causal GQA attention on NPU (Step B). Per q-head:
        S = (Qh * 1/sqrt(d)) @ Khᵀ    [qkt GEMM ELF]
    then all heads' scores are row-softmaxed in ONE masked_softmax dispatch:
        P = softmax(S + mask)          [masked_softmax ELF, batched over heads]
    then per q-head:
        O = P @ Vh                     [pv GEMM ELF]

    The additive mask (0 / BF16_MIN) is head-independent; it is folded into the
    score buffer on host before the batched softmax (the kernel's own mask input
    is left zero). All matmuls + the softmax run on NPU; the host only does
    layout glue (transpose Kh, gather per-head slices).

    Returns attn_out (seq_len, n_heads*head_dim) bf16.
    """
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    hd = config.head_dim
    group = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(hd)

    q_h = np.asarray(q_roped, dtype=np.float32).reshape(seq_len, n_heads, hd)
    k_h = np.asarray(k_roped, dtype=np.float32).reshape(seq_len, n_kv_heads, hd)
    v_h = np.asarray(v, dtype=np.float32).reshape(seq_len, n_kv_heads, hd)

    # --- Step 1: per-head QKᵀ GEMM -> scores (seq, seq) for every head. ---
    scores = np.empty((n_heads, seq_len, seq_len), dtype=bfloat16)
    for h in range(n_heads):
        kv = h // group
        Qh = (q_h[:, h, :] * scale).astype(bfloat16)
        Kh_T = np.ascontiguousarray(k_h[:, kv, :].T).astype(bfloat16)  # (hd, seq)
        s_out = np.zeros(seq_len * seq_len, dtype=bfloat16)
        res = cache.load_and_run(
            "qkt",
            _attn_gemm_backend(),
            Qh.reshape(-1),
            Kh_T.reshape(-1),
            s_out,
            output_indices=[2],
            bo_key="qkt",
        )
        scores[h] = res[2].reshape(seq_len, seq_len)

    # --- Step 2: fold additive mask, batched row-softmax over all heads. ---
    scores_f32 = scores.astype(np.float32) + kernel_mask[None, :, :]
    ms_in = scores_f32.astype(bfloat16).reshape(-1)
    ms_n = n_heads * seq_len * seq_len
    zero_mask = np.zeros(ms_n, dtype=bfloat16)  # mask already folded in
    ms_out = np.zeros(ms_n, dtype=bfloat16)
    res = cache.load_and_run(
        "masked_softmax",
        _masked_softmax_backend(),
        ms_in,
        zero_mask,
        ms_out,
        output_indices=[2],
        intermediate_indices={1},  # zero mask static-once
        bo_key="ms",
    )
    probs = res[2].reshape(n_heads, seq_len, seq_len).astype(bfloat16)

    # --- Step 3: per-head P@V GEMM -> per-head output. ---
    attn_out = np.empty((seq_len, n_heads * hd), dtype=bfloat16)
    for h in range(n_heads):
        kv = h // group
        Ph = np.ascontiguousarray(probs[h]).astype(bfloat16)  # (seq, seq)
        Vh = np.ascontiguousarray(v_h[:, kv, :]).astype(bfloat16)  # (seq, hd)
        o_out = np.zeros(seq_len * hd, dtype=bfloat16)
        res = cache.load_and_run(
            "pv",
            _attn_gemm_backend(),
            Ph.reshape(-1),
            Vh.reshape(-1),
            o_out,
            output_indices=[2],
            bo_key="pv",
        )
        attn_out[:, h * hd : (h + 1) * hd] = res[2].reshape(seq_len, hd)

    return attn_out


# ---------------------------------------------------------------------------
# Transformer block execution
# ---------------------------------------------------------------------------


def run_transformer_block(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    mask,
    positions,
    layer_idx=0,
    cpu_attn=True,
    verbose=False,
):
    """Execute a single SmolVLA transformer block on NPU using cached kernels.

    Args:
        x_bf16: (seq_len, emb_dim) bfloat16 input
        layer_weights: LayerWeights for this layer
        rope_lut_bf16: (seq_len, head_dim) bfloat16 RoPE LUT (per-position cos/sin
            packed the same way as generate_rope_lut for llama-family siblings)
        config: SmolVLABackboneConfig
        cache: KernelCache instance (kernels must be pre-compiled)
        mask: (seq_len, seq_len) additive F32 prefix/padding attention mask
        positions: (seq_len,) int RoPE positions (padding-frozen, see
            smolvla_cpu_helpers.cpu_backbone_forward docstring)
        layer_idx: Layer index for logging / BO-cache keys
        cpu_attn: If True (Step A), run attention on CPU via
            noncausal_attention_reference instead of an NPU kernel.
        verbose: If True, print per-step progress

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

    intermediates = {}

    _arg_cache = getattr(run_transformer_block, "_arg_cache", {})
    run_transformer_block._arg_cache = _arg_cache

    if verbose:
        print(f"  Layer {layer_idx}: Running transformer block...")

    # 1-6. RMSNorm + Q/K/V Projection + RoPE Q+K [6-launch multi-launch ELF]
    kv_dim = n_kv_heads * head_dim
    if verbose:
        print(
            f"    Steps 1-6: RMSNorm + QKV + RoPE [6-launch ELF] "
            f"(Q: {seq_len}x{emb_dim}, K/V: {seq_len}x{kv_dim})"
        )
    _rms_key = f"rms_gemms_rope_L{layer_idx}"
    if _rms_key not in _arg_cache:
        _rms_args = [
            None,  # arg0: x_in (dynamic, replaced each call)
            np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # normed_buf
            np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # q_buf
            np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # k_buf
            np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # v_buf
            np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
            np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # q_roped_buf
            np.zeros((seq_len, kv_dim), dtype=bfloat16),  # k_roped_buf
        ]
        _scratch_arrays, _scratch_inter = _rms_scratch_specs(seq_len, emb_dim, kv_dim)
        _rms_args.extend(_scratch_arrays)
        _arg_cache[_rms_key] = (_rms_args, _scratch_inter)
    cached_args, _scratch_inter = _arg_cache[_rms_key]
    cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

    _rms_inter = {2, 4, 6, 8, 11, 12} | _scratch_inter
    results = cache.load_and_run(
        "rms_gemms_rope",
        _rms_gemms_rope_run_backend(),
        *cached_args,
        output_indices=[8, 11, 12],
        static_input_indices={1, 3, 5, 7, 9, 10},  # weights + LUTs
        intermediate_indices=_rms_inter,
        bo_key=_rms_key,
    )
    v = results[8].reshape(seq_len, kv_dim)
    q_roped = results[11].reshape(seq_len, n_heads * head_dim)
    k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)
    intermediates["v"] = v
    intermediates["k_roped"] = k_roped
    intermediates["q_roped"] = q_roped

    # 7. Attention (non-causal, GQA, prefix-mask + padding-frozen positions).
    # NOT llama32_1b's causal attention_reference -- this is the key SmolVLA
    # divergence. Two paths:
    #   Step A (cpu_attn=True): noncausal_attention_reference on CPU.
    #   Step B (cpu_attn=False): _npu_attention -- per-head QKᵀ/PV GEMMs on NPU +
    #     a batched masked_softmax on NPU (all matmuls + softmax run on NPU;
    #     host only does layout glue).
    if cpu_attn:
        if verbose:
            print(
                f"    Step 7: Attention GQA [CPU non-causal] "
                f"({n_heads}Q/{n_kv_heads}KV heads)"
            )
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            q_h = q_roped.astype(np.float32).reshape(seq_len, n_heads, head_dim)
            k_h = k_roped.astype(np.float32).reshape(seq_len, n_kv_heads, head_dim)
            v_h = v.astype(np.float32).reshape(seq_len, n_kv_heads, head_dim)
            attn_out_f32 = noncausal_attention_reference(
                q_h, k_h, v_h, mask, n_heads, n_kv_heads
            )
            attn_out = attn_out_f32.reshape(seq_len, n_heads * head_dim).astype(
                bfloat16
            )
    else:
        if verbose:
            print(
                f"    Step 7: Attention GQA [NPU: qkt+softmax+pv] "
                f"({n_heads}Q/{n_kv_heads}KV heads)"
            )
        # Convert the additive -inf mask to the kernel's 0 / BF16_MIN form.
        from masked_softmax.masked_softmax import BF16_MIN

        kernel_mask = np.where(np.isneginf(mask), BF16_MIN, 0.0).astype(np.float32)
        attn_out = _npu_attention(
            q_roped, k_roped, v, kernel_mask, config, cache, seq_len, layer_idx
        )
    intermediates["attn_out"] = attn_out

    # 8-15. O GEMM + Residual Add + FFN [8-launch multi-launch ELF]
    if verbose:
        print(
            f"    Steps 8-15: O+FFN [8-launch ELF] "
            f"(O: {seq_len}x{emb_dim}, FFN: {seq_len}x{emb_dim}x{hidden_dim})"
        )
    _offn_key = f"o_ffn_L{layer_idx}"
    if _offn_key not in _arg_cache:
        offn_args = [
            None,  # arg0: attn_out (dynamic)
            np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # proj_buf
            None,  # arg3: x_residual (dynamic)
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # res1_buf
            np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # normed2_buf
            np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
                emb_dim, hidden_dim
            ),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # gate_buf
            np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # up_buf
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # swiglu_buf
            np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
                hidden_dim, emb_dim
            ),
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # down_buf
            np.zeros(n_total, dtype=bfloat16),  # output_buf (arg14)
        ]
        # Registry-driven f32 C-scratch args, in builder order (O,Gate,Up,Down).
        # At SmolVLA's seq_len=256 all 4 GEMMs resolve to drain (0 scratch
        # args); at llama's seq_len=2048 all 4 resolve to fused-cast (4
        # scratch args). Must mirror build_o_ffn_module's own per-shape
        # gemm_registry_config lookup or the args won't match the ELF's
        # declared scratch slots.
        _offn_scratch_arrays, _offn_scratch_inter = _offn_scratch_specs(
            seq_len, emb_dim, hidden_dim
        )
        offn_args.extend(_offn_scratch_arrays)
        _arg_cache[_offn_key] = (offn_args, _offn_scratch_inter)
    cached_args, _offn_scratch_inter = _arg_cache[_offn_key]
    cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
    cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

    _out_idx = 14
    _inter = {2, 4, 6, 8, 10, 11, 13, 14} | _offn_scratch_inter
    results = cache.load_and_run(
        "o_ffn",
        _o_ffn_run_backend(),
        *cached_args,
        output_indices=[_out_idx],
        static_input_indices={1, 5, 7, 9, 12},  # wo, ffn_norm_w, w_gate, w_up, w_down
        intermediate_indices=_inter,
        bo_key=_offn_key,
    )
    output_bf16 = results[_out_idx].reshape(seq_len, emb_dim)
    intermediates["ffn_out"] = output_bf16

    return output_bf16, intermediates


# ---------------------------------------------------------------------------
# Full 16-layer backbone execution
# ---------------------------------------------------------------------------


def run_backbone_prefill(
    prefix_embed_bf16,
    weights,
    config,
    cache,
    mask,
    positions,
    rope_lut,
    cpu_attn=True,
    verbose=False,
    return_kv=False,
):
    """Run the full N-layer SmolVLA backbone prefill on NPU.

    Loops every transformer layer (feeding each layer's output as the next
    layer's input), collects per-layer outputs for cosine diagnosis, then
    applies the final RMSNorm (weights.final_norm). The oracle's
    `final_norm_hidden` is `text_model.norm` applied to the layer-15 output, so
    the final norm here matches that exactly.

    The final RMSNorm runs on CPU (F32) -- the same choice every llama/qwen
    sibling makes (llama32_1b_inference.py:453, verify_adapter.py:219): it is a
    single (seq, emb) RMSNorm on the last hidden state, not inside the per-layer
    hot loop, so it adds no per-token NPU work and reuses the exact F32 reference
    math the oracle was generated with. This is a deliberate, sibling-consistent
    CPU op (not a fallback from a broken NPU path); logged in docs/TODO.md under
    "NPU-execution exceptions".

    Args:
        prefix_embed_bf16: (seq_len, emb_dim) bfloat16 padded prefix embedding.
        weights: backbone weights (weights.layers is the per-layer list,
            weights.final_norm is the final RMSNorm weight).
        config: SmolVLABackboneConfig.
        cache: KernelCache with kernels pre-compiled (compile_all_kernels).
        mask: (seq_len, seq_len) additive F32 prefix/padding attention mask.
        positions: (seq_len,) int RoPE positions (padding-frozen).
        rope_lut: (seq_len, head_dim) bfloat16 RoPE LUT, gathered per token.
        cpu_attn: forwarded to run_transformer_block.
        verbose: per-layer progress printing.
        return_kv: if True, additionally return the per-layer post-RoPE K and
            raw V (5 kv-heads) exactly as the SmolVLA action expert would cache
            them during its prefill (`fill_kv_cache=True`). These come straight
            from the fused RMS+QKV+RoPE ELF's outputs (k_roped, v) that
            run_transformer_block already surfaces in its intermediates dict --
            no recompute. Used by the end-to-end hybrid pipeline
            (smolvla_inference.py) to inject the NPU KV into the CPU expert's
            past_key_values.

    Returns:
        If return_kv is False (default):
            (final_hidden, per_layer_list):
                final_hidden: (seq_len, emb_dim) F32 = RMSNorm(layer[-1] out).
                per_layer_list: list of N (seq_len, emb_dim) bf16 per-layer outputs.
        If return_kv is True:
            (final_hidden, per_layer_list, kv_list) where kv_list is a list of
            N (k_h, v_h) tuples, each (seq_len, n_kv_heads, head_dim) bf16.
            k_h is POST-RoPE (the same quantity apply_rope produces before the
            cache write); v_h is un-rotated.
    """
    x = np.asarray(prefix_embed_bf16, dtype=bfloat16)
    per_layer_list = []
    kv_list = []
    n_layers = len(weights.layers)
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = x.shape[0]
    for layer_idx, lw in enumerate(weights.layers):
        if verbose:
            print(f"\n--- Backbone layer {layer_idx}/{n_layers - 1} ---")
        x, _inter = run_transformer_block(
            x,
            lw,
            rope_lut,
            config,
            cache,
            mask,
            positions,
            layer_idx=layer_idx,
            cpu_attn=cpu_attn,
            verbose=verbose,
        )
        per_layer_list.append(x)
        if return_kv:
            # k_roped: (seq_len, n_kv_heads*head_dim) post-RoPE; v: (seq_len,
            # n_kv_heads*head_dim) un-rotated. Reshape to per-head (L, Hkv, D)
            # -- the (B,L,H,D) layout (minus batch) the expert cache expects.
            k_h = np.asarray(_inter["k_roped"], dtype=bfloat16).reshape(
                seq_len, n_kv_heads, head_dim
            )
            v_h = np.asarray(_inter["v"], dtype=bfloat16).reshape(
                seq_len, n_kv_heads, head_dim
            )
            kv_list.append((k_h, v_h))

    # Final RMSNorm (text_model.norm) on the last hidden state -- CPU F32, same
    # as every llama/qwen sibling (see docstring).
    last_hidden_f32 = np.asarray(x, dtype=np.float32)
    final_hidden = rms_norm(last_hidden_f32, weights.final_norm, config.rms_norm_eps)
    if return_kv:
        return final_hidden, per_layer_list, kv_list
    return final_hidden, per_layer_list

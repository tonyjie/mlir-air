# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Vision-Encoder (SigLIP ViT) 12-layer prefill on MLIR-AIR (NPU2).

Sibling of `smolvla_backbone_prefill.py`, but for the SigLIP ViT (bidirectional
12-layer encoder) instead of the causal language backbone. Correctness-first
(A3-5 Step 2-3): the four heavy ops run on NPU — projection/MLP GEMMs, affine
LayerNorm, GELU-tanh, and non-causal FlashAttention — while the cheap glue
(bias-adds, residual adds, im2col patch-embed) stays on host numpy. Fusing those
host ops onto the device and cutting dispatches is A3-6 (a SEPARATE perf phase);
nothing here is optimized.

SigLIP differences from the backbone (all handled below):
  - Every Linear has a BIAS (q/k/v/out proj + fc1/fc2) — added on host after GEMM.
  - Norm is affine LayerNorm (gamma + beta, eps 1e-6), not RMSNorm. The affine
    `layer_norm` ELF takes a packed [2N] weight||bias buffer.
  - Bidirectional MHA (12 heads, no GQA, no mask, scale 1/8) → the registry FA
    ELF with causal=False. FA applies 1/sqrt(dk)=1/8 internally, matching
    SigLIP's attn_scale exactly, so Q is NOT pre-scaled.
  - GELU-tanh MLP activation (not SwiGLU) → the `gelu` 1D elementwise ELF.

NPU kernels driven (all validated at these exact shapes in A3-1..A3-4, registry):
  gemm_qkvo : 1024x768x768   (q/k/v/o projections, drain tile_m32/tn96)
  gemm_fc1  : 1024x768x3072  (MLP fc1, drain tile_m32/tn128)
  gemm_fc2  : 1024x3072x768  (MLP fc2, drain tile_m32/tn96)
  layer_norm: 1024x768 affine (ln1, ln2, post_layernorm), herd_x=8
  gelu      : N=1024*3072 GELU-tanh, herd 8x2
  flash_attn: 1024/1024 12q/12kv MHA, head_dim=64, non-causal, hpu=2 (full array)
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

from vision_weights import SigLIPVisionConfig
from vision_cpu_helpers import im2col_patch_embed
from shared.infra.cache import KernelCache, Profiler  # noqa: F401 (re-exported)

# ---------------------------------------------------------------------------
# Per-kernel backend presets (match each kernel's own standalone example)
# ---------------------------------------------------------------------------


def _gemm_backend():
    # Standalone external-mm.o GEMM ELF. instance_name MUST equal the module
    # func name (`matmul_bf16`), NOT the cache key.
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "matmul_bf16",
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _ln_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "layer_norm",
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _gelu_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gelu",
        "runtime_loop_tiling_sizes": [4, 4],
    }


_ATTN_BACKEND_KWARGS = {
    "verbose": False,
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


# Vision GEMM tiles (registry, precision="high" → drain). All three resolve to
# tile_m=32 drain; kept as explicit constants so compile-time mm.o bakes match
# the build_module tile args (registry-validated in A3-1, commit 3fc1bb12).
#   gemm_qkvo 1024x768x768 : tile_m32 tk_l2 384 tk_l1 64 tn 96
#   gemm_fc1  1024x768x3072: tile_m32 tk_l2 256 tk_l1 32 tn 128
#   gemm_fc2  1024x3072x768: tile_m32 tk_l2 384 tk_l1 64 tn 96
_GEMM_SHAPES = {
    "gemm_qkvo": dict(
        m=1024, k=768, n=768, tile_m=32, tile_k_l2=384, tile_k_l1=64, tile_n=96
    ),
    "gemm_fc1": dict(
        m=1024, k=768, n=3072, tile_m=32, tile_k_l2=256, tile_k_l1=32, tile_n=128
    ),
    "gemm_fc2": dict(
        m=1024, k=3072, n=768, tile_m=32, tile_k_l2=384, tile_k_l1=64, tile_n=96
    ),
}


# ---------------------------------------------------------------------------
# Kernel compilation
# ---------------------------------------------------------------------------


def compile_all_kernels(cache, config, seq_len=1024, fa_bfp16=True):
    """Pre-compile every unique vision-encoder kernel config to the cache.

    fa_bfp16: FlashAttention microkernel (attn_npu2.o) block-float mode.
        BFP16=True (default) is the only WORKING FA build at this shape: BFP16=False
        (native aie2p bf16 8x8x8 mmul) was tried to remove the systematic
        attention bias but the FA kernel's L1 tiling is sized for the BFP16 mmul
        and the native mmul overflows/mismatches → runtime hang
        (ERT_CMD_STATE_TIMEOUT). Kept as a flag purely to document the experiment.
        See docs/TODO.md "NPU-execution exceptions" for the full precision story.

    Order matters for the GEMM ELFs: `compile_gemm_mm` writes the DIM-baked
    `mm.o` into CWD, then `compile_and_cache` → `prepare_air_project` wipes
    air_project/ fresh and copies the current mm.o into it. So each GEMM ELF's
    mm.o is compiled immediately before its compile_and_cache (mirrors
    smolvla_backbone_prefill._compile_npu_attention_kernels). The LayerNorm /
    GELU / FA ELFs don't link mm.o (a stale copy staged into their air_project
    is harmless).
    """
    from shared.infra.external_kernels import compile_gemm_mm
    from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm
    from layer_norm.layer_norm import build_module as build_layer_norm
    from gelu.gelu import build_module as build_gelu
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_heads  # MHA, no GQA
    head_dim = config.head_dim

    print(f"\n{'='*60}")
    print(f"Compiling vision-encoder kernels (seq_len={seq_len})...")
    print(f"{'='*60}\n")

    # --- 1. Projection / MLP GEMMs (one ELF per distinct shape) ---
    for name, s in _GEMM_SHAPES.items():
        print(
            f"  Compiling {name}: {s['m']}x{s['k']}x{s['n']} (drain tile_m={s['tile_m']} tile_n={s['tile_n']})"
        )
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix="",
            out_name="mm.o",
        )
        mod = build_gemm(
            s["m"],
            s["k"],
            s["n"],
            s["tile_m"],
            s["tile_k_l2"],
            s["tile_k_l1"],
            s["tile_n"],
            8,  # herd_m
            4,  # herd_n
            bfloat16,
            bfloat16,
            arch="aie2p",
            emit_external_call=True,
        )
        cache.compile_and_cache(
            name, mod, {"verbose": cache.verbose, **_gemm_backend()}
        )

    # --- 2. Affine LayerNorm (ln1, ln2, post_layernorm all 1024x768) ---
    print(f"  Compiling layer_norm: {seq_len}x{emb_dim} affine (herd_x=8)")
    ln_mod = build_layer_norm(seq_len, emb_dim, bfloat16, herd_x=8)
    cache.compile_and_cache(
        "layer_norm", ln_mod, {"verbose": cache.verbose, **_ln_backend()}
    )

    # --- 3. GELU-tanh (1D, N = seq_len * hidden_dim) ---
    gelu_n = seq_len * hidden_dim
    print(f"  Compiling gelu: N={gelu_n} (tile_n=4096, herd 8x2)")
    gelu_mod = build_gelu(gelu_n, 4096, bfloat16, herd_x=8, herd_y=2)
    cache.compile_and_cache(
        "gelu", gelu_mod, {"verbose": cache.verbose, **_gelu_backend()}
    )

    # --- 4. Non-causal FlashAttention (12q/12kv MHA, head_dim=64) ---
    # 12 even heads → num_heads_per_unroll=2 divides evenly → FILLS the full 8x4
    # array (A3-4: 1.58x placement win vs hpu=1). Scale 1/sqrt(dk)=1/8 is applied
    # inside the kernel, matching SigLIP attn_scale — Q is NOT pre-scaled.
    num_heads_per_unroll = 2
    num_q_tiles = 4
    assert n_heads % num_heads_per_unroll == 0
    assert num_heads_per_unroll * num_q_tiles <= 8
    print(
        f"  Compiling flash_attn: non-causal, seq={seq_len}, {n_heads}Q/{n_kv_heads}KV, "
        f"head_dim={head_dim}, hpu={num_heads_per_unroll}"
    )
    attn_mod = build_attn(
        lk=seq_len,
        lkp=head_dim,
        lq=seq_len,
        lqp=256,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=False,
        num_heads_per_unroll=num_heads_per_unroll,
    )
    # Pre-build attn_npu2.o with the chosen precision mode and force=True. The
    # compile_and_cache below calls prepare_air_project → compile_all_external_kernels,
    # which rebuilds attn_npu2.o only if absent (force=False) — so this force=True
    # build wins and its .o is the one linked into the FA ELF. fa_bfp16=False gives
    # the native aie2p bf16 mmul (no systematic block-float attention bias).
    from shared.infra.external_kernels import compile_attn_npu2

    compile_attn_npu2(head_dim=head_dim, bfp16=fa_bfp16, force=True)
    print(f"    (FA microkernel BFP16={fa_bfp16})")
    cache.compile_and_cache(
        "flash_attn", attn_mod, {**_ATTN_BACKEND_KWARGS, "verbose": cache.verbose}
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} vision kernels compiled to {cache.cache_dir}/")
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Per-kernel NPU run helpers
# ---------------------------------------------------------------------------


def _run_gemm(cache, name, A, B, M, N, bo_key):
    """y = A @ B on NPU (bf16-in, bf16-out). A:(M,K), B:(K,N). B is the weight
    (static, written once per bo_key)."""
    A = np.ascontiguousarray(np.asarray(A, dtype=bfloat16)).reshape(-1)
    B = np.ascontiguousarray(np.asarray(B, dtype=bfloat16)).reshape(-1)
    C = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        name,
        _gemm_backend(),
        A,
        B,
        C,
        output_indices=[2],
        static_input_indices={1},
        bo_key=bo_key,
    )
    return res[2].reshape(M, N)


def _run_layer_norm(cache, x, weight, bias, M, N, bo_key):
    """Affine LayerNorm on NPU. weight/bias:(N,) packed into a flat [2N] buffer
    ([0:N]=weight, [N:2N]=bias) — the kernel reads it as one DMA."""
    x = np.ascontiguousarray(np.asarray(x, dtype=bfloat16)).reshape(-1)
    param = np.concatenate(
        [np.asarray(weight, dtype=bfloat16), np.asarray(bias, dtype=bfloat16)]
    ).astype(bfloat16)
    out = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        "layer_norm",
        _ln_backend(),
        x,
        param,
        out,
        output_indices=[2],
        static_input_indices={1},
        bo_key=bo_key,
    )
    return res[2].reshape(M, N)


def _run_gelu(cache, x, M, N, bo_key):
    """GELU-tanh elementwise on NPU. x:(M,N) → flat [M*N] 1D kernel."""
    x = np.ascontiguousarray(np.asarray(x, dtype=bfloat16)).reshape(-1)
    out = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        "gelu",
        _gelu_backend(),
        x,
        out,
        output_indices=[1],
        bo_key=bo_key,
    )
    return res[1].reshape(M, N)


def _run_flash_attention(cache, q, k, v, config, seq_len):
    """Non-causal MHA on NPU via FlashAttention. q/k/v:(seq, n_heads*head_dim)
    seq-first (head h occupies columns [h*hd:(h+1)*hd]). Returns (seq, emb)."""
    n_heads = config.n_heads
    head_dim = config.head_dim
    q_attn = np.ascontiguousarray(np.asarray(q, dtype=bfloat16))
    k_attn = np.ascontiguousarray(np.asarray(k, dtype=bfloat16))
    v_attn = np.ascontiguousarray(np.asarray(v, dtype=bfloat16))
    out = np.zeros((seq_len, n_heads * head_dim), dtype=bfloat16)
    res = cache.load_and_run(
        "flash_attn",
        _ATTN_BACKEND_KWARGS,
        q_attn,
        k_attn,
        v_attn,
        out,
        output_indices=[3],
        bo_key="flash_attn",
    )
    return res[3].reshape(seq_len, n_heads * head_dim)


# ---------------------------------------------------------------------------
# One SigLIP encoder block (pre-norm)
# ---------------------------------------------------------------------------


def run_vit_block(
    x_bf16, lw, config, cache, layer_idx=0, verbose=False, attn_mode="flash"
):
    """Execute one SigLIP encoder layer on NPU. x_bf16:(seq, emb). Returns
    output bf16 (seq, emb).

    Heavy ops (LayerNorm, q/k/v/o GEMM, FA, fc1/fc2 GEMM, GELU) run on NPU; the
    bias-adds (every Linear has a bias) and the two residual adds run on host in
    f32 — a correctness-first shortcut (A3-6 fuses them). See module docstring.

    attn_mode: "flash" (default) = non-causal FlashAttention ELF on NPU.
        "cpu" = mha_bidirectional on host (diagnostic / documented fallback).
    """
    seq_len = x_bf16.shape[0]
    emb = config.emb_dim
    hidden = config.hidden_dim

    if verbose:
        print(
            f"  ViT layer {layer_idx}: LN1 -> qkv -> FA -> o -> res -> LN2 -> fc1 -> gelu -> fc2 -> res"
        )

    # --- Attention block (pre-norm) ---
    h = _run_layer_norm(
        cache, x_bf16, lw.ln1_w, lw.ln1_b, seq_len, emb, bo_key=f"ln1_L{layer_idx}"
    )

    q = _run_gemm(cache, "gemm_qkvo", h, lw.wq, seq_len, emb, bo_key=f"wq_L{layer_idx}")
    k = _run_gemm(cache, "gemm_qkvo", h, lw.wk, seq_len, emb, bo_key=f"wk_L{layer_idx}")
    v = _run_gemm(cache, "gemm_qkvo", h, lw.wv, seq_len, emb, bo_key=f"wv_L{layer_idx}")
    # Host bias-add (every SigLIP proj has a bias), cast back to bf16 for FA.
    q = (q.astype(np.float32) + lw.bq.astype(np.float32)).astype(bfloat16)
    k = (k.astype(np.float32) + lw.bk.astype(np.float32)).astype(bfloat16)
    v = (v.astype(np.float32) + lw.bv.astype(np.float32)).astype(bfloat16)

    if attn_mode == "cpu":
        from vision_cpu_helpers import mha_bidirectional

        attn = mha_bidirectional(
            q.astype(np.float32),
            k.astype(np.float32),
            v.astype(np.float32),
            config.n_heads,
            config.head_dim,
            config.attn_scale,
        ).astype(bfloat16)
    else:
        attn = _run_flash_attention(cache, q, k, v, config, seq_len)

    o = _run_gemm(
        cache, "gemm_qkvo", attn, lw.wo, seq_len, emb, bo_key=f"wo_L{layer_idx}"
    )
    o = o.astype(np.float32) + lw.bo.astype(np.float32)

    # Residual (host f32).
    x = x_bf16.astype(np.float32) + o
    x_bf16 = x.astype(bfloat16)

    # --- MLP block (pre-norm) ---
    h = _run_layer_norm(
        cache, x_bf16, lw.ln2_w, lw.ln2_b, seq_len, emb, bo_key=f"ln2_L{layer_idx}"
    )

    h1 = _run_gemm(
        cache, "gemm_fc1", h, lw.w_fc1, seq_len, hidden, bo_key=f"fc1_L{layer_idx}"
    )
    h1 = (h1.astype(np.float32) + lw.b_fc1.astype(np.float32)).astype(bfloat16)

    g = _run_gelu(cache, h1, seq_len, hidden, bo_key=f"gelu_L{layer_idx}")

    h2 = _run_gemm(
        cache, "gemm_fc2", g, lw.w_fc2, seq_len, emb, bo_key=f"fc2_L{layer_idx}"
    )
    h2 = h2.astype(np.float32) + lw.b_fc2.astype(np.float32)

    # Residual (host f32).
    x = x.astype(np.float32) + h2
    return x.astype(bfloat16)


# ---------------------------------------------------------------------------
# Full 12-layer encoder
# ---------------------------------------------------------------------------


def run_vit_encoder(
    patch_embed_or_pixel,
    weights,
    config,
    cache,
    return_per_layer=False,
    do_connector=False,
    verbose=False,
    attn_mode="flash",
):
    """Run the full 12-layer SigLIP ViT encoder on NPU.

    Args:
        patch_embed_or_pixel: either pixel_values (3, 512, 512) — in which case
            the host im2col patch-embed + position-embedding add produces the
            (1024, 768) patch embedding — or a precomputed patch_embed (1024,
            768) fed directly.
        weights: VisionWeights.
        config: SigLIPVisionConfig.
        cache: KernelCache with vision kernels pre-compiled.
        return_per_layer: if True, collect each layer's output.
        do_connector: STUB — the connector (pixel-shuffle + big GEMM) is A3-5
            Step 4, out of scope for this task. Must be False here.

    Returns:
        dict with:
            post_ln: (1024, 768) f32 — LayerNorm(post_layernorm) of the last layer.
            layer_hidden: list of 12 (1024, 768) bf16 [if return_per_layer].
    """
    assert not do_connector, "connector is A3-5 Step 4, out of scope (stub)"

    inp = np.asarray(patch_embed_or_pixel)
    if inp.ndim == 3:
        # pixel_values (3, H, W) → host im2col patch-embed + pos-embed add.
        x = im2col_patch_embed(
            inp,
            weights.patch_w,
            weights.patch_b,
            weights.pos_embed,
            config.patch_size,
        )  # (1024, 768) f32
    else:
        x = inp.astype(np.float32)
    x_bf16 = x.astype(bfloat16)
    seq_len, emb = x_bf16.shape

    per_layer = []
    for layer_idx, lw in enumerate(weights.layers):
        if verbose:
            print(f"\n--- ViT layer {layer_idx}/{len(weights.layers) - 1} ---")
        x_bf16 = run_vit_block(
            x_bf16,
            lw,
            config,
            cache,
            layer_idx=layer_idx,
            verbose=verbose,
            attn_mode=attn_mode,
        )
        if return_per_layer:
            per_layer.append(x_bf16)

    # post_layernorm on NPU (affine LayerNorm, same ELF).
    post_ln = _run_layer_norm(
        cache,
        x_bf16,
        weights.post_ln_w,
        weights.post_ln_b,
        seq_len,
        emb,
        bo_key="post_ln",
    ).astype(np.float32)

    result = {"post_ln": post_ln}
    if return_per_layer:
        result["layer_hidden"] = per_layer
    return result

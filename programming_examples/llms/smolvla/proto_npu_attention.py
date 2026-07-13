# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GOAL 2 prototype: prove ONE attention head runs end-to-end on NPU
(QK^T GEMM -> +mask -> masked_softmax -> P@V GEMM) and matches the CPU
non-causal reference for that head, before wiring the full per-head loop into
run_transformer_block.

Datapath per q-head h (GQA: kv-head = h // (n_heads//n_kv_heads)):
    Qh = q_roped[:, h, :] * (1/sqrt(head_dim))   # pre-scale on host
    S  = Qh @ Kh^T            # GEMM 256x64x256  (B = Kh^T, host transpose)
    P  = masked_softmax(S + mask)   # masked_softmax kernel, row width 256
    Oh = P @ Vh               # GEMM 256x256x64

Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 proto_npu_attention.py
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent
_LLMS = _HERE.parent
for p in (str(_PROG), str(_LLMS), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from smolvla_backbone_weights import (
    load_backbone_weights,
    SmolVLABackboneConfig,
    generate_rope_lut,
)
from smolvla_cpu_helpers import (
    build_prefix_mask,
    noncausal_attention_reference,
    rms_norm,
    _rope_half_split,
)
from shared.infra.cache import KernelCache, Profiler
from shared.infra.external_kernels import compile_gemm_mm
from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm
from masked_softmax.masked_softmax import build_module as build_masked_softmax, BF16_MIN

NPU_SEQ_LEN = 256
N_PREFIX = 240
N_STATE = 1
ORACLE_LEN = 241


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    cfg = SmolVLABackboneConfig()
    hd = cfg.head_dim
    L = NPU_SEQ_LEN
    scale = 1.0 / np.sqrt(hd)

    weights = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    lw = weights.layers[0]
    o = np.load(_HERE / "smolvla_oracle.npz")
    pad_mask_241 = o["prefix_pad_masks"]

    mask_241 = build_prefix_mask(N_PREFIX, N_STATE, pad_mask=pad_mask_241)
    mask_256 = np.full((L, L), -np.inf, dtype=np.float32)
    mask_256[:ORACLE_LEN, :ORACLE_LEN] = mask_241
    pad_mask_256 = np.zeros((L,), bool)
    pad_mask_256[:ORACLE_LEN] = pad_mask_241
    positions = np.clip(np.cumsum(pad_mask_256.astype(np.int64)) - 1, 0, None)

    # Build Q/K/V (F32) via CPU recompute (isolates the attention datapath).
    x = np.zeros((L, cfg.emb_dim), np.float32)
    x[:ORACLE_LEN] = o["prefix_embed"]
    h = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps)
    q = (h @ lw.wq.astype(np.float32)).reshape(L, cfg.n_heads, hd)
    k = (h @ lw.wk.astype(np.float32)).reshape(L, cfg.n_kv_heads, hd)
    v = (h @ lw.wv.astype(np.float32)).reshape(L, cfg.n_kv_heads, hd)
    q = _rope_half_split(q, positions, cfg.rope_base, hd)
    k = _rope_half_split(k, positions, cfg.rope_base, hd)

    # CPU reference attention (all heads).
    attn_cpu = noncausal_attention_reference(
        q, k, v, mask_256, cfg.n_heads, cfg.n_kv_heads
    )

    # Additive bf16 mask for the kernel: 0 where attend, BF16_MIN where masked.
    kernel_mask = np.where(np.isneginf(mask_256), BF16_MIN, 0.0).astype(bfloat16)

    # -- Compile the three NPU kernels --
    cache = KernelCache("proto_attn_cache", verbose=False, profiler=Profiler())

    # masked_softmax over (256 rows x 256 wide); flattened n = 256*256.
    ms_n = L * L
    ms_mod = build_masked_softmax(ms_n, L, 4, bfloat16)
    cache.compile_and_cache(
        "masked_softmax",
        ms_mod,
        {
            "omit_while_true_loop": False,
            "output_format": "elf",
            "instance_name": "masked_softmax",
            "runtime_loop_tiling_sizes": [4, 4],
        },
    )
    # NOTE: compile_and_cache -> prepare_air_project wipes air_project/ and only
    # stages a fixed .o list (mm.o/mm_m32.o/... , NOT masked_softmax.o). Stage it
    # by hand into air_project/ so aiecc links it for THIS ELF.
    import shutil

    shutil.copy2(_HERE / "masked_softmax.o", Path("air_project") / "masked_softmax.o")
    # Recompile+cache masked_softmax now that its .o is staged.
    cache.compile_and_cache(
        "masked_softmax",
        ms_mod,
        {
            "omit_while_true_loop": False,
            "output_format": "elf",
            "instance_name": "masked_softmax",
            "runtime_loop_tiling_sizes": [4, 4],
        },
    )

    print("Prototype: NPU masked_softmax ELF compiled. Running head 0 datapath...")

    # Try just the masked_softmax kernel first (S from CPU) to isolate staging.
    hh = 0
    kv = hh // (cfg.n_heads // cfg.n_kv_heads)
    Qh = (q[:, hh, :] * scale).astype(bfloat16)
    Kh = k[:, kv, :].astype(bfloat16)
    Vh = v[:, kv, :].astype(bfloat16)

    S_cpu = Qh.astype(np.float32) @ Kh.astype(np.float32).T
    S_masked = S_cpu + np.where(np.isneginf(mask_256), BF16_MIN, 0.0)
    S_in = S_masked.astype(bfloat16).reshape(-1)
    zero_mask = np.zeros((L, L), bfloat16).reshape(-1)  # mask already folded in
    out_buf = np.zeros(ms_n, bfloat16)
    res = cache.load_and_run(
        "masked_softmax",
        {
            "omit_while_true_loop": False,
            "output_format": "elf",
            "instance_name": "masked_softmax",
            "runtime_loop_tiling_sizes": [4, 4],
        },
        S_in,
        zero_mask,
        out_buf,
        output_indices=[2],
        bo_key="ms",
    )
    P_npu_isolated = res[2].reshape(L, L).astype(np.float32)
    from smolvla_cpu_helpers import _softmax

    P_cpu = _softmax(S_cpu + mask_256, axis=-1)
    print(
        f"  [isolated] masked_softmax NPU vs CPU P: cosine = "
        f"{cosine(P_npu_isolated[:ORACLE_LEN], P_cpu[:ORACLE_LEN]):.6f}"
    )

    # ---------------------------------------------------------------------
    # Full head-0 chain on NPU: QK^T GEMM -> +mask+softmax -> P@V GEMM.
    # Each GEMM is its OWN standalone ELF that bakes mm.o (DIM_M/N/K) at
    # compile time, so the two configs (n=64 for QK^T, n=16 for P@V) do not
    # collide -- compile_and_cache's prepare_air_project rebuilds mm.o fresh
    # per ELF, and after each we re-stage masked_softmax.o too.
    # ---------------------------------------------------------------------
    gemm_be = {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "runtime_loop_tiling_sizes": [4, 4],
    }

    def build_and_cache_gemm(name, m, k, n, tile_n, tile_k):
        compile_gemm_mm(
            tile_m=32, tile_n=tile_n, tile_k_l1=tile_k, sym_suffix="", out_name="mm.o"
        )
        mod = build_gemm(
            m,
            k,
            n,
            32,
            tile_k,
            tile_k,
            tile_n,
            8,
            4,
            bfloat16,
            bfloat16,
            arch="aie2p",
            emit_external_call=True,
        )
        # GEMM module's entry function is `matmul_bf16`; for ELF output the
        # instance_name MUST match a function in the module.
        cache.compile_and_cache(name, mod, {**gemm_be, "instance_name": "matmul_bf16"})

    # QK^T: (256,64) @ (64,256) -> (256,256).  B = Kh^T (64,256).
    build_and_cache_gemm("qkt", L, hd, L, tile_n=64, tile_k=hd)
    # P@V: (256,256) @ (256,64) -> (256,64).
    build_and_cache_gemm("pv", L, L, hd, tile_n=16, tile_k=64)

    hh = 0
    kv = hh // (cfg.n_heads // cfg.n_kv_heads)
    Qh = (q[:, hh, :] * scale).astype(bfloat16)
    Kh_T = np.ascontiguousarray(k[:, kv, :].T).astype(bfloat16)  # (64,256)
    Vh = np.ascontiguousarray(v[:, kv, :]).astype(bfloat16)  # (256,64)

    # 1. S = Qh @ Kh^T on NPU.
    S_out = np.zeros(L * L, bfloat16)
    res = cache.load_and_run(
        "qkt",
        {**gemm_be, "instance_name": "matmul_bf16"},
        Qh.reshape(-1),
        Kh_T.reshape(-1),
        S_out,
        output_indices=[2],
        bo_key="qkt",
    )
    S_npu = res[2].reshape(L, L).astype(np.float32)
    print(
        f"  [chain] QK^T NPU vs CPU S: cosine = "
        f"{cosine(S_npu[:ORACLE_LEN, :ORACLE_LEN], S_cpu[:ORACLE_LEN, :ORACLE_LEN]):.6f}"
    )

    # 2. +mask, softmax on NPU (fold additive mask into the score input).
    S_masked_npu = (S_npu + np.where(np.isneginf(mask_256), BF16_MIN, 0.0)).astype(
        bfloat16
    )
    out_buf2 = np.zeros(ms_n, bfloat16)
    res = cache.load_and_run(
        "masked_softmax",
        {
            "omit_while_true_loop": False,
            "output_format": "elf",
            "instance_name": "masked_softmax",
            "runtime_loop_tiling_sizes": [4, 4],
        },
        S_masked_npu.reshape(-1),
        zero_mask,
        out_buf2,
        output_indices=[2],
        bo_key="ms",
    )
    P_npu = res[2].reshape(L, L).astype(bfloat16)
    print(
        f"  [chain] softmax NPU vs CPU P: cosine = "
        f"{cosine(P_npu[:ORACLE_LEN].astype(np.float32), P_cpu[:ORACLE_LEN]):.6f}"
    )

    # 3. O = P @ Vh on NPU.
    O_out = np.zeros(L * hd, bfloat16)
    res = cache.load_and_run(
        "pv",
        {**gemm_be, "instance_name": "matmul_bf16"},
        P_npu.reshape(-1),
        Vh.reshape(-1),
        O_out,
        output_indices=[2],
        bo_key="pv",
    )
    O_npu = res[2].reshape(L, hd).astype(np.float32)
    O_cpu = attn_cpu[:, hh, :]
    print(
        f"  [chain] FULL head-0 attention NPU vs CPU O: cosine = "
        f"{cosine(O_npu[:ORACLE_LEN], O_cpu[:ORACLE_LEN]):.6f}"
    )


if __name__ == "__main__":
    main()

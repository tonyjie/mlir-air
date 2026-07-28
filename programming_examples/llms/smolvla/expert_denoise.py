# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NPU2 driver for SmolVLA's 16-layer action expert (phase E4).

One denoise step = 16 layers x 13 dispatches + a final RMSNorm. The 10 steps run
strictly sequentially, so this whole thing executes 10x per inference.

Per layer:
    1   expert_rms_qkv_rope   (EVEN) RMSNorm + Q/K/V GEMM + RoPE Q + RoPE K
        expert_rms_q_rope     (ODD)  RMSNorm + Q GEMM + RoPE Q
   11   decomposed attention  5 qkt + 1 masked_softmax + 5 pv, GQA-group batched
    1   expert_o_ffn          O proj + residual + RMSNorm + gate/up + SiLU + down + residual

Attention is decomposed rather than FlashAttention because BOTH layer kinds
carry a real mask (see `_build_additive_mask`) and the registry FA ELF applies
none; it is additionally broken below lq=256.

Two structural choices, both measured/verified rather than assumed:

**Cross-layer K/V is hoisted out of the denoise loop.** ODD layers re-project
the backbone's constant 241-token K/V cache into expert K/V space. lerobot
recomputes that on all 10 steps even though it cannot change; the oracle dump
asserts step-invariance. Hoisting is exactly equal, and removes 144 of 160 such
GEMMs. `hoist_cross_kv=False` restores the faithful per-step behaviour for A/B.

**The odd layers' fp32 k/v weights are cast to bf16 here, not in the loader.**
`expert_weights.load_expert_weights` stays faithful to the checkpoint (which
really does store those two tensors fp32 while everything else is bf16) so it
can keep serving the CPU oracle. The cast is a port-boundary decision and is
recorded in `deviations()`.

Run:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 expert_denoise.py --selftest
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from expert_fused_builders import (
    _expert_gemm_spec,
    build_expert_o_ffn_module,
    build_expert_rms_q_rope_module,
    build_expert_rms_qkv_rope_module,
    expert_gemm_specs,
)
from expert_weights import SmolVLAExpertConfig, load_expert_weights
from shared.infra.cache import KernelCache, Profiler
from shared.infra.external_kernels import (
    compile_gemm_mm,
    compile_masked_softmax,
    compile_rope,
    compile_silu_and_mul,
)

# The kernel's own sentinel, not a locally invented one:
# masked_softmax.py defines the value its microkernel treats as -inf.
from masked_softmax.masked_softmax import BF16_MIN


def _fused_backend(name):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _attn_backend(name="matmul_bf16"):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _softmax_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "masked_softmax",
        "runtime_loop_tiling_sizes": [4, 4],
    }


def rope_lut(positions, head_dim, base):
    """Concatenated half-split LUT [cos..., sin...] -- rope_halfsplit.cc's layout,
    identical to lerobot's apply_rope (verified to cos 0.9999978)."""
    half = head_dim // 2
    inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.asarray(positions, np.float64), inv)
    lut = np.empty((len(positions), head_dim), np.float64)
    lut[:, :half] = np.cos(ang)
    lut[:, half:] = np.sin(ang)
    return lut.astype(bfloat16)


def _build_additive_mask(bool_mask, seq_pad, kv_pad):
    """(seq_real, kv_real) boolean -> (seq_pad, kv_pad) additive bf16 mask.

    Three kinds of masked-out element, and the third is the one that bites:
      - real positions the model masks (padded language tokens; only 197 of the
        241 prefix tokens are real) -> BF16_MIN
      - padded K/V columns beyond kv_real                          -> BF16_MIN
      - padded query rows beyond seq_real: their output is discarded, but a row
        of all-BF16_MIN makes softmax divide by zero and yields NaN, which then
        propagates through the P@V GEMM into REAL rows via the shared dispatch.
        So pad rows are given exactly one valid column.
    """
    seq_real, kv_real = bool_mask.shape
    add = np.full((seq_pad, kv_pad), BF16_MIN, dtype=np.float32)
    add[:seq_real, :kv_real] = np.where(bool_mask, 0.0, BF16_MIN)
    add[seq_real:, 0] = 0.0  # keep padded rows finite
    return add


class ExpertRunner:
    """Compiles and drives the expert's ELFs. One instance per process."""

    def __init__(
        self,
        weights=None,
        config: SmolVLAExpertConfig = None,
        cache_dir="expert_kernel_cache",
        hoist_cross_kv=True,
        verbose=False,
    ):
        self.cfg = config or SmolVLAExpertConfig()
        self.w = weights
        self.hoist_cross_kv = hoist_cross_kv
        self.cache = KernelCache(cache_dir, verbose=verbose, profiler=Profiler())
        self.verbose = verbose
        c = self.cfg
        # Padding is a port concern, not a model property, so it lives here
        # rather than in SmolVLAExpertConfig (which mirrors the checkpoint).
        self.seq_real = c.chunk_size  # 50 action tokens
        self.seq_pad = 64  # tile_m=16 x herd_m=4; 50 is not tileable
        self.group = c.n_heads // c.n_kv_heads  # 3
        self.group_m = self.group * self.seq_pad  # 192
        self.kv_pad_self = 320  # 241 prefix + 50 action = 291 -> 320
        self.kv_pad_cross = 256  # 241 -> 256
        self._prefix = None
        self._cross_kv = None
        self._deviations = []

    # -- config helpers -----------------------------------------------------
    def deviations(self):
        """Numeric deviations from the CPU reference, for the write-up."""
        return list(self._deviations)

    # -- compilation --------------------------------------------------------
    def compile_all(self):
        c = self.cfg
        print("Staging external objects...")
        specs = expert_gemm_specs(
            seq_len=self.seq_pad,
            emb_dim=c.emb_dim,
            q_dim=c.q_dim,
            kv_dim=c.kv_dim,
            hidden_dim=c.hidden_dim,
            prefix_len_padded=self.kv_pad_cross,
        )
        staged = {}
        for s in specs.values():
            staged[s["obj"]] = s
        for obj, s in sorted(staged.items()):
            compile_gemm_mm(
                tile_m=s["tile_m"],
                tile_n=s["tile_n"],
                tile_k_l1=s["tile_k_l1"],
                sym_suffix=s["sym_suffix"],
                out_name=obj,
            )
        compile_rope()
        compile_silu_and_mul()

        print("Compiling fused layer ELFs...")
        for name, fn, kw in (
            ("expert_rms_qkv_rope", build_expert_rms_qkv_rope_module, {}),
            ("expert_rms_q_rope", build_expert_rms_q_rope_module, {}),
            ("expert_o_ffn", build_expert_o_ffn_module, {}),
        ):
            self.cache.compile_and_cache(
                name,
                (
                    fn(seq_len=self.seq_pad, emb_dim=c.emb_dim, q_dim=c.q_dim, **kw)
                    if name != "expert_o_ffn"
                    else fn(
                        seq_len=self.seq_pad,
                        emb_dim=c.emb_dim,
                        q_dim=c.q_dim,
                        hidden_dim=c.hidden_dim,
                    )
                ),
                _fused_backend(name),
            )

        print("Compiling cross-K/V projection GEMM...")
        self._compile_plain_gemm("kv_cross", specs["kv_cross"])

        print("Compiling attention kernels...")
        for tag, kv_pad in (("self", self.kv_pad_self), ("cross", self.kv_pad_cross)):
            self._compile_attention(tag, kv_pad)

        self.cache._save_manifest()
        print(f"All kernels cached to {self.cache.cache_dir}/")

    def _compile_plain_gemm(self, name, spec):
        """A standalone (non-fused) GEMM ELF from an expert spec."""
        compile_gemm_mm(
            tile_m=spec["tile_m"],
            tile_n=spec["tile_n"],
            tile_k_l1=spec["tile_k_l1"],
            sym_suffix=spec["sym_suffix"],
            out_name=spec["obj"],
        )
        self.cache.compile_and_cache(name, _gemm_module(spec), _attn_backend())

    def _compile_attention(self, tag, kv_pad):
        """qkt / pv / masked_softmax for one K/V length."""
        from masked_softmax.masked_softmax import build_module as build_ms

        c = self.cfg
        hd = c.head_dim
        qkt = _expert_gemm_spec(self.group_m, hd, kv_pad)
        pv = _expert_gemm_spec(self.group_m, kv_pad, hd)

        for nm, spec in ((f"qkt_{tag}", qkt), (f"pv_{tag}", pv)):
            compile_gemm_mm(
                tile_m=spec["tile_m"],
                tile_n=spec["tile_n"],
                tile_k_l1=spec["tile_k_l1"],
                sym_suffix=spec["sym_suffix"],
                out_name=spec["obj"],
            )
            self.cache.compile_and_cache(nm, _gemm_module(spec), _attn_backend())

        ms_n = c.n_heads * self.seq_pad * kv_pad
        assert ms_n % (kv_pad * 4) == 0, (ms_n, kv_pad)
        self.cache.compile_and_cache(
            f"masked_softmax_{tag}",
            build_ms(ms_n, kv_pad, 4, bfloat16),
            _softmax_backend(),
        )


def _gemm_module(spec):
    """A standalone (non-stitched) GEMM Module from an expert spec.

    Built directly rather than via `_gemm_ir` + re-parse: the stitcher needs the
    IR as text, a standalone ELF does not.
    """
    from shared.builders.gemm_builder import _build_gemm_module

    m, k, n = spec["shape"]
    herd_m, herd_n = spec["herd"]
    return _build_gemm_module(
        m,
        k,
        n,
        spec["tile_m"],
        spec["tile_k_l2"],
        spec["tile_k_l1"],
        spec["tile_n"],
        herd_m,
        herd_n,
        **spec["build_kwargs"],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    cfg = SmolVLAExpertConfig()
    r = ExpertRunner(config=cfg, verbose=False)
    print(
        f"SmolVLA action expert on NPU2: {cfg.n_layers} layers, "
        f"emb {cfg.emb_dim}, q_dim {cfg.q_dim}, kv {cfg.kv_dim}, "
        f"hidden {cfg.hidden_dim}, seq {r.seq_real}->{r.seq_pad}, "
        f"KV self 291->{r.kv_pad_self} / cross 241->{r.kv_pad_cross}"
    )
    r.compile_all()

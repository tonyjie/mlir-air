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

    # -- per-inference setup (the hoisted work) ------------------------------
    def set_prefix(
        self, prefix_k, prefix_v, mask_self, mask_cross, pos_self, pos_cross
    ):
        """Everything that depends on the prefix but NOT on the denoise step.

        Called once per inference. This is where the cross-attention K/V
        re-projection is hoisted to: it reads only the backbone's constant
        241-token cache, so lerobot recomputing it on all 10 steps is pure
        waste (the oracle dump asserts step-invariance). 16 GEMMs instead of
        160. `hoist_cross_kv=False` defers them back into run_layer.

        prefix_k/prefix_v: (n_layers, 241, n_kv, head_dim). Already RoPE'd --
        the backbone stored its cache post-RoPE, and SELF layers consume those
        241 columns verbatim. CROSS layers re-project them and apply NO RoPE.
        """
        c = self.cfg
        self._prefix = dict(
            k=np.asarray(prefix_k, np.float32),
            v=np.asarray(prefix_v, np.float32),
            add_self=_build_additive_mask(mask_self, self.seq_pad, self.kv_pad_self),
            add_cross=_build_additive_mask(mask_cross, self.seq_pad, self.kv_pad_cross),
            lut_self=self._lut(pos_self),
            lut_cross=self._lut(pos_cross),
        )
        self._cross_kv = {}
        if self.hoist_cross_kv:
            for li in range(c.n_layers):
                if not c.is_self_attn(li):
                    self._cross_kv[li] = self._project_cross_kv(li)

    def _lut(self, positions):
        """RoPE LUT for the padded sequence. Positions past seq_real are
        continued arithmetically; those rows are masked out anyway."""
        p = np.asarray(positions, np.int64)
        if len(p) < self.seq_pad:
            step = int(p[-1] - p[-2]) if len(p) > 1 else 1
            tail = p[-1] + step * np.arange(1, self.seq_pad - len(p) + 1)
            p = np.concatenate([p, tail])
        return rope_lut(p[: self.seq_pad], self.cfg.head_dim, self.cfg.rope_base)

    def _project_cross_kv(self, li):
        """241x320 backbone cache -> expert K/V space, on NPU (256x320x320).

        Two BO hazards live here, and sharing one `bo_key` between the K and the
        V projection trips both at once (it cost a debugging round):

        - `static_input_indices` uploads a buffer only on the FIRST call for a
          given bo_key. K and V sharing a key means the V call silently reuses
          the K weight and never uploads its own.
        - `load_and_run` returns **zero-copy views** onto the output BO. Sharing
          a key means K's returned view aliases the buffer V then overwrites, so
          by the time this function returns both names point at V's result.

        Hence: a distinct bo_key per (layer, projection), and an explicit copy
        of anything that outlives the next dispatch on the same buffers.
        """
        c = self.cfg
        lw = self.w.layers[li]
        n_p = self._prefix["k"].shape[1]
        out = []
        for which, cache_arr, wt in (
            ("k", self._prefix["k"][li], lw.wk_cross),
            ("v", self._prefix["v"][li], lw.wv_cross),
        ):
            src = np.zeros((self.kv_pad_cross, c.kv_dim), bfloat16)
            src[:n_p] = cache_arr.reshape(n_p, c.kv_dim).astype(bfloat16)
            res = self.cache.load_and_run(
                "kv_cross",
                _attn_backend(),
                src,
                np.asarray(wt, bfloat16),
                np.zeros((self.kv_pad_cross, c.kv_dim), bfloat16),
                output_indices=[2],
                static_input_indices={1},
                bo_key=f"kv_cross_{which}_L{li}",
            )
            out.append(
                res[2].reshape(self.kv_pad_cross, c.n_kv_heads, c.head_dim).copy()
            )
        return out[0], out[1]

    # -- attention -----------------------------------------------------------
    def _attention(self, q_roped, k_h, v_h, add_mask, tag, kv_pad):
        """GQA-group-batched decomposed attention: 5 qkt + 1 softmax + 5 pv.

        Mirrors the backbone's validated `_npu_attention`, with one difference
        that matters: there Q-seq == K-seq, here Q is 64 (50 padded) and K/V is
        320 or 256, so qkt is (192 x 64 x kv_pad) and pv is (192 x kv_pad x 64).
        """
        c = self.cfg
        hd, nh, nkv = c.head_dim, c.n_heads, c.n_kv_heads
        g, gm = self.group, self.group_m
        scale = 1.0 / np.sqrt(hd)

        q_h = np.asarray(q_roped, np.float32).reshape(self.seq_pad, nh, hd)

        scores = np.empty((nh, self.seq_pad, kv_pad), dtype=bfloat16)
        for kv in range(nkv):
            h0 = kv * g
            Q_group = np.concatenate(
                [q_h[:, h0 + j, :] * scale for j in range(g)], axis=0
            ).astype(bfloat16)
            Kh_T = np.ascontiguousarray(k_h[:, kv, :].T).astype(bfloat16)
            res = self.cache.load_and_run(
                f"qkt_{tag}",
                _attn_backend(),
                Q_group,
                Kh_T,
                np.zeros((gm, kv_pad), bfloat16),
                output_indices=[2],
                bo_key=f"qkt_{tag}",
            )
            S = res[2].reshape(gm, kv_pad)
            for j in range(g):
                scores[h0 + j] = S[j * self.seq_pad : (j + 1) * self.seq_pad]

        # Fold the additive mask on host (head-independent) and softmax all
        # heads in one dispatch; the kernel's own mask input stays zero.
        ms_in = (scores.astype(np.float32) + add_mask[None]).astype(bfloat16)
        n = nh * self.seq_pad * kv_pad
        res = self.cache.load_and_run(
            f"masked_softmax_{tag}",
            _softmax_backend(),
            ms_in.reshape(-1),
            np.zeros(n, bfloat16),
            np.zeros(n, bfloat16),
            output_indices=[2],
            intermediate_indices={1},
            bo_key=f"ms_{tag}",
        )
        probs = res[2].reshape(nh, self.seq_pad, kv_pad).copy()

        att = np.empty((self.seq_pad, nh * hd), dtype=bfloat16)
        for kv in range(nkv):
            h0 = kv * g
            P_group = np.concatenate(
                [np.ascontiguousarray(probs[h0 + j]) for j in range(g)], axis=0
            ).astype(bfloat16)
            Vh = np.ascontiguousarray(v_h[:, kv, :]).astype(bfloat16)
            res = self.cache.load_and_run(
                f"pv_{tag}",
                _attn_backend(),
                P_group,
                Vh,
                np.zeros((gm, hd), bfloat16),
                output_indices=[2],
                bo_key=f"pv_{tag}",
            )
            O = res[2].reshape(gm, hd)
            for j in range(g):
                att[:, (h0 + j) * hd : (h0 + j + 1) * hd] = O[
                    j * self.seq_pad : (j + 1) * self.seq_pad
                ]
        return att

    # -- one layer -----------------------------------------------------------
    def run_layer(self, x_pad, li):
        """13 dispatches. x_pad (seq_pad, emb_dim) bf16 -> same."""
        c = self.cfg
        lw = self.w.layers[li]
        is_self = c.is_self_attn(li)
        pf = self._prefix
        n_p = pf["k"].shape[1]

        if is_self:
            res = self.cache.load_and_run(
                "expert_rms_qkv_rope",
                _fused_backend("expert_rms_qkv_rope"),
                x_pad,
                np.asarray(lw.attn_norm, bfloat16),
                np.zeros((self.seq_pad, c.emb_dim), bfloat16),
                np.asarray(lw.wq, bfloat16),
                np.zeros((self.seq_pad, c.q_dim), bfloat16),
                np.asarray(lw.wk, bfloat16),
                np.zeros((self.seq_pad, c.kv_dim), bfloat16),
                np.asarray(lw.wv, bfloat16),
                np.zeros((self.seq_pad, c.kv_dim), bfloat16),
                np.repeat(pf["lut_self"], c.n_heads, axis=0).flatten(),
                np.repeat(pf["lut_self"], c.n_kv_heads, axis=0).flatten(),
                np.zeros((self.seq_pad, c.q_dim), bfloat16),
                np.zeros((self.seq_pad, c.kv_dim), bfloat16),
                output_indices=[8, 11, 12],
                static_input_indices={1, 3, 5, 7, 9, 10},
                intermediate_indices={2, 4, 6, 8, 11, 12},
                bo_key=f"ex_even_L{li}",
            )
            # .copy() out of the zero-copy BO views before any further
            # dispatch can touch those buffers (see _project_cross_kv).
            v_new = res[8].reshape(self.seq_pad, c.n_kv_heads, c.head_dim).copy()
            q_roped = res[11].reshape(self.seq_pad, c.q_dim).copy()
            k_new = res[12].reshape(self.seq_pad, c.n_kv_heads, c.head_dim).copy()

            # K/V = [241 prefix (already RoPE'd, verbatim) | 50 new] -> pad 320
            k_h = np.zeros((self.kv_pad_self, c.n_kv_heads, c.head_dim), bfloat16)
            v_h = np.zeros_like(k_h)
            k_h[:n_p] = pf["k"][li].astype(bfloat16)
            v_h[:n_p] = pf["v"][li].astype(bfloat16)
            k_h[n_p : n_p + self.seq_real] = k_new[: self.seq_real]
            v_h[n_p : n_p + self.seq_real] = v_new[: self.seq_real]
            att = self._attention(
                q_roped, k_h, v_h, pf["add_self"], "self", self.kv_pad_self
            )
        else:
            res = self.cache.load_and_run(
                "expert_rms_q_rope",
                _fused_backend("expert_rms_q_rope"),
                x_pad,
                np.asarray(lw.attn_norm, bfloat16),
                np.zeros((self.seq_pad, c.emb_dim), bfloat16),
                np.asarray(lw.wq, bfloat16),
                np.zeros((self.seq_pad, c.q_dim), bfloat16),
                np.repeat(pf["lut_cross"], c.n_heads, axis=0).flatten(),
                np.zeros((self.seq_pad, c.q_dim), bfloat16),
                output_indices=[6],
                static_input_indices={1, 3, 5},
                intermediate_indices={2, 4, 6},
                bo_key=f"ex_odd_L{li}",
            )
            q_roped = res[6].reshape(self.seq_pad, c.q_dim).copy()
            k_h, v_h = (
                self._cross_kv[li]
                if self.hoist_cross_kv
                else self._project_cross_kv(li)
            )
            att = self._attention(
                q_roped, k_h, v_h, pf["add_cross"], "cross", self.kv_pad_cross
            )

        res = self.cache.load_and_run(
            "expert_o_ffn",
            _fused_backend("expert_o_ffn"),
            np.asarray(att, bfloat16),
            np.asarray(lw.wo, bfloat16),
            np.zeros((self.seq_pad, c.emb_dim), bfloat16),
            x_pad,
            np.zeros((self.seq_pad, c.emb_dim), bfloat16),
            np.asarray(lw.ffn_norm, bfloat16),
            np.zeros((self.seq_pad, c.emb_dim), bfloat16),
            np.asarray(lw.w_gate, bfloat16),
            np.zeros((self.seq_pad, c.hidden_dim), bfloat16),
            np.asarray(lw.w_up, bfloat16),
            np.zeros((self.seq_pad, c.hidden_dim), bfloat16),
            np.zeros((self.seq_pad, c.hidden_dim), bfloat16),
            np.asarray(lw.w_down, bfloat16),
            np.zeros((self.seq_pad, c.emb_dim), bfloat16),
            np.zeros((self.seq_pad, c.emb_dim), bfloat16),
            output_indices=[14],
            static_input_indices={1, 5, 7, 9, 12},
            intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
            bo_key=f"ex_offn_L{li}",
        )
        return res[14].reshape(self.seq_pad, c.emb_dim).copy()

    # -- one denoise step ----------------------------------------------------
    def run_step(self, suffix_emb):
        """(seq_real, emb_dim) -> (seq_real, emb_dim) after the final RMSNorm.

        The final `lm_expert.norm` runs on CPU: one (50,720) RMSNorm outside the
        per-layer loop, 0.58 ms/inference (0.2% of the expert). Same choice
        every llama/qwen sibling makes for its final norm, and the same one the
        SmolVLA backbone port made -- an extra dispatch would buy nothing.
        """
        c = self.cfg
        x = np.zeros((self.seq_pad, c.emb_dim), bfloat16)
        x[: self.seq_real] = np.asarray(suffix_emb, bfloat16)[: self.seq_real]
        for li in range(c.n_layers):
            x = self.run_layer(x, li)
        real = np.asarray(x[: self.seq_real], np.float32)
        w = np.asarray(self.w.final_norm, np.float32)
        rstd = 1.0 / np.sqrt((real**2).mean(-1, keepdims=True) + c.rms_norm_eps)
        return real * rstd * w

    # -- compilation --------------------------------------------------------
    KERNELS = (
        "expert_rms_qkv_rope",
        "expert_rms_q_rope",
        "expert_o_ffn",
        "kv_cross",
        "qkt_self",
        "pv_self",
        "masked_softmax_self",
        "qkt_cross",
        "pv_cross",
        "masked_softmax_cross",
    )

    def ensure_kernels(self):
        """Reuse the on-disk ELF cache when it is COMPLETE, else rebuild.

        Uses the project's own helper so the "partial cache still triggers a
        full rebuild" rule is shared rather than re-implemented.
        SMOLVLA_FORCE_COMPILE=1 forces a rebuild -- needed after editing any
        builder, since the manifest does not track source hashes.
        """
        from bridge_common import ensure_kernels as _ensure

        return _ensure(self.cache, self.KERNELS, self.compile_all, tag="expert")

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

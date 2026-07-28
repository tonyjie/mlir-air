# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Action-Expert Weight Loader

Loads the SmolVLA (`lerobot/smolvla_base`) action-expert weights from
HuggingFace safetensors and provides them as numpy arrays for MLIR-AIR kernel
invocations. Sibling of `smolvla_backbone_weights.py`, which loads the
*backbone* (`...vlm.model.text_model.*`); this one loads only

    model.vlm_with_expert.lm_expert.{norm,layers.N.*}
    model.{action_in_proj,action_time_mlp_in,action_time_mlp_out,action_out_proj}

The expert is a 16-layer Llama-topology transformer at hidden 720 /
intermediate 2048 / GQA 15 q-heads / 5 kv-heads / head_dim 64. Two things make
it structurally different from every backbone this repo has ported so far:

1. q_dim (15*64 = 960) != hidden (720). o_proj is therefore (960 -> 720) and
   the residual stream is 720 wide while attention runs at 960.
2. Alternating attention type (`self_attn_every_n_layers=2`): EVEN layers
   self-attend over 291 = 241 prefix + 50 action tokens; ODD layers cross-attend,
   and their k_proj/v_proj were REPLACED at construction time (see
   SmolVLMWithExpertModel.__init__) by fresh `nn.Linear(320, 320)` that
   re-project the backbone's cached 241-token K/V. Those two matrices are the
   only **fp32** weights in the expert -- everything else is bf16. They are
   exposed as `wk_cross` / `wv_cross` and are None on even layers, while
   `wk` / `wv` (720 -> 320) are None on odd layers.

Weight convention (same as the backbone loader): HF stores linears as
(out, in); we transpose to (in, out) for y = x @ W.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from ml_dtypes import bfloat16

from smolvla_backbone_weights import _load_tensor, _resolve_safetensor_files

_EXPERT_PREFIX = "model.vlm_with_expert.lm_expert."
_HEAD_PREFIX = "model."


@dataclass
class SmolVLAExpertConfig:
    """SmolVLA action-expert hyperparameters (read off the live checkpoint)."""

    n_layers: int = 16
    emb_dim: int = 720  # residual stream width
    n_heads: int = 15
    head_dim: int = 64
    n_kv_heads: int = 5  # GQA group of 3
    hidden_dim: int = 2048  # MLP intermediate
    # RoPE base: smolvlm_with_expert.apply_rope is called WITHOUT max_wavelength
    # on both the self and the cross path, so the effective base is its default
    # 10000 -- not the HF text config's 100000. Same trap as the backbone.
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    self_attn_every_n_layers: int = 2
    chunk_size: int = 50  # action tokens
    num_steps: int = 10  # denoise iterations
    prefix_len: int = 241  # backbone KV-cache length
    action_dim: int = 32  # max_action_dim
    dtype: Any = bfloat16

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim  # 960

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim  # 320

    def is_self_attn(self, layer_idx: int) -> bool:
        return layer_idx % self.self_attn_every_n_layers == 0


@dataclass
class ExpertLayerWeights:
    """One expert layer, y = x @ W convention.

    wk/wv are set on SELF (even) layers only; wk_cross/wv_cross on CROSS (odd)
    layers only. The unused pair is None -- a shape mismatch would otherwise be
    silently absorbed by a wrong-path GEMM.
    """

    attn_norm: np.ndarray  # (720,)
    wq: np.ndarray  # (720, 960)
    wo: np.ndarray  # (960, 720)
    ffn_norm: np.ndarray  # (720,)
    w_gate: np.ndarray  # (720, 2048)
    w_up: np.ndarray  # (720, 2048)
    w_down: np.ndarray  # (2048, 720)
    wk: Optional[np.ndarray] = None  # (720, 320)  self layers
    wv: Optional[np.ndarray] = None  # (720, 320)  self layers
    wk_cross: Optional[np.ndarray] = None  # (320, 320) fp32, cross layers
    wv_cross: Optional[np.ndarray] = None  # (320, 320) fp32, cross layers


@dataclass
class ExpertWeights:
    """Expert stack plus the action-head projections that wrap it.

    action_* are the fp32 heads owned by VLAFlowMatching (not by the expert
    module): embed_suffix uses in/time_mlp_in/time_mlp_out, and denoise_step
    turns the expert's final-norm output into v_t via out_proj.
    """

    layers: List[ExpertLayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (720,)
    action_in_w: np.ndarray = None  # (32, 720)
    action_in_b: np.ndarray = None  # (720,)
    action_time_in_w: np.ndarray = None  # (1440, 720)
    action_time_in_b: np.ndarray = None  # (720,)
    action_time_out_w: np.ndarray = None  # (720, 720)
    action_time_out_b: np.ndarray = None  # (720,)
    action_out_w: np.ndarray = None  # (720, 32)
    action_out_b: np.ndarray = None  # (32,)


def _check(name, arr, expected):
    if arr.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {arr.shape}")


def load_expert_weights(
    model_name_or_path: str = "lerobot/smolvla_base",
    dtype=bfloat16,
    config: Optional[SmolVLAExpertConfig] = None,
) -> ExpertWeights:
    """Load the action-expert weights from safetensors.

    `dtype` applies to the expert stack. The cross-layer k/v projections and
    the action heads are kept in fp32 -- that is their storage dtype in the
    checkpoint, and the real model runs them in fp32, so downcasting them here
    would inject error the reference is supposed to be free of.
    """
    from safetensors import safe_open

    cfg = config or SmolVLAExpertConfig()
    key_to_file = {}
    for path in _resolve_safetensor_files(model_name_or_path):
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = path

    def get(key, as_dtype, transpose):
        if key not in key_to_file:
            raise KeyError(f"Missing weight: {key}")
        with safe_open(key_to_file[key], framework="numpy") as f:
            t = _load_tensor(f, key, as_dtype)
        return np.ascontiguousarray(t.T) if transpose else t

    layers = []
    for i in range(cfg.n_layers):
        p = f"{_EXPERT_PREFIX}layers.{i}."
        lw = ExpertLayerWeights(
            attn_norm=get(p + "input_layernorm.weight", dtype, False),
            wq=get(p + "self_attn.q_proj.weight", dtype, True),
            wo=get(p + "self_attn.o_proj.weight", dtype, True),
            ffn_norm=get(p + "post_attention_layernorm.weight", dtype, False),
            w_gate=get(p + "mlp.gate_proj.weight", dtype, True),
            w_up=get(p + "mlp.up_proj.weight", dtype, True),
            w_down=get(p + "mlp.down_proj.weight", dtype, True),
        )
        if cfg.is_self_attn(i):
            lw.wk = get(p + "self_attn.k_proj.weight", dtype, True)
            lw.wv = get(p + "self_attn.v_proj.weight", dtype, True)
            _check(f"L{i} wk", lw.wk, (cfg.emb_dim, cfg.kv_dim))
            _check(f"L{i} wv", lw.wv, (cfg.emb_dim, cfg.kv_dim))
        else:
            lw.wk_cross = get(p + "self_attn.k_proj.weight", np.float32, True)
            lw.wv_cross = get(p + "self_attn.v_proj.weight", np.float32, True)
            _check(f"L{i} wk_cross", lw.wk_cross, (cfg.kv_dim, cfg.kv_dim))
            _check(f"L{i} wv_cross", lw.wv_cross, (cfg.kv_dim, cfg.kv_dim))
        _check(f"L{i} attn_norm", lw.attn_norm, (cfg.emb_dim,))
        _check(f"L{i} wq", lw.wq, (cfg.emb_dim, cfg.q_dim))
        _check(f"L{i} wo", lw.wo, (cfg.q_dim, cfg.emb_dim))
        _check(f"L{i} w_gate", lw.w_gate, (cfg.emb_dim, cfg.hidden_dim))
        _check(f"L{i} w_up", lw.w_up, (cfg.emb_dim, cfg.hidden_dim))
        _check(f"L{i} w_down", lw.w_down, (cfg.hidden_dim, cfg.emb_dim))
        layers.append(lw)

    final_norm = get(_EXPERT_PREFIX + "norm.weight", dtype, False)
    _check("final_norm", final_norm, (cfg.emb_dim,))

    h = _HEAD_PREFIX
    w = ExpertWeights(
        layers=layers,
        final_norm=final_norm,
        action_in_w=get(h + "action_in_proj.weight", np.float32, True),
        action_in_b=get(h + "action_in_proj.bias", np.float32, False),
        action_time_in_w=get(h + "action_time_mlp_in.weight", np.float32, True),
        action_time_in_b=get(h + "action_time_mlp_in.bias", np.float32, False),
        action_time_out_w=get(h + "action_time_mlp_out.weight", np.float32, True),
        action_time_out_b=get(h + "action_time_mlp_out.bias", np.float32, False),
        action_out_w=get(h + "action_out_proj.weight", np.float32, True),
        action_out_b=get(h + "action_out_proj.bias", np.float32, False),
    )
    _check("action_in_w", w.action_in_w, (cfg.action_dim, cfg.emb_dim))
    _check("action_time_in_w", w.action_time_in_w, (2 * cfg.emb_dim, cfg.emb_dim))
    _check("action_out_w", w.action_out_w, (cfg.emb_dim, cfg.action_dim))
    return w

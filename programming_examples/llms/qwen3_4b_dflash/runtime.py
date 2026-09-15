# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Small host API for prompt preparation and the selected NPU DFlash pair."""

from __future__ import annotations

import time

import numpy as np
from ml_dtypes import bfloat16

from context_prepass import ContextPrepass
from draft_weights import DraftWeights
from model_paths import (
    TARGET_MODEL,
    load_tokenizer,
    TOKENIZER_MODEL,
    default_draft_cache,
    default_draft_snapshot,
    default_target_cache,
)
from prefill_reference import forward_prompt_with_taps
from selected_config import require_selected
from speculative import DFlash
from target_runtime import TargetDecoder


def load_pair(
    target_artifacts,
    draft_artifacts,
    *,
    target_model=TARGET_MODEL,
    target_cache=None,
    draft_cache=None,
    draft_snapshot=None,
):
    target_cache = target_cache or default_target_cache()
    draft_cache = draft_cache or default_draft_cache()
    draft_snapshot = draft_snapshot or default_draft_snapshot()
    target = TargetDecoder(
        target_artifacts,
        packed_cache=target_cache,
        model_source=target_model,
    )
    require_selected(target.config, "target")
    weights = DraftWeights(target_model=target.model, snapshot=draft_snapshot)
    drafter = TargetDecoder(
        draft_artifacts,
        packed_cache=draft_cache,
        model_weights=weights,
    )
    require_selected(drafter.config, "drafter")
    prepass = ContextPrepass(weights=weights, packed_cache=draft_cache)
    return DFlash(target, drafter, prepass, weights.tap_slots, weights.mask_token_id)


def tokenize_prompt(prompt: str, tokenizer_model=TOKENIZER_MODEL, *, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer, tokenizer.encode(text, add_special_tokens=False)


class PrefillWeights:
    """Keep the existing CPU reference's decoded weights resident for chat.

    The weight reader normally dequantizes each layer on every prompt. This
    adapter eagerly stores the exact same arrays once, before the first input.
    It changes preparation cost and memory use, not prefill arithmetic.
    """

    def __init__(self, target: TargetDecoder):
        model = target.model
        self._embed = model.embed_norm_lmhead()
        self._layers = [model.layer_weights(i) for i in range(target.layers)]
        self._rms = [model.layer_rms(i) for i in range(target.layers)]
        self._qk = [model.layer_qk_norm(i) for i in range(target.layers)]
        arrays = list(self._embed)
        arrays += [w for layer in self._layers for w in layer.values()]
        arrays += [w for pair in self._rms + self._qk for w in pair]
        self.nbytes = sum(w.nbytes for w in {id(w): w for w in arrays}.values())

    def embed_norm_lmhead(self):
        return self._embed

    def layer_weights(self, layer):
        return self._layers[layer]

    def layer_rms(self, layer):
        return self._rms[layer]

    def layer_qk_norm(self, layer):
        return self._qk[layer]


def prepare_prompt(target: TargetDecoder, input_ids: list[int], tap_slots: list[int], *, prefill_model=None):
    started = time.perf_counter()
    model = target.model if prefill_model is None else prefill_model
    k, v, logits, boundaries = forward_prompt_with_taps(model, input_ids, tap_slots)
    taps = np.stack([boundaries[index] for index in tap_slots], axis=1).reshape(len(input_ids), -1)
    result = {
        "input_ids": list(input_ids),
        "k": k.astype(bfloat16),
        "v": v.astype(bfloat16),
        "taps": taps.astype(bfloat16),
        "first": int(logits[-1].argmax()),
    }
    return result, time.perf_counter() - started

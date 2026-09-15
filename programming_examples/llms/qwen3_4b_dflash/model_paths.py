# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Portable model and packed-cache defaults for the DFlash example."""

from __future__ import annotations

import os
from pathlib import Path


TARGET_MODEL = os.environ.get("DFLASH_TARGET_MODEL", "FastFlowLM/Qwen3-4B-NPU2")
TOKENIZER_MODEL = os.environ.get("DFLASH_TOKENIZER_MODEL", "Qwen/Qwen3-4B")
DRAFT_MODEL = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_REVISION = "b74e3a329c4d963783143b1e970d95b002be72bd"


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_CACHE_DIR",
            Path.home() / ".cache/qwen3_4b_dflash_npu",
        )
    ).expanduser()


def default_target_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_TARGET_CACHE",
            Path.home() / ".cache/q4nx_qwen3_4b/requant_w2ch.npz",
        )
    ).expanduser()


def default_draft_cache() -> Path:
    return Path(
        os.environ.get(
            "DFLASH_DRAFT_CACHE",
            cache_root() / "draft_requant_w2ch.npz",
        )
    ).expanduser()


def default_draft_snapshot() -> Path:
    configured = os.environ.get("DFLASH_DRAFT_SNAPSHOT")
    if configured:
        return Path(configured).expanduser()
    return (
        Path.home()
        / ".cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots"
        / DRAFT_REVISION
    )

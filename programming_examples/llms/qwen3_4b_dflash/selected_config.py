# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The one validated NPU configuration delivered by this example."""

from __future__ import annotations

from dataclasses import asdict

from target_config import TargetConfig


CAPACITY = 4096
BATCH = 8
TARGET_LAYERS = 36
DRAFT_LAYERS = 5


def selected_config(role: str) -> TargetConfig:
    if role not in ("target", "drafter"):
        raise ValueError("role must be target or drafter")
    return TargetConfig(
        role=role,
        capacity=CAPACITY,
        batch=BATCH,
        layers=TARGET_LAYERS if role == "target" else DRAFT_LAYERS,
        taps=True,
        causal=role == "target",
        stack_bytes=6080,
        l2_pingpong=False,
        chain_pingpong=True,
        activation_memtile_col=4,
        gather_mode="packet",
        token_packets=True,
        dynamic_context=True,
        projection_weight_pingpong=True,
        attention_query_tile=4,
        attention_layout="fused-shards",
        normalization_replay=1,
        normalization_cache=True,
        attention_pair_mode=3,
        vector_attention_norm=True,
        attention_optimization="pv",
        q4k_unpack_fma=False,
        q4k_unpack_unroll=0,
    )


def require_selected(config: TargetConfig, role: str) -> None:
    expected = asdict(selected_config(role))
    observed = asdict(config)
    if observed != expected:
        differences = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if expected[key] != observed[key]
        }
        raise ValueError(f"Artifact is not the selected {role} configuration: {differences}")


def baseline_config() -> TargetConfig:
    """Historical C4096 B1 GEMV performance denominator, without hidden taps."""
    return TargetConfig(capacity=4096, batch=1, layers=36, taps=False,
                        stack_bytes=10240, dynamic_context=True,
                        vector_attention_norm=True)

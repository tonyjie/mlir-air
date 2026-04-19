# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness metrics for Phase 2 / 3 gates.

LESSON 1 (llama32_3b deployment, 2026-04-18): the per-position cosine
threshold scales with head_dim because BF16 accumulation noise grows with
sqrt(head_dim) * sqrt(K). Use `head_dim_scaled_per_pos_threshold(head_dim)`
to pick the right gate value per model.
"""

import numpy as np


def cosine_sim(a, b):
    """Whole-tensor cosine similarity (≈ Pearson correlation for ~zero-mean data).

    Range [-1, 1]; 1 = perfect alignment. The Phase 2 gate is > 0.99.
    """
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    return float(
        np.dot(a_flat, b_flat)
        / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )


# Alias: phase3 tests historically called this `_whole_cosine`.
whole_cosine = cosine_sim


def mae(a, b):
    """Mean absolute error. Informational only; NOT a gate (LESSON 1).

    Use for diagnostic comparison vs the reference deployment's measured MAE.
    Production BF16-output GEMMs land at ~0.005-0.025 per single block.
    """
    return float(
        np.mean(
            np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
        )
    )


def per_pos_cosine_min(a, b):
    """Minimum per-position (per-row) cosine sim across the seq_len axis.

    Gates the Phase 2 "no per-row dropout" check. Computed as cosine for each
    token row (over the feature dim), then take the min across all rows.
    """
    a2 = np.asarray(a, dtype=np.float32).reshape(a.shape[0], -1)
    b2 = np.asarray(b, dtype=np.float32).reshape(b.shape[0], -1)
    cos = (a2 * b2).sum(axis=-1) / (
        np.linalg.norm(a2, axis=-1) * np.linalg.norm(b2, axis=-1) + 1e-12
    )
    return float(cos.min())


def head_dim_scaled_per_pos_threshold(head_dim):
    """Phase 2 per-position cosine threshold, scaled by head_dim per LESSON 1.

    BF16 accumulation noise scales with sqrt(head_dim) * sqrt(K). Empirical
    thresholds from the 3 reference deployments:
      head_dim=64  (llama3-1B, smollm2): per-pos min ~0.998     → gate 0.99
      head_dim=128 (llama32_3b):         per-pos min ~0.980-0.99 → gate 0.98
      head_dim=256 (extrapolated for future):                    → gate 0.97
    """
    if head_dim <= 64:
        return 0.99
    if head_dim <= 128:
        return 0.98
    return 0.97

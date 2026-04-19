# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical prompt set for Phase 3 full-model correctness gate.

LESSON 2 (llama32_3b deployment, 2026-04-18): for deep models (n_layers ≥ 24)
or wide models (head_dim ≥ 128), BF16 accumulation reorders close-prob top
tokens. The Phase 3 gate splits prompts into:

- DECISIVE: CPU top-1 prob > 0.5 → strict NPU top-1 == CPU top-1 match required
- COMPETITIVE: CPU top-1 prob ≤ 0.5 → top-5 overlap (CPU top-1 ∈ NPU top-5
  AND NPU top-1 ∈ CPU top-5) — accepts BF16 reorder of tied head tokens

Use these as defaults; override via `--prompts` for instruct/chat models.
"""

# CPU top-1 prob > 0.5 with the LlamaForCausalLM-family base models tested:
DECISIVE_PROMPTS = [
    "1 + 1 =",  # CPU top-1 ' ', p ≈ 0.74 (numeric-completion)
    "2 + 2 =",  # CPU top-1 ' ', p ≈ 0.53
    "Water freezes at",  # CPU top-1 ' ', p ≈ 0.71
    "The largest ocean is the",  # CPU top-1 ' Pacific', p ≈ 0.82
]

# CPU top-1 prob ≤ 0.5 — multiple plausible continuations within close probs:
COMPETITIVE_PROMPTS = [
    "The capital of France is",  # CPU top-1 ' Paris' p≈0.25, top-2 ' the' p≈0.14
    "The sky is",  # CPU top-1 ' the' p≈0.33, top-2 ' falling' p≈0.07
]

# Combined canonical set used by phase3 tests
CANONICAL_PROMPTS = DECISIVE_PROMPTS + COMPETITIVE_PROMPTS

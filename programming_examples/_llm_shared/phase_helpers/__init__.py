# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for LLM deployment phase tests + inference runners.

These modules consolidate code that's identical across LlamaForCausalLM-class
deployments (llama3, smollm2_1_7b, llama32_3b). Each helper takes the
deployment's CPU reference module (`<model>_reference`) and config as
parameters, so the same helper code drives any compatible deployment.

Module map:
- metrics.py         : cosine, MAE, per-position cosine, head_dim-scaled threshold
- decode_setup.py    : decode-side preload, KV-cache seeding, NPU LM Head GEMV
- prefill_runner.py  : prefill orchestration (cold/warm/preload, KV extraction)
- canonical_prompts.py : LESSON 2 canonical prompt set (decisive + competitive)
- headfirst_fa.py    : Option C head-first FA wrapper for head_dim ≥ 128
                       (works around upstream seq-first dk_chunks > 1 hang)

Usage pattern from a per-model phase test:
    from llama32_3b_weights import LlamaConfig, load_weights, generate_rope_lut
    import llama32_3b_reference  # the model-specific CPU forward pass
    from _llm_shared.phase_helpers import metrics, decode_setup, prefill_runner

    config = LlamaConfig()
    weights = load_weights(model_path, config=config)
    # ... helper calls take config + ref_module as args
"""

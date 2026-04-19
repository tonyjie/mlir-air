# `_llm_shared/` — Shared Infrastructure for LLM Deployments

This directory contains the kernel builders, host helpers, and orchestration
code that every LLM deployment in `programming_examples/` reuses. If you're
starting a new LlamaForCausalLM-class deployment on NPU2, **invoke
`/deploy-new-llm <hf_model_id>` from Claude Code** (the skill walks the
7-phase deployment workflow with gates) rather than rolling your own.

## Module map

```
_llm_shared/
├── kernel_builder/                    # NPU kernel compilation
│   ├── external_kernels.py            # compile_all_external_kernels(head_dim);
│   │                                    compile_attn_npu2(head_dim);
│   │                                    compile_attn_npu2_split(lqp, lkp,
│   │                                      dk, dv, num_q_tiles=4)  ← Option C
│   ├── rope_halfsplit.cc              # RoPE half-split AIE2P kernel
│   ├── ffn_swiglu/silu_and_mul.cc     # SwiGLU AIE2P kernel
│   ├── cache.py                       # KernelCache (compile-time cache)
│   ├── gemm_builder.py                # GEMM IR builder helpers
│   └── stitching.py                   # Multi-launch ELF MLIR stitching
├── phase_helpers/                     # Per-deployment phase test helpers
│   ├── metrics.py                     # cosine_sim, mae, per_pos_cosine_min,
│   │                                    head_dim_scaled_per_pos_threshold
│   ├── canonical_prompts.py           # DECISIVE_PROMPTS + COMPETITIVE_PROMPTS
│   │                                    (LESSON 2 Phase 3 set)
│   ├── orchestration.py               # compile_block_kernels,
│   │                                    preload_block_weights,
│   │                                    evaluate_prompt
│   ├── prefill_runner.py              # embed_and_pad, npu_full_prefill,
│   │                                    run_npu_full_prefill,
│   │                                    run_cpu_full_prefill,
│   │                                    npu_prefill_with_kv_extraction
│   ├── decode_setup.py                # pre_transpose_decode_weights,
│   │                                    npu_lm_head_gemv,
│   │                                    seed_kv_cache_via_cpu_prefill
│   └── headfirst_fa.py                # Option C head-first FA wrapper for
│                                        head_dim ≥ 128 (LESSON 3)
└── docs/                              # Shared pattern + kernel design docs
    │                                  # (cited by the skill chain;
    │                                  # see docs/README.md for the topic map)
    ├── README.md
    ├── explain.md                     # compilation pipeline overview
    ├── perf_optimization.md           # the 18.67s → 1.30s perf journey
    ├── kernels/                       # per-kernel design (GEMM, GEMV, RoPE,
    │                                  # RMSNorm, FlashAttn, SwiGLU, …)
    ├── multi-launch/                  # multi-launch ELF stitching patterns
    └── compiler_issues/               # MLIR-AIR compiler quirks + workarounds
```

## How a deployment uses this

A per-model deployment lives in
`programming_examples/<model>/` with this skeleton (~10 files including docs):

```
<model>/
├── <model>_weights.py        # Config dataclass + HF safetensors loader + RoPE LUT
├── <model>_reference.py      # CPU F32 reference forward pass
├── <model>_phase2_test.py    # ~100 LOC: imports config + ref_module + helpers
├── <model>_phase3_test.py    # ~120 LOC: same
├── <model>_phase4_test.py    # ~150 LOC: same
├── <model>_phase5_test.py    # ~150 LOC: same
├── <model>_inference.py      # ~200 LOC: end-to-end NPU runner
├── Makefile                  # llama3-style targets
├── CLAUDE.md, README.md, TODO.md
└── docs/development_progress/{progress,LESSONS,debug_log,phase{1..6}_*}.md
```

The phase scripts pass the model's CPU reference module (e.g.
`smollm2_reference` or `llama32_3b_reference`) into helper calls so the same
helper code drives any compatible deployment.

## Reference deployments

Three end-to-end validated deployments, in order of architectural divergence
from the original `llama3/`:

| Model | Layers | head_dim | n_kv_heads | vocab | Notable | Status |
|---|---|---|---|---|---|---|
| `llama3/`         | 16 | 64  | 8 (GQA g=4) | 128256 | original baseline (1.30 s prefill, 92 ms/tok) | reference |
| `smollm2_1_7b/`   | 24 | 64  | 32 (MHA)    |  49152 | MHA, smaller vocab, tied embeddings, rope_θ=130k | Tier-A |
| `llama32_3b/`     | 28 | **128** | 8 (GQA g=3) | 128256 | head_dim=128 (Option C FA), tied embeddings | Tier-A |

When in doubt, look at how `llama32_3b/` uses the helpers — it exercises the
full helper API including the head-first-FA wrapper.

## Key lessons baked into the helpers

- **LESSON 1** (`metrics.py`): per-position cosine threshold scales with head_dim
  (≤64 → 0.99; 128 → 0.98; ≥256 → 0.97). BF16 noise grows with `sqrt(head_dim) ·
  sqrt(K)`.
- **LESSON 2** (`canonical_prompts.py`, `orchestration.evaluate_prompt`):
  Phase 3 gate splits prompts into decisive (CPU top-1 p > 0.5 → strict
  top-1 match) and competitive (top-5 overlap). Per-layer cos > 0.95 demoted
  to informational for n_layers ≥ 24 OR head_dim ≥ 128.
- **LESSON 3** (`headfirst_fa.py`, `kernel_builder/external_kernels.py:
  compile_attn_npu2_split`): seq-first FA `dk_chunks > 1` is broken
  upstream; head-first FA + host transposes is the workaround
  (`install_headfirst_fa_wrapper()` + `compile_headfirst_fa_kernel(...)`).
  Per-tile flag conventions (`-Dlqp=LQP/NUM_Q_TILES`, `-Ddk=LKP`) are baked
  into `compile_attn_npu2_split`.

Full lesson writeups in
`programming_examples/llama32_3b/docs/development_progress/LESSONS.md`.

## When NOT to use these helpers

- **Non-LlamaForCausalLM architectures** (MoE, sliding-window attention, MLA,
  encoder-decoder, QKV-bias). The `deploy-new-llm` skill rejects these in
  Step 2; the helpers assume RMSNorm + SwiGLU + RoPE + GQA/MHA.
- **NPU1 (AIE2)**. The kernel sources are AIE2P-specific; `target_device='npu2'`
  is hardcoded throughout.
- **Non-BF16**. The helpers assume `bfloat16` weights and activations end-to-end.

# SmolVLA on NPU2 (mlir-air) — Architecture & Porting Feasibility

A feasibility study for deploying **SmolVLA** (a Vision-Language-Action model)
onto AMD **NPU2** (Strix, AIE2P) using the **mlir-air** compiler flow and the
existing `programming_examples/kernel_registry` kernel set.

> **Status:** research note, not yet an implementation. This document records
> what SmolVLA *is*, what operators it needs, and how much of that the registry
> already covers — so a port can start from a shape-accurate plan rather than
> guesswork.
>
> **Method:** synthesized from two independent deep-research passes (one on
> Claude Opus 4.8, one on Sonnet 4.6), each fan-out web search + 3-vote
> adversarial claim verification, cross-checked against the local
> `kernel_registry/supported_kernels.md`. Where the two passes disagreed, the
> disagreement is called out explicitly below (see *Contested: backbone size*).

---

## TL;DR

- **Language backbone port: HIGH feasibility.** SmolVLA's LLM is a Llama-class
  decoder (RMSNorm + RoPE + GQA attention + SwiGLU/SiLU MLP). Every leaf kernel
  it needs is already verified on NPU2 in the registry (GEMM, GEMV,
  FlashAttention, RMSNorm, RoPE, SiLU-and-Mul, Element-wise Add). This is the
  same kernel-reuse path already validated twice (SmolLM2-1.7B, Qwen3-1.7B).
- **Full pipeline port: MEDIUM feasibility.** The vision encoder (SigLIP) and
  the action expert add a handful of **incremental** ops not yet in the
  registry — none are a fundamentally new compute paradigm, but they are real
  work: non-causal (bidirectional) attention, cross-attention, GELU, LayerNorm,
  pixel-shuffle, and a 10-step iterative denoising loop.
- **Closest prior art exists and is encouraging.** TileFuse (IEEE FCCM 2026)
  already runs quantized LLM GEMM/GEMV on XDNA2 (= NPU2) via the MLIR-AIE/IRON
  layer that mlir-air compiles through. No VLA has been deployed on NPU2 yet —
  this would be a first.

---

## 1. What SmolVLA is

SmolVLA is an open-source **Vision-Language-Action** model from the LeRobot /
Hugging Face ecosystem. It maps camera views + a natural-language instruction +
robot proprioceptive state to a **chunk of continuous robot actions**. It has
two parts:

| Part | What it is | Role |
|---|---|---|
| **VLM backbone** (SmolVLM2) | SigLIP vision encoder → SmolLM2 language decoder | Perception: turn pixels + text into conditioning features |
| **Action expert** (~100M params) | flow-matching transformer, interleaved cross-/self-attention | Generate a continuous action chunk by iterative denoising |

Sources: SmolVLA paper [arXiv:2506.01844](https://arxiv.org/abs/2506.01844);
LeRobot docs [huggingface.co/docs/lerobot/smolvla](https://huggingface.co/docs/lerobot/smolvla);
HF blog [huggingface.co/blog/smolvla](https://huggingface.co/blog/smolvla).

### Contested: backbone size (read the checkpoint config, do not trust the paper prose)

The two research passes disagreed on backbone size, both citing the same paper:

- **Pass A (Opus):** the deployed `lerobot/smolvla_base` checkpoint uses
  **SmolVLM2-500M-Video-Instruct** → **SmolLM2-360M** decoder, with
  `num_vlm_layers=16` (half of 32), and the action expert conditions on an
  **intermediate VLM layer (~L/2)**, discarding upper layers.
- **Pass B (Sonnet):** reported the larger **SmolVLM2-2.2B → SmolLM2-1.7B**
  backbone; its adversarial verifier *killed* the 450M/360M/L-2 claims (13 of 25
  claims refuted — an over-aggressive verifier that discarded correct claims).

**Resolution:** Pass A is correct *for the deployed checkpoint*. This is a
primary-source fact from `lerobot/smolvla_base`'s `config.json`
(`vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `num_vlm_layers=16`,
`load_vlm_weights=true`), not a question of paper wording. The **2.2B / 1.7B**
figures describe the *largest SmolVLM2 variant*, not SmolVLA's deployed backbone.

> **Action item before any port:** load `lerobot/smolvla_base`, read its
> `config.json`, and dump per-layer shapes. Do not size kernels from the paper
> prose — checkpoint revisions change `num_vlm_layers`, `chunk_size`, etc.

---

## 2. Compute graph (three stages)

```
[camera frames] → SigLIP ViT → pixel-shuffle (space-to-depth) → MLP proj → visual tokens
                                                                              │
[instruction] → text embed ───────────────────────────────────────────── concat
                                                                              ▼
                          SmolLM2 decoder (16 layers) ── take features @ ~L/2 ── VLM prefix
                                                                              │ (as cross-attn K/V)
[action noise] → Action Expert (interleaved CA / causal-SA) ◄── 10× flow-matching denoise ──► action chunk
                                                                                              (chunk_size=50)
```

Key structural facts (all 3-0 verified across both passes):

- **Action expert interleaves** cross-attention (action tokens attend to the
  VLM prefix as K/V) and **causal** self-attention (action tokens attend to
  earlier action tokens). Each block is *either* CA *or* SA, not both. Causal SA
  masking measured best in the paper's ablations.
- **Flow matching**, not autoregression: it predicts a chunk of `chunk_size=50`
  continuous actions non-autoregressively, with **inference fixed at 10
  denoising (Euler ODE) steps**.
- Action-expert hidden size = **0.75×** the VLM hidden dim.
- ⚠️ Note `chunk_size=50` (actions per chunk) is *distinct* from the 10
  denoising steps — a common misread. 50 = output width; 10 = solver iterations.

---

## 3. Operator inventory vs. the kernel registry

### Already covered — direct reuse (the language backbone)

SmolLM2 is a Llama2-style decoder, so its operator *set* is identical-in-type to
Llama-3.2-1B, which is already deployed on NPU2 via mlir-air. Every kernel maps
to an existing, verified registry entry:

| SmolLM2 op | Registry kernel | Status |
|---|---|---|
| Q/K/V/O & MLP projections (prefill) | GEMM BF16 (in/fp32-out, in/bf16-out) | ✅ |
| same, decode (batch=1) | GEMV BF16 | ✅ |
| self-attention | FlashAttention BF16 (GQA, causal) | ✅ |
| pre-norm | RMSNorm BF16 | ✅ |
| positional encoding | RoPE BF16 (half-split) | ✅ |
| SwiGLU activation | SiLU-and-Mul BF16 | ✅ |
| residual adds | Element-wise Add BF16 | ✅ |

This is the same kernel-first reuse path already validated on SmolLM2-1.7B and
Qwen3-1.7B. **HIGH feasibility** for this stage.

### Not yet covered — incremental new kernels/variants

| Gap | Needed by | Difficulty | Notes |
|---|---|---|---|
| **Bidirectional (non-causal) attention** | SigLIP ViT | **Medium–High** | The registry FA kernel's tile config is *causal-pinned* and cannot currently do `lq≠lk` / non-causal cases without redesign. SigLIP is bidirectional. **This is the hardest single gap.** |
| **Cross-attention (distinct K/V source)** | action expert | Medium-Low | Reuses the same attention/GEMM/softmax primitives; only the K/V source differs (the cached VLM prefix). |
| **GELU + LayerNorm** | SigLIP | Low | SigLIP uses LayerNorm/GELU (not RMSNorm/SiLU). Same eltwise/reduction shapes as existing SiLU/RMSNorm kernels — analogous to write. Note: `programming_examples/gelu/` and `layer_norm/` already exist upstream as starting points. |
| **Pixel-shuffle (space-to-depth)** | vision→LLM connector | Low | Pure reshape / DMA-stride op, no arithmetic. |
| **10-step denoising loop scheduling** | action expert | Medium | Host-driven 10× forward passes over a cached VLM prefix vs. on-device loop — a control-flow / runtime question, not a kernel. |
| **New shape coverage** | action expert + SigLIP | Medium | Small action-expert GEMM/attention shapes and SigLIP patch-grid attention fall **outside** the registry's currently-tested shapes (which target llama-3.2-1B projections). Must be measured. |

**MEDIUM feasibility** for the full pipeline: incremental additions, not new
paradigms, but real and untested.

---

## 4. Performance regime (why the two stages need different strategies)

Roofline measurements from VLA-Perf ([arXiv:2602.18397](https://arxiv.org/html/2602.18397v1))
show the two stages sit in opposite regimes:

| Stage | Operator intensity | Regime | NPU2 implication |
|---|---|---|---|
| VLM backbone | ~543 FLOPs/Byte | **compute-bound** | target GEMM/FA kernels for throughput |
| Action expert | ~54 FLOPs/Byte | **memory-bound** | bandwidth-bound; ~50 tokens × 10 steps |

This mirrors the registry's own split exactly: GEMM/FA are compute-bound
(GFLOP/s reported), while SiLU/RoPE/RMSNorm/EltwiseAdd are memory-bound
(GB/s reported).

> ⚠️ **Caveat:** the 543 / 54 figures are measured on pi0/Gemma (768 visual
> tokens), **not SmolVLA** (64 tokens/frame). The exact numbers do not transfer
> — SmolVLA's backbone intensity will be lower — but the *qualitative* pattern
> (compute-bound backbone, memory-bound action expert) is robust.

---

## 5. Prior art on the same hardware / toolchain

- **TileFuse** ([arXiv:2606.11357](https://arxiv.org/html/2606.11357v1), IEEE
  FCCM 2026) — *strongest precedent, 3-0 verified.* AWQ-style W4A16/W8A16
  quantized GEMM/GEMV compiled via **MLIR-AIE/IRON** and run on **XDNA2 (=
  NPU2/Strix)**, +281% GEMV and 2.0× lower LLM prefill latency. The IRON layer
  it uses is the same one mlir-air compiles through → the registry's BF16
  kernels are on the same feasibility frontier, and a quantized future path has
  a concrete precedent.
- **vla.cpp** ([arXiv:2606.08094](https://arxiv.org/html/2606.08094)) — first
  ggml-class C++ runtime to natively serve flow-matching/diffusion VLA
  inference: **caches the vision-language prefix** and runs the cross-attending
  action expert across iterative solver steps. Validates the execution model a
  NPU2 port should adopt (cache VLM prefix once, reuse across the 10 denoising
  steps) — i.e. exactly the "pre-compiled fused kernels + prefix reuse"
  direction the registry encourages. (CPU/GPU runtime, not NPU — indirect.)
- **No VLA has been deployed on AMD NPU2 / mlir-air.** This would be a first.

---

## 6. Recommended next step

Per the kernel-first methodology, the first step is **not** writing kernels —
it's getting shape-accurate. Load `lerobot/smolvla_base`, dump per-layer tensor
shapes for all three stages (SigLIP encoder / SmolLM2-360M decoder @ 16 layers /
~100M action expert @ 0.75× hidden), and produce a table:

> *covered ✅ (existing kernel + already-tested shape) · needs new shape ⚠️
> (existing kernel, untested shape) · needs new kernel ❌*

This resolves both passes' #1 open question (exact shapes) and the backbone-size
dispute in one pass.

---

## Open questions

1. Exact per-layer shapes of (a) SmolLM2-360M @ `num_vlm_layers=16`, (b) the
   SigLIP encoder, (c) the ~100M action expert @ 0.75× hidden — to map onto /
   extend the registry's tested GEMM & FlashAttention shape coverage.
2. Can the causal-pinned FlashAttention registry kernel be generalized to
   non-causal/bidirectional (SigLIP) and to cross-attention with a distinct K/V
   source (action expert), or do these need separate kernels? What is the L1
   tiling feasibility for SigLIP patch-grid attention tensors?
3. How best to map the 10-step flow-matching loop onto NPU2 — host-driven 10×
   dispatch over a cached VLM prefix vs. an on-device loop — and the resulting
   end-to-end latency at robot control frequency (10–50 Hz)?
4. Are there reference IRON/mlir-aie kernels for bidirectional attention, GELU,
   and LayerNorm that can be adapted rather than written from scratch?

---

## Sources

Primary:
- SmolVLA paper — https://arxiv.org/abs/2506.01844
- SmolVLM paper — https://arxiv.org/abs/2504.05299
- LeRobot SmolVLA docs — https://huggingface.co/docs/lerobot/smolvla
- HF SmolVLA blog — https://huggingface.co/blog/smolvla
- SmolVLM2-2.2B model card — https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- VLA-Perf (roofline) — https://arxiv.org/html/2602.18397v1
- TileFuse (XDNA2 / IRON) — https://arxiv.org/html/2606.11357v1
- vla.cpp (runtime) — https://arxiv.org/html/2606.08094

Cross-checked against local `programming_examples/kernel_registry/supported_kernels.md`.

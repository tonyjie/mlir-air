# Edge LLM Candidates for AMD NPU2 Deployment

**Date:** 2026-04-17 (living doc — update as new models release / kernels expand)

Inventory of decoder-only LLMs we could deploy on AMD NPU2, mapped against our current kernel inventory (built during the Llama-3.2-1B work). Goal: prioritize the next pilot deployment(s) and clarify which kernel additions unlock the most models per unit of effort.

---

## TL;DR — Recommended pilot sequence

| Order | Model | Tier | Why |
|-------|-------|------|-----|
| **1** | `HuggingFaceTB/SmolLM2-1.7B` | A | Architectural twin to Llama-3.2-1B (same emb/head_dim/hidden/RoPE/RMSNorm/SwiGLU). Only differences: tied embeddings, MHA (handled as degenerate GQA), smaller vocab. Validates pipeline portability with **near-zero kernel work**. |
| **2** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | A/B | Same emb_dim=2048 and head_dim=64. New `hidden_dim=5632` GEMM shape and 32:4 GQA ratio. Easy second data point. |
| **3** | `HuggingFaceTB/SmolLM2-135M` and `360M` | A/B | Tiny — useful for fast iteration / debugging / profiling. Same kernel set, smaller shapes. Optional. |
| **4** | `meta-llama/Llama-3.2-3B` | B | Same family + tooling. Forces **head_dim=128 RoPE+FA** generalization — the single most reusable kernel upgrade — which unlocks Qwen2.5, Llama-3.1, OLMo-2, InternLM2.5, Phi-4-mini. Memory tight (~6.4 GB BF16). |
| **5** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | C | Reasoning-distilled, rides on Qwen2.5-1.5B arch. Adds **QKV bias** epilogue + needs head_dim=128 from #4. Demonstrates "reasoning model" support. |
| **6** | `Qwen/Qwen2.5-1.5B` (or `0.5B`) | C | Most popular sub-2B Llama alternative. Same QKV-bias change unlocks all of Qwen2.5 family. |
| **7** | `google/gemma-3-1b-it` | D | Smallest "frontier" architecture: GeGLU + head_dim=256 + MQA + QK-Norm + 5:1 sliding-window. Stretch goal — bundles three reusable additions (GeGLU, QK-Norm, SWA) that also enable Qwen3 and OLMo-2. |

**Sequencing logic:** 1–3 prove portability with zero/minimal new kernels. 4 generalizes head_dim=128 (the gating dependency for almost everything else). 5–6 add QKV-bias (a small epilogue). 7 is the first true "new architecture family" investment.

**Out of scope (this iteration):** all gpt-oss (MoE+MXFP4+attention sinks), DeepSeek-V*/R1 base (MLA+MoE), Gemma 3 ≥4B (vision tower), all 7B+ BF16 (memory-infeasible without Q4 — currently shelved).

---

## Compatibility Tier Legend

| Tier | Meaning | What's needed |
|------|---------|---------------|
| **A** | Drop-in: same kernels, same shapes | Just config swap. Best pilot candidates. |
| **B** | Drop-in: same kernels, **new shapes** | Add new shape to existing kernel sweep. Minor. |
| **C** | **Minor kernel adaptation** — new variant of existing kernel | E.g., GeGLU (vs SwiGLU), QKV bias add, alternate RoPE convention. Each is a small focused C++ or Python addition. |
| **D** | **Significant kernel additions** | Sliding window attention, logit softcap, partial RoPE, QK-Norm, post-norm, parallel attn/MLP. |
| **E** | **Out of scope** | MoE expert routing, MLA, multi-modal, encoder-decoder. Fundamental new infrastructure. |

## Memory budget on NPU2

| Marker | BF16 size | Verdict |
|--------|-----------|---------|
| ✅ | ≤ ~3B params (~6 GB) | Fits comfortably |
| ⚠ | 3–6B params (6–12 GB) | Tight (KV cache + activations on 24 GB shared) |
| ❌ | > 6B BF16 | Needs quantization (Q4/INT8 — not currently supported) |

---

## 1. Llama family

**Architecture:** RMSNorm pre-norm, SwiGLU FFN, RoPE half-split, GQA (3.x+), no QKV bias, untied embeddings (3.1 8B) / tied (3.2 1B/3B). Llama 3.1+ adds long-context "llama3" RoPE scaling (factor 32, low/high freq factors 1/4) — mostly inert for prefill ≤8192.

| Model | Params | L | emb | nH | nKV | head_dim | hidden | vocab | rope_θ | Special | BF16 | Tier | Mem |
|-------|--------|---|-----|----|----|----------|--------|-------|--------|---------|------|------|-----|
| `meta-llama/Llama-3.2-1B` | 1.24B | 16 | 2048 | 32 | 8 | 64 | 8192 | 128256 | 500k | tied emb | 2.5 GB | A (current) | ✅ |
| `meta-llama/Llama-3.2-3B` | 3.21B | 28 | 3072 | 24 | 8 | **128** | 8192 | 128256 | 500k | tied emb | 6.4 GB | B | ⚠ |
| `meta-llama/Llama-3.1-8B` | 8.03B | 32 | 4096 | 32 | 8 | 128 | 14336 | 128256 | 500k | — | 16 GB | B | ❌ |
| `meta-llama/Llama-3.3-70B` | 70B | 80 | 8192 | 64 | 8 | 128 | 28672 | 128256 | 500k | — | 140 GB | B | ❌ |

**3.2-3B kernels needed:** GEMM at M=K=3072, N∈{3072, 8192, 1024}; **head_dim=128** generalization for RoPE + FlashAttention.

## 2. Qwen family

**Architecture:** Qwen2.5 — **QKV bias** (vs none in Llama), SwiGLU, RMSNorm, RoPE half-split, GQA, rope_θ=1e6, vocab 151936/152064. Qwen3 — **removes QKV bias**, **adds QK-Norm** (RMSNorm on Q and K before attention), otherwise same recipe. Qwen3 also has MoE variants (out of scope).

| Model | Params | L | emb | nH | nKV | head_dim | hidden | vocab | rope_θ | Special | BF16 | Tier | Mem |
|-------|--------|---|-----|----|----|----------|--------|-------|--------|---------|------|------|-----|
| `Qwen/Qwen2.5-0.5B` | 0.49B | 24 | 896 | 14 | 2 | 64 | 4864 | 151936 | 1e6 | QKV bias, tied | 1.0 GB | C | ✅ |
| `Qwen/Qwen2.5-1.5B` | 1.54B | 28 | 1536 | 12 | 2 | 128 | 8960 | 151936 | 1e6 | QKV bias, tied | 3.1 GB | C | ✅ |
| `Qwen/Qwen2.5-3B` | 3.09B | 36 | 2048 | 16 | 2 | 128 | 11008 | 151936 | 1e6 | QKV bias, tied | 6.2 GB | C | ⚠ |
| `Qwen/Qwen2.5-7B` | 7.61B | 28 | 3584 | 28 | 4 | 128 | 18944 | 152064 | 1e6 | QKV bias, untied | 15.2 GB | C | ❌ |
| `Qwen/Qwen2.5-14B` | 14.7B | 48 | 5120 | 40 | 8 | 128 | 13824 | 152064 | 1e6 | QKV bias, untied | 29 GB | C | ❌ |
| `Qwen/Qwen3-0.6B` | 0.6B | 28 | 1024 | 16 | 8 | 128 | 3072 | 151936 | 1e6 | **QK-Norm**, tied | 1.2 GB | D | ✅ |
| `Qwen/Qwen3-1.7B` | 1.7B | 28 | 2048 | 16 | 8 | 128 | 6144 | 151936 | 1e6 | QK-Norm, tied | 3.4 GB | D | ✅ |
| `Qwen/Qwen3-4B` | 4.0B | 36 | 2560 | 32 | 8 | 128 | 9728 | 151936 | 1e6 | QK-Norm | 8 GB | D | ⚠ |
| `Qwen/Qwen3-8B`+ | ≥8B | — | — | — | — | 128 | — | — | — | QK-Norm | ≥16 GB | D | ❌ |
| Qwen3-30B-A3B / 235B-A22B | MoE | — | — | — | — | — | — | — | — | MoE routing | — | **E** | ❌ |

**Qwen2.5 missing:** head_dim=128 RoPE/FA (except 0.5B which is 64); **QKV bias add** (trivial — extend GEMM epilogue or eltwise-add kernel). **Qwen3 additionally:** **QK-Norm** (small RMSNorm along head_dim before RoPE).

## 3. Gemma family

**Architecture:**
- **Gemma 1**: GeGLU (gelu_pytorch_tanh), RMSNorm, MQA (2B) or MHA (7B), head_dim=256.
- **Gemma 2**: + **interleaved sliding-window/global attention** (1:1, window 4096), **logit softcap** on attention (50) and final logits (30), query pre-attn scale.
- **Gemma 3**: removes softcap, adds **QK-Norm**, **5:1 local:global** sliding window 1024, RoPE θ=1M. 4B/12B/27B are vision-language; only 1B is text-only.

| Model | Params | L | emb | nH | nKV | head_dim | hidden | vocab | rope_θ | Special | BF16 | Tier | Mem |
|-------|--------|---|-----|----|----|----------|--------|-------|--------|---------|------|------|-----|
| `google/gemma-2b` (v1) | 2.5B | 18 | 2048 | 8 | 1 (MQA) | 256 | 16384 | 256000 | 10k | **GeGLU**, tied | 5 GB | C | ✅ |
| `google/gemma-7b` (v1) | 8.5B | 28 | 3072 | 16 | 16 (MHA) | 256 | 24576 | 256000 | 10k | GeGLU | 17 GB | C | ❌ |
| `google/gemma-2-2b` | 2.6B | 26 | 2304 | 8 | 4 | 256 | 9216 | 256000 | 10k | GeGLU + **SWA 4096** + softcap 50/30 | 5.2 GB | **D** | ✅ |
| `google/gemma-2-9b` | 9.2B | 42 | 3584 | 16 | 8 | 256 | 14336 | 256000 | 10k | same | 18 GB | D | ❌ |
| `google/gemma-2-27b` | 27B | 46 | 4608 | 32 | 16 | 128 | 36864 | 256000 | 10k | same | 54 GB | D | ❌ |
| `google/gemma-3-1b-it` | 1.0B | 26 | 1152 | 4 | 1 | 256 | 6912 | 262144 | 1M | **QK-Norm**, SWA 1024 (5:1), MQA | 2 GB | D | ✅ |
| `google/gemma-3-4b-it` | 4B | — | — | — | — | 256 | — | 262144 | 1M | + SigLIP vision | 8 GB | **E** (vision) | ⚠ |
| `google/gemma-3-12b/27b` | 12/27B | — | — | — | — | 256 | — | 262144 | 1M | + vision | 24+ GB | E | ❌ |

**Gemma kernels missing:** **GeGLU** (need GELU-tanh activation; structurally same as SwiGLU); **head_dim=256** (4× current — possibly L1 capacity concerns); for Gemma 2 also **SWA with interleaved layer scheduling** + **tanh softcap**. Gemma 3-1B drops softcap but adds QK-Norm and 5:1 SWA.

## 4. gpt-oss family — TIER E (out of scope)

| Model | Total | Active | L | Experts | Top-K | Notes |
|-------|-------|--------|---|---------|-------|-------|
| `openai/gpt-oss-20b` | 20.9B | 3.6B | 24 | 32 | 4 | MXFP4 MoE weights, **attention sinks**, alternating full + 128-tok SWA |
| `openai/gpt-oss-120b` | 116.8B | 5.1B | 36 | 128 | 4 | same |

Requires MoE routing infra, attention sinks (per-head learned softmax denominator bias), 128-token SWA, and MXFP4 dequant (or BF16 explosion). Memory: ❌ either way.

## 5. DeepSeek family

| Model | Class | Notes |
|-------|-------|-------|
| DeepSeek-V2 / V2.5 / V3 | MoE + **MLA** | Multi-head Latent Attention is fundamentally different. Tier **E**. |
| DeepSeek-R1 | V3 base (671B MoE) | E |
| **`DeepSeek-R1-Distill-Qwen-1.5B`** | **Qwen2.5-Math-1.5B** arch | Same as Qwen2.5-1.5B → **Tier C**, ✅ memory |
| `DeepSeek-R1-Distill-Qwen-7B` | Qwen2.5-Math-7B | C, ❌ memory |
| `DeepSeek-R1-Distill-Llama-8B` | Llama-3.1-8B | B, ❌ memory |
| `DeepSeek-R1-Distill-Qwen-14B/32B` | Qwen2.5 | C, ❌ |
| `DeepSeek-R1-Distill-Llama-70B` | Llama-3.3 | B, ❌ |

The distilled small models inherit base architecture (no MLA, no MoE) — they collapse onto Llama/Qwen tiers with only tokenizer/weight changes. **R1-Distill-Qwen-1.5B is a real candidate** because reasoning capability is downstream-valuable.

## 6. Phi family

All use Phi3Config (`Phi3ForCausalLM`): SwiGLU, RMSNorm, RoPE half-split with **longrope** scaling for 128K variants. Phi-3 originally ships with **fused QKV** (single weight for q/k/v) and **fused gate_up** in MLP — packing detail, not algorithmic. Phi-4-mini introduces **partial/fractional RoPE** (25% of head_dim is position-agnostic) and tied embeddings.

| Model | Params | L | emb | nH | nKV | head_dim | hidden | vocab | rope_θ | Special | BF16 | Tier | Mem |
|-------|--------|---|-----|----|----|----------|--------|-------|--------|---------|------|------|-----|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 32 | 3072 | 32 | 32 (MHA) | 96 | 8192 | 32064 | 10k | fused QKV/gate_up | 7.6 GB | C | ❌/⚠ |
| `microsoft/Phi-3-mini-128k-instruct` | 3.8B | 32 | 3072 | 32 | 32 | 96 | 8192 | 32064 | 10k | + longrope | 7.6 GB | C | ❌ |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | 32 | 3072 | 32 | 32 | 96 | 8192 | 32064 | 10k | longrope, 128k ctx | 7.6 GB | C | ❌ |
| `microsoft/Phi-4-mini-instruct` | 3.8B | 32 | 3072 | 24 | 8 (GQA) | 128 | 8192 | 200064 | 10k | **partial RoPE** (75% rotated), tied emb, longrope | 7.6 GB | **D** | ❌ |

Phi-3 family is actually MHA (not GQA) — distinct from Llama 3.x. **head_dim=96** (Phi-3) is unusual. All 3.8B → 7.6 GB BF16, beyond comfort budget.

## 7. Smaller / edge-specific

| Model | Params | L | emb | nH | nKV | head_dim | hidden | vocab | rope_θ | Norm | FFN | Special | BF16 | Tier | Mem |
|-------|--------|---|-----|----|----|----------|--------|-------|--------|------|-----|---------|------|------|-----|
| `TinyLlama/TinyLlama-1.1B` | 1.1B | 22 | 2048 | 32 | 4 | 64 | 5632 | 32000 | 10k | RMSNorm | SwiGLU | Llama-2 arch | 2.2 GB | **A/B** | ✅ |
| `HuggingFaceTB/SmolLM2-135M` | 135M | 30 | 576 | 9 | 3 | 64 | 1536 | 49152 | 100k | RMSNorm | SwiGLU | tied emb, depth>width | 0.27 GB | A/B | ✅ |
| `HuggingFaceTB/SmolLM2-360M` | 360M | 32 | 960 | 15 | 5 | 64 | 2560 | 49152 | 100k | RMSNorm | SwiGLU | tied emb | 0.72 GB | A/B | ✅ |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | 24 | 2048 | 32 | 32 (MHA) | 64 | 8192 | 49152 | 130k | RMSNorm | SwiGLU | tied emb, MHA | 3.4 GB | **A** | ✅ deployed 2026-04-17 |
| `allenai/OLMo-2-0425-1B` | 1.5B | 16 | 2048 | 16 | 16 (MHA) | 128 | 8192 | 100k+ | 500k | RMSNorm | SwiGLU | **post-norm**, **QK-Norm** | 3 GB | D | ✅ |
| `allenai/OLMo-2-1124-7B` | 7B | 32 | 4096 | 32 | 32 (MHA) | 128 | 11008 | — | — | RMSNorm | SwiGLU | post-norm, QK-Norm | 14 GB | D | ❌ |
| `internlm/internlm2_5-7b` | 7.7B | 32 | 4096 | 32 | 8 | 128 | 14336 | 92544 | 1e6 | RMSNorm | SwiGLU | wqkv fused | 15 GB | C | ❌ |
| `stabilityai/stablelm-2-1_6b` | 1.6B | 24 | 2048 | 32 | 32 (MHA) | 64 | 5632 | 100352 | 10k | **LayerNorm** (with bias) | SwiGLU | parallel attn/MLP | 3.2 GB | **C** (LayerNorm) | ✅ |

**SmolLM2-1.7B** is the closest architectural twin to Llama-3.2-1B: same emb_dim=2048, head_dim=64, hidden=8192, RMSNorm + SwiGLU, RoPE half-split. Differences: MHA (degenerate GQA), tied embeddings, smaller vocab, rope_θ=130k. **Tier A** strictly. **Validated by deployment 2026-04-17** (`programming_examples/smollm2_1_7b/`): per-layer prefill/decode rate at parity with llama3 (79 ms/layer prefill, 5.7 ms/layer decode); ~1.88 s prefill / 136 ms-per-token decode for the deeper 24-layer stack; only one bug fixed (stale path in `_llm_shared/kernel_builder/external_kernels.py:99`).

**TinyLlama-1.1B**: emb=2048, head_dim=64, hidden=5632 (new GEMM N), nKV=4 (new GQA ratio). Tier B (one new GEMM shape and a 32:4 GQA pattern).

---

## Kernel Gap Matrix — what to build to unlock what

This table reads top-to-bottom as effort, left-to-right as models unlocked. Pick the gap with highest models-per-unit-effort.

| Kernel addition | Effort | Models unlocked (in scope) |
|-----------------|--------|----------------------------|
| **Tied embeddings handling** (lm_head shares embed weights) | trivial (host-side only) | SmolLM2-*, Llama-3.2-1B-Instruct, Qwen2.5-≤3B, Qwen3-≤1.7B, Gemma-3-1B, others |
| **Smaller vocab GEMV/GEMM** for LM Head (e.g., 32k, 49k) | minor (just new shape) | TinyLlama, SmolLM2, Phi-3 |
| **MHA as degenerate GQA** | trivial (config flag) | SmolLM2-1.7B, Phi-3, OLMo-2, StableLM-2 |
| **head_dim=128 RoPE + FlashAttention** | medium (new C++ kernel for RoPE; FA generalization) | Llama-3.2-3B, Llama-3.1-8B, Qwen2.5 ≥1.5B, Qwen3 ≥0.6B, OLMo-2, InternLM-2.5, Phi-4-mini, R1-Distill-Qwen-* |
| **QKV bias add** (eltwise add post-GEMM) | trivial (eltwise add already exists; just plumbing) | All Qwen2.5 + R1-Distill-Qwen-* |
| **GeGLU** (replace SiLU activation with tanh-GELU) | small (new C++ activation kernel; same outer GEMM structure) | Gemma-1, Gemma-2, Gemma-3 (text-only) |
| **head_dim=256** (Gemma) | medium (L1 capacity check; possibly tile differently) | Gemma family |
| **QK-Norm** (RMSNorm on Q and K before attention) | small-medium (insert RMSNorm in the attention pipeline) | Qwen3, Gemma-3, OLMo-2 |
| **Sliding-window attention** (window mask + interleaved layer scheduling) | large (new attention variant + dispatch logic) | Gemma-2/3, Mistral, gpt-oss (still E for other reasons) |
| **Logit softcap** (tanh epilogue) | small (post-projection eltwise) | Gemma-2 only |
| **Partial RoPE** (rotate only first 75% of head dims) | small (RoPE kernel variant) | Phi-4-mini |
| **Post-norm** (norm after residual instead of before) | trivial (just where we apply RMSNorm) | OLMo-2 |
| **LayerNorm with bias** | small (LayerNorm kernel + bias) | StableLM-2 |
| **Q4/INT8 dequantization** | large (already explored, see Q4 trial; shelved) | Unlocks 7B+ across all families |
| **MoE expert routing** (top-k gate + sparse expert dispatch) | very large | gpt-oss, Mixtral, Qwen3-MoE, DeepSeek-V*/R1 base |
| **MLA** (Multi-head Latent Attention) | very large | DeepSeek-V*/R1 base |

---

## Sources

- [HF Llama docs](https://huggingface.co/docs/transformers/en/model_doc/llama), Llama 3.2 [1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [3B](https://huggingface.co/meta-llama/Llama-3.2-3B) cards
- [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/pdf/2412.15115); [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/pdf/2505.09388); [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- [HF Gemma docs](https://huggingface.co/docs/transformers/en/model_doc/gemma), [Gemma 2 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma2), [Gemma 3 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3); [Welcome Gemma 2 blog](https://huggingface.co/blog/gemma2); [Welcome Gemma 3 blog](https://huggingface.co/blog/gemma3)
- [OpenAI gpt-oss announcement](https://openai.com/index/introducing-gpt-oss/); [gpt-oss model card PDF](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- [DeepSeek-R1 card](https://huggingface.co/deepseek-ai/DeepSeek-R1); [R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B); [R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [HF Phi-3 docs](https://huggingface.co/docs/transformers/en/model_doc/phi3); [Phi-4-Mini Technical Report (arXiv:2503.01743)](https://arxiv.org/html/2503.01743v1)
- [SmolLM2-1.7B card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B); [SmolLM2-135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama)
- [OLMo 2 blog](https://allenai.org/blog/olmo2); [OLMo 2 tech report (arXiv:2501.00656)](https://arxiv.org/pdf/2501.00656); [StableLM 2 1.6B Technical Report](https://arxiv.org/html/2402.17834v1)

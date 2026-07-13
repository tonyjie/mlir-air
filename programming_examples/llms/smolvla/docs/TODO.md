# SmolVLA backbone — deployment TODO / execution notes

## NPU-execution exceptions

These are the operations that run on CPU rather than NPU, with the concrete
reason for each. Everything else in the per-layer hot loop (RMSNorm, Q/K/V
GEMMs, RoPE, O-proj, residual, SwiGLU FFN) runs on NPU via the fused
`rms_gemms_rope` and `o_ffn` ELFs.

- **Final RMSNorm** (`text_model.norm` on the layer-15 output, in
  `run_backbone_prefill`): CPU F32.
  Reason: this is a deliberate, sibling-consistent choice, NOT a fallback from a
  broken NPU path. Every llama/qwen sibling applies the final RMSNorm on CPU
  outside the per-token hot loop (llama32_1b_inference.py:453,
  llama32_1b/verify_adapter.py:219). It is a single (seq, emb) RMSNorm on the
  last hidden state — no per-token NPU work — and reuses the exact F32 reference
  math the oracle (`final_norm_hidden`) was generated with. Moving it to NPU
  would add an extra ELF dispatch for zero throughput benefit.

- **Attention**: two selectable paths.
  - Step A (`cpu_attn=True`, the default): CPU non-causal
    (`noncausal_attention_reference`). Retained as the low-risk default and the
    per-substep diagnostic reference.
  - Step B (`cpu_attn=False`): attention runs on NPU — `_npu_attention` does per
    q-head `S = (Q·1/√d) @ Kᵀ` (qkt GEMM ELF), a single batched
    `masked_softmax` over all heads' (seq×seq) scores (masked_softmax ELF), and
    per q-head `O = P @ V` (pv GEMM ELF). ALL matmuls + the softmax run on NPU;
    the host only does layout glue (transpose Kh, gather per-head slices, fold
    the additive mask into the score buffer). Verified: full 16-layer backbone
    with `cpu_attn=False` matches the oracle to the same tier as Step A
    (final-hidden cosine 0.99614 vs Step A's 0.99617; per-head O cosine
    0.999992). No CPU attention fallback in this path.

## ELFs per layer

- Prefill, Step A (cpu_attn=True): **2 fused ELFs/layer** — `rms_gemms_rope`
  (6-launch: RMSNorm + Q/K/V GEMM + RoPE Q/K) and `o_ffn` (8-launch: O-proj +
  residual + FFN). Attention on CPU.
- Prefill, Step B (cpu_attn=False): the 2 fused ELFs above **plus** the NPU
  attention dispatches per layer: 15 `qkt` + 1 batched `masked_softmax` + 15
  `pv` = **31 attention dispatches/layer** (33 ELF dispatches/layer total).
  Per-head dispatch is a deliberate, correct-first design; fusing the 15 heads
  into one multi-launch attention ELF (kernel-first) is a Phase-4/5 optimization
  follow-up, not a correctness blocker.

## NPU-attention GEMM shapes (not in the large-shape registry sweep)

`256×64×256` (S=Q@Kᵀ) and `256×256×64` (O=P@V) are thin attention shapes absent
from the registry's large-GEMM sweep. Validated directly on NPU by Pearson
correlation vs FP32 (0.99995 each) in `validate_attn_gemms.py` — the phase-1
"no-harness" lens (there is no per-shape harness with a tuned rtol/atol; the
element-wise diffs are the expected bf16/BFP16 tier and these outputs feed
softmax). Recorded as rows in `kernel_registry/details/GEMM_bf16_in_bf16_out.{md,json}`.

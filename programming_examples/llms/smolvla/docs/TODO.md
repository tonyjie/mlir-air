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

- **Attention (Step A, cpu_attn=True default)**: CPU non-causal
  (`noncausal_attention_reference`).
  Status: Step B (attention on NPU via per-head GEMM + masked_softmax) —
  see the "attention on NPU" section below for the current status.

## ELFs per layer

- Prefill, Step A (cpu_attn=True): **2 fused ELFs/layer** — `rms_gemms_rope`
  (6-launch: RMSNorm + Q/K/V GEMM + RoPE Q/K) and `o_ffn` (8-launch: O-proj +
  residual + FFN). Attention on CPU.

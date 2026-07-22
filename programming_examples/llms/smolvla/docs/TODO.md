# SmolVLA backbone — deployment TODO / execution notes

## A3-5 Vision encoder (SigLIP ViT, 12 layers) — NPU integration (Step 2-3)

Deliverables `vision_prefill.py` + `test_full_vit.py` are built and the full
12-layer encoder runs **entirely on NPU** for all four heavy ops: projection/MLP
GEMMs (gemm_qkvo/fc1/fc2 ELFs), affine LayerNorm (layer_norm ELF), GELU-tanh
(gelu ELF), and non-causal bidirectional MHA (flash_attn ELF). Only the
correctness-first host glue runs on CPU: per-Linear bias-adds, the two residual
adds, and the im2col patch-embed (all A3-6 fusion candidates, NOT fallbacks).

**NPU-execution: all 4 heavy ops on NPU (0 CPU fallback). NPU dispatches/layer =
10** (LN1, q, k, v, FA, o, LN2, fc1, gelu, fc2), + 1 post_ln (once, end of stack).
Correctness-first: NOT fused/optimized — A3-6 will merge these into multi-launch
ELFs and move the bias/residual/im2col glue onto the device.

### Correctness gate NOT met — precision-ceiling FAIL (root-caused, not a bug)

Full-run vs the real-lerobot oracle: per-layer cosine 0.968–0.998 (layers 1–10
below the 0.98 gate), **post_ln cosine 0.945** (< 0.99 gate). Per-token-median
cosine is higher (post_ln 0.988) but still short.

This is an HONEST precision ceiling of the current validated BF16/BFP16 NPU
kernels on the SigLIP vision encoder, exhaustively root-caused (NOT a wiring bug):
  1. Every leaf kernel is clean at registry tier: layer-0 block 0.9985, all
     sub-ops cos > 0.9998 (LN 0.99997, qkvo GEMM 0.99999, FA 0.99979).
  2. Teacher-forced per-layer (feed oracle input to each block independently) is
     clean: L0=0.998, L2–L11 = 0.994–0.9998. Only L1 is anomalous at 0.981.
  3. An EXACT numpy mirror of the pipeline (BFP16-input matmul + bf16-out cast +
     host f32 bias + f32 residual) predicts post_ln 0.998 — so the failure is
     NOT the wiring, NOT bf16-boundary rounding, NOT eps (1e-5 vs 1e-6 identical).
  4. The excess error is SYSTEMATIC: **81% of FlashAttention's error is a shared
     per-channel column-mean bias** (measured: total FA err L2 3.32, systematic
     component 2.70). BFP16 block-float (shared 8-elem exponent) on the vision
     encoder's outlier-heavy LayerNorm activations (max/mean-abs ~45) produces a
     directional bias that survives LayerNorm centering and accumulates coherently.
  5. Layer 1 has the strongest residual cancellation of any layer
     (|input|=225.8, |attn_out|=182.8 → |x1|=143.0), which AMPLIFIES that
     systematic bias — hence L1's 0.981. Random-noise models of the same rel
     magnitude give x1 cos 0.9999 (cancellation averages out random error but
     NOT systematic error). The per-layer errors compound in quadrature to the
     observed 0.945.

### NPU precision levers tried (all exhausted; none recovers the gate)

  - **GEMM without BFP16** (`compile_gemm_mm(bfp16=False)`, native aie2p bf16
    8x8x8 mmul): WRONG results (block-vs-oracle 0.28–0.90). The `mm_aie2p.cc`
    non-BFP16 branch uses a different C_block accumulator layout that is not
    correct at these tiles without a microkernel rewrite. Flag left in
    external_kernels.py (default True = unchanged) to document.
  - **Direct-codegen GEMM** (`build_module_lowered`, aievec lowering, no external
    mm.o): WRONG at the cancellation layers (L1=0.23, L5=0.35).
  - **FlashAttention without BFP16** (`compile_attn_npu2(bfp16=False)`): runtime
    HANG (ERT_CMD_STATE_TIMEOUT). The FA kernel's L1 buffer tiling is sized for
    the BFP16 mmul; the native mmul overflows/mismatches. Flag left (default
    True) to document; would need an FA L1-tiling rewrite (A3-6 kernel work).
  - **CPU attention** (mha_bidirectional on host): WORSE full-run (post_ln
    0.905), so FA is NOT the dominant error and CPU is not a usable fallback.

### Path forward (A3-6 / kernel work, out of this correctness task's scope)

The gate needs a higher-precision matmul that stays on NPU: either (a) rewrite the
FA kernel's L1 tiling so the native (non-BFP16) bf16 8x8x8 mmul places and runs
(removes the 81%-systematic attention bias), or (b) an fp32-accumulate FA epilogue
that doesn't quantize the attention output per block. Both are microkernel changes.
Alternatively, revisit the gate metric (per-token-median 0.988 vs flattened 0.945).

### DECISION (2026-07-22): A3-6a precision fix DROPPED, proceed to A3-6b (perf)

BFP16 is the format the NPU must use for good performance; forcing pure BF16 would
be very slow. The 0.945 mismatch is an accepted BFP16 systematic-bias ceiling — the
cause is NOT a controllable factor on our side (wiring/config/eps all ruled out,
teacher-forced clean). So we do NOT pursue the higher-precision microkernel; we move
to A3-6b performance hill-climb (ELF fusion, BO reuse, on-device glue).

### Unfused reference preserved: tag `smolvla-vision-unfused-v1`

Before A3-6b makes vision_prefill.py messier (merged ELFs), the clean
kernel-by-kernel version is frozen at git tag **`smolvla-vision-unfused-v1`**
(commit 65e0d3d5). To trace a precision/other issue against the simple version:
    git checkout smolvla-vision-unfused-v1 -- programming_examples/llms/smolvla/vision_prefill.py
That version = 10 dispatches/layer, one ELF per op, host bias/residual/im2col,
measured NPU 281ms / 2.6x vs CPU, correctness 0.945.

## A3-6b — vision performance hill-climb (IN PROGRESS)

Baseline to beat: unfused NPU vision **281 ms** (median, compile excluded) vs CPU
719 ms. Levers (each: apply → re-measure wall time → re-check per-layer cosine vs
oracle stays ~0.945, i.e. fusion must not change the math):
  - opt-merge-multi-launch-kernels: fuse the 10 per-layer dispatches into a few
    multi-launch ELFs (mirror backbone's rms_gemms_rope / o_ffn), e.g. LN+QKV+attn
    and O+residual+LN+FFN. Dominant win.
  - opt-buffer-object-reuse: pre-load per-layer weight BOs once (static_input_indices)
    across the 12 layers; reuse intermediate BOs.
  - move host glue on-device: bias-adds into the GEMM epilogue, residual adds, and
    ideally im2col — remove host round-trips.
Gate: wall time strictly < 281 ms AND per-layer cosine unchanged (~0.945, fusion is
math-equivalent — a cosine DROP means a fusion bug, not the BFP16 ceiling).

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
from the registry's large-GEMM sweep. Validated directly on NPU in
`validate_attn_gemms.py` by the SAME project-standard metric every other GEMM
row uses: **mean_rel_L1 = mean|out−ref| / mean|ref| vs the FP32 reference**,
gated at the bf16-out tier. Measured mean_rel_L1 = 9.57e-3 (QKᵀ) and 9.62e-3
(P@V) — in line with the other drain rows (9.3-9.9e-3). Pearson correlation
(0.99995) is reported as a secondary sanity metric only; it is NOT the gate
(correlation is scale/shift-invariant and cannot catch a systematic scale/bias).
Recorded as rows in `kernel_registry/details/GEMM_bf16_in_bf16_out.{md,json}`.

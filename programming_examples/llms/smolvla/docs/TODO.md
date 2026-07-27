# SmolVLA backbone — deployment TODO / execution notes

## A3-5 Vision encoder (SigLIP ViT, 12 layers) — NPU integration (Step 2-3)

Deliverables `vision_prefill.py` + `test_full_vit.py` are built and the full
12-layer encoder runs **entirely on NPU** for all four heavy ops: projection/MLP
GEMMs (gemm_qkvo/fc1/fc2 ELFs), affine LayerNorm (layer_norm ELF), GELU-tanh
(gelu ELF), and non-causal bidirectional MHA (flash_attn ELF). Only the
correctness-first host glue runs on CPU: per-Linear bias-adds, the two residual
adds, and the im2col patch-embed (all A3-6 fusion candidates, NOT fallbacks).

**NPU-execution: all ops on NPU (0 CPU fallback).**
- Unfused reference (`fused=False`, tag smolvla-vision-unfused-v1): 10
  dispatches/layer (LN1, q, k, v, FA, o, LN2, fc1, gelu, fc2) + 1 post_ln.
- **Fused production path (A3-6b, `fused=True`, default): 3 dispatches/layer**
  (`vit_ln_qkv`, `flash_attn`, `vit_o_ffn`) + 1 post_ln `layer_norm` = 37 total.
  All per-Linear bias-adds + both residual adds run ON-DEVICE inside the two
  fused ELFs (no host f32 glue). Only im2col patch-embed stays on host (one-time,
  pre-loop, not hot-loop work — see A3-6b Lever 3 remainder). See A3-6b section
  above for the 368 ms → 141.6 ms result and the post_ln 0.945 → 0.990 rise.

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

## A3-6b — vision performance hill-climb (DONE — gate met)

Baseline to beat: unfused NPU vision **281 ms** (median, compile excluded) vs CPU
719 ms. **Result: fused NPU vision 141.6 ms** (median of 10, this harness) — 2.0x
faster than the 281 ms unfused baseline, gate (< 281 ms) cleared decisively.

Measurement note: on the current machine this harness reproduces the *unfused*
baseline at ~368 ms (not 281 ms — machine/contention variance; the 281 ms figure
is from the original A3-6a run). The apples-to-apples delta on THIS machine is
368 ms → 141.6 ms (2.6x). Either way the fused wall is far below the 281 ms gate.

### Levers applied (each re-measured + re-verified against the oracle)

| Lever | What | Wall (this harness) | Dispatches | post_ln cos |
|---|---|---|---|---|
| baseline (unfused) | 10 ELF/layer, host bias/residual | 368 ms | 121 | 0.9455 |
| L1: BO intermediate reuse | mark kernel-overwritten output BOs `intermediate_indices` (skip host upload) | 368 ms | 121 | 0.9455 |
| L2+L3: fuse ELFs + on-device glue | 2 fused ELFs (vit_ln_qkv, vit_o_ffn) + FA; bias-adds + residuals moved on-device | **141.6 ms** | **37** | **0.9906** |

- **L1 (opt-buffer-object-reuse)**: applied. Weights were already static via
  `static_input_indices`; the remaining redundant traffic was re-uploading the
  zeroed output/scratch BOs every call. Marking them `intermediate_indices`
  removed that. Small standalone effect (368→368 ms, within noise — the per-op
  intermediate uploads were tiny vs the 1.75 ms/dispatch python/XRT overhead),
  but it is retained and is inherited by the fused block runner. Cosine held at
  0.9455.
- **L2 (opt-merge-multi-launch-kernels)** + **L3 (host glue on-device)**: the
  dominant win, applied together. Two fused ELFs built in `vit_fused_builders.py`
  (mirroring the backbone's rms_gemms_rope / o_ffn), stitched via
  `shared/infra/stitching.stitch_elf`:
    - `vit_ln_qkv`  (7 launches): affine LN1 + Q/K/V drain GEMM + Q/K/V on-device
      broadcast bias-add → q_b/k_b/v_b.
    - `flash_attn`  (1 launch, unchanged registry ELF).
    - `vit_o_ffn`  (10 launches): O GEMM + O bias + residual + affine LN2 + fc1
      GEMM + fc1 bias + GELU-tanh + fc2 GEMM + fc2 bias + residual → output.
  Per-layer weight/bias BOs pre-loaded once (`static_input_indices` +
  `bo_key=f"...L{i}"`); every scratch/output BO is `intermediate_indices`. Result:
  **121 → 37 dispatches** (3/layer + 1 post_ln), host-gap 212 ms → 4 ms, all
  per-Linear bias-adds + both residuals on NPU (no host f32 glue).

### Why post_ln cosine IMPROVED (0.945 → 0.990), not a masked bug

The gate says a cosine DROP = fusion bug. This is a RISE, and it is expected, not
suspicious: (1) each fused ELF was validated in isolation vs a faithful numpy
mirror of its exact sub-graph (`test_vit_fused_elfs.py`: vit_ln_qkv q/k/v cos
0.9999, vit_o_ffn out cos 0.9999) BEFORE end-to-end wiring, so the math is exactly
as designed; (2) the improvement comes from doing the bias-adds + residuals
on-device in bf16 vector lanes instead of the unfused path's host round-trip
(bf16→f32 add→bf16 recast per op), which removed a layer of intermediate
re-quantization that had been compounding the BFP16 systematic bias. The fused
path is strictly the more-accurate arithmetic, so the accepted 0.945 BFP16 ceiling
lifts. Correctness gate now PASSES outright (all per-layer > 0.98, post_ln > 0.99).

### Fused-ELF gotcha (recorded for reuse)

Both fused ELFs use DRAIN GEMMs, but at seq=1024 they resolve to two distinct
tile_n (qkvo/o/fc2 → 96, fc1 → 128). `compile_gemm_mm` bakes DIM_N into the
`mm.o` at compile time, and the plain `_m32` suffix is keyed only on method — so
a GEMM built in isolation would link a stale generic `mm_m32.o` (wrong DIM_N) and
produce garbage (first attempt: vit_ln_qkv cos ~0.07). Fix: force the
tile_n-keyed suffix (`_m32_n96` / `mm_m32_n96.o`) via
`vit_fused_builders._force_tile_n_suffix`, mirroring `disambiguate_by_tile_n`, so
every ELF links the correctly-baked object. This is the same class as the
backbone o_ffn's O/Down(80) vs Gate/Up(128) disambiguation.

### Lever 3 remainder (NOT done — out of scope, no gate impact)

im2col patch-embed stays on host: it is a ONE-TIME op before the layer loop (not
per-layer hot-loop work), so it costs nothing in the per-inference number the gate
measures. Moving it on-device would be a Conv/im2col kernel with no throughput
benefit here. bias-adds + residuals (the per-layer host glue) ARE all on-device.

### Deliverables

- `vit_fused_builders.py` — the two fused-ELF builders (+ _build_gelu_2d,
  _force_tile_n_suffix). Reuses shared bias-add (rms_qkv_bias_rope_multi) +
  residual add (o_ffn_multi) + stitch_elf.
- `vision_prefill.py` — `compile_all_kernels(..., fused=True)` (default) compiles
  the 3-ELF fused path; `run_vit_block_fused` drives 3 dispatches/layer with
  static weight BOs. `fused=False` keeps the frozen unfused path for A/B + diag.
- `test_vit_fused_elfs.py` — standalone NPU correctness of each fused ELF vs numpy.
- `bench_vision.py` — median-of-10 NPU vs CPU bench (mirrors bench_backbone.py).

## A3-5 Step 4-5 — connector on NPU + NPU vision spliced into the hybrid (DONE)

**Step 4 — connector (modality projection) on NPU.** `run_vit_encoder(...,
do_connector=True)` now runs the real thing: host `pixel_shuffle` (a pure
space-to-depth RESHAPE, no arithmetic, bit-exact vs HF) turns post_ln
(1024, 768) into (64, 12288), then the new **`gemm_connector` ELF**
(64x12288x960) does the projection on NPU. Registry row: drain, tile_m=16,
tile_k_l2=256, tile_k_l1=32, tile_n=80, **herd 4x4** — the registry's per-shape
herd override, needed because M=64 is only 4 tile_m rows and `build_module`
asserts `m % (tile_m*herd_m) == 0`. tile_m=16 also differs from the `_m32`
drain default, so the ELF links its own `mm_m16_n80.o` (symbol suffix
`_m16_n80`) and cannot collide with the encoder's `mm_m32_n{96,128}.o`.

Measured vs `vision_oracle.npz` (`make vision-connector` / `test_vit_connector.py`):

| check | cosine |
|---|---|
| ELF in isolation (oracle post_ln in) vs `connector` | **0.999988** |
| end-to-end (NPU encoder + NPU connector) vs `connector` | **0.996624** |
| scale sanity `NPU*sqrt(960)` vs `connector_scaled` | 0.996624 (norm ratio 0.968) |

The isolation number proves the kernel is clean; the e2e number is just the
encoder's own 0.9906 post_ln BFP16 drift propagated (see the A3-6b section).
The connector output returned is RAW — lerobot's `embed_prefix` applies the
`sqrt(960)` scale AFTER `embed_image`, so it must not be pre-applied here.

**Step 5 — spliced into the hybrid.** `run_npu_vision.py` (new bridge, mirrors
`run_npu_backbone.py`) encodes **all 3 cameras in ONE subprocess invocation**;
`smolvla_inference.run_hybrid_forward(npu_vision=True)` runs it once from inside
the wrapped `embed_prefix` and swaps `vlm_with_expert.embed_image` to serve the
3 precomputed results. Everything downstream — the sqrt(960) scale, language and
state tokens, prefix assembly, the NPU backbone, the CPU action expert — is
untouched lerobot code.

Gate (`make verify-npu-vision`, same pure-CPU lerobot action chunk as baseline):

| config | chunk cosine | nmse | gate |
|---|---|---|---|
| NPU backbone only (`make verify`) | 0.997144 | 0.007783 | **PASS** |
| NPU vision + NPU backbone | **0.993975** | **0.012201** | **PASS** |

Adding NPU vision costs ~0.003 of chunk cosine and still clears cos>=0.99 /
nmse<=0.04 with margin.

### Host-BLAS thread contention (new, reusable finding)

The vision driver loop is HOST-BOUND (37 dispatches/image, ~1.75 ms of
python+XRT each). OpenBLAS worker threads busy-spin after the one host matmul we
do (im2col), preempting the dispatch thread: encoder 135 ms -> 178 ms per image.
`bridge_common.limit_blas_threads()` (called before numpy is imported in BOTH
bridges) pins BLAS to 1 thread: costs ~4 ms on im2col, buys back ~40 ms/image
(~120 ms/inference over 3 cameras). Only the NPU bridge processes do this — the
lerobot driver keeps all threads for the CPU action expert, which IS BLAS-bound.

### ELF-cache reuse in the bridges

A bridge process is spawned per inference, so the old unconditional
`compile_all_kernels` rebuilt every ELF every call (~68 s for the backbone).
`bridge_common.ensure_kernels` reuses the on-disk cache when its manifest
resolves AND contains every expected kernel name, else rebuilds.
`SMOLVLA_FORCE_COMPILE=1` forces a rebuild (the manifest does not track source
hashes — set it after editing any kernel builder).

### End-to-end measurement (`make bench-e2e`, median of 5, fixed zero noise)

| config | as measured | bridge overhead | compute-only |
|---|---|---|---|
| pure CPU lerobot | **967 ms** | — | 967 ms |
| hybrid, NPU backbone | 2158 ms | 596 ms | 1562 ms |
| hybrid, NPU vision + backbone | 2589 ms | 1163 ms | 1425 ms |

"bridge overhead" = process spawn + imports + safetensors weight load + XRT/ELF
load + npz round-trip, measured by the bridges themselves; it is an artifact of
the two-disjoint-venv setup, not of NPU deployment.

Per-stage, the honest picture:

| stage | CPU (lerobot) | NPU (ours) |
|---|---|---|
| vision, per image | 222 ms | 148 ms warm / 335 ms first (cold XRT+BO) |
| vision, 3 cameras | 666 ms | 444 ms warm / 631 ms as the bridge runs it |
| connector (3x) | 5.3 ms | 1.6 ms/image (inside the above) |
| backbone prefill | 91 ms | 229 ms warm / ~378 ms cold |
| action expert + rest | 307 ms | (stays on CPU) |

**The vision tower is a genuine 1.5x NPU win** (148 vs 222 ms/image). The
end-to-end pipeline is still SLOWER than pure CPU, for three reasons, none of
which the vision work regressed:
  1. Our NPU backbone (229 ms warm) is 2.5x SLOWER than lerobot's torch CPU
     backbone (91 ms) at seq=256 — this was known and is Route 3 headroom
     (fuse the 11 attention dispatches/layer into the per-layer ELF).
  2. Each NPU stage is a fresh subprocess, so it always pays COLD-start NPU cost
     (XRT context + ELF load + full weight upload) instead of the warm cost.
  3. The CPU action expert (307 ms, 10 denoise steps) is untouched by A1/A3.

Splice artifact worth noting: the hybrid still lets lerobot run its own CPU
backbone prefill (91 ms) and then OVERWRITES the resulting KV cache with the NPU
K/V. That redundant work is the price of reusing lerobot's exact cache
structure; removing it is a follow-up, not a correctness issue.

## NPU-execution exceptions

These are the operations that run on CPU rather than NPU, with the concrete
reason for each. Everything else in the per-layer hot loop (RMSNorm, Q/K/V
GEMMs, RoPE, O-proj, residual, SwiGLU FFN) runs on NPU via the fused
`rms_gemms_rope` and `o_ffn` ELFs.

- **Connector `pixel_shuffle`** (A3-5 Step 4): host. NOT a fallback — it is a
  pure space-to-depth RESHAPE/transpose with zero arithmetic (`reshape` +
  `transpose` + `reshape`), done once per image outside any loop. The
  connector's only math, the 64x12288x960 projection, runs on NPU
  (`gemm_connector`). Verified bit-exact vs HF.

- **Vision im2col patch-embed**: host, one-time pre-loop (see A3-6b Lever 3
  remainder above). Unchanged by Step 4-5.

- **Prefix assembly, language/state token embedding, and the flow-matching
  action expert** (10 denoise steps, 307 ms): CPU lerobot, BY DESIGN. The A1/A3
  scope is the backbone + the vision tower; the expert is explicitly out of
  scope and runs the real lerobot code so the gate compares against an
  unmodified reference downstream.

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
- Vision (A3-6b fused, `fused=True`): **3 ELF dispatches/layer** (`vit_ln_qkv`,
  `flash_attn`, `vit_o_ffn`) x 12 layers + 1 `layer_norm` (post_ln) + 1
  `gemm_connector` = **38 dispatches per image**, 5 distinct ELFs. With 3
  cameras that is 114 dispatches per inference, all in one bridge invocation.

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

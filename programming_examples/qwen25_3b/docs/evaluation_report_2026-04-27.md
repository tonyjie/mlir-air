# Evaluation Report: Qwen2.5-3B BF16 on NPU2

**Reference deployment**: `../qwen25_1_5b/` (closest sibling — same Qwen2
family, hd=128, Option C wrapper, QKV bias, GQA padding helpers).
**Audit date**: 2026-04-27. **Auditor**: Independent fresh-context
evaluator (Phase 7 skill).

---

## 1. Current Status

### Verified (auditor re-run, Apr 27, 2026)

| Check | Result |
|---|---|
| `make verify` (NPU prefill + CPU LM-head workaround vs CPU F32) | **NPU top-1 == CPU top-1** (` Paris`, id=12095). Final logits cos **0.991**. **Greedy 5/5 PASS.** |
| `make run` × 2 byte-identical reproducibility | Yes — both runs: `'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain'` |
| Adversarial 3 prompts (NPU first token vs CPU top-1) | 3/3 EXACT match (`Light travels at`→` a`(264), `DNA stands for`→` de`(409), `The Pacific Ocean is the`→` largest`(7772)) |
| Phase 5 standalone NPU LM Head GEMV (W2 differential) | **FLAKY**: 1st run 0/4 (garbage `'�Paris]les'`), 2nd run 3/3 PASS, 3rd run 0/4 (same garbage). State-dependent. |
| W2 workaround code audit (anti-reward-hacking) | **HONEST**: `lm_head_f32 = np.asarray(orig_weights.lm_head, dtype=np.float32)` followed by genuine `(last_normed_f32.flatten() @ lm_head_f32.T)` numpy matmul. No hardcoded answer. |
| Anti-fallback (kernel ELFs present, prefill is real NPU) | OK: 3 prefill ELFs (rms_gemms_rope=3.3MB, flash_attn=2.4MB, o_ffn=5.6MB) + 3 decode ELFs (lm_head=3.0MB, o_gemv_ffn=1.3MB, rms_gemv_rope=0.6MB). Per-layer prefill 109 ms (much faster than CPU forward → real NPU). |

### Performance (auditor measurement, single trial)

| Phase | Per-layer | Total |
|---|---:|---:|
| NPU prefill (36 layers, NPU FA Option C, head_dim=128, 5-token prompt → 2048 padded) | **109 ms/layer** | **3.92 s** [TTFT] |
| First LM Head GEMV (CPU workaround) | — | **~21 ms** |
| Decode steady-state (median over 19 tokens; CPU LM head workaround) | 6.7 ms/layer | **240 ms/token (4.2 tok/s)** |
| Decode mean (incl. cold first-token) | — | 297 ms/token (3.4 tok/s) |

Per-layer prefill 109 ms tracks well with the family: vs Llama-3.2-3B (28L,
hd=128, hidden=8192, 126 ms/layer) and Qwen2.5-1.5B (28L, hd=128,
hidden=8960, 86 ms/layer warm). 36-layer model goes deeper; depth scaling
roughly linear.

### Manual Verify Commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/qwen25_3b

# (1) End-to-end smoke (~30 s warm cache, several min cold compile):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=20 \
    PROMPT="The capital of France is"
#   Expected first token: ' Paris' (id=12095)
#   Expected text: ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain'
#   Expected prefill:    ~3.9 s (109 ms/layer × 36L)
#   Expected decode:     240 ms/token (4.2 tok/s) median

# (2) NPU vs CPU F32 numerical verify (~1 min wall):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris'), logits cos ≈ 0.991
#   Greedy multi-token match: 5/5
#   K/V drift: L0 cos 1.000 → L35 cos 0.95–0.98 (BF16 deep-stack noise)
#   Note: 63/72 per-layer "WARN" (cos 0.94–0.99) — informational only;
#   final logits + greedy match remain stable.

# (3) Phase 5 standalone NPU LM Head differential (W2 probe):
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 qwen25_3b_phase5_test.py \
    --n-tokens 5 --cpu-verify
#   ⚠ FLAKY — observed both 0/4 garbage and 3/3 PASS in same audit session.
#   Behaviour appears to depend on prior NPU traffic.
```

If first token is **not** ` Paris` (id=12095) — first reaction:
`make clean` and re-run (stale-cache trap).

---

## 2. Architectural Differences vs Reference (Qwen2.5-1.5B)

| Field | Qwen2.5-1.5B | Qwen2.5-3B | Why it matters |
|---|---:|---:|---|
| n_layers | 28 | **36** | Deepest in catalog; per-layer BO arrays sized to 36 |
| emb_dim | 1536 (padded → 2048) | **2048** (no pad) | 1024-aligned natively → no emb padding cost; saves ~33% Q/O GEMM |
| n_heads | 12 | **16** | Wider parallelism, still even ✓ FA OK |
| head_dim | 128 | 128 | Same — Option C head-first FA wrapper reused |
| n_kv_heads | 2 (GQA group=6) | 2 (**GQA group=8**) | Same shape as qwen25_0_5b; W1 risk noted but cleared at hd=128 |
| hidden_dim (decode) | 8960 (padded → 8960) | **11008** | Larger FFN; needs new K-split factor for Down GEMV |
| hidden_dim (prefill) | 8960 | **12288** (padded from 11008) | New — padded for prefill BD-pool fit. Note: code uses 12288 (≠ CLAUDE.md’s 11264 plan; 12288 = 12×1024 chosen for tile fit). |
| vocab_size | 151936 | 151936 | Same — but partitions differ (see below) |
| LM head partitions | 10 × 16384 | **11 × 13824** | NEW — at K=2048 the 10×16384 tile_m=16 herd_m=8 config exceeds Rule D L2 cap by 256B (C buffer overhead); switch to tile_m=8 m_input=8 herd_m=8 with 11 partitions (152 064 ≥ 151 936) |
| Down GEMV K-split | 70 (K=8960) | **86** (K=11008) | NEW — Rule B/D forces split factor; mv_k11008.o uses DIM_M_OUTPUT=2 |
| QKV bias | Yes | Yes | Same — `qwen25_bias` host wrapper reused |
| Tied embeddings | Yes | Yes | Same |
| rope_θ | 1 000 000 | 1 000 000 | Same |

**Deltas that triggered new code**: (i) the LM-head re-partition to
11×13824 (driven by Rule D L2 cap), (ii) `mv_k11008.o` for Down GEMV at
K=11008 (driven by Rule B + Rule D), and (iii) the 11008→12288 prefill
padding through `qwen25_pad`. Everything else is parametric.

---

## 3. Implementation: Reused vs New

| What | Reused from | New (qwen25_3b-specific) |
|---|---|---|
| Per-layer prefill orchestration | `llama3_prefill.run_transformer_block`, `npu_prefill_with_kv_extraction` | — |
| Per-layer decode orchestration | `llama3_decode.run_decode_block` | — |
| Multi-launch ELF builders | `llama3.multi_launch_builder.{rms_gemms_rope, o_ffn, rms_gemv_rope, o_gemv_ffn, lm_head_gemv}` | — |
| KernelCache, BO preload, intermediate skip | `_llm_shared/kernel_builder/cache.py` | — |
| Option C head-first FA (hd=128) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| QKV bias post-RoPE add | `qwen25_bias.*` (from qwen25_1_5b) | — |
| GQA-aware hidden_dim padding | `qwen25_pad.*` (from qwen25_1_5b) | — |
| KV extraction in prefill | `_llm_shared/phase_helpers/prefill_runner.py` | — |
| Pre-transpose decode weights | `_llm_shared/phase_helpers/decode_setup.py` | — |
| **mv_k11008.o (Down GEMV K=11008)** | builds via `_compile_kernel(mv.cc)` with `-DDIM_M_OUTPUT=2 -Dmatvec_vectorized_bf16_bf16=dg_…` | **NEW: `qwen25_3b_decode_setup.ensure_mv_k11008_o`** |
| **11×13824 LM head GEMV** | builds via `build_lm_head_gemv_module(emb=2048, n_partitions=11, tile_m=8, m_input=8, herd_m=8)` | **NEW: `qwen25_3b_decode_setup.{compile_qwen25_3b_decode_kernels, preload_qwen25_3b_lm_head, qwen25_3b_npu_lm_head_gemv}`** |
| Weights / CPU reference / runner | — | model-specific scaffold (`qwen25_3b_{weights,reference,inference}.py`) |

The kernel topology (3 prefill XRT calls / layer; 2 decode XRT calls /
layer + CPU attention; final NPU LM Head — but currently CPU-fallback in
inference.py per W2) is identical to llama3-1B. The new code paths are
strictly the two extra-shape kernels (mv_k11008 + 11-partition LM head)
plus the prefill padding helpers.

---

## 4. End-to-End Inference Workflow

### Setup (one-time)

```
Compile (warm cache: ~3 s; cold: several min)
  external .o: silu_and_mul, rope, attn_npu2, mv (head_dim=128 variant),
               + mv_k11008.o (renamed-symbol Down GEMV K=11008)
  prefill ELFs (padded shapes emb=2048, hidden=12288):
    rms_gemms_rope.elf (6 launches merged)
    flash_attn.elf      (Option C head-first builder, hd=128)
    o_ffn.elf           (8 launches merged)
  decode ELFs (orig shapes emb=2048, hidden=11008):
    rms_gemv_rope.elf
    o_gemv_ffn.elf      (uses mv_k11008.o for Down GEMV)
    lm_head_gemv.elf    (11 partitions × 13824)

Weights (~15 s)
  HF safetensors → 36 layers (incl. QKV bias). lm_head TIED to embed_table.
  Pad weights for prefill (hidden 11008→12288 via GQA reindex).

BO preload + bias registry (~15 s)
  install_qkv_bias_wrapper()
  _register_all_layer_biases (precomputed RoPE-applied bias)
  pre_transpose_decode_weights()
  preload_prefill_weights()       (36 × 9 weight BOs)
  preload_qwen25_3b_lm_head()     (11 × 13824 LM head partitions)
```

### Prefill — runs 36 times, then once at end

```
PER LAYER (3 XRT calls): same as llama32_3b/qwen25_1_5b
  rms_gemms_rope (6 launches)  → flash_attn (Option C) → o_ffn (8 launches)

KV extraction host code:
  k_cache[layer] = results["k_roped"].reshape(seq, n_kv, hd).transpose(1,0,2)
  v_cache[layer] = results["v"]     .reshape(seq, n_kv, hd).transpose(1,0,2)

AFTER 36 LAYERS — first generated token (CURRENT WORKAROUND PATH):
  HOST: CPU final RMSNorm on 1×2048 vector
  HOST: numpy GEMV (1, 2048) @ (151936, 2048).T → 151936 logits  (~21 ms)
  HOST: argmax → first token id
```

### Decode — per token

```
PER LAYER (2 XRT calls + CPU attention): same as llama32_3b/qwen25_1_5b
  rms_gemv_rope → CPU GQA attention with KV cache → o_gemv_ffn

PER TOKEN (after 36 layers) — CURRENT WORKAROUND PATH:
  HOST: CPU final RMSNorm on 1×2048
  HOST: numpy GEMV (CPU LM head — W2 workaround)
  HOST: argmax → next token
```

### What's on NPU vs CPU

**On NPU**: All matmuls (GEMM/GEMV), all RMSNorm, all RoPE, FlashAttention
(prefill), SiLU×mul. **NPU LM Head ELF compiled and BO-preloaded but NOT
invoked** in `inference.py` — the W2 workaround routes the LM head through
CPU numpy.

**On CPU**: Tokenization, embedding lookup, Option C head-first transposes
(prefill FA wrap, ~3-5 ms/layer), KV cache reshape/transpose, decode-time
single-query GQA attention (per llama3 design), final RMSNorm at last
position, **W2 workaround LM head GEMV (~21 ms/token cost — would be ~150
ms NPU if NPU LM head were reliable)**, argmax over 151936-vocab.

---

## Notes

### W1 disposition (NPU FlashAttention precision)

Not observed at hd=128 + GQA group=8. Auditor confirmed K/V cache per-layer
cosine drift L0=1.000 → L35≈0.95-0.98 across 36 layers — within BF16
deep-stack accumulation noise band. **W1 (per-pos cos ≈ 0.94 collapse)
remains seq-first-FA-specific** as the deployment claimed; Option C
head-first wrapper at hd=128 sidesteps it cleanly. `make verify` shows
many per-layer WARN entries (cos 0.94-0.99) but final logits cos = 0.991
and 5/5 greedy match holds — these warnings are informational, not failures.

### W2 disposition — partial refutation of deployment narrative

The deployment recorded W2 as: *"NPU LM Head returns garbage when called
via inference.py path"*, with the claim *"Phase 5 standalone proves NPU
LM head works (4/4 NPU/CPU match)"*.

**Auditor finding**: Phase 5 standalone is **flaky, not always-passing**.
Three back-to-back runs in this audit session:

| Run | Result | Generated text |
|---|---|---|
| Phase 5 #1 (immediately after make verify) | 0/4 FAIL | `'…Paris�Paris]les'` (token id 112) |
| Phase 5 #2 (immediately after #1) | 3/3 PASS | `'…Paris. The capital'` |
| Phase 5 #3 (immediately after #2) | 0/4 FAIL | `'…Paris�Paris]les'` (same garbage) |

This **refutes** the deployment's claim that Phase 5 standalone always
gives 4/4. The actual symptom is a **state-dependent NPU LM Head GEMV
correctness bug** that affects both Phase 5 standalone and inference.py
runner. The CPU LM head workaround in `qwen25_3b_inference.py` (lines
214–229, 337–340) is **honest** — it computes a real numpy matmul against
`orig_weights.lm_head`, no hardcoded answer — and gives correct output,
which is why `make verify` still passes 5/5 greedy.

**Severity**: HIGH for paper-citable correctness. The deployment's
"Phase 5 4/4 match" data point is **not reliably reproducible** as a
proof that NPU LM Head works. Recommended actions:

1. **Do not cite "Phase 5 standalone NPU LM head 4/4"** as a stable
   correctness result without first reproducing it ≥ 5 times in a row.
2. The W2 root cause hypothesis (BO state contamination) deserves
   investigation: try re-`load_and_run` with `static_input_indices`
   detached, or invalidate the cached BOs between calls.
3. Until W2 is root-caused, the CPU LM head workaround is the only
   reliable production path. The 11×13824 NPU LM head ELF + preload code
   exists and compiles but is effectively dead code in production.

### Anti-reward-hacking audit (W2 workaround)

`qwen25_3b_inference.py` line 215: `lm_head_f32 = np.asarray(orig_weights.lm_head, dtype=np.float32)` — real weights.
Lines 222-223: `last_normed_f32 = qwen25_3b_reference.rms_norm(last_hidden, orig_weights.final_norm); first_logits = (last_normed_f32.flatten() @ lm_head_f32.T).astype(np.float32)` — real GEMV.
Line 338: `next_logits = (x_normed_f32.flatten() @ lm_head_f32.T).astype(np.float32)` — same in decode loop.
**No hardcoded token ids, no answer caching, no shortcut.** The W2
workaround is honest and produces real CPU logits.

### Anti-fallback heuristics

- All 6 ELFs present and recently built (Apr 27 14:13–14:15).
- Per-layer prefill 109 ms — much faster than a 3 GB CPU forward (would
  be O(seconds) per layer at this shape) → confirms NPU is doing the
  work.
- TTFT 3.92 s vs Llama-3.2-3B's 3.5 s and Qwen2.5-1.5B's 2.4 s — fits
  the 36L × hd=128 family scaling.

### Cross-deployment regression (Step 7)

The audit branch (`llm_mapping`) carries 107 changed files under
`programming_examples/(_llm_shared/|matrix_vector_multiplication/|llama3/)`
relative to `main`. Re-running every sibling deployment is out of scope
for a Phase 7 audit; recommend a separate regression sweep (running
`make verify` on each of llama3, llama32_3b, smollm2_1_7b, qwen25_0_5b,
qwen25_1_5b, qwen3_0_6b, qwen3_1_7b) before merging to main.

---

## File Map

| File | Role | Lines |
|---|---|---:|
| `qwen25_3b_inference.py`     | End-to-end runner (`make run`/`make verify` entry; CPU LM-head workaround for W2) | 423 |
| `qwen25_3b_weights.py`       | HF safetensors loader (incl. QKV bias, tied embeddings) | — |
| `qwen25_3b_reference.py`     | CPU F32 reference (used for `make verify`) | — |
| `qwen25_3b_decode_setup.py`  | NEW: mv_k11008.o + 11×13824 LM head GEMV (compile/preload/run) | 261 |
| `qwen25_3b_phase{1,2,3,4,5}_test.py` | Per-phase validation scripts | — |
| `Makefile`                   | `make run` / `make verify` / `make verify-decode` / `make compile` / `make clean` | — |
| `build/{prefill,decode}_kernel_cache/` | 3 prefill + 3 decode ELFs + manifest.json | — |

Imports from siblings: `qwen25_bias.*` and `qwen25_pad.*` (sys.path
import from `../qwen25_1_5b/`); `llama3_prefill.*`, `llama3_decode.*`
(from `../llama3/`); `_llm_shared.{kernel_builder,phase_helpers}.*`.

---

## Final Verdict

**PASS-with-warnings.**

The production path (`make run` + `make verify` with the W2 CPU LM head
workaround) is correct, reproducible, and matches the CPU F32 reference
on the canonical and adversarial prompts (1 + 3/3 adversarials, 5/5
greedy multi-token, byte-identical between runs). The W2 workaround is
honest (real numpy matmul, no shortcuts).

**Warning 1 (HIGH)**: The W2 root cause is **more severe than the
deployment narrative suggests**. Phase 5 standalone is flaky (observed
0/4 → 3/3 → 0/4 in three back-to-back runs), so the claim "Phase 5
proves NPU LM head works" is not reproducible. The 11×13824 NPU LM
head ELF is effectively unused production code until W2 is root-caused.

**Warning 2 (LOW)**: 63/72 per-layer K/V cosine "WARN" entries
(cos 0.94-0.99) — informational, not failures. Final logits cos=0.991
and 5/5 greedy match are the real gates and they pass.

**Warning 3 (PROCESS)**: 107 cross-deployment changes on this branch
were not regression-tested in this Phase 7 audit (out of scope). A
sibling-deployment sweep is recommended before merging.

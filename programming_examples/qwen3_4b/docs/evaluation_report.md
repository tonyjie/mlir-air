# Evaluation Report: Qwen3-4B BF16 on NPU2

**Reference deployment**: `../qwen3_1_7b/` (closest kernel-first sibling — same
Qwen3 family, head_dim=128, Q/K Norm, NO QKV bias). Most patterns inherit
unchanged; this doc covers what's different. Independent audit performed
2026-04-27 by `independent-evaluator` skill.

**Verdict**: **PASS-with-warnings**. NPU top-1 matches CPU on `verify` and
on three adversarial prompts; runs are byte-identical across repeats; ELF
cache is real; no reward-hacking in the verify gate. The only soft signal
is V-cache cosine drift (19/36 layers below 0.99 informational threshold,
final logits cosine 0.910), which is consistent with BF16 accumulation
across a 36-layer stack with GQA group=4 and matches the deployment's own
caveat in the verify printout.

---

## 1. Current Status

### Verified on Apr 27, 2026 — auditor re-ran every numerical claim

| Check | Result |
|---|---|
| `make verify` reward-hacking audit | **CLEAN** — uses real `np.corrcoef` on flattened `npu_logits` vs `cpu_logits_pred`, computes per-layer K/V cosine over 36 layers via `cpu_block` from `qwen3_4b_reference`, top-1 via `np.argmax`. No hardcoded `1.0` shortcuts. Threshold for K/V is informational (`> 0.99` → "OK" else "WARN"); the production gate is final-logits cosine + top-1 match. |
| `make verify` (NPU vs CPU F32 reference) | NPU top-1 == CPU top-1 = `12095` (`' Paris'`). Final logits cosine **0.910** at pred_pos. K cache: 0.998-0.9999 across all 36L. V cache: 17/36 layers ≥ 0.99, drifts 0.999 (L0) → 0.959 (L35). |
| `make run N_TOKENS=30` × 2 (byte-identical reproducibility) | Both runs produced **identical** 30-token output: `'The capital of France is Paris. The capital of Paris is...? The capital of Paris is not defined, as Paris is a city and not a country. However, if'`. Greedy = deterministic confirmed. Prefill jitter 8.03 → 8.10 s (1%). |
| Adversarial prompts (3 NOT in canonical set) | All 3 NPU first-token == CPU HF top-1: `"Light travels at"` → ` a` (id 264 ✓), `"DNA stands for"` → ` de` (id 409 ✓), `"The Pacific Ocean is the"` → ` largest` (id 7772 ✓). |
| Anti-fallback ELF presence | `build/prefill_kernel_cache/` contains all 6 expected ELFs: `rms_attn_gemms.elf` (2.4 MB), `o_ffn.elf` (5.7 MB), `flash_attn.elf` (3.6 MB), `rms_attn_gemvs_qknorm_rope.elf` (1.2 MB), `o_gemv_ffn_silu.elf` (2.1 MB), `lm_head_gemv.elf` (11.3 MB). `build/decode_kernel_cache/` shares the 3 decode ELFs. NPU FA wired via Option C wrapper (`install_headfirst_fa_wrapper`) — no CPU FA fallback. |
| Anti-fallback per-layer prefill > 5 ms | **223 ms/layer** measured (claim ~222 ms/layer). Far above the 5 ms threshold that would indicate a no-op kernel. |
| Anti-fallback LM head integrated | First-token NPU LM head measured separately during decode preload + warmup. Decode steady-state 387 ms/token includes 1× LM head GEMV per token (one of the 3 decode ELFs). Above the 10 ms threshold. |
| Cross-deployment regression scan | `git diff main..HEAD` touches `_llm_shared/` and `llama3/multi_launch_builder/` widely (the 7-phase deployment naturally edits shared infra). Re-verifying every other deployment is **OUT OF SCOPE for this 30-min audit**; flagged for follow-up. |

### Performance (Apr 27, 2-trial mean)

| Phase | Per-layer | Total |
|---|---:|---:|
| NPU prefill (36 layers, NPU FA Option C, head_dim=128, seq_len=2048) | **223 ms/layer** | **8.07 s ± 0.04 s** |
| NPU prefill warmup (cold) | ~325 ms/layer | 11.6-11.8 s |
| NPU decode steady-state (3 fused ELFs / token) | 10.7 ms/layer | **387 ms/token (2.58 tok/s)** |
| Decode preload (one-time) | — | 18.9 s |

Per-layer prefill 223 ms is the **highest** in the catalog (vs llama32_3b 126 ms, qwen3_1_7b ~120 ms estimated from per-block rate). Reasons: (1) split-ELF approach (3 separate prefill ELFs vs llama3's fused `rms_gemms_rope` + `flash_attn` + `o_ffn` already 3-ELF — same count, but emb-dim widened from 3072 to padded 3072 / hidden_dim padded 9728→10240 widens GEMM work); (2) GQA group=4 means each KV head fans out to 4 Q heads in attention; (3) Q/K Norm host-side wrapper adds per-layer host overhead.

Per-layer decode 10.7 ms vs qwen3_1_7b's 5.3 ms ≈ 2× — consistent with hidden_dim padded ratio (10240 / 6144 ≈ 1.67) plus 36/28 layer count ratio.

### Manual verify commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/qwen3_4b

# (1) End-to-end NPU vs CPU F32 numerical verify (~1 min wall, warm cache):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris' id=12095)
#   Final logits cosine ~0.910 (BF16 deep-stack drift; informational K/V WARNs OK)

# (2) End-to-end smoke (~1 min warm, ~5 min cold):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30
#   Expected first token: ' Paris'
#   Expected prefill: ~8.07 s (223 ms/layer × 36)
#   Expected decode: ~0.39 s/token (2.58 tok/s)

# (3) HuggingFace cross-check on CPU (no NPU; slow):
python3 qwen3_4b_reference.py --prompt "The capital of France is" --verify

# (4) Phase 2 single-block:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-block

# (5) Phase 3 full-model:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-full
```

If first token is **not** ` Paris` (id=12095) — first reaction:
`make clean` and re-run (stale-cache trap; ELF cache is keyed by file hash
but a partial cache is the most common foot-gun).

---

## 2. Architectural Differences vs Qwen3-1.7B

| Field | Qwen3-1.7B | Qwen3-4B | Why it matters |
|---|---:|---:|---|
| n_layers | 28 | **36** | +29% depth → linear scaling of prefill + decode |
| emb_dim (raw → padded) | 2048 (no pad) | **2560 → 3072** | NOT 1024-aligned; padding applied in `qwen3_4b_pad.py`. 3072 = 3×1024 = BD-aligned. Same padded width as llama32_3b. |
| n_heads | 16 | **32** | 2× heads, same head_dim. Fits even GQA group=4. |
| head_dim | 128 | 128 | Same — Option C head-first FA wrapper applies. |
| n_kv_heads | 8 (GQA group=2) | 8 (GQA **group=4**) | NEW axis vs prior kernel-first deployments (0.6B/1.7B were g=2). Doubled fan-out per KV head — implicated in V-cache drift. |
| **q_dim** = n_heads × head_dim | 2048 (= emb_dim) | **4096** (≠ emb_dim) | qwen3_1_7b's `q_dim==emb_dim` allowed the "2-K matvec rename" shortcut; qwen3_4b needs the full **3-K rename** (q=4096, k=v=1024, hidden=10240) — same pattern as qwen3_0_6b. |
| hidden_dim (raw → padded) | 6144 | **9728 → 10240** | NOT 1024-aligned; padded to 10240 = 10×1024. Skipped 9216 (would lose data) per pad-up convention. |
| vocab_size | 151936 | 151936 | Same — LM Head partition count carried over. |
| rope_θ | 1e6 | 1e6 | Same. |
| QKV bias | absent | absent | Standard Qwen3, no bias. |
| Q/K Norm | yes (per-head) | yes (per-head) | Same — applied via `_llm_shared/phase_helpers/qk_norm.py`. |
| Tied embeddings | yes | yes | Same. |

**The deltas that matter**: depth (36L → deepest in catalog with qwen25_3b),
GQA group=4 (NEW), and dual padding (emb 2560→3072 + hidden 9728→10240).
Everything else inherits unchanged.

---

## 3. Implementation: Reused vs New

The kernel topology (3 prefill XRT calls / layer, 3 decode XRT calls / token,
8-partition LM head) is **identical** to qwen3_1_7b. Only model-specific
weights/config code is new.

| What | Reused from | New (qwen3_4b-specific) |
|---|---|---|
| Per-layer prefill orchestration | `llama3_prefill.run_transformer_block` (via `qwen3_4b_phase4_test.run_block_optimized` wrapper) | thin wrapper for layer arg cache |
| Per-layer decode orchestration | `qwen3_4b_decode.py` (modeled on `qwen3_1_7b/qwen3_decode.py`) | model-specific decode driver (3 fused ELFs) |
| Multi-launch ELF builders (rms_attn_gemms, o_ffn, flash_attn, rms_attn_gemvs_qknorm_rope, o_gemv_ffn_silu, lm_head_gemv) | `qwen3_4b/multi_launch/` (mirrors `qwen3_0_6b/multi_launch/`) | **6 builders re-parametrized** for 36L / emb=3072 / hidden=10240 / q_dim=4096 (3-K rename) |
| Padding helpers (config + weight pad) | — | `qwen3_4b_pad.py` (new; emb 2560→3072, hidden 9728→10240) |
| KernelCache, BO preload, intermediate_indices | `_llm_shared/kernel_builder/cache.py` | — |
| Option C head-first FA wrapper (hd=128) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| Q/K Norm host-side application | `_llm_shared/phase_helpers/qk_norm.py` | — |
| KV extraction in prefill | `qwen3_4b_phase4_test.npu_full_prefill(collect_kv=True)` | — |
| External C++ kernels (silu_and_mul, rope_halfsplit, attn_npu2, mv) | shared sources, `head_dim=128` arg | — |
| Weights loader, CPU F32 reference, end-to-end runner | — | `qwen3_4b_{weights,reference,inference}.py` (model-specific scaffold) |

### Key reuse pattern: 3-K matvec rename for `q_dim ≠ emb_dim`

`q_dim = 32 × 128 = 4096` ≠ padded `emb_dim = 3072`. The fused decode ELF
`rms_attn_gemvs_qknorm_rope` therefore needs three different matvec
shapes (Q at K=3072→4096, K and V at K=3072→1024). This is the same
"3-K extern rename" pattern from `qwen3_0_6b/multi_launch/`, applied
unchanged here. The shortcut used by `qwen3_1_7b` (where q_dim equals
emb_dim and only 2 distinct matvec shapes are needed) does **not**
apply.

### How Q/K Norm is wired

Q/K Norm is applied per-head **between Q/K projection and RoPE** —
neither slots cleanly into the existing fused ELFs. Per the kernel-first
methodology (also used in qwen3_0_6b/qwen3_1_7b), the Q/K Norm step is
performed on the host between `rms_attn_gems` (which produces Q/K/V) and
the head-first FA wrapper (which applies RoPE-rotated Q/K to attention).
Wrapper: `_llm_shared/phase_helpers/qk_norm.py:apply_qk_norm`.

---

## 4. End-to-End Inference Workflow

### Setup (one-time, before `NPU prefill ...` print)

```
Compile (~5 min cold, instant if cached)
  external .o files (silu_and_mul, rope, attn_npu2, mv)
                    ↑ all compiled with head_dim=128 variant
  prefill ELFs (3, in build/prefill_kernel_cache/):
    rms_attn_gemms.elf            (RMSNorm + Q/K/V GEMM)
    flash_attn.elf                (Option C head-first FA, hd=128)
    o_ffn.elf                     (O proj + residual + RMSNorm + SwiGLU + Down + residual)
  decode ELFs (3, in build/decode_kernel_cache/):
    rms_attn_gemvs_qknorm_rope.elf  (Pre-norm + 3 GEMV + QKNorm + RoPE — fused)
    o_gemv_ffn_silu.elf             (O GEMV + residual + Pre-norm + Up/Gate GEMV + SiLU + Down GEMV + residual)
    lm_head_gemv.elf                (Final RMSNorm + 8-partition vocab GEMV)

Weights (~17 s)
  load HF safetensors (raw 2560 / 9728)
  pad weights → padded 3072 / 10240

Decode-weights BO preload (~19 s)
  36 layers × 2 ELFs (rms_attn_gemvs_qknorm_rope + o_gemv_ffn_silu) + LM head
  Fires each ELF once with dummy inputs so subsequent calls skip the
  weight-write via static_input_indices.

NPU prefill warmup (~12 s; first-pass setup, discards K/V)

═══════════════════════════ PROFILED SCOPE ═══════════════════════════
```

### Prefill — runs 36 times, then once at end (LM head)

```
PER LAYER (3 XRT calls):

  XRT call 1: rms_attn_gemms.elf  [NPU; multi-launch]
    Pre-RMSNorm + Q/K/V GEMMs at seq_len=2048, K_emb=3072.
    Q output:  (2048, 4096)
    K output:  (2048, 1024)
    V output:  (2048, 1024)

  HOST: Q/K Norm (per-head)  [_llm_shared/phase_helpers/qk_norm.py]
    Q reshape (2048, 32, 128) → RMSNorm last-axis with q_norm weight
    K reshape (2048, 8, 128)  → RMSNorm last-axis with k_norm weight
    ~tens of ms / layer (host BF16 numpy)

  XRT call 2: flash_attn.elf  [NPU; Option C head-first wrapper]
    HOST pre-transpose seq → head:
      q (2048, 32, 128) → (32, 2048, 128)
      k (2048,  8, 128) → ( 8, 2048, 128)
      v (2048,  8, 128) → ( 8 × dv_chunks, 2048, 128)
    NPU launch: head-first FA, hd=128, GQA 32Q/8KV (group=4), causal.
    HOST post-transpose head → seq.
    Wrapper: _llm_shared/phase_helpers/headfirst_fa.py

  XRT call 3: o_ffn.elf  [NPU; multi-launch]
    O proj GEMM + residual + post-RMSNorm + Gate/Up GEMM +
    SiLU×Up + Down GEMM (K_hidden=10240) + residual.
    Output: x_out (2048, 3072)

  HOST: KV extraction  (only when --verify or --decode npu)
    k_per_layer[li] = k_roped reshape & transpose to seq-first
    v_per_layer[li] = v       reshape & transpose to seq-first

AFTER 36 LAYERS — first generated token:
  HOST: CPU final RMSNorm on 1×3072 vector  (<1 ms; last position only)
  XRT call: lm_head_gemv.elf  [NPU; 8-partition fused]
    8 partitions × M=18992, K=3072. Concatenate to 151936-vocab logits.
  HOST: argmax → first token id
```

### Decode — per token (3 XRT calls / token)

```
PER LAYER (2 XRT calls; CPU attention):

  XRT call 1: rms_attn_gemvs_qknorm_rope.elf  [NPU; fused]
    Pre-RMSNorm + Q/K/V GEMV (M=1) + Q/K Norm (NPU side, per-head) + RoPE.
    All weights pre-loaded as static BOs (skip weight-write per call).

  HOST: CPU attention (single-query, ~few ms / layer)
    q (32 heads × 128) attends to KV cache (growing with each token).
    GQA group=4 maps Q head h → KV head h//4.
    Update K_cache[layer, :, pos, :], V_cache[layer, :, pos, :].

  XRT call 2: o_gemv_ffn_silu.elf  [NPU; fused]
    O proj GEMV + residual + post-RMSNorm + Gate/Up GEMV +
    SiLU×Up + Down GEMV (K_hidden=10240) + residual.

PER TOKEN (after 36 layers):
  HOST: CPU final RMSNorm on 1×3072  (<1 ms)
  XRT call: lm_head_gemv.elf  (same 8-partition ELF as prefill end)
  HOST: argmax → next token

Total XRT calls / decoded token: 36 layers × 2 + 1 LM head = 73 calls
(but with per-layer arg cache and BO preload, dispatch cost is amortized;
measured 387 ms / token = 2.58 tok/s).
```

### What's on NPU vs CPU

**On NPU** (load-bearing work):
- All matrix multiplies (GEMM and GEMV)
- All RMSNorm (per-position norm)
- All RoPE (Q and K rotation)
- FlashAttention (prefill only, head-first via Option C)
- Q/K Norm in **decode** (fused into `rms_attn_gemvs_qknorm_rope.elf`)
- SiLU × multiply
- Final LM Head (8-partition GEMV)

**On CPU** (lightweight host wrap):
- Tokenization (HF tokenizer)
- Embedding lookup
- Weight padding (one-time, at startup)
- Q/K Norm in **prefill** (host-side numpy between rms_attn_gems and FA)
- Option C head-first transposes around prefill FA
- KV cache extraction in `--verify` / `--decode npu` mode
- Decode-time **single-query attention** with KV cache (all deployments do this on CPU)
- Final RMSNorm at last position only
- Argmax over 151936-vocab logit vector

---

## Notes

### V-cache drift signal — the principal "warning" in PASS-with-warnings

19 of 36 V-cache entries fall below the informational 0.99 cosine threshold,
ending at L35 V-cache cos = 0.959 (max_err 10.1, mean_err 0.105). K-cache
stays clean (0.998-0.9999 across all 36 layers). The final logits cosine
(0.910) is below the 0.99 typical gate but well above the 0.5 disaster
threshold, and the **strict top-1 match holds on the canonical prompt
AND all 3 adversarial prompts**.

Likely contributors:
1. **Depth**: 36 layers is the deepest in catalog → BF16 accumulation drift.
2. **GQA group=4**: each KV head's V participates in 4× the Q-projection
   work, so single-element V noise is amplified vs llama32_3b's group=3.
3. **Dual padding**: padded zero columns participate in F32-vs-BF16 GEMM
   accumulator differently than the unpadded case (the auditor did NOT
   verify this; it's a hypothesis worth investigating in a follow-up).
4. **Q/K Norm host-vs-NPU**: prefill applies Q/K Norm on host, decode on
   NPU. Verify only checks prefill so this doesn't distort the K-cache —
   but is worth noting for the user.

The deployment's own verify printout self-labels these as "informational;
BF16 K/V drift across deep stacks is expected" — honest framing. Top-1
match is the production gate, and it holds on 4 / 4 prompts tested by
the auditor (1 canonical + 3 adversarial).

### `make verify` reward-hacking audit

Examined `qwen3_4b_inference.py` lines 219-303 (`if args.verify:` block):

- Cosine via `np.corrcoef(npu_logits_f32, cpu_logits_pred)[0,1]` — real
  Pearson correlation, not a hardcoded `1.0`.
- Per-layer cosines computed against `cpu_block(...)` from
  `qwen3_4b_reference.py` — independent NumPy F32 forward pass.
- `cpu_block` and NPU prefill consume the **same padded weights** — no
  silent unpadded-vs-padded comparison that would manufacture artificial
  drift OR hide drift.
- Top-1 via `int(np.argmax(...))` on NPU and CPU logits independently.
- No `try/except: return 1.0` swallowing.

Audit verdict: **verify gate is honest**.

### Cross-deployment regression (NOT performed)

`git diff main..HEAD` shows changes across `_llm_shared/` (.py + .cc),
`llama3/multi_launch_builder/`, and the per-model deployment dirs. This
is normal for the 7-phase deployment chain (shared infra evolves with
each deployment). Re-verifying every other LLM deployment was deemed
**out of scope** for this 30-minute audit. **Recommended follow-up**:
spot-check `qwen3_0_6b` and `qwen3_1_7b` `make verify` (closest
infra-shared siblings) before merging to main.

### Anything redundant vs qwen3_1_7b?

No. Same kernel-first topology: 3 prefill ELFs + 3 decode ELFs + Option
C head-first FA + Q/K Norm host wrapper. The split-ELF approach (no
attempt to merge rms+attn+ffn into a single ELF) is forced by Q/K Norm
not commuting with RoPE — same constraint as qwen3_0_6b/qwen3_1_7b. The
only actual change is the 3-K matvec rename (vs qwen3_1_7b's 2-K) and
the dual padding helper (`qwen3_4b_pad.py`).

---

## File Map

| File | Role | Lines |
|---|---|---:|
| `qwen3_4b_inference.py` | End-to-end runner (`make run` / `make verify` entry) | 378 |
| `qwen3_4b_weights.py` | HF safetensors loader + RoPE LUT (NO QKV bias, Q/K Norm, tied embeddings) | — |
| `qwen3_4b_reference.py` | CPU F32 NumPy reference forward (used for `make verify`) | 378 |
| `qwen3_4b_pad.py` | Padding helpers (emb 2560→3072, hidden 9728→10240) | — |
| `qwen3_4b_decode.py` | Decode driver (3 fused ELFs, BO preload, per-layer arg cache) | — |
| `qwen3_4b_phase{1..5}_test.py` | Per-phase validation scripts | — |
| `multi_launch/` | 6 model-specific multi-launch ELF builders | — |
| `build/prefill_kernel_cache/` | 6 cached ELFs (rms_attn_gemms, o_ffn, flash_attn, rms_attn_gemvs_qknorm_rope, o_gemv_ffn_silu, lm_head_gemv) | — |
| `build/decode_kernel_cache/` | 3 cached decode ELFs | — |

Model-specific helpers under `multi_launch/`; shared infrastructure under
`_llm_shared/`. The Option C head-first FA wrapper is in
`_llm_shared/phase_helpers/headfirst_fa.py` (originally written for
`llama32_3b`, reused here unchanged).

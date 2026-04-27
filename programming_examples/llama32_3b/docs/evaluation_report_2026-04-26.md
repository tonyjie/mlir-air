# Evaluation Report: Llama-3.2-3B BF16 on NPU2

**Reference deployment**: `../llama3/` (Llama-3.2-1B). Most things are
inherited unchanged; this doc covers what's different.

---

## 1. Current Status

### Verified ✓ (Apr 26, 2026 — 5 of 5 protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: independent-evaluator`) | Apr 25: **PASS-with-warnings** (see legacy `evaluation_report_2026-04-25.md` content for full audit categories). Warnings were perf-related, not correctness. |
| `make run` smoke (Apr 26) | First token ` Paris` (id=12366). 5-trial mean prefill **3.518 s ± 4 ms**. |
| `make verify` (NPU vs CPU F32 reference) | NPU top-1 == CPU top-1 (` Paris`). Final logits cosine **0.993** at pred_pos. K/V cache cosine drift **0.999 → 0.987 over 28 layers** (within head_dim-scaled BF16 noise floor). |
| HuggingFace F32 cross-check on CPU reference | top-1 ` Paris`, logits correlation > 0.9999 vs HF |
| Code review | inference / weights / reference all clean; no silent CPU fallback; cpu_attn defaults False (NPU FA via Option C) |

### Performance (Apr 26, 5-trial mean)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (28 layers, NPU FA Option C, head_dim=128) | 126 ms/layer | **3.518 s ± 4 ms** |
| First LM Head GEMV (cold; warm ≈ 22 ms) | — | ~22 ms |
| Decode steady-state | 7.7 ms/layer | **215 ms/token** (4.7 tok/s) |

Per-layer prefill 126 ms vs llama3-1B's 79 ms → **1.6× scaling**, tracks
the K-width ratio (3072/2048 = 1.5×) almost linearly. Per-layer decode
7.7 ms vs llama3-1B's 5.75 ms → **1.34×** at the same K-width ratio,
indicating kernels at the per-byte ceiling.

### Manual Verify Commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama32_3b

# ⚠️ FIRST: if you (or anyone) recently edited _llm_shared/ or
# llama3/multi_launch_builder/, do this to force ELF rebuild:
flock -x -w 1800 /tmp/mlir-air-npu.lock make clean

# (1) End-to-end smoke (~30 s warm cache, ~4 min cold compile):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30 PROMPT="The capital of France is"
#   Expected first token: ' Paris' (id=12366)
#   Expected prefill:     3.5 s ± 5 ms (5-trial mean)
#   Expected decode:      215 ms/token (4.7 tok/s)

# (2) NPU vs CPU F32 numerical verify (~1 min wall):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris'), logits cos ≈ 0.993
#   K/V drift: L0 cos 0.999 → L27 cos 0.987 (well within BF16 noise)

# (3) HuggingFace cross-check on CPU side (no NPU needed):
python3 llama32_3b_reference.py --prompt "The capital of France is" --verify
#   Expected: top-1 ' Paris', logits correlation > 0.9999 vs HF F32

# (4) Phase 2 single-block (~30 s; head-first FA correctness):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-block
#   Expected: cosine ≥ 0.9996 (real), per-pos min ≥ 0.998

# 1-minute sanity:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=5 PROMPT="The capital of France is" 2>&1 | grep -E "first|Tok"
```

If first token is **not** ` Paris` (id=12366) — first reaction:
`make clean` and re-run (stale-cache trap).

---

## 2. Architectural Differences vs Llama-3.2-1B

| Field | Llama-3.2-1B | Llama-3.2-3B | Why it matters |
|---|---:|---:|---|
| n_layers | 16 | **28** | Linear scaling; per-layer BO arrays sized to 28 |
| emb_dim | 2048 | **3072** | 1.5× wider K in all projections; 3 × 1024 (still BD-aligned) |
| n_heads | 32 | 24 | Same total Q work as 1B (24 × 128 = 32 × 64 = wide) |
| **head_dim** | 64 | **128** | Triggers seq-first FA hang at dk_chunks>1 → forces Option C wrapper |
| n_kv_heads | 8 (GQA group=4) | 8 (GQA group=3) | Slightly tighter sharing; KV cache size moderate |
| hidden_dim | 8192 | 8192 | Same — FFN ELFs identical |
| vocab_size | 128256 | 128256 | Same — LM Head 8-partition pattern unchanged |
| rope_θ | 500k | 500k | Same |
| QKV bias | absent | absent | Standard Llama, no bias |
| Tied embeddings | yes | yes | Same |

**The single delta that matters**: `head_dim=128`. Everything else is
parametric (tile configs absorb the 1.5× K-width without retuning). The
hd=128 change is what forced the Option C head-first FA wrapper —
detailed in §3.

---

## 3. Implementation: Reused vs New

The kernel topology (3 XRT calls / prefill layer, 2 / decode layer + CPU
attention, 8-partition LM head) is **identical** to llama3. There are
**no model-specific helpers in this directory** — every operation goes
through shared infrastructure.

| What | Reused from | New (llama32_3b-specific) |
|---|---|---|
| Per-layer prefill orchestration | `llama3_prefill.run_transformer_block` | — |
| Per-layer decode orchestration | `llama3_decode.run_decode_block` | — |
| Multi-launch ELF builders (rms_gemms_rope, o_ffn, rms_gemv_rope, o_gemv_ffn, lm_head_gemv) | `llama3.multi_launch_builder.*` | — |
| KernelCache, BO preload, `intermediate_indices` skip | `_llm_shared/kernel_builder/cache.py` | — |
| **Option C head-first FA wrapper (hd=128)** | **`_llm_shared/phase_helpers/headfirst_fa.py`** | **— (originally written here, now shared with qwen25/qwen3)** |
| KV extraction in prefill | `_llm_shared/phase_helpers/prefill_runner.py` | — |
| Pre-transpose decode weights, NPU LM Head GEMV | `_llm_shared/phase_helpers/decode_setup.py` | — |
| External C++ kernels (silu_and_mul.cc, rope.cc, attn_npu2.cc, mv.cc) | shared sources, compiled with `head_dim=128` arg | — |
| Weights loader, CPU F32 reference, end-to-end runner | — | model-specific scaffold (`llama32_3b_{weights,reference,inference}.py`) |

### How Option C (head-first FA) actually works

**The problem**: the seq-first FlashAttention kernel
(`flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`) used by
llama3-1B hangs at runtime when `dk_chunks > 1` (required for hd=128 +
seq_len=2048 at L1 budget 64 KB). Symptom: `ERT_CMD_STATE_TIMEOUT` on
first `run.wait2()`. Root cause: the dk_chunks>1 shim-DMA path was
never lit-tested upstream and is broken (LESSON 3, discovered by this
deployment 2026-04-18).

**The workaround (Option C)**: the **head-first** FA kernel
(`attn_npu2.py`, lit-tested by llama3-8b) works fine at the same
config. Wrapper monkey-patches `llama3_prefill._attn_backend_kwargs`
and `_run_cached` to:
1. Take seq-first q (2048×3072), k (2048×1024), v (2048×1024)
2. Reshape + transpose to head-first:
   - q: (2048, 24, 128) → (24, 2048, 128)
   - k: (2048, 8, 128) → (8, 2048, 128)
   - v: (2048, 8, 2×128) → (16, 2048, 128) [dv split into 2 chunks]
3. Invoke head-first FA kernel
4. Transpose output back to seq-first

**Where**: code lives in `_llm_shared/phase_helpers/headfirst_fa.py`.
Originally written for this deployment 2026-04-18, then promoted to
shared infra and now reused by qwen25_1_5b, qwen3_0_6b, qwen3_1_7b
(all hd=128 deployments).

**Cost**: ~3-5 ms/layer host transpose. With 28 layers, that's ~100 ms
out of 3.5 s prefill (~3% overhead). Compare to CPU FA fallback:
250-300 ms/layer × 28 = ~8 s — Option C is **~4× faster** than CPU.

**Where to look in the code**:
- Wrapper: `_llm_shared/phase_helpers/headfirst_fa.py`
- Active by default in `llama32_3b_inference.py` (`cpu_attn=False`);
  confirmed via `make run` printout `attn=NPU FA (Option C)`

### Other deltas (no new code needed)

- **emb_dim=3072 vs 2048**: parametric — `compile_block_kernels(config)`
  builds ELFs sized to 3072. Tile configs ([8,4] GEMM herds) absorb
  the 1.5× K-width without retuning.
- **28 layers vs 16**: per-layer BO arrays sized at preload from
  `config.n_layers`. No code change.
- **GQA group=3 vs 4**: invariant — kernels see (n_heads, n_kv_heads)
  as a tuple, GQA group is implicit.

---

## 4. End-to-End Inference Workflow

### Setup (one-time, before `[1/3] NPU prefill ...` print)

```
Compile (~4 min cold, instant if cached)
  external .o files (silu_and_mul, rope, attn_npu2, mv, mv_k8192)
                    ↑ all compiled with head_dim=128 variant
  prefill ELFs:
    rms_gemms_rope.elf  (6 launches merged)
    flash_attn.elf      (Option C head-first builder; head_dim=128)
    o_ffn.elf           (8 launches merged)
  decode ELFs:
    rms_gemv_rope.elf   (6 launches merged)
    o_gemv_ffn.elf      (8 launches merged)
    lm_head_gemv.elf    (8 launches merged, 8 partitions × M=16384)

Weights (~3 s)
  load HF safetensors → weights (28 layers, no QKV bias)

BO preload (~5 s, writes to NPU DRAM)
  pre_transpose_decode_weights(weights)
  preload_prefill_weights(weights, ...)            ← 28 × 9 weight BOs
  llama3_inference._preload_decode_weights(...)    ← incl. LM head warmup call

═══════════════════════════ PROFILED SCOPE ═══════════════════════════
```

### Prefill — runs 28 times, then once at end

```
PER LAYER (3 XRT calls):

  XRT call 1: rms_gemms_rope.elf  [NPU, 6 launches merged into 1 ELF]
    ┌─────────────────────────────────────────────────┐
    │   1. RMSNorm    [8,1]  attn_norm × x → normed   │
    │   2. Q GEMM     [8,4]  normed @ wq  → q         │
    │   3. K GEMM     [8,4]  normed @ wk  → k         │
    │   4. V GEMM     [8,4]  normed @ wv  → v         │
    │   5. RoPE Q     [8,1]  q  → q_roped             │
    │   6. RoPE K     [8,1]  k  → k_roped             │
    │ Intermediates flow through DDR (no CPU sync).   │
    │ NPU2 grid: [8,4] = 32 cores; [8,1] = 8 cores.   │
    └─────────────────────────────────────────────────┘

  XRT call 2: flash_attn.elf  [NPU, 1 launch, head-first via Option C]
    ┌─────────────────────────────────────────────────┐
    │ HOST pre-transpose seq → head:                  │
    │   q (2048, 24, 128) → (24, 2048, 128)           │
    │   k (2048, 8,  128) → (8,  2048, 128)           │
    │   v (2048, 8,  128) → (8*dv_chunks, 2048, 128)  │
    │   ~3-5 ms / layer                               │
    │ NPU launch: attn_npu2.py builder, hd=128,       │
    │   dk_chunks=2, GQA 24Q/8KV (group=3), causal.   │
    │ HOST post-transpose head → seq.                 │
    │ Wrapper: _llm_shared/phase_helpers/             │
    │   headfirst_fa.py                               │
    └─────────────────────────────────────────────────┘

  XRT call 3: o_ffn.elf  [NPU, 8 launches merged into 1 ELF]
    ┌─────────────────────────────────────────────────┐
    │   1. O GEMM     [8,4]  attn_out @ wo → proj     │
    │   2. Add        [8,1]  proj + x_residual → res1 │
    │   3. RMSNorm    [8,1]  ffn_norm × res1 → normed2│
    │   4. Gate GEMM  [8,4]  normed2 @ w_gate → gate  │
    │   5. Up GEMM    [8,4]  normed2 @ w_up   → up    │
    │   6. SiLU × mul [8,1]  silu(gate) * up → swiglu │
    │   7. Down GEMM  [8,4]  swiglu @ w_down → down   │
    │   8. Add        [8,1]  res1 + down → x_out      │
    └─────────────────────────────────────────────────┘
    ↓ KV extraction host code:
        k_cache[layer] = results["k_roped"].reshape & transpose
        v_cache[layer] = results["v"].reshape & transpose

AFTER 28 LAYERS — first generated token:
  HOST: CPU final RMSNorm on 1×3072 vector  (<1 ms; last position only)
  XRT call: lm_head_gemv.elf  [NPU, 8 launches merged]
    8 partitions × M=16384, K=3072. Each partition computes its slice of
    the 128256-row vocab logit vector. Concatenated and clipped.
  HOST: argmax → first token id
```

### Decode — per token

```
PER LAYER (2 XRT calls + CPU attention):

  XRT call 1: rms_gemv_rope.elf  [NPU, 6 launches merged]
    Same 6-step structure as prefill rms_gemms_rope but GEMV at M=1.
    All [1,1] or [8,1] herds.

  HOST: CPU attention (single-query, ~3 ms / layer)
    q (24 heads × 128) attends to KV cache.
    GQA group=3 maps Q head h → KV head h//3.
    Update K_cache[layer, :, pos, :], V_cache[layer, :, pos, :].

  XRT call 2: o_gemv_ffn.elf  [NPU, 8 launches merged]
    Same 8-step structure as prefill o_ffn but GEMV.
    Uses mv_k8192.o for the Down GEMV (K=hidden_dim=8192).

PER TOKEN (after 28 layers):
  HOST: CPU final RMSNorm on 1×3072  (<1 ms)
  XRT call: lm_head_gemv.elf  (same 8-partition ELF as prefill end)
  HOST: argmax → next token
```

### What's on NPU vs CPU

**On NPU** (the load-bearing work):
- All matrix multiplies (GEMM and GEMV)
- All RMSNorm (per-position norm)
- All RoPE (Q and K rotation)
- FlashAttention (prefill only, head-first via Option C)
- SiLU × multiply
- Final LM Head (8-partition GEMV)

**On CPU** (lightweight host wrap):
- Tokenization (HuggingFace tokenizer)
- Embedding lookup (single index into table)
- Option C **head-first transposes** around prefill FA — ~7-10 ms / layer
- KV cache extraction (reshape + transpose intermediate K/V into cache)
- Decode-time **single-query attention** with KV cache — *all*
  deployments do this on CPU because softmax over a long, growing
  context isn't a great NPU fit at M=1
- Final RMSNorm at last position only — tiny single-vector op
- Argmax over 128256-vocab logit vector

---

## Notes

### Why per-layer K/V cosine drift looks reasonable here

K/V cache cosine drifts from 0.999 (L0) to 0.987 (L27) — a clean
trajectory consistent with BF16 deep-stack accumulation. No noise
sources beyond what llama3-1B incurs. Compare to qwen25's 0.999 → 0.5
drift, which has 3 extra noise sources (padded compute path, RMSNorm
pre-scaling, host bias add); none of those apply here.

The Option C host transposes don't add precision loss — they're
exact reshape + transpose, no arithmetic.

### Anything redundant vs llama3-1B?

No. Same 3-prefill-XRT / 2-decode-XRT topology. Same multi-launch ELF
merging. Same per-layer BO scheme. Same KernelCache + manifest. Same
LM Head GEMV reuse pattern. No explicit warmup pass (NPU prefill keeps
device warm into decode).

The only "extra" cost is the Option C host transpose (~3-5 ms/layer),
which is a strict requirement to dodge the seq-first FA hang at hd=128.
Cannot be eliminated without fixing the upstream FA bug (which is a
separate, unrelated effort).

### Note on first-prefill heap-churn fix (Apr 26)

A previous version of this deployment had an unexplained 6× slowdown
on the first prefill of a fresh process (~21 s vs 3.5 s warm steady).
Initially mis-attributed to NPU clock state. Real cause: the Option C
wrapper allocated a fresh ~48 MB transpose buffer per FA call (~510 ms
of glibc heap growth + page faults on cold pages × 28 layers).

**Fix**: per-shape buffer cache `_HF_BUF_CACHE` in
`_llm_shared/phase_helpers/headfirst_fa.py` — replace
`np.ascontiguousarray(transpose(...))` with
`np.copyto(cached_buf, view)`. Eliminates per-call allocation.
Result: first prefill matches warm steady within 1%.

This fix benefited llama32_3b directly; was the primary contributor
to the 4.94 s → 3.518 s prefill improvement.

---

## File Map

| File | Role | Lines |
|---|---|---:|
| `llama32_3b_inference.py` | End-to-end runner (`make run` entry; --verify wired Apr 26) | 348 |
| `llama32_3b_weights.py`   | HF safetensors loader (no QKV bias, tied embeddings) | — |
| `llama32_3b_reference.py` | CPU F32 reference (used for `make verify`) | — |
| `llama32_3b_phase{2,3,4,5}_test.py` | Per-phase validation scripts | — |

No model-specific host helpers — Option C wrapper is in `_llm_shared/`.

# Evaluation Report: SmolLM2-1.7B BF16 on NPU2

**Reference deployment**: `../llama3/` (Llama-3.2-1B). Most things are
inherited unchanged; this doc covers what's different.

---

## 1. Current Status

### Verified ✓ (Apr 26, 2026 — 5 of 5 protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: evaluate-deployment`) | Apr 25: **PASS** (see legacy `evaluation_report_2026-04-25.md` content for the full audit). Phase 2 cosine 0.9992 (real-tok), Phase 3 6/6 strict top-1, adversarial 2/2 in CPU top-5. |
| `make run` smoke (Apr 26) | First token ` Paris` (id=7042). 30-token greedy: `'The capital of France is Paris.\n\nThe capital of France is Paris.\n...'` (greedy loop on a small model — semantically and topically correct). |
| `make verify` (NPU vs CPU F32 reference, wired Apr 26) | NPU top-1 == CPU top-1 (` Paris`). Final logits cosine **0.9966** at pred_pos. K/V cache cosine drift **0.9999 → 0.9931 over 24 layers** (clean BF16 noise floor). 0 layer warnings. |
| HuggingFace F32 cross-check on CPU reference | top-1 ` Paris`, logits correlation **0.99999978** vs HF F32 |
| Code review | inference / weights / reference all clean; tied-embed handled correctly at `smollm2_weights.py:285-297`; cpu_attn defaults False (NPU FA seq-first) |

### Performance (Apr 26 single-shot)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (24 layers, NPU FA seq-first, head_dim=64) | 87 ms/layer | **2.08 s** |
| First LM Head GEMV | — | 17 ms |
| Decode steady-state | 5.7 ms/layer | **137 ms/token** (7.3 tok/s) |

Per-layer prefill 87 ms vs llama3-1B's 79 ms → **1.10× scaling** (almost
flat — K width 2048 unchanged; the small bump is from MHA's 4× larger
K/V projections). Per-layer decode 5.7 ms vs llama3-1B's 5.75 ms →
**parity** at the same K-width, kernels at the per-byte ceiling.

### Manual Verify Commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/smollm2_1_7b

# ⚠️ FIRST: if you (or anyone) recently edited _llm_shared/ or
# llama3/multi_launch_builder/, do this to force ELF rebuild:
flock -x -w 1800 /tmp/mlir-air-npu.lock make clean

# (1) End-to-end smoke (~30 s warm cache, ~2 min cold compile):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30 PROMPT="The capital of France is"
#   Expected first token: ' Paris' (id=7042)
#   Expected prefill:     2.08 s (87 ms/layer)
#   Expected decode:      137 ms/token (7.3 tok/s)

# (2) NPU vs CPU F32 numerical verify (~30 s wall):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris'), logits cos ≈ 0.997
#   K/V drift: L0 cos 0.99996 → L23 cos 0.998 (K) / 0.993 (V)
#   0 per-layer warnings

# (3) HuggingFace cross-check on CPU side (no NPU needed):
python3 smollm2_reference.py --prompt "The capital of France is" --verify
#   Expected: top-1 ' Paris', logits correlation ≈ 0.99999978 vs HF F32

# (4) Phase 2 single-block (~30 s):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-block
#   Expected: cosine ≥ 0.999 (real), per-pos min ≥ 0.997

# 1-minute sanity:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=5 PROMPT="The capital of France is" 2>&1 | grep -E "first|Tok"
```

If first token is **not** ` Paris` (id=7042) — first reaction:
`make clean` and re-run (stale-cache trap).

---

## 2. Architectural Differences vs Llama-3.2-1B

| Field | Llama-3.2-1B | SmolLM2-1.7B | Why it matters |
|---|---:|---:|---|
| n_layers | 16 | **24** | Linear scaling; per-layer BO arrays sized to 24 |
| emb_dim | 2048 | 2048 | Same — most ELFs reused identically |
| n_heads | 32 | 32 | Same |
| head_dim | 64 | 64 | Same — standard seq-first FA path (no Option C needed) |
| **n_kv_heads** | 8 (GQA group=4) | **32 (MHA, group=1)** | K/V projections 4× wider; absorbed by existing tile config |
| hidden_dim | 8192 | 8192 | Same — FFN ELFs identical |
| **vocab_size** | 128256 | **49152** | 5/8 partitions used in LM Head; 3 partitions are zero-padded waste |
| **rope_θ** | 500k | **130k** | RoPE LUT regen only — no kernel change |
| QKV bias | absent | absent | Standard, no bias add path |
| Tied embeddings | yes | yes | Same fallback (lm_head ← embed_table) |

**No deltas force kernel changes.** The 4× wider K/V projections (MHA)
fit inside the existing `[8,4]` GEMM herd; the smaller vocab fits the
8-partition LM Head with some waste; the new RoPE θ is just an LUT
regen. This is the simplest model migration in the deployment set.

---

## 3. Implementation: Reused vs New

**100% kernel reuse, no model-specific host helpers.** This is the
canonical Tier-A deployment pattern: minimal scaffold + sys.path
imports.

| What | Reused from | New (smollm2-specific) |
|---|---|---|
| Per-layer prefill orchestration | `llama3_prefill.run_transformer_block` | — |
| Per-layer decode orchestration | `llama3_decode.run_decode_block` | — |
| Multi-launch ELF builders (rms_gemms_rope, o_ffn, rms_gemv_rope, o_gemv_ffn, lm_head_gemv) | `llama3.multi_launch_builder.*` | — |
| KernelCache, BO preload, `intermediate_indices` skip | `_llm_shared/kernel_builder/cache.py` | — |
| FlashAttention path | seq-first (no Option C; head_dim=64 fits L1 at dk_chunks=1) | — |
| KV extraction in prefill | `_llm_shared/phase_helpers/prefill_runner.py` | — |
| Pre-transpose decode weights, NPU LM Head GEMV | `_llm_shared/phase_helpers/decode_setup.py` | — |
| External C++ kernels (silu_and_mul.cc, rope.cc, attn.cc, mv.cc) | shared sources, compiled with `head_dim=64` | — |
| Weights loader, CPU F32 reference, end-to-end runner | — | model-specific scaffold (`smollm2_{weights,reference,inference}.py`) |

### How MHA gets absorbed without touching kernels

**The "delta"**: with `n_kv_heads=32` instead of 8, K and V projections
are 4× wider in the OUTPUT dim (out=2048 instead of out=512). Naively
this looks like "the K/V kernels need different tile configs."

**Why it's a no-op**: the parametric `rms_gemms_rope` builder takes
`(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim)` as args. It builds
each K/V GEMM tile based on `kv_dim`. The existing `[8,4]` GEMM herd
(32 cores) was already sized for emb_dim=2048 outputs (Q proj); going
from `512 → 2048` for K/V just means the K and V GEMMs now look
**identical** to the Q GEMM in tile structure. Compute herds were
already at full grid utilization. Total work per layer goes up
(~10%), but no per-tile work changes.

**Result**: same `rms_gemms_rope.elf` builder, just compiled with
`(2048, 2048, 32, 32, 64)` instead of `(2048, 512, 32, 8, 64)`. Per-layer
prefill 87 ms vs llama3-1B's 79 ms (10% bump from 4× more K/V bytes,
not 4× more time — bandwidth ceiling not hit).

### How the smaller vocab is absorbed

**The "delta"**: vocab=49152 instead of 128256.

**The pattern**: the shared `npu_lm_head_gemv` (in
`_llm_shared/phase_helpers/decode_setup.py`) hard-codes
`_LM_N_PARTITIONS = 8` and `_LM_N_PART = 16384`, giving
`8 × 16384 = 131072` logical rows. SmolLM2's vocab=49152 fits in 3 of
these 8 partitions; the other 5 partitions are zero-padded and computed
anyway (62.5% partition-level waste).

**Cost of the waste**: ~3 ms/token on decode (the LM Head GEMV is 17 ms
total, ~62.5% of which is on padded partitions). This is the only known
optimization opportunity for this deployment, tracked as a TODO. Not
a correctness or critical perf issue.

### Decode CPU attention overhead

MHA (n_kv_heads=32) means the decode-time CPU single-query attention
loop processes 4× more KV cache reads per token vs llama3-1B's GQA(g=4).
Per-token overhead ≈ 15 ms vs llama3's 5 ms. Absorbed by the budget;
NPU dominates the per-token cost regardless.

---

## 4. End-to-End Inference Workflow

### Setup (one-time, before `[1/3] NPU prefill ...` print)

```
Compile (~2 min cold, instant if cached)
  external .o files (silu_and_mul, rope, attn, mv, mv_k8192)
                    ↑ all compiled with head_dim=64 (default)
  prefill ELFs:
    rms_gemms_rope.elf  (6 launches merged)
    flash_attn.elf      (seq-first builder; head_dim=64, MHA group=1)
    o_ffn.elf           (8 launches merged)
  decode ELFs:
    rms_gemv_rope.elf   (6 launches merged)
    o_gemv_ffn.elf      (8 launches merged; mv_k8192.o for Down)
    lm_head_gemv.elf    (8 launches merged, 8 partitions × M=16384)

Weights (~9 s)
  load HF safetensors → weights (24 layers, no QKV bias)
  Tied embedding: lm_head.weight not in safetensors → lm_head ← embed_table

BO preload (~12 s, writes to NPU DRAM)
  pre_transpose_decode_weights(weights)
  preload_prefill_weights(weights, ...)            ← 24 × 9 weight BOs
  llama3_inference._preload_decode_weights(...)    ← incl. LM head warmup call

═══════════════════════════ PROFILED SCOPE ═══════════════════════════
```

### Prefill — runs 24 times, then once at end

```
PER LAYER (3 XRT calls):

  XRT call 1: rms_gemms_rope.elf  [NPU, 6 launches merged into 1 ELF]
    ┌─────────────────────────────────────────────────┐
    │   1. RMSNorm    [8,1]  attn_norm × x → normed   │
    │   2. Q GEMM     [8,4]  normed @ wq  → q  (32 heads × 64 hd)│
    │   3. K GEMM     [8,4]  normed @ wk  → k  (32 KV × 64 hd)   │
    │      ↑ 4× wider than llama3 because MHA         │
    │   4. V GEMM     [8,4]  normed @ wv  → v  (same)│
    │   5. RoPE Q     [8,1]  q  → q_roped             │
    │   6. RoPE K     [8,1]  k  → k_roped             │
    │ Intermediates flow through DDR (no CPU sync).   │
    │ NPU2 grid: [8,4] = 32 cores; [8,1] = 8 cores.   │
    └─────────────────────────────────────────────────┘

  XRT call 2: flash_attn.elf  [NPU, 1 launch, seq-first]
    ┌─────────────────────────────────────────────────┐
    │ NO host transposes — head_dim=64 stays inside   │
    │ L1 budget at dk_chunks=1, so the standard       │
    │ seq-first FA path works directly.               │
    │                                                 │
    │ NPU launch: attn.py builder, hd=64,             │
    │   MHA (32Q/32KV, group=1), causal.              │
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
    │      ↑ uses mv_k8192.o (K=hidden_dim=8192)      │
    │   8. Add        [8,1]  res1 + down → x_out      │
    └─────────────────────────────────────────────────┘
    ↓ KV extraction host code:
        k_cache[layer] = results["k_roped"].reshape & transpose
        v_cache[layer] = results["v"].reshape & transpose

AFTER 24 LAYERS — first generated token:
  HOST: CPU final RMSNorm on 1×2048 vector  (<1 ms; last position only)
  XRT call: lm_head_gemv.elf  [NPU, 8 launches merged]
    8 partitions × M=16384, K=2048. Real vocab 49152 covers ~3
    partitions; the other 5 are zero-padded (62.5% waste). Concatenated
    and clipped to 49152.
  HOST: argmax → first token id (' Paris', id=7042)
```

### Decode — per token

```
PER LAYER (2 XRT calls + CPU attention):

  XRT call 1: rms_gemv_rope.elf  [NPU, 6 launches merged]
    Same 6-step structure as prefill rms_gemms_rope but GEMV at M=1.
    All [1,1] or [8,1] herds.
    K/V GEMVs project to 2048 (MHA) vs llama3's 512 (GQA).

  HOST: CPU attention (single-query, ~15 ms / layer)
    q (32 heads × 64) attends to KV cache.
    MHA: each Q head has its own KV head (group=1 → kv_idx=h).
    Update K_cache[layer, :, pos, :], V_cache[layer, :, pos, :].
    ~4× longer than llama3's GQA(g=4) — this is MHA's tax on decode.

  XRT call 2: o_gemv_ffn.elf  [NPU, 8 launches merged]
    Same 8-step structure as prefill o_ffn but GEMV.
    Uses mv_k8192.o for the Down GEMV (K=8192).

PER TOKEN (after 24 layers):
  HOST: CPU final RMSNorm on 1×2048  (<1 ms)
  XRT call: lm_head_gemv.elf  (same 8-partition ELF as prefill end)
  HOST: argmax → next token
```

### What's on NPU vs CPU

**On NPU** (the load-bearing work):
- All matrix multiplies (GEMM and GEMV)
- All RMSNorm (per-position norm)
- All RoPE (Q and K rotation)
- FlashAttention (prefill only, seq-first — no transposes needed)
- SiLU × multiply
- Final LM Head (8-partition GEMV)

**On CPU** (lightweight host wrap):
- Tokenization (HuggingFace tokenizer)
- Embedding lookup (single index into table)
- KV cache extraction (reshape + transpose intermediate K/V into cache)
- Decode-time **single-query attention** with KV cache — *all*
  deployments do this on CPU; smollm2's MHA makes it ~4× longer per
  token vs llama3
- Final RMSNorm at last position only — tiny single-vector op
- Argmax over 49152-vocab logit vector

---

## Notes

### Why per-layer K/V cosine drift is so clean here

K/V cache cosine drifts from 0.9999 (L0) to 0.9931 (V at L23) — the
cleanest deep-stack drift in the deployment set. No noise sources beyond
what llama3-1B incurs: same head_dim, same seq-first FA, no host bias
add, no padded compute. Compare to qwen25's 0.999 → 0.5 drift, which
has 3 extra noise sources (padded compute, RMSNorm pre-scaling, host
bias add); none of those apply here.

### Anything redundant vs llama3-1B?

No. Same 3-prefill-XRT / 2-decode-XRT topology. Same multi-launch ELF
merging. Same per-layer BO scheme. Same KernelCache + manifest. Same
LM Head GEMV reuse pattern. No explicit warmup pass.

This deployment has the **least** custom code of any (zero
model-specific host helpers). The 3 model-local files
(`smollm2_{weights,reference,inference}.py`) are the minimum scaffold
required by the `deploy-new-llm` skill chain.

### Known optimization opportunity

LM Head GEMV partition right-sizing: `vocab=49152` only needs ~3
partitions, not 8. Current ELF zero-pads. Saving estimate: ~3
ms/token (~2% of decode wall). Tracked in TODO; not blocking.

---

## File Map

| File | Role |
|---|---|
| `smollm2_inference.py` | End-to-end runner (`make run` entry; --verify wired Apr 26) |
| `smollm2_weights.py`   | HF safetensors loader (no QKV bias, tied embeddings) |
| `smollm2_reference.py` | CPU F32 reference (used for `make verify`) |
| `smollm2_phase{2,3,4,5}_test.py` | Per-phase validation scripts |

No model-specific host helpers — every operation goes through
`../llama3/` or `../_llm_shared/`.

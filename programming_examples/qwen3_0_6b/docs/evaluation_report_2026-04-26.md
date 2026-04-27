# Evaluation Report: Qwen3-0.6B BF16 on NPU2

**Reference deployments**: `../llama3/` (Llama-3.2-1B) for the canonical
kernel sequence; `../llama32_3b/` for Option C head-first FA at hd=128.
This is the **first kernel-first deployment** in the set: model-specific
fused multi-launch ELFs live in `qwen3_0_6b/multi_launch/`.

---

## 1. Current Status

### Verified ✓ (Apr 26, 2026 — 5 of 5 protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: evaluate-deployment`) | Apr 26: deployment ran cleanly post-`make clean` (Apr 21 cached ELFs invalidated, recompiled fresh against current shared host code). No stale-cache trap as on qwen25. |
| `make run` smoke (Apr 26) | First token ` Paris` (id=12095). 30-token greedy: `'The capital of France is Paris, ...'` (semantically correct). |
| `make verify` (NPU vs CPU F32 reference, wired Apr 26) | NPU top-1 == CPU top-1 (` Paris`). Final logits cosine **0.9903** at pred_pos. Per-layer K/V drift larger than llama3-family (L0 cos 0.9999 → L27 K cos 0.96 / V cos 0.72; 50 informational warnings) — see Q1 in Notes. |
| HuggingFace F32 cross-check on CPU reference | top-1 ` Paris`, logits correlation > 0.9999 vs HF |
| Code review | inference / weights / reference / decode / 4 multi_launch builders all clean; cpu_attn defaults False; Option C wrapper installed; Q/K Norm flows correctly through prefill (host) and decode (NPU-fused) |

### Performance (Apr 26 single-shot, post `make clean`)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (28 layers, NPU FA Option C, head_dim=128) | 81.5 ms/layer | **2.28 s** (warm) |
| First LM Head GEMV (10 partitions × 16384, vocab=151936) | — | ~111 ms (cold; warm ≈ 22 ms) |
| Decode steady-state (Phase B fused ELFs) | 3.4 ms/layer | **~95 ms/token** (10.5 tok/s) |

Per-layer prefill 81.5 ms is slightly under llama32_3b's 126 ms because
of narrower K (1024 vs 3072). Per-layer decode **3.4 ms** is the fastest
in the deployment set — the 3-ELF kernel-first decode path eliminates
per-call Python overhead (BO preload + per-layer arg-list cache +
pre-transposed weights).

### Manual Verify Commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/qwen3_0_6b

# ⚠️ FIRST: if you (or anyone) recently edited _llm_shared/ or
# llama3/multi_launch_builder/, do this to force ELF rebuild:
flock -x -w 1800 /tmp/mlir-air-npu.lock make clean

# (1) End-to-end smoke (~30 s warm cache, ~3 min cold compile):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30 PROMPT="The capital of France is"
#   Expected first token: ' Paris' (id=12095)
#   Expected prefill:     2.28 s warm (81.5 ms/layer)
#   Expected decode:      ~95 ms/token (10.5 tok/s)

# (2) NPU vs CPU F32 numerical verify (~2 min wall):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris'), logits cos ≈ 0.99
#   K/V drift: L0 cos 0.9999 → L27 K cos 0.96 / V cos 0.72 (50 WARN's; OK)

# (3) HuggingFace cross-check on CPU side (no NPU needed):
python3 qwen3_reference.py --prompt "The capital of France is" --verify
#   Expected: top-1 ' Paris', logits correlation > 0.9999 vs HF F32

# (4) Phase 2 single-block (~30 s):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-block

# (5) Decode-side per-token NPU vs CPU top-1 verify (Phase B fused decode):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify-decode
#   Note: ~25 s/token CPU verify; default 8 tokens.

# 1-minute sanity:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=5 PROMPT="The capital of France is" 2>&1 | grep -E "first|Tok|Paris"
```

If first token is **not** ` Paris` (id=12095) — first reaction:
`make clean` and re-run (stale-cache trap; same one that bit qwen25).

---

## 2. Architectural Differences vs Llama-3.2-1B

| Field | Llama-3.2-1B | Qwen3-0.6B | Why it matters |
|---|---:|---:|---|
| n_layers | 16 | **28** | Linear scaling; per-layer BO arrays sized to 28 |
| emb_dim | 2048 | **1024** | Half-width — but BD-aligned (1 × 1024), no padding needed |
| n_heads | 32 | 16 | Half — same total Q work as 1B (16 × 128 = 32 × 64) |
| **head_dim** | 64 | **128** | Triggers Option C head-first FA wrapper (same as llama32_3b / qwen25) |
| n_kv_heads | 8 (GQA group=4) | 8 (GQA group=2) | Tighter sharing (only 2 Q heads per KV head) |
| hidden_dim | 8192 | **3072** | Smaller FFN — BD-aligned (3 × 1024) |
| vocab_size | 128256 | **151936** | LM Head needs 10 partitions (8 doesn't fit) |
| rope_θ | 500k | **1M** | RoPE LUT regen only — no kernel change |
| QKV bias | absent | absent | Standard (unlike Qwen2.5 which DOES have bias) |
| **Q/K Norm** | absent | **present** | NEW — per-head RMSNorm on Q and K BEFORE RoPE; cannot fuse into shared `rms_gemms_rope` ELF via linearity → forces kernel-first decode |
| Tied embeddings | yes | yes (also explicit in safetensors) | Same code path |

**The single delta that drives the implementation strategy**: **Q/K Norm**.
Unlike Qwen2.5's QKV bias (which RoPE linearity lets us fuse via host
post-processing), Q/K Norm is a non-linear per-head normalization that
RoPE doesn't commute with — `RoPE(rms_norm(q + ε))` ≠
`RoPE(rms_norm(q)) + RoPE(ε)`. This means the bias-trick used by qwen25
won't work; we have to either run Q/K Norm on the host between two NPU
calls (split-ELF approach, used in **prefill**) or build new model-specific
fused ELFs that do Q/K Norm on the tile (kernel-first approach, used in
**decode**).

All other deltas are absorbed by parametric ELF builders or RoPE LUT
regen — same story as smollm2.

---

## 3. Implementation: Reused vs New

This is a **hybrid** deployment: prefill uses the **split-ELF inheritance
approach** (predecessor llama3 builder + host Q/K Norm + host RoPE);
decode uses the **kernel-first approach** (model-specific fused ELFs in
`qwen3_0_6b/multi_launch/`).

| What | Reused from | New (qwen3-specific) |
|---|---|---|
| Per-layer prefill orchestration | `qwen3_phase4_test.run_block_optimized` (split-ELF runner; not run_transformer_block) | — |
| Per-layer decode orchestration | — | `qwen3_decode.py` (decode_loop_from_kv) |
| **Prefill ELF: rms_attn_gemms** (no RoPE) | `llama3.multi_launch_builder.superseded.rms_attn_gemms_multi` | — |
| **Prefill ELF: o_ffn** | `llama3.multi_launch_builder.o_ffn_multi` | — |
| Prefill ELF: flash_attn (Option C, head-first) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| **Decode ELF: rms_attn_gemvs_qknorm_rope** (8 launches: RMSNorm + Q/K/V GEMV + **Q/K Norm** + RoPE Q/K) | builder factory pattern from llama3 | **`qwen3_0_6b/multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py`** |
| **Decode ELF: o_gemv_ffn_silu** (8 launches; uses mv_og.o + mv_dg_qwen3.o renamed kernels) | builder factory pattern from llama3 | **`qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py`** |
| Decode ELF: lm_head_gemv (**10** partitions × 16384) | `llama3.multi_launch_builder.lm_head_gemv_multi.build_lm_head_gemv_module` (parameterized n_partitions) | — |
| KernelCache, BO preload, `intermediate_indices` skip | `_llm_shared/kernel_builder/cache.py` | — |
| KV extraction in prefill | `qwen3_phase4_test.npu_full_prefill(collect_kv=True)` | — |
| **Q/K Norm host helper** | `_llm_shared/phase_helpers/qk_norm.apply_qk_norm` | — |
| Option C head-first FA (hd=128) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| Renamed mv kernels (`mv_og.o`, `mv_dg_qwen3.o`) | `_llm_shared/kernel_builder/external_kernels.{compile_mv_og, compile_mv_dg_qwen3}` | — |
| Weights loader (Q/K Norm weights), CPU F32 reference, end-to-end runner | — | model-specific scaffold (`qwen3_{weights,reference,inference,decode}.py`) |

### Why Q/K Norm forces the hybrid approach

**Math**: Qwen3 applies per-head RMSNorm to Q and K **after** projection
but **before** RoPE:
```
q = (normed @ wq).reshape(seq, n_heads, head_dim)
q_normed = rms_norm(q, q_norm_weight, axis=-1)        # per-head norm on head_dim
q_roped = RoPE(q_normed)
# similarly for K
```

The shared `rms_gemms_rope` ELF was designed for bias-free Llama and
emits the path `q_roped = RoPE(normed @ wq)`. Forking it to add a
per-head RMSNorm in the middle requires either a new on-tile kernel
(non-trivial C++ work) OR routing Q/K through the host between two NPU
calls.

**Prefill: split-ELF (host Q/K Norm)** — the simpler approach.
`qwen3_phase4_test.run_block_optimized` decomposes a layer into 5 NPU
calls instead of 3:
1. `rms_attn_gemms.elf` (NPU) — RMSNorm + Q/K/V GEMMs (NO RoPE)
2. **HOST**: `apply_qk_norm` — per-head RMSNorm on Q and K
3. **HOST**: RoPE Q/K (predecessor `rope_qk_multi` has interleaved LUT
   mismatch with our half-split convention)
4. `flash_attn.elf` (NPU, Option C head-first)
5. `o_ffn.elf` (NPU)

Cost: ~10-20 ms/layer host overhead vs llama3 (negligible at hd=128
where NPU FA dominates anyway).

**Decode: kernel-first (NPU-fused Q/K Norm)** — the better approach for
M=1 tight loops where host overhead matters more. The
`rms_attn_gemvs_qknorm_rope` ELF fuses RMSNorm + Q/K/V GEMV + Q/K Norm +
RoPE Q/K all on the NPU in one 8-launch ELF. Trick: reuse the
`weighted_rms_norm` kernel via the **heads-as-M trick** — treat
`(n_heads, head_dim)` as `(M, K)` and run RMSNorm on each row.

### Other deltas (no new code needed)

- **emb_dim=1024 vs 2048**: parametric — kernels see emb_dim as a
  builder arg. No tile config retune.
- **GQA group=2**: invariant — kernels see (n_heads, n_kv_heads) as a
  tuple, GQA group is implicit.
- **vocab=151936 → 10 partitions**: shared
  `build_lm_head_gemv_module(n_partitions=10, tile_m=16, m_input=16,
  herd_m=8)`. Zero-pad waste in last partition: `163840 - 151936 =
  11904 rows`, ~7.3%.
- **rope_θ=1M**: `generate_rope_lut(rope_base=1e6)` — host-side LUT
  regen only.
- **NO QKV bias**: skip the `qwen25_bias.install_qkv_bias_wrapper()`
  step; saves a host post-processing step per layer vs qwen25.

---

## 4. End-to-End Inference Workflow

### Setup (one-time, before `Warming up NPU prefill...` print)

```
Compile (~3 min cold, instant if cached)
  external .o files (silu_and_mul, rope, attn_npu2, mv, mv_k8192, mv_og, mv_dg_qwen3)
                    ↑ all compiled with head_dim=128 variant
  prefill ELFs (predecessor split-ELF approach):
    rms_attn_gemms.elf  (split: RMSNorm + Q/K/V GEMMs, NO RoPE)
    flash_attn.elf      (Option C head-first; head_dim=128, GQA group=2)
    o_ffn.elf           (8 launches merged)
  decode ELFs (kernel-first; in qwen3_0_6b/multi_launch/):
    rms_attn_gemvs_qknorm_rope.elf  (8 launches; fuses Q/K Norm + RoPE!)
    o_gemv_ffn_silu.elf             (8 launches; mv_og.o + mv_dg_qwen3.o)
    lm_head_gemv.elf                (10 partitions × M=16384)

Weights (~3 s)
  load HF safetensors → weights (28 layers, NO QKV bias; Q/K Norm weights present)

Warmup NPU prefill (~3 s, discards K/V — runs first compile pass)

BO preload (~1.5 s)
  qwen3_decode.preload_decode_weights:
    For each layer + LM head: fire each fused ELF once with dummy inputs
    so subsequent calls skip weight-write via static_input_indices.
    Pre-transposes decode weights (q_t, k_t, v_t, o_t, gate_t, up_t, down_t)
    onto layer_weights._wq_t etc. (cached).

═══════════════════════════ PROFILED SCOPE ═══════════════════════════
```

### Prefill — runs 28 times, then once at end

```
PER LAYER (5 NPU calls + 2 host ops — split-ELF approach):

  XRT call 1: rms_attn_gemms.elf  [NPU, 6 launches merged into 1 ELF]
    ┌─────────────────────────────────────────────────┐
    │   1. RMSNorm    [8,1]  attn_norm × x → normed   │
    │   2. Q GEMM     [8,4]  normed @ wq  → q (raw)   │
    │   3. K GEMM     [8,4]  normed @ wk  → k (raw)   │
    │   4. V GEMM     [8,4]  normed @ wv  → v         │
    │   ↑ NO RoPE in this ELF (split point)           │
    └─────────────────────────────────────────────────┘

  HOST: apply_qk_norm (qwen3-specific)
    ┌─────────────────────────────────────────────────┐
    │ q_normed = rms_norm(q.reshape(seq, n_heads, hd),│
    │                     q_norm_weight, axis=-1)     │
    │ k_normed = rms_norm(k.reshape(seq, n_kv, hd),   │
    │                     k_norm_weight, axis=-1)     │
    │ Reuses _llm_shared/phase_helpers/qk_norm.py     │
    │ ~5 ms / layer for seq_len=2048                  │
    └─────────────────────────────────────────────────┘

  HOST: RoPE Q + RoPE K
    ┌─────────────────────────────────────────────────┐
    │ q_roped = apply_rope(q_normed, rope_lut)        │
    │ k_roped = apply_rope(k_normed, rope_lut)        │
    │ Half-split convention (HuggingFace Llama).      │
    │ ~5-10 ms / layer (predecessor rope_qk_multi has │
    │   interleaved LUT mismatch — host RoPE bypass). │
    └─────────────────────────────────────────────────┘

  XRT call 2: flash_attn.elf  [NPU, 1 launch, head-first via Option C]
    ┌─────────────────────────────────────────────────┐
    │ HOST pre-transpose seq → head:                  │
    │   q (2048, 16, 128) → (16, 2048, 128)           │
    │   k (2048, 8, 128)  → (8, 2048, 128)            │
    │   v (2048, 8, 128)  → (8*dv_chunks, 2048, 128)  │
    │ NPU launch: attn_npu2.py builder, hd=128,       │
    │   dk_chunks=2, GQA 16Q/8KV (group=2), causal.   │
    │ HOST post-transpose head → seq.                 │
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
    ↓ KV extraction host code (npu_full_prefill collect_kv=True):
        k_per_layer.append(k_roped)  ← post-Q/K-Norm + post-RoPE
        v_per_layer.append(v)

AFTER 28 LAYERS — first generated token:
  HOST: rms_norm(npu_hidden[real_len-1], final_norm) on 1×1024 vector
  XRT call: lm_head_gemv.elf  [NPU, 10 launches merged]
    10 partitions × M=16384, K=1024. Output clipped to vocab=151936.
  HOST: argmax → first token id (12095 = ' Paris')
```

### Decode — per token (kernel-first, fully fused)

```
PER LAYER (2 NPU calls + CPU attention):

  XRT call 1: rms_attn_gemvs_qknorm_rope.elf  [NPU, 8 launches merged]
    ┌─────────────────────────────────────────────────┐
    │ Fuses ALL of these in one ELF — qwen3-specific: │
    │   1. RMSNorm   [1,1]  attn_norm × x → normed    │
    │   2. Q GEMV    [8,1]  normed @ wq_t → q (raw)   │
    │   3. K GEMV    [8,1]  normed @ wk_t → k (raw)   │
    │   4. V GEMV    [8,1]  normed @ wv_t → v         │
    │   5. **Q Norm**[8,1]  weighted_rms_norm(q,      │
    │                       q_norm_weight) per head   │
    │     ↑ heads-as-M trick: treat (n_heads, hd) as  │
    │       (M, K) for the rms_norm kernel            │
    │   6. **K Norm**[8,1]  weighted_rms_norm(k, ...) │
    │   7. RoPE Q    [1,1]  q_normed → q_roped        │
    │   8. RoPE K    [1,1]  k_normed → k_roped        │
    │ All intermediates flow through DDR (no CPU sync)│
    └─────────────────────────────────────────────────┘

  HOST: CPU attention (single-query, GQA group=2)
    Update K_cache[layer, :, pos, :], V_cache[layer, :, pos, :].
    16 Q heads attend to 8 KV heads (kv_idx = h // 2).

  XRT call 2: o_gemv_ffn_silu.elf  [NPU, 8 launches merged]
    ┌─────────────────────────────────────────────────┐
    │ Same 8-step structure as o_ffn but GEMV at M=1. │
    │ Linker imports: mv.o (default) + mv_og.o        │
    │   (renamed for O GEMV, K=2048) + mv_dg_qwen3.o  │
    │   (renamed for Down GEMV, K=hidden=3072,        │
    │   tile_m=8). 3 distinct .o files coexist in the │
    │   same ELF via -D symbol renames.               │
    └─────────────────────────────────────────────────┘

PER TOKEN (after 28 layers):
  HOST: rms_norm(x, final_norm) on 1×1024  (<1 ms)
  XRT call: lm_head_gemv.elf  (same 10-partition ELF as prefill end;
                                kept BO-preloaded across decode steps via
                                npu_lm_head._part_weights cache)
  HOST: argmax → next token

Per-token total: ~95 ms / token (10.5 tok/s)
  - rms_attn_gemvs_qknorm_rope:  ~5 ms × 28 = 140 ms? No — actually
    much faster per layer due to fusion: ~1.5 ms / layer × 28 = 42 ms
  - o_gemv_ffn_silu:              ~1.9 ms × 28 = 53 ms
  - lm_head_gemv:                  ~14 ms
  - CPU attn + host:               ~few ms
```

### What's on NPU vs CPU

**On NPU** (the load-bearing work):
- All matrix multiplies (GEMM and GEMV)
- All RMSNorm (final + per-layer attn_norm + ffn_norm)
- **Q/K Norm — on host in prefill, on NPU (fused) in decode**
- RoPE — on host in prefill, on NPU (fused) in decode
- FlashAttention (prefill only, head-first via Option C)
- SiLU × multiply
- Final LM Head (10-partition GEMV)

**On CPU** (lightweight host wrap):
- Tokenization (HuggingFace tokenizer)
- Embedding lookup
- **Prefill-time Q/K Norm + RoPE** — these are split out of the prefill ELF
  because Q/K Norm doesn't fuse cleanly into the predecessor
  `rms_attn_gemms` builder. Cost: ~10-15 ms / layer total.
- Option C head-first transposes around prefill FA — ~7-10 ms / layer
- Decode-time single-query attention with KV cache
- Final RMSNorm at last position — tiny single-vector op
- Argmax over 151936-vocab logit vector

---

## Notes

### Q1: Why are some per-layer K/V cosines low?

Same compounded-residual-drift effect as the other deployments: NPU and
CPU run two independent 28-layer forwards; by L27 the residual streams
have diverged enough that the K/V projections show large per-layer
divergence even though the K/V kernels themselves are correct.

qwen3's drift profile is **between** llama32_3b's clean trajectory and
qwen25's heavy drift:

| Deployment | L0 cos | L27 cos | Layer warns |
|---|---:|---:|---:|
| llama32_3b   | 0.9999 | K=0.987, V=0.987 |  25 |
| smollm2_1_7b | 0.9999 | K=0.998, V=0.993 |   0 |
| qwen3-0.6B   | 0.9999 | K=0.96,  V=0.72  |  50 |
| qwen25-1.5B  | 0.9999 | K=0.61,  V=0.50  |  48 |

qwen3's extra drift sources (vs llama3-family but not as bad as qwen25):
1. Host Q/K Norm in prefill — extra `f32 → bf16` quantization round per
   layer (similar to qwen25's host bias add but smaller in magnitude)
2. Host RoPE in prefill — also adds quantization noise
3. The kernel-first decode ELFs are EQUIVALENT to host-side math for
   verify purposes (both produce the same bf16 outputs) — drift here is
   from the prefill side.

**End-to-end correctness is preserved**: NPU top-1 == CPU top-1
(` Paris`), logits cos 0.9903, 30-token greedy generation is
semantically perfect.

### Q2: Anything redundant vs llama3-1B?

Mostly no — but the explicit warmup pass (`Warming up NPU prefill...`)
**is** unique to qwen3 and is necessary because:
- It primes BO allocations + does the cold compile pass for the
  split-ELF prefill (which has more launches than llama3's fused path)
- It's run-then-discarded (not in the timed scope)

This is a deliberate design choice for the split-ELF approach, not a
bug. llama3 / llama32_3b / smollm2 / qwen25 don't need explicit warmup
because their unified prefill path keeps the NPU warm from compile to
inference automatically.

### Note on the Apr 21 cached ELFs

Pre-`make clean` (Apr 26) the cache had Apr 21 ELFs from before the
recent shared-infra changes. These would have hit the same stale-cache
trap as qwen25 if used directly — `make clean` triggered a full rebuild
against current shared host code, and the deployment ran cleanly.

---

## File Map

| File | Role |
|---|---|
| `qwen3_inference.py` | End-to-end runner (`make run` entry; --verify wired Apr 26) |
| `qwen3_weights.py`   | HF safetensors loader (Q/K Norm weights, no QKV bias, tied embeddings) |
| `qwen3_reference.py` | CPU F32 reference with Q/K Norm BEFORE RoPE |
| `qwen3_decode.py`    | Decode pipeline + npu_lm_head + preload_decode_weights + decode_loop_from_kv |
| `qwen3_phase{1,2,3,4,5}_test.py` | Per-phase validation scripts; phase4 is the production prefill runner |
| `qwen3_verify_decode.py` | Decode-side per-token NPU vs CPU top-1 check (`make verify-decode`) |
| `qwen3_canonical_sweep.py` | 6-prompt full NPU pipeline sweep |
| `multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py` | **NEW**: kernel-first decode ELF (fuses Q/K Norm + RoPE on NPU) |
| `multi_launch/o_gemv_ffn_silu_qwen3.py` | **NEW**: kernel-first decode ELF (uses mv_og.o + mv_dg_qwen3.o renamed kernels) |
| `multi_launch/lm_head_gemv_qwen3_test.py` | LM head builder wrapper (uses shared `build_lm_head_gemv_module` with n_partitions=10) |

Imported from `_llm_shared/`: KernelCache, prepare_air_project,
compile_all_external_kernels, headfirst_fa.install_headfirst_fa_wrapper,
qk_norm.apply_qk_norm.

Imported from `llama3/`: multi_launch_builder.superseded.rms_attn_gemms_multi
(predecessor split-ELF prefill builder), multi_launch_builder.o_ffn_multi,
multi_launch_builder.lm_head_gemv_multi (parameterized n_partitions).

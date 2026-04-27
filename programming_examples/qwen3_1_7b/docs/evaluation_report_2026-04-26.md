# Evaluation Report: Qwen3-1.7B BF16 on NPU2

**Reference deployments**: `../qwen3_0_6b/` (first Qwen3 deployment;
canonical Q/K Norm + kernel-first methodology); `../llama3/` (canonical
kernel sequence); `../llama32_3b/` (Option C head-first FA at hd=128).
This is the **second validation** of the kernel-first methodology — same
hybrid prefill/decode pattern as 0.6B, scaled up.

---

## 1. Current Status

### Verified ✓ (Apr 26, 2026 — 5 of 5 protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: evaluate-deployment`) | Apr 21 PASS-with-warnings (3 audit warnings, 2 already fixed). Re-run today reproduces clean. |
| `make run` smoke (Apr 26, post `make clean`) | First token ` Paris` (id=12095). Generated text: `'The capital of France is Paris'` — semantically correct. |
| `make verify` (NPU vs CPU F32 reference, wired Apr 26) | NPU top-1 == CPU top-1 (` Paris`). Final logits cosine **0.9863** at pred_pos. K/V cache cosine drift **0.9999 → 0.995 over 28 layers** (cleaner than qwen3-0.6B's 0.96 because emb_dim=2048 attenuates per-layer noise vs 0.6B's emb_dim=1024). 29 informational layer warnings (vs 50 on 0.6B). |
| HuggingFace F32 cross-check on CPU reference | top-1 ` Paris`, logits correlation > 0.9999 vs HF |
| Code review | inference / weights / reference / decode / multi_launch builders all clean; cpu_attn defaults False; Option C wrapper installed; same code path as qwen3-0.6B with shape constants bumped |

### Performance (Apr 26 single-shot, post `make clean`)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (28 layers, NPU FA Option C, head_dim=128) | 98.3 ms/layer | **2.75 s** (warm) |
| First LM Head GEMV (19 partitions × 8192, vocab=151936) | — | ~22 ms warm |
| Decode steady-state (Phase B fused ELFs) | 5.3 ms/layer | **~149 ms/token** (6.7 tok/s) |

Per-layer prefill 98.3 ms vs qwen3-0.6B's 81.5 ms (1.21×) tracks the
2× wider K (2048 vs 1024) — sub-linear because much of the per-layer
cost is FA + LM head fixed overhead. Per-layer decode 5.3 ms vs 0.6B's
3.4 ms (1.56×) reflects the wider K-loop in the FFN GEMVs.

### Manual Verify Commands

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/qwen3_1_7b

# ⚠️ FIRST: if you (or anyone) recently edited _llm_shared/ or
# llama3/multi_launch_builder/, do this to force ELF rebuild:
flock -x -w 1800 /tmp/mlir-air-npu.lock make clean

# (1) End-to-end smoke (~30 s warm cache, ~3 min cold compile):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30 PROMPT="The capital of France is"
#   Expected first token: ' Paris' (id=12095)
#   Expected prefill:     2.75 s warm (98.3 ms/layer)
#   Expected decode:      ~149 ms/token (6.7 tok/s)

# (2) NPU vs CPU F32 numerical verify (~2 min wall):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
#   Expected: NPU top-1 == CPU top-1 (' Paris'), logits cos ≈ 0.99
#   K/V drift: L0 cos 0.9999 → L27 cos 0.995 (29 WARN's; OK)

# (3) HuggingFace cross-check on CPU side (no NPU needed):
python3 qwen3_reference.py --prompt "The capital of France is" --verify
#   Expected: top-1 ' Paris', logits correlation > 0.9999 vs HF F32

# (4) Phase 2 single-block (~30 s):
flock -x -w 1800 /tmp/mlir-air-npu.lock make run-block

# (5) Decode-side per-token NPU vs CPU top-1 verify (Phase B fused decode):
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify-decode

# 1-minute sanity:
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=5 PROMPT="The capital of France is" 2>&1 | grep -E "first|Tok|Paris"
```

If first token is **not** ` Paris` (id=12095) — first reaction:
`make clean` and re-run (stale-cache trap; same one that bit qwen25).

---

## 2. Architectural Differences vs Llama-3.2-1B

| Field | Llama-3.2-1B | Qwen3-1.7B | Why it matters |
|---|---:|---:|---|
| n_layers | 16 | **28** | Linear scaling |
| emb_dim | 2048 | 2048 | Same — most ELFs reused parametrically |
| n_heads | 32 | 16 | Half — same total Q work as 1B (16 × 128 = 32 × 64) |
| **head_dim** | 64 | **128** | Triggers Option C head-first FA wrapper (same as llama32_3b / qwen25 / qwen3-0.6B) |
| n_kv_heads | 8 (GQA group=4) | 8 (GQA group=2) | Tighter sharing (only 2 Q heads per KV head) |
| hidden_dim | 8192 | **6144** | Smaller FFN — BD-aligned (6 × 1024) |
| vocab_size | 128256 | **151936** | LM Head needs **19 × 8192** partition layout (not 10 × 16384 like 0.6B; see §3) |
| rope_θ | 500k | **1M** | RoPE LUT regen only — no kernel change |
| QKV bias | absent | absent | Same (unlike Qwen2.5) |
| **Q/K Norm** | absent | **present** | NEW — per-head RMSNorm on Q and K BEFORE RoPE; same handling as qwen3-0.6B (host in prefill, NPU-fused in decode) |
| Tied embeddings | yes | yes (also explicit in safetensors) | Same code path |

### Vs Qwen3-0.6B (the closest reference)

The two qwen3 deployments share the same kernel-first methodology, but
the shape arithmetic differs in one important way:

- qwen3-0.6B: `q_dim (n_heads × head_dim = 16 × 128 = 2048) ≠ emb_dim (1024)`
  → FFN's `o_gemv_ffn_silu` needed 3 distinct matvec symbol-renamed `.o`
  files (mv.o for K=1024, mv_og.o for K=2048 O proj, mv_dg_qwen3.o for
  K=hidden Down proj).
- **qwen3-1.7B: `q_dim == emb_dim == 2048`** → only 2 distinct K-widths
  needed (K=2048 standard `mv.o`, K=hidden=6144 renamed `mv_dg_qwen3.o`).
  Llama3's shared `o_gemv_ffn_multi.build_o_gemv_ffn_module` works
  directly without the `mv_og.o` rename. **Saves one `.o` file vs 0.6B.**

This makes qwen3-1.7B's decode-side reuse **even cleaner than 0.6B's** —
the entire `o_gemv_ffn` ELF is shared with llama3 (parametrized).

---

## 3. Implementation: Reused vs New

Same hybrid pattern as qwen3-0.6B: **split-ELF prefill** (host Q/K Norm
+ host RoPE) + **kernel-first decode** (NPU-fused Q/K Norm + RoPE).

| What | Reused from | New (qwen3_1_7b-specific) |
|---|---|---|
| Per-layer prefill orchestration | `qwen3_phase4_test.run_block_optimized` (split-ELF runner) | — |
| Per-layer decode orchestration | `qwen3_decode.py` (decode_loop_from_kv) | — (mostly same as 0.6B with shape constants bumped) |
| **Prefill ELF: rms_attn_gemms** (no RoPE) | `llama3.multi_launch_builder.superseded.rms_attn_gemms_multi` | — |
| **Prefill ELF: o_ffn** | `llama3.multi_launch_builder.o_ffn_multi` | — |
| Prefill ELF: flash_attn (Option C, head-first) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| **Decode ELF: rms_attn_gemvs_qknorm_rope** (8 launches: RMSNorm + Q/K/V GEMV + Q/K Norm + RoPE Q/K) | builder factory pattern from qwen3-0.6B's same-name builder | **`qwen3_1_7b/multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py`** (shape constants bumped) |
| **Decode ELF: o_gemv_ffn_silu** | **`llama3.multi_launch_builder.o_gemv_ffn_multi.build_o_gemv_ffn_module`** ← uses llama3 directly! No qwen3 fork needed (q_dim == emb_dim simplifies) | — |
| Decode ELF: lm_head_gemv (**19** partitions × 8192) | `llama3.multi_launch_builder.lm_head_gemv_multi.build_lm_head_gemv_module` (parameterized n_partitions + tile_m + m_input + herd_m) | — (just a different builder call) |
| KernelCache, BO preload, `intermediate_indices` skip | `_llm_shared/kernel_builder/cache.py` | — |
| KV extraction in prefill | `qwen3_phase4_test.npu_full_prefill(collect_kv=True)` | — |
| Q/K Norm host helper (prefill) | `_llm_shared/phase_helpers/qk_norm.apply_qk_norm` | — |
| Option C head-first FA (hd=128) | `_llm_shared/phase_helpers/headfirst_fa.py` | — |
| Renamed mv kernel (`mv_dg_qwen3.o` for K=hidden=6144 Down GEMV) | `_llm_shared/kernel_builder/external_kernels.compile_mv_dg_qwen3` | — |
| Weights loader, CPU F32 reference, end-to-end runner | — | model-specific scaffold (`qwen3_{weights,reference,inference,decode}.py`) |

### Why the LM Head needs 19 partitions × 8192 (vs 0.6B's 10 × 16384)

Same vocab (151936) as 0.6B, but **emb_dim differs**: 0.6B has K=1024,
1.7B has K=2048. The standard LM head tile config (tile_m=16, m_input=16,
herd_m=8) works for K=1024 but breaches L2 by 256 B at K=2048:

```
A buffer (per-tile):  herd_m × tile_m × K × bytes/element
                    = 8 × 16 × 2048 × 2 = 524 288 B = exactly 512 KB cap
                    + C buffer = 256 B → trips L2
```

Fix for 1.7B: **halve M_part to 8192, tile_m to 8, m_input to 8, keep
herd_m=8**:
```
A buffer = 8 × 8 × 2048 × 2 = 262 144 B ✓
B-DMA fires per partition = 8192 / (8 × 8) = 128 ≤ 255 ✓
n_partitions = ceil(151936 / 8192) = 19 (3712 padding rows in last partition)
```

Cost: more partitions = more launch overhead. Benefit: fits L2 ✓.

### Why `o_gemv_ffn_silu` doesn't need a qwen3-specific fork (vs 0.6B)

In 0.6B, `q_dim != emb_dim` (2048 != 1024), so the O GEMV (input dim =
q_dim, output = emb_dim) and the FFN Gate/Up GEMVs (input = emb_dim,
output = hidden) needed **different K dimensions** in the same ELF.
The shared llama3 `o_gemv_ffn_multi` builder hardcodes `mv.o` for the
internal K-uniform case; couldn't accommodate. So 0.6B added
`mv_og.o` (renamed for O GEMV, tile_m=8) + `mv_dg_qwen3.o` (renamed
for Down GEMV, K=hidden, tile_m=8) — **3 distinct .o files in one ELF**.

In 1.7B, `q_dim == emb_dim == 2048`, so the O GEMV's K is the same as
Gate/Up's K = 2048. Llama3's default `mv.o` handles both. Only the
Down GEMV (K=hidden=6144) needs the renamed `mv_dg_qwen3.o`. **2
distinct .o files** — same pattern as llama3-1B.

### Down GEMV K=6144 L2 budget

Standard tile_m=8 gives `A = 8 × 8 × 6144 × 2 = 786 432 B > 512 KB`.
Reduced to `down_tile_m=2, down_m_input=1` → `A = 8 × 2 × 6144 × 2 =
196 608 B ✓`. K=6144 ≤ 8160 (auto-split limit) so no `down_k_split`
knob needed (unlike qwen25 at K=8960 which needed `down_k_split=70`).

---

## 4. End-to-End Inference Workflow

Same workflow shape as qwen3-0.6B (split-ELF prefill + kernel-first
decode). Only shape constants differ.

### Setup (one-time, before `Warming up NPU prefill...` print)

```
Compile (~3 min cold, instant if cached)
  external .o files (silu_and_mul, rope, attn_npu2, mv, mv_dg_qwen3)
                    ↑ all compiled with head_dim=128 variant
                    ↑ NO mv_og.o needed (q_dim == emb_dim simplification)
  prefill ELFs (predecessor split-ELF approach):
    rms_attn_gemms.elf   (split: RMSNorm + Q/K/V GEMMs, NO RoPE)
    flash_attn.elf       (Option C head-first; head_dim=128, GQA group=2)
    o_ffn.elf            (8 launches merged)
  decode ELFs (kernel-first; in qwen3_1_7b/multi_launch/):
    rms_attn_gemvs_qknorm_rope.elf  (8 launches; fuses Q/K Norm + RoPE)
    o_gemv_ffn_silu.elf             (8 launches; uses llama3's builder
                                     directly, mv.o + mv_dg_qwen3.o)
    lm_head_gemv.elf                (19 partitions × M=8192;
                                     tile_m=8/m_input=8/herd_m=8)

Weights (~3 s)
  load HF safetensors → weights (28 layers, NO QKV bias; Q/K Norm weights present)

Warmup NPU prefill (~3 s, discards K/V — runs first compile pass)

BO preload (~1.5 s)
  qwen3_decode.preload_decode_weights:
    For each layer + LM head: fire each fused ELF once with dummy inputs
    so subsequent calls skip weight-write via static_input_indices.
    Pre-transposes decode weights onto layer_weights._wq_t etc.

═══════════════════════════ PROFILED SCOPE ═══════════════════════════
```

### Prefill — runs 28 times, then once at end

Same structure as qwen3-0.6B:

```
PER LAYER (5 NPU calls + 2 host ops — split-ELF approach):
  XRT call 1: rms_attn_gemms.elf  [NPU, RMSNorm + Q/K/V GEMMs, no RoPE]
  HOST: apply_qk_norm  (per-head RMSNorm on Q/K)
  HOST: RoPE Q + RoPE K  (half-split convention)
  XRT call 2: flash_attn.elf  [NPU, Option C head-first; hd=128, GQA group=2]
  XRT call 3: o_ffn.elf  [NPU, 8 launches merged]
  ↓ KV extraction host code

AFTER 28 LAYERS — first generated token:
  HOST: rms_norm(npu_hidden[real_len-1], final_norm) on 1×2048 vector
  XRT call: lm_head_gemv.elf  [NPU, 19 launches merged]
    19 partitions × M=8192, K=2048. Output clipped to vocab=151936.
  HOST: argmax → first token id (12095 = ' Paris')
```

See [`../qwen3_0_6b/docs/evaluation_report_2026-04-26.md`](../qwen3_0_6b/docs/evaluation_report_2026-04-26.md)
§4 for the per-launch breakdown of each ELF (identical structure here,
just different shape constants).

### Decode — per token (kernel-first, fully fused)

```
PER LAYER (2 NPU calls + CPU attention):
  XRT call 1: rms_attn_gemvs_qknorm_rope.elf  [NPU, 8 launches merged]
    Fuses RMSNorm + Q/K/V GEMV + Q/K Norm (per-head via heads-as-M trick)
    + RoPE Q/K. All intermediates flow through DDR.

  HOST: CPU attention (single-query, GQA group=2)
    16 Q heads, 8 KV heads, kv_idx = h // 2.
    Update K_cache[layer, :, pos, :], V_cache[layer, :, pos, :].

  XRT call 2: o_gemv_ffn_silu.elf  [NPU, 8 launches merged]
    Linker imports: mv.o (K=2048 for O + Gate + Up) + mv_dg_qwen3.o
    (K=6144 for Down). 2 distinct .o files coexist in same ELF.
    NB: 1.7B uses llama3's o_gemv_ffn_multi builder directly (no
    qwen3-specific fork like 0.6B's o_gemv_ffn_silu_qwen3.py).

PER TOKEN (after 28 layers):
  HOST: rms_norm(x, final_norm) on 1×2048  (<1 ms)
  XRT call: lm_head_gemv.elf  (19-partition ELF; preloaded)
  HOST: argmax → next token

Per-token total: ~149 ms / token (6.7 tok/s)
```

### What's on NPU vs CPU

**On NPU**: all matmuls (GEMM + GEMV), all RMSNorm, **Q/K Norm in decode
only** (host in prefill), RoPE in decode only (host in prefill),
FlashAttention (prefill only, head-first via Option C), SiLU × multiply,
Final LM Head (19-partition GEMV).

**On CPU**: tokenization, embedding lookup, **prefill-time Q/K Norm + RoPE**
(~10-15 ms / layer), Option C head-first transposes around prefill FA
(~7-10 ms / layer), KV cache extraction, decode-time single-query
attention, final RMSNorm at last position, argmax over 151936-vocab.

---

## Notes

### Q1: Why per-layer K/V cosine drift is cleaner than qwen3-0.6B

| Deployment | L0 cos | L27 cos | Layer warns | Logits cos |
|---|---:|---:|---:|---:|
| qwen3-1.7B   | 0.9999 | K=0.995, V=0.995 |  29 | 0.9863 |
| qwen3-0.6B   | 0.9999 | K=0.96, V=0.72   |  50 | 0.9903 |

Same kernel topology, but 1.7B's per-layer drift is ~10× cleaner. Most
likely cause: **emb_dim=2048 vs 1024**. Wider residual stream means each
per-layer BF16 quantization round contributes a smaller fraction of the
total signal, so compounded drift over 28 layers is less aggressive.
End-to-end logits cos is similar (0.99 vs 0.99) because the log-sum-exp
of the LM head smooths out the per-layer K/V noise.

### Q2: Why decode tok/s lower than 0.6B (6.7 vs 10.5)

Wider K-loop in FFN GEMVs: 0.6B's hidden_dim=3072 vs 1.7B's 6144 (2×).
The Down GEMV is the bottleneck (K=hidden = 6144 here). Per-layer
decode time scales as K × n_kv_heads ratio: 5.3 ms / 3.4 ms = 1.56×,
which roughly matches the K-width ratio + the smaller n_partitions (19
vs 10) overhead.

### Anything redundant vs qwen3-0.6B?

Per the audit: `qwen3_inference.py:45` imports `install_headfirst_fa_wrapper`
but never calls it directly (the wrapper is installed transitively via
`compile_block_kernels`). Dead import; low priority.

### Note on Apr 21 cached ELFs and old leftover dirs

Pre-`make clean` (Apr 26) the deployment had 5 stale cache dirs at root:
`prefill_kernel_cache/`, `prefill_kernel_cache_2048/`,
`prefill_kernel_cache_2048.bak/`, `prefill_kernel_cache_512/`,
`decode_kernel_cache/` — accumulated across iterations. After cleanup
+ post-build/-refactor the new layout is just `build/` + `air_project/`
+ `air.*` (per-deployment standard).

---

## File Map

| File | Role |
|---|---|
| `qwen3_inference.py` | End-to-end runner (`make run` entry; --verify wired Apr 26) |
| `qwen3_weights.py`   | HF safetensors loader (Q/K Norm weights, no QKV bias, tied embeddings) |
| `qwen3_reference.py` | CPU F32 reference with Q/K Norm BEFORE RoPE |
| `qwen3_decode.py`    | Decode pipeline (uses **llama3's** o_gemv_ffn_multi builder, NOT 0.6B's fork) |
| `qwen3_phase{2,4,5}_test.py`, `qwen3_kernel_registry_test.py` | Per-phase / per-leaf validation scripts; phase4 is the production prefill runner |
| `qwen3_verify_decode.py` | Decode-side per-token NPU vs CPU top-1 check (`make verify-decode`) |
| `qwen3_canonical_sweep.py` | 6-prompt full NPU pipeline sweep |
| `multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py` | Phase B fused decode ELF (shape constants for 1.7B) |

Imported from `_llm_shared/`: KernelCache, prepare_air_project,
compile_all_external_kernels, headfirst_fa.install_headfirst_fa_wrapper,
qk_norm.apply_qk_norm.

Imported from `llama3/`: multi_launch_builder.superseded.rms_attn_gemms_multi
(predecessor split-ELF prefill builder), multi_launch_builder.o_ffn_multi,
multi_launch_builder.o_gemv_ffn_multi (decode FFN builder — used directly,
no qwen3 fork), multi_launch_builder.lm_head_gemv_multi (parameterized
n_partitions / tile_m / m_input / herd_m).

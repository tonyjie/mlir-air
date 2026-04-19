# qwen25_1_5b — deployment progress log

## Phase 0: Bootstrap (PASSED 2026-04-18)

- HF model: `Qwen/Qwen2.5-1.5B`
- Config: n_layers=28, emb_dim=1536, n_heads=12, n_kv_heads=2 (GQA group=6),
  head_dim=128, hidden_dim=8960, vocab=151936, rope_base=1e6, rms_norm_eps=1e-6,
  qkv_bias=True, tied embeddings.
- Files produced:
  - `qwen25_weights.py` — config + safetensors loader (handles QKV bias + tied embeddings)
  - `qwen25_reference.py` — CPU F32 forward pass with QKV-bias addition before RoPE
- Smoke test: weights load cleanly (28 layers, all shape asserts pass; bq=(1536,),
  bk=bv=(256,); lm_head correctly tied to embed_table).
- Reference verification (`--prompt "The capital of France is" --verify`):
  - Top-1: " Paris" (matches HF F32)
  - Max abs err vs HF F32: 0.0061
  - Mean abs err: 0.0010
  - Logits correlation: 0.99999992
  - **VERIFICATION PASSED**

### Notes for downstream phases
- QKV bias is applied BEFORE RoPE (matches HF reference impl). Phase 1
  kernel design must add the bias eltwise into the existing GEMM output
  before passing to the half-split RoPE kernel.
- head_dim=128 → Phase 2 will need the Option C head-first FA wrapper
  (`_llm_shared/phase_helpers/headfirst_fa.py`), same as `llama32_3b/`.

## Phase 1: Per-kernel shape audit (PASSED 2026-04-18)

See `phase1_kernel_shapes.md` for the full classification table and risk audit.

- **Variant audit** (Step 0): production code uses
  `_llm_shared/phase_helpers/headfirst_fa.patch_run_cached_for_headfirst_fa`
  → `attn_npu2.py` (head-first), which IS lit-tested at head_dim=128 by
  `run_npu2_makefile_peano_llama3_8b.lit`. No FA coverage gap.
- **Drop-in**: RMSNorm at emb_dim=1536, RoPE at head_dim=128, residual add,
  SiLU+mul, all GEMM/GEMV builders (parametric).
- **Recompile** (unique baked-config): rms_gemms_rope (emb_dim=1536, head=12,
  kv=2, hd=128), o_ffn (hidden_dim=8960), o_gemv_ffn (K=8960), FA via
  `compile_attn_npu2_split(lqp=256, lkp=64, dk=dv=128, num_q_tiles=4)`.
- **NEW work for Phase 2** (3 items, all with plans):
  1. **QKV bias add** — Option A: extend `rms_gemms_rope_multi` with
     `qkv_bias` flag, broadcast-add bq/bk/bv (1-D) to GEMM output before RoPE.
     Same `_build_add_2d_to_2d` helper, generalized to 1-D broadcast on M.
  2. **LM Head partition for vocab=151936** — 10 × 16384 = 163840 (pad 11904);
     verify `lm_head_multi.build_lm_head_module(n_partitions=10)` accepts >8.
  3. **`mv_k8960.o` renamed kernel** — copy `_ensure_mv_k8192_o` from
     `llama3_decode.py`, parameterize on K.
- **Risk for Phase 2**: NPU FA at GQA group=6 (n_heads=12, n_kv_heads=2)
  is untested; `llama32_3b/` validated group=3. Use `debug-fa-runtime-failure`
  recipe with `(n_heads=12, n_kv_heads=2, lq=lk=2048, dk=128)` if it fails.

## Phase 2: Single-block correctness (PASS @ seq_len=512, 2026-04-19)

See `phase2_block.md` for the full discussion + tile-config audit.

- **Whole-tensor cosine vs Qwen2.5 CPU ref**: **0.9986** (gate >0.99 ✓)
- **Per-position cosine min**: **0.9977** (gate >0.98 for head_dim=128 ✓)
- **MAE**: 0.042; **NaN**: none
- **Path**: NPU rms_gemms_rope (bias-free) → host QKV-bias add (RoPE-linearity
  trick) → CPU attention → NPU o_ffn
- **Tile config**: `rms_gemms_rope` built with `tile_n=64, herd_n=4`
  (`tile_n*herd_n=256` fits both Q's N=1536 and K/V's N=256); `o_ffn` with
  `gate_tile_n=64, swiglu_tile_n=2240` (fits hidden_dim=8960).
- **Bias path**: Qwen2 QKV bias added on host via `qwen25_bias.py`'s
  `_run_cached` monkey-patch — same pattern as Option C head-first FA wrapper.
  Justified by RoPE's linearity. Zero changes to shared multi-launch builders.

### Lessons captured (`LESSONS.md`)
- L1: RoPE-linearity trick lets us add Qwen2 QKV bias post-RoPE on the host.
- L2: GEMM tile config must satisfy `N%(tile_n*herd_n)==0` AND
  `M≥tile_m*herd_m`, or kernel SILENTLY produces garbage.
- L3: At emb_dim=1536, the 6-launch `rms_gemms_rope` ELF exhausts shim BD
  pool at seq_len=2048 — Phase 3 needs the 2-ELF split.

## Phase 3: Full-model correctness (PASS, 2026-04-19)

See `phase3_full.md` for the per-prompt table.

- **Gate**: decisive 3/3 top-1, competitive 3/3 top-5 overlap, no NaN ✓
- **Strict top-1**: 5/6 (the one miss is `'The sky is'` competitive: NPU=' ',
  CPU=' blue', cpu_p=0.265 — expected BF16 reorder per LESSON 2)
- **Decisive matches**: ' ', ' ', ' ' for `1+1`, `2+2`, `Water freezes at`
- **Competitive top-1 successes**: ' Pacific', ' Paris' for `largest ocean`
  and `France` (despite cpu_p=0.34/0.27 — competitive but stable enough)
- **NPU prefill (CPU-attn)**: ~10 s / 28 layers (~360 ms/layer); CPU
  reference: ~18 s/prompt
- **Padding stability**: GQA-reindexed padding propagates correctly
  through all 28 layers — no structural drift from the padding scheme
  (the small per-prompt corr range 0.91–0.99 is normal BF16 accumulation
  across deep stack at head_dim=128, matching llama32_3b's behavior).

## Phase 4: Prefill perf (PASS, 2026-04-19)

See `phase4_prefill.md` for the full pattern table + per-pattern measurements.

- **5/5 patterns applied with NPU FA**:
  1. Multi-launch merging (INHERITED — rms_gemms_rope=6 + o_ffn=8 launches)
  2. BO pre-loading (APPLIED — `preload_prefill_weights` works unchanged with
     padded weights; 1.78 s setup)
  3. Intermediate buffer reuse (INHERITED)
  4. Seq-first layout (INHERITED)
  5. **NPU FA via Option C head-first wrapper** (APPLIED — works clean at
     padded GQA group=8)
- **Headline numbers** (warm avg of 3, NPU FA path):
  - NPU layers: **2.4 s** (85 ms/layer) — 4.2× faster than CPU-attn (10.1 s)
  - Wall (incl. CPU LM head): **4.1 s** — 2.9× faster than CPU-attn (11.8 s)
- **Top-1 token preserved** (' Paris') on cold + warm + CPU reference — no
  regression introduced by NPU FA path through padded shapes.
- **Vs prior deployments** (warm ms/layer): llama3 81, smollm2 79,
  llama32_3b 115, **qwen25_1_5b 85** — sits cleanly between smaller-emb
  and same-depth models, confirming the padding overhead is small.

## Phase 5: Decode perf (PASS, 2026-04-19)

See `phase5_decode.md` for the full per-pattern table + timing.

- **Headline**: **216 ms/token (4.6 tok/s)** steady-state, 7.7 ms/layer at
  Qwen2.5-1.5B (28 layers, head_dim=128). Per-layer rate matches
  llama32_3b exactly (also 7.7 ms/layer at same depth + head_dim).
- **Top-1 NPU/CPU match**: 5/6 (83%) > 80% gate ✓
- **Generated**: "The capital of France is Paris, and the capital of France"
- **5/5 patterns** (with `[PARTIAL]` on attention same as llama3 design):
  1. Multi-launch merging (INHERITED — rms_gemv_rope=6, o_gemv_ffn=8)
  2. Static weight BOs (INHERITED)
  3. **NPU LM Head GEMV** (APPLIED — 10×16384 partition, vocab=151936;
     `tile_m=16, m_input=16` to fit B-DMA repeat-count limit)
  4. **Extern kernel rename** (APPLIED — `mv_k8960.o` for Down GEMV K=8960
     plus new `down_k_split=70` matvec knob to fit K-DMA repeat-count)
  5. CPU→NPU op promotion (PARTIAL — attention stays on CPU, llama3 design)

### NEW reusable infra (back-compat verified, default-off)
- `matrix_vector_multiplication/bf16/matvec.py`: added optional `k_split`
  parameter. Default None preserves existing behavior byte-for-byte.
- `llama3/multi_launch_builder/o_gemv_ffn_multi.py`: added optional
  `down_k_split` parameter forwarded to matvec.
- `qwen25_bias.py`: refactored to monkey-patch `KernelCache.load_and_run`
  (covers BOTH prefill `rms_gemms_rope` and decode `rms_gemv_rope`); added
  `set_decode_position(pos)` API.

### Lesson 5 captured (LESSONS.md)
- Two distinct K-DMA `repeat_count > 255` walls at hidden_dim ≥ ~8200.
- `k_split` knob in matvec.py is reusable for any future model with
  hidden_dim > 8160 (incl. Llama-3-8B at 14336).

### Phase 3 BLOCKERS — RESOLVED 2026-04-19
1. **seq_len=2048 BD allocator exhaustion** — RESOLVED via `qwen25_pad.py`
   GQA-aware reindexed padding (emb_dim 1536→2048, hidden_dim 8960→9216,
   Q heads reindexed inside KV groups). Phase 2 cosine = 0.9988 at
   seq_len=2048 (matches the unpadded seq_len=512 PASS).
2. **seq_len < 512 silently corrupts GEMM outputs** — runner must pad
   short prompts to ≥512 (still applies for the unpadded path; padded
   path inherits the same constraint with tile_m*herd_m=512).

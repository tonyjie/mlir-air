# Qwen3-1.7B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen3-1.7B` inference on AMD NPU2 (AIE2P). **Second
validation** of the kernel-first methodology developed during qwen3-0.6B.
Reference deployments:
- [`../qwen3_0_6b/`](../qwen3_0_6b/) — first Qwen3 deployment (full Q/K Norm
  design, 3-K matvec rename, all the kernel-first methodology details)
- [`../llama3/`](../llama3/) — canonical kernel sequence, multi-launch ELF,
  KernelCache, BO pre-loading, seq-first layout
- [`../llama32_3b/`](../llama32_3b/) — head_dim=128 + Option C head-first
  FA wrapper (this model also requires it)

## Status

**Production-ready** (2026-04-21, independently audited PASS-with-warnings).
End-to-end full-NPU inference operational via `qwen3_inference.py`
(`make run`).

Performance (validated):
- **NPU Prefill**: 2.81 s warm @ seq_len=2048 (100 ms/layer × 28 layers)
- **NPU Decode**: 0.149 s/token (6.7 tok/s)
- 3 decode ELFs: `rms_attn_gemvs_qknorm_rope` (8 launches, fuses RMSNorm +
  Q/K/V GEMV + Q/K Norm + RoPE Q/K), `o_gemv_ffn_silu` (8 launches via
  llama3's **2-K** matvec rename — no qwen3 fork needed), `lm_head_gemv`
  (19 partitions × 8192)

Functional gates verified:
- 6/6 canonical prompts PASS dynamic decisive/competitive gate
- 8/8 NPU decode tokens within CPU top-5 (7/8 exact top-1)
- 4/4 adversarial prompts (out-of-canonical) PASS top-5 — see audit report

See [`docs/development_progress/progress.md`](docs/development_progress/progress.md)
for the full phase results table and methodology validation log.

## Model config

28 layers, emb_dim=2048, n_heads=16, **head_dim=128**, n_kv_heads=8 (GQA group=2),
hidden_dim=6144, vocab=151936, BF16, **rope_θ=1,000,000**, **tied embeddings**
(but `lm_head.weight` also stored explicitly in safetensors), **NO QKV bias**,
**NEW Q/K Norm** (per-layer per-head RMSNorm BEFORE RoPE).

## Divergences from prior deployments

### Vs qwen3-0.6B (the closest reference)

1. **q_dim == emb_dim** (2048 == 2048) — 0.6B had q_dim (2048) ≠ emb_dim (1024)
   because n_heads × head_dim ≠ emb_dim there. For 1.7B they coincide, which
   means:
   - **2-K matvec rename instead of 3-K**. The `compile_mv_og` /
     `compile_mv_dg_qwen3` helpers added during 0.6B aren't needed here.
     llama3's `o_gemv_ffn_multi.build_o_gemv_ffn_module` works directly
     (default `mv.o` for K=2048 + renamed `mv_dg_qwen3.o` for K=6144 Down).
   - No `q_dim` / `o_in_dim` kwarg overrides on the shared builders —
     they default to emb_dim and just work.
   - `qwen3_decode.py` calls llama3's builder directly, NOT 0.6B's
     `o_gemv_ffn_silu_qwen3` fork.
   - **arg layout follows llama3**: `wo` at arg0, `attn_out` at arg1
     (qwen3-0.6B fork swapped these — be careful when copying decode glue).

2. **LM head GEMV reshaped: 19 partitions × 8192** (vs 0.6B's 10 × 16384).
   At emb_dim=2048 the standard 16/16/8 tile config breaches L2 by 256 B
   (A = herd_m·tile_m·K·2 = 8·16·2048·2 = 524288 B = exactly the cap, then
   C = 256 B trips it). Halving M_part to 8192 with `tile_m=8 / m_input=8 /
   herd_m=8` gives A = 262144 B ✓ and per-partition B-DMA fires =
   8192/(8·8) = 128 ≤ 255 ✓. Padding = 19·8192 − 151936 = 3712 rows.

3. **Down GEMV K=6144 L2 budget**: in `o_gemv_ffn` the standard tile_m=8
   gives A = 8·8·6144·2 = 786432 B > 512 KB. Reduced to
   `down_tile_m=2, down_m_input=1` → A = 8·2·6144·2 = 196608 B ✓.
   K=6144 ≤ 8160 (auto-split limit) so no `down_k_split` needed. Llama-3-8B
   class K=14336+ would need k_split.

### Vs other deployments

4. **NEW Q/K Norm** (Qwen3-only, vs Qwen2.5 / Llama family): same design
   as 0.6B — fused into `rms_attn_gemvs_qknorm_rope` via on-tile per-head
   `weighted_rms_norm` (heads-as-M trick).
5. **NO QKV bias** (vs Qwen2.5 / qwen25_1_5b which has bias) — skip the
   bias-add wrapper.
6. **head_dim=128** (same as `llama32_3b/`, `qwen25_1_5b/`, `qwen3_0_6b/`):
   use Option C head-first FA wrapper from
   `_llm_shared/phase_helpers/headfirst_fa.py` — seq-first FA is broken
   for `dk_chunks > 1`.
7. **GQA group=2** (vs Qwen2.5 group=6, llama32_3b group=3): minimal sharing
   per KV head. Group is a power of 2 → no GQA-reindex padding (unlike
   qwen25 which needed phantom-Q-head padding).
8. **BD-friendly shapes**: emb_dim=2048 and hidden_dim=6144 are clean
   multiples of 1024 — no padding gymnastics at GEMM tile config.

## File layout convention

Minimal scaffold + sys.path imports (recommended pattern; see
`deploy-new-llm` skill). This directory contains **only Qwen3-1.7B-specific
code** (mostly shape-constant copies of qwen3_0_6b/); orchestration helpers
and multi-launch ELF builders are imported from `../llama3/`,
`../qwen3_0_6b/` (transitively), and `../_llm_shared/` at runtime.

Qwen3-1.7B-specific code:
- `qwen3_weights.py` — config dataclass + HF safetensors loader (Q/K Norm + no QKV bias) + RoPE LUT, defaults updated for 1.7B
- `qwen3_reference.py` — CPU F32 reference forward pass (Q/K Norm BEFORE RoPE)
- `qwen3_decode.py` — decode block helper + `compile_decode_kernels` (uses **llama3's** o_gemv_ffn_multi, NOT the qwen3-0.6B fork)
- `qwen3_inference.py` — end-to-end NPU runner
- `qwen3_phaseN_test.py`, `qwen3_kernel_registry_test.py`, `qwen3_verify_decode.py`, `qwen3_canonical_sweep.py` — per-phase / per-leaf validation
- `multi_launch/rms_attn_gemvs_qknorm_rope_qwen3.py` — Phase B fused decode ELF #1 builder

Imported from `../llama3/` (do not copy):
- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_inference._preload_decode_weights`
- `multi_launch_builder.{rms_gemms_rope, o_ffn, lm_head_gemv, o_gemv_ffn_multi, ...}` (current production)
- `multi_launch_builder.superseded.{rms_attn_gemms_multi, rope_qk_multi}` (split-ELF prefill path for Q/K Norm)

Imported from `../_llm_shared/`:
- `KernelCache`, `prepare_air_project`
- `compile_all_external_kernels` (rope_halfsplit, silu_and_mul, attn_npu2, mv)
- `compile_attn_npu2_split` (per-tile NPU FA for head_dim=128)
- `phase_helpers.headfirst_fa.patch_run_cached_for_headfirst_fa` (Option C wrapper)
- `phase_helpers.qk_norm.apply_qk_norm` (host helper for split-ELF Q/K Norm)
- `phase_helpers.{metrics, canonical_prompts, decode_setup, ...}`

## Documentation

| Doc | Content |
|---|---|
| [README.md](README.md) | Newcomer overview |
| [TODO.md](TODO.md) | Phase status, key 1.7B-specific decisions |
| [`docs/development_progress/progress.md`](docs/development_progress/progress.md) | Phase results table + kernel-first methodology validation |
| [`docs/evaluation_report.md`](docs/evaluation_report.md) | Independent audit (re-derived gates + 3 warnings) |
| [`../qwen3_0_6b/CLAUDE.md`](../qwen3_0_6b/CLAUDE.md) | First Qwen3 deployment (Q/K Norm design, 3-K rename) |
| [`../llama3/CLAUDE.md`](../llama3/CLAUDE.md) | Canonical kernel sequence, multi-launch design |
| [`../llama32_3b/CLAUDE.md`](../llama32_3b/CLAUDE.md) | head_dim=128 + Option C FA wrapper reference |

## Audit warnings (open follow-ups)

Per [`docs/evaluation_report.md`](docs/evaluation_report.md):
1. ~~Missing `README.md` and `CLAUDE.md`~~ — fixed by this commit.
2. `Makefile` `make run-full` target was broken — fixed (redirected to sweep).
3. `qwen3_inference.py:45` imports `install_headfirst_fa_wrapper` but never
   calls it — dead import. Low priority; the wrapper is installed
   transitively via `compile_block_kernels`. Cleanup deferred.

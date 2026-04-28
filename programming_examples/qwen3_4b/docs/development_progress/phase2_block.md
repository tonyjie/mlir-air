# Phase 2 — Single-Block Validation (PASS, 2026-04-27)

## Result

| Metric | Value | Gate |
|---|---|---|
| Whole-tensor cosine (real-token) | **0.998753** | ≥ 0.99 ✓ |
| Per-position cosine min | **0.998232** | ≥ 0.98 (hd=128) ✓ |
| MAE | 0.020888 | informational |
| max_abs | 1.1790 | informational |
| NaN | False | ✓ |

**Verdict**: PASS at `seq_len=2048` (production target) with padded
`emb_dim=3072`, `hidden_dim=10240`. Confirmed seq=512 and seq=2048
produce identical numbers.

## Bisect (NPU rms_attn_gemms outputs vs CPU)

| Tensor | Cosine | max_abs | mean_abs |
|---|---|---|---|
| normed | 0.999973 | 0.0383 | 0.0002 |
| q | 0.999805 | 0.0969 | 0.0014 |
| k | 0.999859 | 0.0706 | 0.0018 |
| v | 0.999886 | 0.0259 | 0.0009 |

All Q/K/V GEMMs producing correct numerical output via the fused
`rms_attn_gemms` ELF.

## Integration path

**Kernel-first split-ELF + host Q/K Norm + host RoPE** (mirror of
qwen3_1_7b/qwen3_0_6b):

```
NPU rms_attn_gemms (predecessor builder, no RoPE) → produces normed, q, k, v
  → host apply_qk_norm (Qwen3 NEW: per-head RMSNorm before RoPE)
  → host RoPE (BF16 inside, F32 internally — predecessor rope_qk_multi LUT incompatible)
  → NPU flash_attn (head-first wrapper for hd=128, Option C)
  → NPU o_ffn (with o_in_dim=q_dim=4096; emb_padded=3072; hidden_padded=10240)
```

3 NPU launches per block (rms_attn_gemms, flash_attn, o_ffn) + 2 host
launches (apply_qk_norm, RoPE).

## Padding workaround

`qwen3_4b_pad.py` (NEW; qwen3-specific simpler than qwen25_pad):
- `emb_dim 2560 → 3072` (3×1024-aligned). 1.20× compute inflation on
  Q/K/V/O GEMMs touching emb.
- `hidden_dim 9728 → 10240` (10×1024-aligned). 1.05× FFN inflation.
- RMSNorm weight rescaling by `sqrt(2560/3072) ≈ 0.9128` for
  `attn_norm`, `ffn_norm`, `final_norm`.
- `n_heads / n_kv_heads / head_dim / q_dim / kv_dim` UNCHANGED — Qwen3-4B
  has `q_dim ≠ emb_dim`, so no GQA-aware reindex is needed (simpler than
  qwen25_pad's phantom-Q-head trick).
- CPU sanity: padded forward vs orig forward cosine **0.999998573** ✓.

## Debug timeline (notable_events for paper §6 dev_min)

3 NPU compile+run cycles (~1.5 min compile + ~1 min runtime each ≈ 8 min
NPU work, on top of ~7 min agent dev). Honest record of which axes
were genuinely hard:

1. **Attempt #1**: emb=2560 unpadded, seq_len=256, default tile config.
   → RMSNorm cos 0.999975 ✓ but Q/K/V GEMM all cos=**0.000000**
   (deterministic garbage, max_abs ≈ 1-2). End-to-end cos 0.082.
2. **Attempt #2**: emb=2560 unpadded, seq_len=256, tile_k_l2=256
   (Phase-1 standalone GEMM used this config and PASSed). → BYTE-IDENTICAL
   garbage to #1 (max_abs 1.9935 vs 1.9914). Tile config not the cause.
3. **Attempt #3**: emb=3072 padded + hidden=10240 padded, seq_len=256,
   default tile config (tile_n=128). → BYTE-IDENTICAL garbage again
   (max_abs 1.9914 still). Padding not the cause either!
4. **Attempt #4**: emb=3072 padded + hidden=10240 padded, **seq_len=512**.
   → **PASS**. q/k/v cos 0.99980+, end-to-end cos 0.998753.

**Root cause** (re-derived from the bisect): not Rule A non-1024-alignment.
The actual trigger was **seq_len=256 < tile_m × herd_m = 64 × 8 = 512**.
With seq_len < herd_m × tile_m, the herd's M-axis under-utilization causes
silent wrong-data-read in the L2-shared `normed` buffer between RMSNorm
and Q/K/V GEMMs. seq=512 = 1 clean herd round across all 8 tiles
restores correctness.

**Why qwen3_1_7b's Phase 2 default seq=256 didn't hit this**: emb=2048
(K=2048) at seq=256 uses different tile coordination — probably an
under-utilization mode that happens to be safe at K=2048 but breaks at
K≥2560.

**Padding decision: KEEP**. The 3072/10240 padding is independently
useful for downstream phases — non-1024-aligned emb/hidden in the FUSED
multi-launch ELF often hits Rule A at production seq_len=2048. Padding
preempts that. Cost: 1.20× emb-side GEMM compute (acceptable). Mirrors
qwen25_3b's hidden 11008→12288 precedent.

## Post-paper followup (deferred)

- Test unpadded at seq_len=512: would tell us whether padding is
  strictly needed at production seq_len=2048, or just a precaution.
  Skipped here for time.
- Surface seq_len ≥ tile_m × herd_m as a Phase 2 prerequisite in
  `single-block-validation` skill (current default seq_len=256 is
  unsafe at K ≥ 2560).

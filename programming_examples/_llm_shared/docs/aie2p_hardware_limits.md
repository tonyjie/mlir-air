# AIE2P shim DMA + lowering limits — single-source-of-truth

This doc consolidates the AIE2P shim DMA / lowering limits we've discovered
during LLM deployments. Future deployments should consult this BEFORE
selecting tile configs / multi-launch packing, to avoid the "compile fails
with cryptic error" → bisect detours we've paid for.

The numbers below are **observed** during deployments on AMD NPU2 (Strix,
AIE2P architecture). They are NOT taken from any official spec doc — they're
the failure thresholds the lowering and aiecc surface in practice. Treat
them as conservative; if a future deployment proves a higher value works,
update this doc.

## The four limits we've hit

| # | Limit | Observed bound | Surfaces as | Discovered by |
|---|---|---|---|---|
| 1 | Shim BD inner-dim `size` | ≤ 1024 (768 OK, 1024 borderline) | DMA auto-splits dim into multi-D pattern | qwen25_1_5b @ emb_dim=1536 (Phase 2) |
| 2 | Shim BD chain `repeat_count` | **≤ 255** | `'aiex.npu.push_queue' op Repeat count exceeds the [0:255] range` | qwen25_1_5b @ K=8960 / M=8960 (Phase 5) |
| 3 | Shim BD pool per channel | ~16 BDs simultaneously in-flight | `'aiex.dma_configure_task' op Allocator exhausted available buffer descriptor IDs` | qwen25_1_5b @ seq_len=2048 + emb_dim=1536 (Phase 2) |
| 4 | AIE2P BF16 vector width | 32 lanes | sets `tile_k_inner = 32` in matvec (forces K-DMA outer dim count = K / 32) | matvec.cc convention |

## Derived "BD-friendly" rules of thumb

These follow directly from the limits above. Use them when picking dims for
a new model / new tile config:

### Rule A: emb_dim and hidden_dim should be n × 1024

If they aren't, the DMA pattern for "M rows × N cols" splits into a 2D
`(size=A, stride=B)` BD pattern where the outer A can balloon. At
seq_len=2048 + multi-launch ELFs (4+ stitched launches), this can exhaust
the BD pool (limit 3).

**Examples**:
- `2048 = 2 × 1024` ✓ — outer dim is 2, packs cleanly
- `3072 = 3 × 1024` ✓ — outer 3
- `1536 = 1.5 × 1024` ✗ — splits as `(size=512, stride=768)` outer 512
- `8960 = 8.75 × 1024` ✗ — splits as `(size=140, stride=64)` outer 140

**Workaround if not 1024-aligned**: see `qwen25_1_5b/qwen25_pad.py` for
GQA-aware reindexed padding (lift emb_dim 1536→2048, hidden 8960→9216).

### Rule B: GEMV K_max via auto-split = vector_width × repeat_max = 32 × 255 = 8160

Anything larger forces the SHIM B-DMA outer dim past 255 → fails compile.

**Examples**:
- llama3 K=2048 ✓ (auto-split outer = 64)
- llama3 K=8192 ✓ (right at the edge — auto-split outer ≈ 256, just under)
- qwen25 K=8960 ✗ (auto-split outer 280)
- llama3-8B K=14336 ✗ (auto-split outer 448 — needs `k_split` knob)

**Workaround**: use the `k_split` parameter in
`matrix_vector_multiplication/bf16/matvec.py` (added 2026-04-19, default
None preserves existing behavior). Pre-splits the K-DMA dim so the
lowering doesn't auto-split. For K=8960 → `k_split=70` gives `(70, 128)`
outer 70 ≤ 255 ✓.

### Rule C: Per-GEMV B-input read count = launch_count × (tile_m / m_input)

The B input vector read DMA fires `launch_count × inner_loop_count` times
per GEMV. Multiple GEMVs in the same multi-launch ELF that share the same
input share a channel — counts ADD UP.

  launch_count = M / (tile_m × herd_m)
  inner_loop_count = tile_m / m_input

**Surface**: `repeat_count > 255` on a SHIM read-channel BD.

**Examples**:
- llama3 Gate at M=8192, default (tile_m=8, m_input=4, herd_m=8):
  launch=128, inner=2 → 256 fires per GEMV ≈ at edge
- llama3 Gate + Up combined: 256 × 2 = 512 → ❌
  (llama3 Gate and Up may use separate channels or different shape)
- qwen25 Gate at M=8960, default: launch=140, inner=2 → 280 ✗
- qwen25 Gate at M=8960, tile_m=16 m_input=16: launch=70, inner=1 → 70 ✓

**Rule of thumb**: pick `tile_m == m_input` (inner_loop=1) and
`tile_m × herd_m ≥ M / 127` to keep per-GEMV ≤ 127 (so two combined ≤ 254).

### Rule D: L2 capacity per MemTile = 512 KiB

Tile config must satisfy: `K × herd_m × tile_m × bytes_per_elem ≤ 512 KiB`
for the staged A buffer.

**Surface**: `assert a_l2_bytes + c_l2_bytes <= L2_CAPACITY` in matvec.

For BF16 GEMV: `K × herd_m × tile_m ≤ 256 KiB / 2B = 131072 elements`.

**Examples**:
- K=1536, herd=8, tile_m=16: 1536 × 128 = 196608 ✓ (just under)
- K=8960, herd=8, tile_m=4: 8960 × 32 = 286720 / 2B = 143360 elements ✓
- K=8960, herd=8, tile_m=8: 8960 × 64 = 573440 elements > 131072 ✗

## Cross-references to deployment LESSONS files

These rules were discovered during these deployments — see those LESSONS
for full debug narrative:

- Rule A (BD inner-dim 1024): `qwen25_1_5b/docs/development_progress/LESSONS.md`
  Lessons 3–4
- Rule B (K_max via auto-split = 8160): `qwen25_1_5b/.../LESSONS.md` Lesson 5
- Rule C (B-input read count): `qwen25_1_5b/.../LESSONS.md` Lesson 5
- Rule D (L2 capacity): asserted in `matrix_vector_multiplication/bf16/matvec.py`
  build_module; surfaced when tuning Down GEMV at K=8960

## "BD-friendliness audit" checklist for new models

Before starting Phase 2 (single block) on a new model, compute:

```python
# Rule A — flag dims that aren't 1024-aligned
for name, dim in [("emb_dim", config.emb_dim),
                  ("hidden_dim", config.hidden_dim),
                  ("kv_dim", config.n_kv_heads * config.head_dim)]:
    if dim % 1024 != 0:
        print(f"⚠ {name}={dim} not 1024-aligned — Rule A")
        # → likely needs padding (see qwen25_pad.py) at seq_len ≥ 2048

# Rule B — flag if Down GEMV K > 8160
if config.hidden_dim > 8160:
    print(f"⚠ hidden_dim={config.hidden_dim} > 8160 — Down GEMV needs k_split")
    # → use `down_k_split=N` where hidden_dim % N == 0 and N ≤ 255

# Rule C — flag if any GEMV's M needs care
for name, M in [("Q",      config.n_heads * config.head_dim),
                ("K/V",    config.n_kv_heads * config.head_dim),
                ("Gate/Up", config.hidden_dim),
                ("LM-head-partition", 16384)]:
    # Default tile_m=8, m_input=4, herd_m=8 → launch_count × 2
    default_fires = (M // 64) * 2
    if default_fires > 127:  # 2 GEMVs sharing channel = 254 cap
        print(f"⚠ {name} M={M}: default fires {default_fires} > 127 "
              f"— needs tile_m=m_input bump (Rule C)")

# Rule D — flag if Down GEMV L2 doesn't fit
down_l2 = config.hidden_dim * 8 * 2 * 2  # default herd=8 tile_m=2 BF16
if down_l2 > 512 * 1024:
    print(f"⚠ Down L2 {down_l2}B > 512KB — need smaller tile_m or herd_m")
```

Adding this audit to Phase 1 (`validate-per-kernel-shapes`) catches all
four wall types BEFORE compile time, saving ~30 min of debug per failure.

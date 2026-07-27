# SmolVLA Backbone NPU2 Port — Technical Design

(Named `ARCHITECTURE.md`, not `CLAUDE.md` — the repo's top-level `.gitignore`
excludes `CLAUDE.md`, so it would silently not ship.)

This document covers the technical design of the A1 backbone port: kernel
mapping, the non-causal attention design, KV-cache injection, the RoPE-base
gotcha, sequence padding, and the correctness methodology. For scope/how-to-run,
see [`README.md`](README.md). For the pre-port feasibility research, see
[`FEASIBILITY.md`](FEASIBILITY.md). For the approved design spec, see
[`docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md`](docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md).

## Model facts (measured from the checkpoint, not paper prose)

`lerobot/smolvla_base` (450M) backbone = SmolLM2-360M, first 16 of its 32
layers (`text_model.layers[:16]`):

| Param | Value |
|---|---|
| emb_dim (hidden_size) | 960 |
| hidden_dim (intermediate_size) | 2560 |
| n_heads / n_kv_heads | 15 / 5 (GQA group = 3) |
| head_dim | 64 |
| rope_base | **10000** (see [gotcha](#rope-base-gotcha) below) |
| rms_norm_eps | 1e-5 |
| prefix seq_len | 241 (3 cameras x 64 visual tokens + 48 language tokens + 1 state token), padded to **256** on NPU |

## Kernel mapping

One genuinely new kernel; everything else is an existing `kernel_registry`
kernel re-validated at SmolVLA's shapes (recorded in `docs/PROGRESS.md`'s
tested-shapes table).

| Backbone op | Kernel | Shape (seq padded to 256) | ELF |
|---|---|---|---|
| input / post RMSNorm | RMSNorm BF16 (fp32 reduce) | 256x960 | `rms_gemms_rope` |
| q_proj | GEMM bf16-out (drain, tile_m=32) | 256x960x960 | `rms_gemms_rope` |
| k/v_proj | GEMM bf16-out (drain, tile_m=32) | 256x960x320 | `rms_gemms_rope` |
| RoPE (Q, K) | RoPE BF16 half-split, host LUT | head_dim=64, θ=10000 | `rms_gemms_rope` |
| **non-causal attention** | QKᵀ GEMM + **new** masked-softmax + PV GEMM | 15x(256x64x256), (15,256,256), 15x(256x256x64) | `qkt` / `masked_softmax` / `pv` |
| o_proj | GEMM bf16-out (fused-cast) | 256x960x960 | `o_ffn` |
| residual add | Element-wise Add BF16 | 256x960 | `o_ffn` |
| gate/up (SwiGLU) | GEMM bf16-out (fused-cast) | 256x960x2560 | `o_ffn` |
| SiLU-and-Mul | SiLU-and-Mul BF16 | 256x2560 | `o_ffn` |
| down (SwiGLU) | GEMM bf16-out (fused-cast) | 256x2560x960 | `o_ffn` |
| final RMSNorm (post layer 16) | — | 256x960 | CPU F32, outside the hot loop (sibling-consistent; see `docs/TODO.md`) |

Two fused multi-launch ELFs per layer carry the non-attention work (mirroring
the llama/qwen siblings' fusion strategy):
- `rms_gemms_rope` (6 launches): input RMSNorm + Q/K/V GEMM + RoPE on Q and K.
- `o_ffn` (8 launches): O-projection + residual add + post-RMSNorm + gate/up
  GEMM + SiLU-and-Mul + down GEMM + residual add.

Attention is **not** fused into either ELF — it dispatches per q-head (see
below), for **31 attention ELF dispatches/layer** (15 `qkt` + 1 batched
`masked_softmax` + 15 `pv`) on top of the 2 fused ELFs, i.e. 33 ELF
dispatches/layer total. Fusing the 15 per-head dispatches into one
multi-launch attention ELF is a correctness-first-then-optimize deferral
(Phase 4/5 territory), not a blocker — see `docs/TODO.md`.

## Non-causal attention: approach B (GEMM + masked full-row softmax)

The registry's FlashAttention kernel supports both causal and non-causal
attention (and GQA, head_dim 64), but its mask is a single **boolean** switch
(`causal` on/off) — it cannot express SmolVLA's attention mask
(`make_att_2d_masks`: image+language tokens attend **bidirectionally**, the
state token starts a new causal block — this is prefix-LM masking, i.e. a
mixed pattern that is neither fully causal nor fully bidirectional). Two
approaches were on the table:

- **Approach A (rejected):** fork FlashAttention's masking logic to accept an
  arbitrary mask. Rejected because SmolVLA's seq=241 (padded 256) is small
  enough that a single-head QKᵀ score matrix (256x256xfp32 ~= 262 KB)
  comfortably fits L2 — flash's online-softmax tiling exists to avoid
  materializing the full score matrix for long sequences, which is not a
  constraint here. FA's tile config is also a near-fixed point tuned for
  seq≈2048–16384. Reworking the kernel's masking for no benefit at this shape
  was judged not worth the risk.
- **Approach B (built):** a dedicated **non-flash masked attention**, built as
  a first-class `kernel_registry` entry (reusable by A2's action-expert
  attention, which needs the same prefix-mask semantics) out of three
  building blocks, run per q-head with GQA reuse of the 5 kv-heads (group=3):

  ```
  per q-head h (kv = h // 3):
    S = (Qh * 1/sqrt(64)) @ Khᵀ         qkt GEMM ELF   (256,64) x (64,256) -> (256,256)
  (all 15 heads' S folded with the additive mask on host, then one batched call:)
    P = softmax(S + mask)               masked_softmax ELF, batched over all 15 heads x 256 x 256
  per q-head h:
    O = P_h @ Vh                        pv GEMM ELF    (256,256) x (256,64) -> (256,64)
  ```

  All three steps run on NPU; the host only does layout glue (transpose
  `Kh`, gather per-head slices from the fused QKV output, fold the additive
  mask into the score buffer before the single batched softmax dispatch). No
  CPU attention fallback exists in this path (`cpu_attn=False`, the path
  `run_npu_backbone.py` — and therefore `make run`/`make verify` — actually
  uses).

  A CPU reference implementation of the same math
  (`noncausal_attention_reference` in `smolvla_cpu_helpers.py`) exists
  behind `cpu_attn=True` as a de-risking step (Step A) that isolated the NPU
  GQA/RoPE assembly from the NPU-attention risk during Phase 2/3
  development; it is not the production path.

### Host additive mask, `finfo.min` semantics

SmolVLA's own mask construction (`make_att_2d_masks`) produces a boolean
attend/no-attend matrix; the reference math converts it to an **additive**
mask (0 where attention is allowed, a large negative constant where it is
not) before the softmax, exactly like the standard `attn_scores + mask` /
`torch.finfo(dtype).min` pattern. The NPU port follows the same recipe but
substitutes the kernel's native low value: the host builds a `(256,256)`
additive `float32` mask with `-inf` in masked positions (mirroring
`torch.finfo`), then before folding it into the score buffer the driver
rewrites every `-inf` to `BF16_MIN` (`masked_softmax.py`'s bf16 sentinel) —
`np.isneginf(mask)` -> `BF16_MIN`, else `0.0` — because the softmax kernel
operates in bf16 and an unconverted `-inf` would not survive the bf16 cast
cleanly. The kernel's own mask input is left all-zero; the mask is folded
into the score buffer on the host instead, once, before the single batched
softmax call (see `_npu_attention` in `smolvla_backbone_prefill.py`).

Padding tokens (the 256-241=15 pad slots) are excluded identically: they get
`-inf` (-> `BF16_MIN`) rows/columns in the same additive mask, and their RoPE
position IDs are frozen at the last real position
(`positions = cumsum(pad_mask) - 1`, see [padding](#241-to-256-padding) below)
so they never perturb a real token's RoPE phase.

## KV-cache injection mechanism

SmolVLA's action expert does not recompute the prefix on every denoise step —
it reuses a per-layer K/V cache filled once during the backbone's prefill
(`modeling_smolvla.py`'s `fill_kv_cache=True` call). This is the seam the A1
hybrid pipeline exploits:

1. NPU backbone prefill (`run_backbone_prefill(..., return_kv=True)`) returns,
   for every one of the 16 layers, the **post-RoPE** K (5 kv-heads, head_dim
   64) and the **un-rotated** V — the exact tensors `run_transformer_block`
   already produces as `k_roped`/`v` from the fused `rms_gemms_rope` ELF, no
   recompute needed.
2. `smolvla_inference.run_hybrid_forward` wraps `vlm_with_expert.forward`: when
   lerobot calls it with `fill_kv_cache=True`, the wrapper lets the real
   forward pass build a correctly-shaped `past_key_values` dict (structure,
   dtypes, batch dim all lerobot-native), then **overwrites** each layer's
   `key_states`/`value_states` entries in place with the NPU-computed K/V
   (reshaped to lerobot's `(1, L, 5, 64)` layout, dtype-matched to the
   original cached tensor).
3. The unchanged 10-step flow-matching denoise loop reads this cache exactly
   as it would in a pure-CPU run — the expert code has no NPU awareness.

**Execution model (A3-7): SINGLE PROCESS.** The lerobot venv turned out to
have `air`/`aircc`/`pyxrt` too (with the mlir-air env sourced), so steps 1-3 all
run in one interpreter: `smolvla_npu_runtime.BackboneRuntime` holds the weights,
the ELF `KernelCache`, the XRT context and the device BOs for the process
lifetime. The original **bridged** model — the NPU stage as a subprocess in the
worktree python exchanging `.npz` files (`run_npu_backbone.py` /
`run_npu_vision.py`) — is kept as a fallback behind
`run_hybrid_forward(bridge=True)` for environments where the driver venv really
cannot import `air`; it costs ~570-585 ms per NPU stage per inference in process
spawn, safetensors reload and ELF/XRT load.

**This whole KV-injection mechanism is DISABLED in the production config.**
`npu_backbone=False` (the default since A3-7) leaves lerobot's own torch prefill
result in place — nothing is hooked and nothing is overwritten — because the CPU
prefill is measurably faster at seq=256 (76-93 ms vs 238-252 ms on NPU). The
mechanism above is still exercised by `make verify-npu-backbone`.
The outer caller (`make verify`/`make run`, via the Makefile's own recipe)
holds `flock /tmp/mlir-air-npu.lock`; the subprocess does not re-acquire that
same path (would self-deadlock) — its own `KernelCache` uses a distinct inner
`filelock`. Note this Makefile's targets self-lock internally, so do not wrap
`make verify`/`make run` in a redundant external `flock
/tmp/mlir-air-npu.lock` — see the README's "How to run" note; two flocks on
the identical path from nested processes deadlock until the 30-minute
timeout, surfacing as `make: *** [Makefile:NN: verify] Error 1`.

## RoPE-base gotcha

The HF checkpoint config reports `rope_theta=100000`, but the actual runtime
RoPE application (`smolvlm_with_expert.py:28`'s `apply_rope()`) defaults
`max_wavelength=10_000` and **no call site overrides it** — so the real
runtime rope_base is **10000**, not the config's 100000. This was caught by
diffing against the config value rather than trusting paper/config prose, and
verified via the Phase-0 CPU-reference cosine gate (a 100000 base would have
produced a visible RoPE-phase mismatch by layer 1). `rope_interleaved=False`,
so RoPE uses the half-split (not interleaved-pair) convention, matching the
registry's RoPE kernel.

## 241 -> 256 padding

The real prefix length is 241 (192 visual + 48 language + 1 state — see
README diagram). The registry's GEMM/RMSNorm/attention kernels at this shape
were validated at **256** (a friendlier tile boundary; 241 is not a nice
power-of-two and would force awkward tile remainders). The padding is
transparent to correctness:

- The extra 15 rows are zero-initialized hidden states.
- The additive attention mask is `-inf` (`BF16_MIN`) for every row/column
  touching a padded position, so padded tokens neither attend to nor are
  attended to by real tokens.
- RoPE position IDs for padded slots are frozen at the last real position
  (`positions_256 = clip(cumsum(pad_mask_256) - 1, 0, None)`), so they never
  advance the RoPE phase past the real content, though this is moot in
  practice since they're fully masked out of attention anyway.
- Only the first 241 rows of the final backbone output are read back into the
  CPU expert's K/V cache (`run_npu_backbone.py`'s `oracle_len` slicing).

## Dual-track correctness

Both tracks are required to call a phase done — a passing e2e gate alone
would not catch a regression that happens to cancel out end-to-end, and a
passing per-layer cosine alone would not catch an integration bug in the
CPU/NPU handoff.

1. **Diagnosis lens (per-layer cosine, `make diagnosis` ->
   `test_full_backbone.py --npu-attn`):** each of the 16 layers' NPU output
   compared to the CPU oracle's forward-hook hidden state at that exact layer
   (`smolvla_oracle.npz`'s per-layer array). Used to **localize** a
   regression to a specific layer/op when something drifts, not as the
   ship/no-ship gate. Measured: full-backbone final-hidden cosine **0.9962**
   (Step B / NPU attention) vs **0.9962**-tier (Step A / CPU attention) — the
   two attention paths track each other closely, confirming the NPU attention
   kernel is not the accuracy bottleneck.
2. **PASS/FAIL gate (end-to-end, `make verify`, mirrors the sibling LLMs'
   `make verify` convention but with a continuous-output regression gate
   instead of the token-set inclusion gate autoregressive models use):** the
   full hybrid pipeline's (1,50,6) action chunk vs the pure-CPU baseline
   chunk (`verify_adapter.py`'s `regression_gate`), same fixed zero noise on
   both sides for determinism. Gated on cosine (primary, `cos_min=0.99`) AND
   a magnitude-invariant normalized MSE (secondary,
   `nmse = mse / mean(ref**2)`, `nmse_max=0.04`) — normalized specifically so
   the gate does not silently drift PASS/FAIL as action magnitude changes
   across prompts. Measured: median per-position cosine **0.9971**, raw MSE
   **9.13e-4**, normalized MSE **0.0078**, max_abs **0.054** -> **PASS**.

SmolVLA does not use the shared `verify` subsystem's token-set gate
(`compute_topk_set_check`/`verify_runner.py`'s `HfRunner` harness) because it
is not a token-generation model — its output is a continuous action chunk,
hence the dedicated `regression_gate` comparator (added to
`verify/comparators.py` in Task 3.2) and a bespoke `verify_adapter.py` that is
structurally parallel to the sibling LLMs' adapters but drives
`run_hybrid_forward` instead of an autoregressive decode loop.

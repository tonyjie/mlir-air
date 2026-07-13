# SmolVLA Backbone Hybrid Inference on NPU2 (MLIR-AIR)

This example ports the language backbone of **SmolVLA**
([`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), a
Vision-Language-Action model from the LeRobot ecosystem) to AMD **NPU2**
(Strix, AIE2P) using the mlir-air compiler flow and the
`programming_examples/kernel_registry` kernel set.

Milestone **A1**: the **16-layer SmolLM2-360M language backbone prefill**
(RMSNorm, GQA QKV projections, RoPE, non-causal attention, O-projection,
residual add, SwiGLU FFN — every op in the per-layer hot loop) runs on NPU2.
Vision (SigLIP) and the action-expert flow-matching denoise loop stay on CPU.
This is a **hybrid CPU/NPU pipeline**, not a full-model port — see [Scope](#scope-a1)
below.

For the feasibility study that preceded this port, see [`FEASIBILITY.md`](FEASIBILITY.md).
For the design spec and execution plan, see
[`docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md`](docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md)
and [`docs/superpowers/plans/2026-07-13-smolvla-backbone-npu-port.md`](docs/superpowers/plans/2026-07-13-smolvla-backbone-npu-port.md).
For the technical design of the NPU backbone itself, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## The CPU/NPU split

```
[CPU] 3 cameras -> SigLIP (frozen) -> connector -> 192 visual tokens  ┐
[CPU] language tokenize -> 48 tokens                                  ├─> prefix (241,960), padded to 256
[CPU] state -> 1 token                                                ┘        │
                                                                                ▼
                                              ┌─────────────────────────────────────────────┐
                                              │ [NPU2] SmolLM2-360M backbone, 16-layer prefill│
                                              │  RMSNorm -> Q/K/V GEMM -> RoPE -> non-causal  │
                                              │  attention -> O-proj -> residual -> SwiGLU FFN│
                                              └─────────────────────────────────────────────┘
                                                                                │
                                                            per-layer post-RoPE K/V cache
                                                                                │
                                                                                ▼
                             [CPU] inject NPU K/V into lerobot's past_key_values,
                                   run the UNCHANGED action-expert 10-step
                                   flow-matching denoise loop
                                                                                ▼
                                                                (1, 50, 6) action chunk
```

Vision and the action expert are intentionally left on CPU for A1: vision is
frozen, the highest additional engineering cost (bidirectional 1024x1024
attention + 3 new kernels — GELU, LayerNorm, pixel-shuffle), and the lowest
ROI, since it runs once per episode rather than in the per-step hot loop. The
action expert is the natural **A2** follow-on (see [Future work](#future-work-a2-a3)).

## How to run

Two Python environments are involved (the two are bridged automatically by
`smolvla_inference.run_hybrid_forward`, which drives from the lerobot venv and
spawns the NPU backbone as a subprocess in the worktree python):

- `LEROBOT_PYTHON` (default `~/Projects/smolvla_playground/.venv/bin/python`) —
  has `torch` + `lerobot`; drives the CPU prefix assembly + action expert +
  the correctness gates.
- `NPU_PYTHON` (default the worktree `sandbox/bin/python3`) — has `air` +
  `pyxrt`; runs the NPU backbone prefill.

All NPU-touching targets are wrapped with `flock -x -w 1800 /tmp/mlir-air-npu.lock`
**inside** the Makefile itself (this machine has one NPU shared across
sessions) — unlike some sibling examples, whose Makefiles do NOT self-lock and
therefore expect the caller to wrap `make run`/`make profile` in an external
`flock`. **Do not wrap `make run`/`make verify`/`make export-kv`/`make
diagnosis` in an extra outer `flock /tmp/mlir-air-npu.lock`** — the target's
own internal lock will then wait on itself and time out after 30 minutes
(`make: *** [Makefile:NN: verify] Error 1`), because the outer process already
holds the same lock. Just run `make verify` directly.

```bash
make help        # target list
make oracle      # (re)generate smolvla_oracle.npz: pure-CPU fixture (prefix,
                  # per-layer hidden states, action chunk) — no NPU, lerobot venv only
make export-kv   # Step-1 gate: NPU-exported per-layer post-RoPE K/V vs the CPU KV cache
make run         # one hybrid forward end to end; prints the action chunk
make verify      # the production PASS/FAIL gate: e2e action-chunk regression vs CPU baseline
make diagnosis   # informational: full 16-layer per-layer backbone cosine vs oracle (NPU attention)
make clean       # remove kernel cache + build artifacts
```

`make oracle` must be run at least once (it writes `smolvla_oracle.npz`, the
fixture every other target reads) before `make verify`, `make run`, or
`make diagnosis`.

## Correctness gates (dual-track)

Two independent tracks, both required to call a phase done — see
[`ARCHITECTURE.md`](ARCHITECTURE.md#dual-track-correctness) for the rationale.

1. **Diagnosis lens (per-layer cosine, `make diagnosis`)** — each of the 16
   backbone layers' output vs the CPU oracle's forward-hook hidden state at
   that layer, using the real NPU attention path (`--npu-attn`). Used to
   localize regressions to a specific layer/op. Measured: full-backbone final
   hidden cosine **0.9962** vs the CPU oracle.
2. **PASS/FAIL gate (end-to-end, `make verify`)** — the full hybrid pipeline's
   (1, 50, 6) action chunk vs the pure-CPU baseline chunk (same fixed zero
   noise), gated on cosine (primary) and a magnitude-invariant normalized MSE
   (secondary). Measured: median per-position cosine **0.9971**, normalized
   MSE **0.0078** (gate: cosine >= 0.99, nmse <= 0.04) — **PASS**.

Both numbers are measured on real NPU2 hardware, not simulated.

## Scope (A1)

- **On NPU:** all 16 backbone layers' RMSNorm, Q/K/V GEMM projections, RoPE,
  non-causal (prefix-LM) attention (QKᵀ GEMM + masked softmax + PV GEMM, all
  three on-device), O-projection, residual add, SwiGLU FFN (gate/up GEMM,
  SiLU-and-Mul, down GEMM).
- **On CPU:** vision encoder (SigLIP, frozen), the multimodal connector,
  prefix/token assembly, the final backbone RMSNorm (a single (seq, emb)
  reduction outside the per-layer hot loop — the same choice every
  llama/qwen sibling makes), the action-expert transformer, and the 10-step
  flow-matching denoise loop.
- Prefill-only: SmolVLA's backbone is a one-shot prefill (no autoregressive
  decode, no token sampling, no `lm_head` in the action path); its output is
  a per-layer K/V cache consumed by the CPU action expert, not a token.

See `docs/TODO.md` for the itemized NPU-execution exceptions and their
justifications, and `docs/PROGRESS.md` for the phase-by-phase report
(kernel validation numbers, tile configs, non-obvious findings).

## Future work (A2, A3)

- **A2:** port the action expert (cross-attention to the VLM K/V + causal
  self-attention, 10-step flow-matching) to NPU, reusing the non-causal
  masked-attention kernel built for A1.
- **A3 (stretch, likely never):** the SigLIP vision encoder — hardest
  (1024x1024 bidirectional attention + GELU/LayerNorm/pixel-shuffle, none yet
  in the registry) and lowest ROI since it is frozen and runs once per
  episode.

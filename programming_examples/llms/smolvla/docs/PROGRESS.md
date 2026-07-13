# SmolVLA Backbone Port — Progress

Target: lerobot/smolvla_base 16-layer SmolLM2-360M backbone on NPU2.
Spec: docs/superpowers/specs/2026-07-13-smolvla-backbone-npu-port-design.md

## Phase status
- [ ] Phase 0: CPU reference + oracle hooks
- [ ] Phase 1: kernel validation (7 existing shapes + non-causal attn)
- [ ] Phase 2: single-block validation
- [ ] Phase 3: full-backbone + end-to-end action-chunk gate

## Tested (kernel, shape) — filled in Phase 1

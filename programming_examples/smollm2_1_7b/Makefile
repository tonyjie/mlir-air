# ============================================================
# SmolLM2-1.7B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile         — Compile prefill + decode kernels (one-time, ~2 min)
#   make run             — End-to-end demo: NPU prefill seed + NPU decode + NPU LM Head
#   make profile         — Prefill perf measurement (cold vs warm with preload)
#   make verify          — Decode with NPU/CPU top-1 verification
#
# Performance (NPU2, AMD Strix, validated 2026-04-17):
#   Prefill:  1.88s NPU layers / 2.41s wall (24 layers, seq_len=2048)
#   Decode:   136 ms/token (7.3 tok/s)
#
# Per-layer rates at parity with Llama-3.2-1B despite MHA's 4x larger K/V GEMVs.
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Configurable parameters (override via `make VAR=value`)
N_TOKENS ?= 8
PROMPT   ?= The capital of France is
SEQ_LEN  ?= 2048
MAX_SEQ  ?= 128
MODEL    ?= HuggingFaceTB/SmolLM2-1.7B

# Phase test scripts
PHASE0     := smollm2_reference.py
PHASE2     := smollm2_phase2_test.py
PHASE3     := smollm2_phase3_test.py
PHASE4     := smollm2_phase4_test.py
PHASE5     := smollm2_phase5_test.py
INFERENCE  := smollm2_inference.py

.PHONY: help compile run profile verify clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " SmolLM2-1.7B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile         Compile prefill + decode kernels (~2 min, one-time)"
	@echo "  make run             End-to-end NPU inference: prefill + decode ($(N_TOKENS) tokens)"
	@echo "  make profile         Prefill perf measurement (cold + warm)"
	@echo "  make verify          Decode with NPU/CPU top-1 match check"
	@echo ""
	@echo "Compile only:"
	@echo "  make compile         All kernels (prefill + decode)"
	@echo "  make compile-prefill Prefill kernels only (~80 s first time)"
	@echo "  make compile-decode  Decode kernels only (~22 s first time)"
	@echo ""
	@echo "Individual run targets:"
	@echo "  make run             smollm2_inference.py — NPU prefill (with KV extraction) + NPU decode"
	@echo "  make run-prefill     Phase 4  — NPU prefill perf (cold vs warm) — no decode"
	@echo "  make run-decode-only Phase 5  — CPU prefill SEED + NPU decode (older path; uses CPU prefill)"
	@echo "  make run-block       Phase 2  — single-block correctness (NPU FA)"
	@echo "  make run-full        Phase 3  — full 24-layer prefill + top-1 match (3 prompts)"
	@echo "  make run-reference   Phase 0  — CPU reference vs HuggingFace F32"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           Remove kernel caches and build artifacts"
	@echo ""
	@echo "Options (override with 'make VAR=value'):"
	@echo "  N_TOKENS=$(N_TOKENS)         Decode tokens to generate"
	@echo "  PROMPT='...'         Input prompt (default: '$(PROMPT)')"
	@echo "  SEQ_LEN=$(SEQ_LEN)       Prefill sequence length"
	@echo "  MAX_SEQ=$(MAX_SEQ)         Decode max position (run-decode-only only)"
	@echo "  MODEL=$(MODEL)"
	@echo "                       HF model id or local snapshot path"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=20"
	@echo "  make profile PROMPT='The meaning of life is'"
	@echo "  make verify N_TOKENS=4    # CPU verify is slow (~25 s/token)"

# ============================================================
# Compile (one-time, ~2 min total)
# ============================================================

## Compile all kernels (prefill + decode)
compile:
	python3 $(srcdir)/$(INFERENCE) --compile-only --seq-len $(SEQ_LEN)

## Compile prefill kernels only (rms_gemms_rope + o_ffn + flash_attn)
compile-prefill:
	python3 $(srcdir)/$(PHASE4) --compile-only --seq-len $(SEQ_LEN)

## Compile decode kernels only (rms_gemv_rope + o_gemv_ffn + lm_head_gemv)
compile-decode:
	python3 $(srcdir)/$(PHASE5) --compile-only

# ============================================================
# Run (uses cached kernels if present)
# ============================================================

## End-to-end NPU inference: NPU prefill + KV extraction + NPU decode + NPU LM Head
run:
	python3 $(srcdir)/$(INFERENCE) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) --profile

## Alias kept for back-compat (older Phase-5-only decode path with CPU prefill seed)
run-decode-only:
	python3 $(srcdir)/$(PHASE5) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--max-seq $(MAX_SEQ) --model $(MODEL) --no-cpu-verify

## Alias for run (back-compat with previous Makefile)
run-decode: run

## Phase 4 — prefill perf (cold + warm with preload_prefill_weights)
run-prefill:
	python3 $(srcdir)/$(PHASE4) --seq-len $(SEQ_LEN) --model $(MODEL)

## Phase 2 — single transformer block on NPU (uses NPU FlashAttention)
run-block:
	python3 $(srcdir)/$(PHASE2) --no-preload --npu-attn --seq-len $(SEQ_LEN) --model $(MODEL)

## Phase 3 — full 24-layer NPU prefill, NPU/CPU top-1 match on 3 canonical prompts
run-full:
	python3 $(srcdir)/$(PHASE3) --seq-len $(SEQ_LEN) --model $(MODEL)

## Phase 0 — CPU reference forward pass + HuggingFace F32 verification
run-reference:
	python3 $(srcdir)/$(PHASE0) --prompt "$(PROMPT)" --model $(MODEL) --verify

# ============================================================
# Profile / Verify wrappers
# ============================================================

## Prefill perf measurement (alias for run-prefill — Phase 4 already prints timings)
profile: run-prefill

## Decode with NPU/CPU top-1 verification (slow: ~25 s/token CPU verify; default 4 tokens)
verify:
	python3 $(srcdir)/$(PHASE5) --n-tokens 4 --prompt "$(PROMPT)" --max-seq $(MAX_SEQ) \
		--model $(MODEL) --cpu-verify

# ============================================================
# Clean
# ============================================================

## Remove kernel caches and build artifacts (keeps source + docs)
clean:
	rm -rf prefill_kernel_cache decode_kernel_cache air_project build_peano build_chess __pycache__
	rm -f *.o air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

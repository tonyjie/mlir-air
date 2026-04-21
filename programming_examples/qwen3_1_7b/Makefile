# ============================================================
# Qwen3-1.7B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile         — Compile prefill + decode kernels (one-time, ~3 min)
#   make run             — NPU prefill + NPU decode end-to-end (10.7 tok/s)
#   make profile         — Prefill perf measurement (cold + warm)
#   make verify          — NPU vs CPU decode per-token top-1 match (qwen3_verify_decode.py)
#   make sweep           — Canonical 6-prompt sweep through full NPU pipeline
#
# Status (2026-04-20): Production-ready. NPU prefill 2.09 s @ seq_len=2048;
# NPU decode 0.09 s/token (10.7 tok/s) via 3 fused ELFs.
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Configurable parameters (override via `make VAR=value`)
N_TOKENS ?= 100
PROMPT   ?= The capital of France is
SEQ_LEN  ?= 2048
MAX_SEQ  ?= 256
MODEL    ?= Qwen/Qwen3-1.7B

# Phase test scripts
PHASE0     := qwen3_reference.py
PHASE2     := qwen3_phase2_test.py
PHASE3     := qwen3_phase3_test.py
PHASE4     := qwen3_phase4_test.py
PHASE5     := qwen3_phase5_test.py
INFERENCE  := qwen3_inference.py

.PHONY: help compile run profile verify clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " Qwen3-1.7B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile         Compile prefill + decode kernels (one-time)"
	@echo "  make run             NPU prefill + NPU decode ($(N_TOKENS) tokens, ~10.7 tok/s)"
	@echo "  make profile         Prefill perf measurement (cold + warm)"
	@echo "  make verify          Decode with NPU/CPU top-1 match check"
	@echo ""
	@echo "Compile only:"
	@echo "  make compile         All kernels (prefill + decode)"
	@echo "  make compile-prefill Prefill kernels only"
	@echo "  make compile-decode  Decode kernels only"
	@echo ""
	@echo "Individual run targets:"
	@echo "  make run             qwen3_inference.py — NPU prefill (with KV extraction) + NPU decode"
	@echo "  make run-prefill     Phase 4  — NPU prefill perf (cold vs warm) — no decode"
	@echo "  make run-decode-only Phase 5  — CPU prefill SEED + NPU decode (older path)"
	@echo "  make run-block       Phase 2  — single-block correctness (NPU FA)"
	@echo "  make run-full        Phase 3  — full 28-layer prefill + top-1 match"
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

# ============================================================
# Compile (one-time)
# ============================================================

compile:
	python3 $(srcdir)/$(INFERENCE) --compile-only --seq-len $(SEQ_LEN)

compile-prefill:
	python3 $(srcdir)/$(PHASE4) --compile-only --seq-len $(SEQ_LEN)

compile-decode:
	python3 $(srcdir)/$(PHASE5) --compile-only

# ============================================================
# Run (uses cached kernels if present)
# ============================================================

run:
	python3 $(srcdir)/$(INFERENCE) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) --profile

run-decode-only:
	python3 $(srcdir)/$(PHASE5) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--max-seq $(MAX_SEQ) --model $(MODEL) --no-cpu-verify

run-decode: run

run-prefill:
	python3 $(srcdir)/$(PHASE4) --seq-len $(SEQ_LEN) --model $(MODEL)

run-block:
	python3 $(srcdir)/$(PHASE2) --no-preload --seq-len $(SEQ_LEN) --model $(MODEL)

run-full:
	python3 $(srcdir)/$(PHASE3) --seq-len $(SEQ_LEN) --model $(MODEL)

run-reference:
	python3 $(srcdir)/$(PHASE0) --prompt "$(PROMPT)" --model $(MODEL) --verify

# ============================================================
# Profile / Verify wrappers
# ============================================================

profile: run-prefill

## NPU vs CPU decode per-token top-1 verify (uses Phase B fused decode)
##   --prefill-cache name encodes seq_len (LESSON L1: kernel cache name doesn't
##   carry seq_len, so we MUST pass the right cache for the requested SEQ_LEN
##   or we'll read from a stale-shape ELF and get garbage outputs).
verify:
	python3 $(srcdir)/qwen3_verify_decode.py --n-tokens 8 --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) \
		--prefill-cache prefill_kernel_cache_$(SEQ_LEN)

## 6 canonical prompts through full NPU pipeline (NPU prefill + NPU LM head)
sweep:
	python3 $(srcdir)/qwen3_canonical_sweep.py --seq-len $(SEQ_LEN) --model $(MODEL) \
		--prefill-cache prefill_kernel_cache_$(SEQ_LEN)

# ============================================================
# Clean
# ============================================================

clean:
	rm -rf prefill_kernel_cache decode_kernel_cache air_project build_peano build_chess __pycache__
	rm -f *.o air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

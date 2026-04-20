# ============================================================
# Qwen2.5-1.5B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile         — Compile prefill + decode kernels (one-time)
#   make run             — End-to-end demo: NPU prefill seed + NPU decode + NPU LM Head
#   make profile         — Prefill perf measurement (cold vs warm with preload)
#   make verify          — Decode with NPU/CPU top-1 verification
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Configurable parameters (override via `make VAR=value`)
N_TOKENS ?= 100
PROMPT   ?= The capital of France is
SEQ_LEN  ?= 2048
MAX_SEQ  ?= 128
MODEL    ?= Qwen/Qwen2.5-1.5B

# Phase test scripts
PHASE0     := qwen25_reference.py
PHASE2     := qwen25_phase2_test.py
PHASE3     := qwen25_phase3_test.py
PHASE4     := qwen25_phase4_test.py
PHASE5     := qwen25_phase5_test.py
INFERENCE  := qwen25_inference.py

.PHONY: help compile run profile verify clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " Qwen2.5-1.5B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile         Compile prefill + decode kernels (one-time)"
	@echo "  make run             End-to-end NPU inference: prefill + decode ($(N_TOKENS) tokens)"
	@echo "  make profile         Prefill perf measurement (cold + warm)"
	@echo "  make verify          Decode with NPU/CPU top-1 match check"
	@echo ""
	@echo "Compile only:"
	@echo "  make compile         All kernels (prefill + decode)"
	@echo "  make compile-prefill Prefill kernels only"
	@echo "  make compile-decode  Decode kernels only"
	@echo ""
	@echo "Individual run targets:"
	@echo "  make run             qwen25_inference.py — NPU prefill (with KV extraction) + NPU decode"
	@echo "  make run-prefill     Phase 4  — NPU prefill perf (cold vs warm) — no decode"
	@echo "  make run-decode-only Phase 5  — CPU prefill SEED + NPU decode (older path)"
	@echo "  make run-block       Phase 2  — single-block correctness (NPU FA via Option C)"
	@echo "  make run-full        Phase 3  — full 28-layer prefill + top-1 / top-5 match"
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
# Profile / Verify
# ============================================================

profile: run-prefill

verify:
	python3 $(srcdir)/$(PHASE5) --n-tokens 4 --prompt "$(PROMPT)" --max-seq $(MAX_SEQ) \
		--model $(MODEL) --cpu-verify

# ============================================================
# Clean
# ============================================================

clean:
	rm -rf prefill_kernel_cache decode_kernel_cache air_project build_peano build_chess __pycache__
	rm -f *.o air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

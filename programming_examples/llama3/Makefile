# ============================================================
# LLAMA-3.2-1B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile        — Compile all kernels (~4 min, cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown
#
# Performance (NPU2, AMD Strix):
#   Prefill:  1.26s (seq_len=2048, 16 layers)
#   Decode:   92ms/token (10.8 tokens/sec)
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Build directory
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= The capital of France is
MODEL    ?= base

.PHONY: help compile run profile verify chat clean \
        compile-prefill compile-decode run-prefill run-decode

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "Unified pipeline (recommended):"
	@echo "  make compile          Compile prefill + decode kernels"
	@echo "  make run              NPU prefill + NPU decode"
	@echo "  make profile          With per-kernel timing"
	@echo "  make verify           With CPU reference verification"
	@echo ""
	@echo "Individual pipelines:"
	@echo "  make compile-prefill  Compile prefill kernels only"
	@echo "  make compile-decode   Compile decode kernels only"
	@echo "  make run-prefill      Run prefill only (16 layers)"
	@echo "  make run-decode       Run decode only (CPU prefill + NPU decode)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Number of decode tokens"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  MODEL=base|instruct   Model variant (default: base)"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=50"
	@echo "  make profile PROMPT=\"The meaning of life is\""
	@echo "  make run MODEL=instruct PROMPT=\"What is the capital of France?\""
	@echo "  make verify N_TOKENS=10"

# ============================================================
# Unified Pipeline (NPU prefill + NPU decode)
# ============================================================

## Compile all kernels (prefill + decode, ~4 min)
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Run with CPU reference verification
verify:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --n-tokens $(N_TOKENS) --verify --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

## Compile and run in one step
all: compile profile

# ============================================================
# Individual Pipelines
# ============================================================

## Compile prefill kernels only (~4 min)
compile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_prefill.py --compile-only --profile

## Compile decode kernels only (~25s)
compile-decode:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_decode.py --compile-only

## Run prefill only (16 layers, no decode)
run-prefill:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_prefill.py \
		--run-only --n-layers 16 --profile

## Run decode only (CPU prefill for KV cache + NPU decode)
run-decode:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama3_decode.py \
		--run-only --n-tokens $(N_TOKENS) --profile

## Run CPU-only reference (no NPU)
run-reference:
	python3 $(srcdir)/llama3_reference.py

# ============================================================
# Clean
# ============================================================

## Remove all build artifacts
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	@echo "Build directory removed. Run 'make compile' to rebuild."

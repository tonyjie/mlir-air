# ============================================================
# Qwen2.5-0.5B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile         — Compile prefill + decode kernels (one-time)
#   make run             — End-to-end demo: NPU prefill seed + NPU decode + NPU LM Head
#   make profile         — Prefill perf measurement (cold vs warm with preload)
#   make verify          — End-to-end NPU prefill vs CPU F32: per-layer K/V cosine + final logits
#
# Model config (per HF):
#   24 layers, emb_dim=896, n_heads=14, n_kv_heads=2 (GQA g=7),
#   head_dim=64, ffn_hidden=4864, vocab=151936, rope_theta=1000000,
#   tied embeddings, QKV bias, BF16
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
MODEL    ?= Qwen/Qwen2.5-0.5B

# Phase test scripts
PHASE0     := qwen25_0_5b_reference.py
PHASE2     := qwen25_0_5b_phase2_test.py
PHASE3     := qwen25_0_5b_phase3_test.py
PHASE4     := qwen25_0_5b_phase4_test.py
PHASE5     := qwen25_0_5b_phase5_test.py
INFERENCE  := qwen25_0_5b_inference.py

.PHONY: help compile run profile verify verify-decode clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

help:
	@echo "============================================================"
	@echo " Qwen2.5-0.5B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile         Compile prefill + decode kernels (one-time)"
	@echo "  make run             End-to-end NPU inference ($(N_TOKENS) tokens)"
	@echo "  make profile         Prefill perf measurement"
	@echo "  make verify          End-to-end NPU vs CPU F32: per-layer K/V cosine + final logits"
	@echo "  make verify-decode   Decode-side NPU/CPU top-1 (slow)"
	@echo ""
	@echo "Compile only:"
	@echo "  make compile         All kernels (prefill + decode)"
	@echo "  make compile-prefill Prefill kernels only"
	@echo "  make compile-decode  Decode kernels only"
	@echo ""
	@echo "Individual run targets:"
	@echo "  make run             qwen25_0_5b_inference.py — NPU prefill (with KV extraction) + NPU decode"
	@echo "  make run-prefill     Phase 4  — NPU prefill perf (cold vs warm)"
	@echo "  make run-decode-only Phase 5  — CPU prefill SEED + NPU decode"
	@echo "  make run-block       Phase 2  — single-block correctness (NPU FA)"
	@echo "  make run-full        Phase 3  — full 24-layer prefill + top-1 match"
	@echo "  make run-reference   Phase 0  — CPU reference vs HuggingFace F32"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           Remove kernel caches and build artifacts"
	@echo ""
	@echo "Options (override with 'make VAR=value'):"
	@echo "  N_TOKENS=$(N_TOKENS)  PROMPT='...'  SEQ_LEN=$(SEQ_LEN)  MAX_SEQ=$(MAX_SEQ)"
	@echo "  MODEL=$(MODEL)"

# Compile (one-time)
compile:
	python3 $(srcdir)/$(INFERENCE) --compile-only --seq-len $(SEQ_LEN)

compile-prefill:
	python3 $(srcdir)/$(PHASE4) --compile-only --seq-len $(SEQ_LEN)

compile-decode:
	python3 $(srcdir)/$(PHASE5) --compile-only

# Run
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

# Profile / Verify
profile: run-prefill

verify:
	python3 $(srcdir)/$(INFERENCE) --n-tokens 5 --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) --verify

verify-decode:
	python3 $(srcdir)/$(PHASE5) --n-tokens 4 --prompt "$(PROMPT)" --max-seq $(MAX_SEQ) \
		--model $(MODEL) --cpu-verify

# Clean
clean:
	rm -rf build/ air_project/ __pycache__/ && rm -f air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

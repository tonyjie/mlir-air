# ============================================================
# Qwen2.5-3B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================
#
# Quick start:
#   make compile         — Compile prefill + decode kernels (one-time)
#   make run             — End-to-end demo: NPU prefill seed + NPU decode + NPU LM Head
#   make profile         — Prefill perf measurement (cold vs warm with preload)
#   make verify          — End-to-end NPU prefill vs CPU F32: per-layer K/V cosine + final logits
#
# Model config (per HF):
#   36 layers, emb_dim=2048 (1024-aligned, no emb pad needed),
#   n_heads=16, n_kv_heads=2 (GQA g=8), head_dim=128,
#   ffn_hidden=11008 (NOT 1024-aligned; pad → 11264), vocab=151936,
#   rope_theta=1000000, tied embeddings, QKV bias, BF16
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

N_TOKENS ?= 100
PROMPT   ?= The capital of France is
SEQ_LEN  ?= 2048
MAX_SEQ  ?= 128
MODEL    ?= Qwen/Qwen2.5-3B

PHASE0     := qwen25_3b_reference.py
PHASE2     := qwen25_3b_phase2_test.py
PHASE3     := qwen25_3b_phase3_test.py
PHASE4     := qwen25_3b_phase4_test.py
PHASE5     := qwen25_3b_phase5_test.py
INFERENCE  := qwen25_3b_inference.py

.PHONY: help compile run profile verify verify-decode clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

help:
	@echo "============================================================"
	@echo " Qwen2.5-3B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile         Compile prefill + decode kernels"
	@echo "  make run             End-to-end NPU inference ($(N_TOKENS) tokens)"
	@echo "  make profile         Prefill perf measurement"
	@echo "  make verify          End-to-end NPU vs CPU F32 verification"
	@echo "  make verify-decode   Decode-side NPU/CPU top-1 (slow)"
	@echo ""
	@echo "Compile only:  make compile / compile-prefill / compile-decode"
	@echo ""
	@echo "Phase tests:   make run-block (Phase 2) / run-full (Phase 3)"
	@echo "               make run-prefill (Phase 4) / run-decode-only (Phase 5)"
	@echo "               make run-reference (Phase 0)"
	@echo ""
	@echo "Maintenance:   make clean"
	@echo ""
	@echo "Options:       make VAR=value"
	@echo "  N_TOKENS=$(N_TOKENS)  PROMPT='...'  SEQ_LEN=$(SEQ_LEN)  MAX_SEQ=$(MAX_SEQ)"
	@echo "  MODEL=$(MODEL)"

compile:
	python3 $(srcdir)/$(INFERENCE) --compile-only --seq-len $(SEQ_LEN)

compile-prefill:
	python3 $(srcdir)/$(PHASE4) --compile-only --seq-len $(SEQ_LEN)

compile-decode:
	python3 $(srcdir)/$(PHASE5) --compile-only

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

profile: run-prefill

verify:
	python3 $(srcdir)/$(INFERENCE) --n-tokens 5 --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) --verify

verify-decode:
	python3 $(srcdir)/$(PHASE5) --n-tokens 4 --prompt "$(PROMPT)" --max-seq $(MAX_SEQ) \
		--model $(MODEL) --cpu-verify

clean:
	rm -rf build/ air_project/ __pycache__/ && rm -f air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

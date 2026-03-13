# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
progdir := $(realpath $(srcdir)/..)

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# SwiGLU kernel parameters
SWIGLU_N ?= 65536
SWIGLU_TILE_N ?= 1024

# Peano compiler flags
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# Flash attention parameters (for kernel compilation)
LQP ?= 256
LQP_TILE := $(shell echo $$(($(LQP) / 4)))
LKP ?= 64
DK ?= 64
DV ?= 64

all: run-swiglu

# ============================================================
# External kernel compilation (needed for prefill)
# ============================================================

compile-external-kernels:
	mkdir -p $(BUILD_DIR)
	@echo "=== Compiling external kernels ==="
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR is required."; exit 1; \
	fi
	@echo "  [1/3] swiglu_activation.o"
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${srcdir}/swiglu_activation.cc -o $(BUILD_DIR)/swiglu_activation.o
	@echo "  [2/3] rope.o"
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${progdir}/rope_lut/rope.cc -o $(BUILD_DIR)/rope.o
	@echo "  [3/3] attn.o"
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${progdir}/flash_attention/kernel_fusion_based/attn.cc -o $(BUILD_DIR)/attn.o \
		-Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
	@echo "=== External kernels compiled ==="
	@echo "  Copying to air_project/ for aircc..."
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_activation.o $(BUILD_DIR)/air_project/
	cp $(BUILD_DIR)/rope.o $(BUILD_DIR)/air_project/
	cp $(BUILD_DIR)/attn.o $(BUILD_DIR)/air_project/

# ============================================================
# SwiGLU activation kernel (standalone test)
# ============================================================

compile-swiglu-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${srcdir}/swiglu_activation.cc -o $(BUILD_DIR)/swiglu_activation.o; \
	else \
		echo "Error: PEANO_INSTALL_DIR is required for SwiGLU kernel compilation."; \
		exit 1; \
	fi

run-swiglu: compile-swiglu-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/swiglu_activation.py \
		--n $(SWIGLU_N) --tile-n $(SWIGLU_TILE_N) $(OUTPUT_FORMAT_FLAG)

print-swiglu:
	${powershell} python3 ${srcdir}/swiglu_activation.py $(OUTPUT_FORMAT_FLAG) -p

# ============================================================
# CPU reference (no NPU needed)
# ============================================================

run-reference:
	python3 ${srcdir}/llama3_reference.py

run-reference-verify:
	python3 ${srcdir}/llama3_reference.py --verify

# ============================================================
# Weight loading test
# ============================================================

test-weights:
	python3 ${srcdir}/llama3_weights.py

# ============================================================
# Full LLAMA prefill (Phase 2+)
# ============================================================

# LLAMA model path (HuggingFace ID or local dir)
LLAMA_MODEL ?= /home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
SEQ_LEN ?= 2048
N_LAYERS ?= 16

# Single-block test (1 layer, with verification)
run-prefill-1layer: compile-external-kernels
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/llama3_prefill.py \
		--model $(LLAMA_MODEL) --seq-len $(SEQ_LEN) --n-layers 1 --verify

# Full model prefill (16 layers)
run-prefill: compile-external-kernels
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/llama3_prefill.py \
		--model $(LLAMA_MODEL) --seq-len $(SEQ_LEN) --n-layers $(N_LAYERS)

# Full model prefill with verification
run-prefill-verify: compile-external-kernels
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/llama3_prefill.py \
		--model $(LLAMA_MODEL) --seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --verify

clean:
	rm -rf $(BUILD_DIR) __pycache__ .kernel_cache

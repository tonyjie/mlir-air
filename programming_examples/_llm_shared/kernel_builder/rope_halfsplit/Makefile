# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Peano compiler flags (mirror ffn_swiglu Makefile)
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

# Compile rope.o from the half-split RoPE kernel (one level up).
# Output named `rope.o` to match the @rope symbol the herd link_with's.
compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-c $(srcdir)/../rope_halfsplit.cc -o $(BUILD_DIR)/rope.o

# Print MLIR module
print:
	${powershell} python3 ${srcdir}/run.py -p

# Compile + run correctness test (cosine vs CPU F32 numpy reference)
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py

# Compile + profile (timing across N iterations)
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__

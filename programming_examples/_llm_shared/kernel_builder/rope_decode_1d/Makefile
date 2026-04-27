# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Peano flags (mirror rope_halfsplit/Makefile)
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

# Compile rope.o (same C++ kernel as rope_halfsplit/, just placed in this BUILD_DIR)
compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-c $(srcdir)/../rope_halfsplit.cc -o $(BUILD_DIR)/rope.o

print:
	${powershell} python3 ${srcdir}/run.py -p

# Default: Q-shape (n_rows=32). Override with --n-rows for K (n_rows=8).
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py

profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__

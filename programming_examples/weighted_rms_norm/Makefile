# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Default LLAMA dimensions
M ?= 2048
N ?= 2048
HERD_X ?= 1

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

all: run

print:
	${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) -p --M $(M) --N $(N) --herd-x $(HERD_X)

# Correctness test (Python)
run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x $(HERD_X)

# Python profiling
profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x $(HERD_X) --profile

# Build C++ test executable
build-test-exe:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && g++ ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

# C++ profiling (compile xclbin first, then run test.exe)
profile-cpp: build-test-exe
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/weighted_rms_norm.py --output-format xclbin \
		--M $(M) --N $(N) --herd-x $(HERD_X) --compile-mode compile-only
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-M $(M) -N $(N)

clean:
	rm -rf $(BUILD_DIR) __pycache__ test_2048 test_llama test_llama_dims

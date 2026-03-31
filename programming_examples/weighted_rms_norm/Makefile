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

all: run

print:
	${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) -p --M $(M) --N $(N) --herd-x $(HERD_X)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x $(HERD_X)

profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x $(HERD_X) --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__ test_2048 test_llama test_llama_dims

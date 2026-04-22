# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# K2: gemm (64, 2048) x (2048, 2048) -> (64, 2048) bf16 standalone test.
#   make run     - build + run on NPU2; exits 0 on PASS
#   make clean   - remove build artifacts

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

all: run

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 $(srcdir)/kernel.py

clean:
	rm -rf $(BUILD_DIR) __pycache__

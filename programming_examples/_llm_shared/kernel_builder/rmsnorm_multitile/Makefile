# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

all: run

# RMSNorm has no external C++ kernel — pure MLIR
print:
	${powershell} python3 ${srcdir}/run.py -p

# Compile + run correctness (cosine vs CPU F32 numpy reference)
run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py

# Compile + 5 warmup + 20 timed iterations
profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__

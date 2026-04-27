# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

all: run

# Run cosine-based correctness on one llama3 shape (default Q/O 2048^3)
# Pass --shape to pick: qo, kv, gateup, down, all
run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --shape $(or $(SHAPE),qo)

run-all:
	$(MAKE) run SHAPE=all

# 5 warmup + 20 iter timing on a single shape (default qo)
profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --shape $(or $(SHAPE),qo) --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# K7: RoPE chunk K standalone test.
#   x: (64, 512) bf16, lut: (32768,) bf16  -> y: (64, 512) bf16

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

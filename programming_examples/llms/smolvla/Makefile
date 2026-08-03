# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolVLA on NPU2 (MLIR-AIR) — SigLIP vision encoder + connector on the NPU.
#
# Pipeline: NPU vision tower + connector -> CPU prefix assembly -> CPU language
# backbone -> CPU action expert, all in ONE process. The backbone and the action
# expert stay on the CPU because they measure SLOWER on the NPU at their shapes
# (seq 256 and seq 50); see README.md.
#
# Correctness is a continuous-output REGRESSION gate on the (1,50,6) action
# chunk against the unmodified CPU model — not the token-set gate the
# autoregressive LLM siblings use, because SmolVLA does not emit tokens.
#
# One process: with the mlir-air environment sourced and requirements.txt
# installed, one interpreter has torch + lerobot AND air + pyxrt, so the NPU
# stage runs in-process through smolvla_runtime, with weights, ELFs and the XRT
# context loaded once.
#
# NOTE: every recipe here already self-locks the NPU (NPU_LOCK). Do NOT wrap
# `make ...` in an outer flock on the same file — it self-deadlocks.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# The interpreter that runs the model. It needs torch + lerobot (installed by
# requirements.txt) AND air + pyxrt (from the sourced mlir-air environment) in
# ONE process, which is what makes the in-process NPU path work. Defaults to
# the environment's python3; override if your lerobot install lives elsewhere:
#     make verify LEROBOT_PYTHON=/path/to/venv/bin/python
LEROBOT_PYTHON ?= python3
NPU_PYTHON     ?= python3
NPU_LOCK       := flock -x -w 1800 /tmp/mlir-air-npu.lock
BUILD_DIR      ?= build

.PHONY: help compile oracle run verify profile clean

help:
	@echo "=================================================================="
	@echo " SmolVLA on NPU2 — SigLIP vision encoder on the NPU, one process"
	@echo "=================================================================="
	@echo "  make compile   Build every vision ELF (no NPU dispatch, no download)"
	@echo "  make oracle    Regenerate smolvla_oracle.npz, the CPU baseline (no NPU)"
	@echo "  make run       One end-to-end forward; prints the action chunk"
	@echo "  make verify    THE GATE: action chunk vs the pure-CPU model (PASS/FAIL)"
	@echo "  make profile   Per-stage wall clock, NPU vision vs pure CPU"
	@echo "  make clean     Remove the kernel cache and build artifacts"
	@echo ""
	@echo "  LEROBOT_PYTHON=$(LEROBOT_PYTHON)"
	@echo "  (needs torch + lerobot; with the mlir-air env sourced it also has"
	@echo "   air/pyxrt, which is what makes the single-process path work)"

## Build every vision ELF through AIR -> AIE -> aiecc -> Peano.
## No NPU dispatch and no HuggingFace download, so this runs anywhere the
## toolchain is installed.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(NPU_PYTHON) $(srcdir)/smolvla_inference.py --compile-only

## Regenerate the pure-CPU reference the gate compares against.
## CPU only — no NPU, so no lock.
oracle:
	cd $(srcdir) && $(LEROBOT_PYTHON) smolvla_prefix.py

## One end-to-end forward in the shipping configuration.
run:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) smolvla_inference.py

## THE GATE. Action chunk vs the unmodified CPU model: cosine >= 0.99 and
## normalized MSE <= 0.04. Exits non-zero on FAIL.
verify:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py

## Per-stage wall clock, NPU vision against pure CPU, same input and fixed
## noise. Run back to back under one lock so the two are comparable.
profile:
	cd $(srcdir) && $(NPU_LOCK) sh -c '\
	  $(LEROBOT_PYTHON) smolvla_inference.py --cpu; \
	  $(LEROBOT_PYTHON) smolvla_inference.py'

clean:
	rm -rf $(srcdir)/$(BUILD_DIR) $(srcdir)/vision_kernel_cache \
	       $(srcdir)/air_project $(srcdir)/__pycache__ \
	       $(srcdir)/*.elf $(srcdir)/*.xclbin $(srcdir)/*.o $(srcdir)/*.mlir \
	       $(srcdir)/*.insts.bin

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolVLA backbone hybrid inference on NPU2 (MLIR-AIR).
#
# Hybrid CPU/NPU pipeline: CPU vision+prefix -> NPU 16-layer backbone prefill
# -> CPU action expert 10-step denoise. Correctness = a continuous-output
# REGRESSION gate on the (1,50,6) action chunk vs the pure-CPU baseline (NOT the
# token-set gate used by autoregressive LLM siblings).
#
# Two Python envs (BRIDGED execution):
#   LEROBOT_PYTHON  - has torch+lerobot, drives CPU prefix/expert + the gate.
#   NPU_PYTHON      - has air+pyxrt, runs the NPU backbone (spawned as a
#                     subprocess by smolvla_inference.run_hybrid_forward).
# make targets run the LEROBOT_PYTHON driver; the NPU subprocess is internal.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LEROBOT_PYTHON ?= $(HOME)/Projects/smolvla_playground/.venv/bin/python
NPU_PYTHON     ?= /home/jiajli/apps/mlir-air/sandbox/bin/python3
NPU_LOCK       := flock -x -w 1800 /tmp/mlir-air-npu.lock
export SMOLVLA_NPU_PYTHON = $(NPU_PYTHON)

.PHONY: help oracle export-kv run verify diagnosis clean

help:
	@echo "============================================================"
	@echo " SmolVLA backbone hybrid inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo "  make oracle      Regenerate smolvla_oracle.npz (lerobot venv, CPU)"
	@echo "  make export-kv   Verify NPU-exported per-layer K/V vs CPU cache"
	@echo "  make run         Run one hybrid forward, print the action chunk"
	@echo "  make verify      E2E action-chunk regression gate (PASS/FAIL)"
	@echo "  make diagnosis   Full 16-layer per-layer backbone cosine vs oracle"
	@echo "  make clean       Remove kernel cache + build artifacts"
	@echo ""
	@echo "  LEROBOT_PYTHON=$(LEROBOT_PYTHON)"
	@echo "  NPU_PYTHON=$(NPU_PYTHON)"

## Regenerate the pure-CPU oracle fixture (no NPU; lerobot venv only).
oracle:
	cd $(srcdir) && $(LEROBOT_PYTHON) smolvla_prefix.py

## Step-1 gate: NPU-exported post-RoPE K/V vs the CPU KV cache (in-process NPU).
export-kv:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_export_kv.py

## Run one hybrid forward end to end.
run:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) smolvla_inference.py

## The production gate: e2e action-chunk regression vs the pure-CPU baseline.
verify:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py

## Informational: full 16-layer per-layer backbone cosine vs oracle (NPU attn).
diagnosis:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_full_backbone.py --npu-attn

clean:
	rm -rf $(srcdir)/smolvla_block_kernel_cache $(srcdir)/air_project \
	       $(srcdir)/__pycache__ $(srcdir)/.pytest_cache
	@echo "Kernel cache + build artifacts removed."

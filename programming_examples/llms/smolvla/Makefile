# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolVLA hybrid inference on NPU2 (MLIR-AIR).
#
# PRODUCTION pipeline: NPU SigLIP vision tower + connector -> CPU prefix
# assembly -> CPU backbone prefill -> CPU action expert 10-step denoise, all in
# ONE process. Correctness = a continuous-output REGRESSION gate on the
# (1,50,6) action chunk vs the pure-CPU baseline (NOT the token-set gate used
# by the autoregressive LLM siblings).
#
# SINGLE PROCESS: with the mlir-air env sourced, LEROBOT_PYTHON has torch +
# lerobot AND air + pyxrt, so the NPU stages run in-process via
# smolvla_npu_runtime (weights/ELFs/XRT loaded once). NPU_PYTHON is only used
# by the worktree-only NPU tests and by the legacy `*-bridge` targets, which
# keep the old two-process subprocess path alive as a fallback reference.
#
# NOTE: every recipe below already self-locks the NPU (NPU_LOCK). Do NOT wrap
# `make ...` in an outer flock on the same file — it self-deadlocks.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LEROBOT_PYTHON ?= $(HOME)/Projects/smolvla_playground/.venv/bin/python
NPU_PYTHON     ?= /home/jiajli/apps/mlir-air/sandbox/bin/python3
NPU_LOCK       := flock -x -w 1800 /tmp/mlir-air-npu.lock
export SMOLVLA_NPU_PYTHON = $(NPU_PYTHON)

.PHONY: help oracle export-kv run verify verify-npu-backbone verify-bridge \
        vision-connector bench-e2e bench-e2e-all diagnosis clean

help:
	@echo "============================================================"
	@echo " SmolVLA hybrid inference on NPU2 (MLIR-AIR), single process"
	@echo "============================================================"
	@echo "  make oracle      Regenerate smolvla_oracle.npz (lerobot venv, CPU)"
	@echo "  make export-kv   Verify NPU-exported per-layer K/V vs CPU cache"
	@echo "  make run         Run one forward (NPU vision + CPU backbone), print the chunk"
	@echo "  make verify      PRODUCTION gate: NPU vision + CPU backbone (PASS/FAIL)"
	@echo "  make verify-npu-backbone"
	@echo "                   Same gate with the 16-layer prefill ALSO on NPU"
	@echo "  make verify-bridge"
	@echo "                   Same gate through the legacy two-process bridge"
	@echo "  make vision-connector"
	@echo "                   A3-5 Step 4 gate: connector on NPU vs vision oracle"
	@echo "  make bench-e2e   End-to-end wall-time table: CPU / NPU-vision / +NPU-backbone"
	@echo "  make bench-e2e-all"
	@echo "                   ... plus the BLAS-clamp A/B and the legacy bridge rows"
	@echo "  make diagnosis   Full 16-layer per-layer backbone cosine vs oracle"
	@echo "  make clean       Remove kernel cache + build artifacts"
	@echo ""
	@echo "  LEROBOT_PYTHON=$(LEROBOT_PYTHON)   (also has air/pyxrt -> single process)"
	@echo "  NPU_PYTHON=$(NPU_PYTHON)"

## Regenerate the pure-CPU oracle fixture (no NPU; lerobot venv only).
oracle:
	cd $(srcdir) && $(LEROBOT_PYTHON) smolvla_prefix.py

## Step-1 gate: NPU-exported post-RoPE K/V vs the CPU KV cache (in-process NPU).
export-kv:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_export_kv.py

## Run one forward end to end in the production config.
run:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) smolvla_inference.py

## THE production gate: e2e action-chunk regression vs the pure-CPU baseline,
## NPU vision + connector, CPU backbone, single process.
verify:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py

## Same gate with the ACTION EXPERT (16 layers x 10 denoise steps) on NPU.
verify-npu-expert:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py --npu-expert

## Expert on NPU, CPU vision -- isolates the expert's own error budget.
verify-expert-only:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py --npu-expert --cpu-vision

## Tiered teacher-forced gate for the expert (per layer / per step / free-run).
expert-gate:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_expert_npu.py --level 123

## Same gate with the 16-layer backbone prefill ALSO on NPU: the full
## vision+backbone error budget (slower, kept for the record).
verify-npu-backbone:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py --npu-backbone

## Same gate driven through the legacy two-process subprocess bridge, to prove
## that fallback path still works when air/pyxrt are absent from the venv.
verify-bridge:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) verify_adapter.py \
	    --npu-backbone --bridge

## A3-5 Step 4 gate: connector ELF (64x12288x960) vs the vision oracle.
vision-connector:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_vit_connector.py

## The A3-7 deliverable: true end-to-end predict_action_chunk wall time for
## pure-CPU / NPU-vision / NPU-vision+backbone, single process, interleaved.
bench-e2e:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) bench_e2e.py --iters 5 \
	    --configs cpu,vision,vision+backbone

## ... plus the scoped-BLAS-clamp A/B and the legacy bridge rows (slow).
bench-e2e-all:
	cd $(srcdir) && $(NPU_LOCK) $(LEROBOT_PYTHON) bench_e2e.py --iters 5 --blas-ab \
	    --configs cpu,vision,vision+backbone,bridge:vision+backbone

## Informational: full 16-layer per-layer backbone cosine vs oracle (NPU attn).
diagnosis:
	cd $(srcdir) && $(NPU_LOCK) $(NPU_PYTHON) test_full_backbone.py --npu-attn

clean:
	rm -rf $(srcdir)/smolvla_block_kernel_cache $(srcdir)/air_project \
	       $(srcdir)/__pycache__ $(srcdir)/.pytest_cache
	@echo "Kernel cache + build artifacts removed."

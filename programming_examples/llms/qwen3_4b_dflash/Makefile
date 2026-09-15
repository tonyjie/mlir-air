# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Selected Qwen3-4B DFlash B8 inference on AMD NPU2.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD_DIR ?= $(srcdir)/build
TARGET_RAW := $(BUILD_DIR)/target_raw
DRAFT_RAW := $(BUILD_DIR)/draft_raw
TARGET_ARTIFACTS ?= $(BUILD_DIR)/target
DRAFT_ARTIFACTS ?= $(BUILD_DIR)/draft
CAPACITY := 4096
PREFILL ?= npu
PREFILL_CTX ?= 2048
PREFILL_ARTIFACTS ?= $(BUILD_DIR)/prefill_C$(PREFILL_CTX)
PER_DATASET ?= 16
RESUME ?= 0
N_TOKENS ?= 64
STREAM ?= 1
CHAT_LOG ?= $(BUILD_DIR)/chat.jsonl
PROMPT ?= Explain why speculative decoding can improve latency.
BENCHMARK_BUNDLE ?= $(BUILD_DIR)/benchmark_bundle
STUDY_DIR ?= $(BUILD_DIR)/study
STUDY_TOKENS ?= 2048
STUDY_LABEL ?= reviewer
BASELINE_ARTIFACTS ?= $(BUILD_DIR)/baseline
INPUTS ?= $(BENCHMARK_BUNDLE)/inputs
OUTPUT ?= $(BUILD_DIR)/benchmark.jsonl
MODE ?= spec

TARGET_MODEL ?= FastFlowLM/Qwen3-4B-NPU2
TOKENIZER_MODEL ?= Qwen/Qwen3-4B
TARGET_CACHE ?= $(HOME)/.cache/q4nx_qwen3_4b/requant_w2ch.npz
DRAFT_CACHE ?= $(HOME)/.cache/qwen3_4b_dflash_npu/draft_requant_w2ch.npz
DRAFT_SNAPSHOT ?=

export DFLASH_PROMPT := $(PROMPT)
export DFLASH_TARGET_MODEL := $(TARGET_MODEL)
export DFLASH_TOKENIZER_MODEL := $(TOKENIZER_MODEL)
export DFLASH_TARGET_CACHE := $(TARGET_CACHE)
export DFLASH_DRAFT_CACHE := $(DRAFT_CACHE)
ifneq ($(strip $(DRAFT_SNAPSHOT)),)
export DFLASH_DRAFT_SNAPSHOT := $(DRAFT_SNAPSHOT)
endif

RUN := $(srcdir)/env.sh $(PYTHON)
NPU_RUN := flock --exclusive /tmp/mlir-air-npu.lock $(RUN)
SELECTED := --capacity $(CAPACITY) --batch 8 --taps --stack-bytes 6080 \
  --no-l2-pingpong --chain-pingpong --activation-memtile-col 4 \
  --gather-mode packet --token-packets --dynamic-context \
  --projection-weight-pingpong --attention-query-tile 4 \
  --attention-layout fused-shards --normalization-replay 1 \
  --normalization-cache --vector-attention-norm \
  --attention-optimization pv --attention-pair-mode 3

.PHONY: help weights compile compile-target compile-draft compile-prefill run chat verify verify-host benchmark benchmark-report clean

help:
	@echo "Qwen3-4B DFlash B8 on AMD NPU2"
	@echo ""
	@echo "  make weights          Prepare/validate target and draft packed caches"
	@echo "  make compile          Build target, drafter, and NPU prefill"
	@echo "  make compile-prefill  Build only the standard Q4NX NPU prefill kernels"
	@echo "  make run              Stream one response as tokens are committed"
	@echo "  make chat             Prepare once, then stream independent prompts"
	@echo "  make verify-host      Test acceptance/bonus/EOS logic without the NPU"
	@echo "  make verify           Repeatability + same-B8 oracle + committed-KV gate"
	@echo "  make compile-study    Build selected B8 pair and the historical B1 baseline"
	@echo "  make benchmark-data   Download and verify the frozen 80-prompt study"
	@echo "  make benchmark-study  One paired B1/DFlash pass; STUDY_TOKENS=2048"
	@echo "  make study-report     Summarize STUDY_DIR again"
	@echo "  make benchmark INPUTS=/path/to/prepared_inputs N_TOKENS=2048"
	@echo "  make benchmark-report OUTPUT=/path/to/results.jsonl"
	@echo ""
	@echo "Options: N_TOKENS=64 PROMPT=\"...\" STREAM=1|0 MODE=spec|control"
	@echo "         PREFILL=npu|cpu PREFILL_CTX=2048|4096 PER_DATASET=16 RESUME=0|1"
	@echo "         CHAT_LOG=$(BUILD_DIR)/chat.jsonl (one JSON record per completed prompt)"
	@echo "         TARGET_MODEL=... TARGET_CACHE=... DRAFT_CACHE=... DRAFT_SNAPSHOT=..."

weights:
	$(RUN) $(srcdir)/prepare_weights.py $(if $(DRAFT_SNAPSHOT),--draft-snapshot $(DRAFT_SNAPSHOT),)

compile: compile-target compile-draft $(if $(filter npu,$(PREFILL)),compile-prefill,)

compile-prefill:
	$(NPU_RUN) $(srcdir)/npu_prefill.py --seq-len $(PREFILL_CTX) --cache-dir "$(PREFILL_ARTIFACTS)"

compile-target:
	@if [ -f $(TARGET_ARTIFACTS)/manifest.json ]; then \
	  echo "using existing $(TARGET_ARTIFACTS)"; \
	else \
	  set -e; \
	  test ! -e $(TARGET_RAW) && test ! -e $(TARGET_ARTIFACTS); \
	  mkdir -p $(BUILD_DIR); \
	  $(NPU_RUN) $(srcdir)/build_target.py --output $(TARGET_RAW) --role target --layers 36 --causal $(SELECTED); \
	  $(NPU_RUN) $(srcdir)/route_artifact.py --source $(TARGET_RAW) --output $(TARGET_ARTIFACTS) \
	    --fix-shim-arbiter --isolate-main-output; \
	fi

compile-draft:
	@if [ -f $(DRAFT_ARTIFACTS)/manifest.json ]; then \
	  echo "using existing $(DRAFT_ARTIFACTS)"; \
	else \
	  set -e; \
	  test ! -e $(DRAFT_RAW) && test ! -e $(DRAFT_ARTIFACTS); \
	  mkdir -p $(BUILD_DIR); \
	  $(NPU_RUN) $(srcdir)/build_target.py --output $(DRAFT_RAW) --role drafter --layers 5 --no-causal $(SELECTED); \
	  $(NPU_RUN) $(srcdir)/route_artifact.py --source $(DRAFT_RAW) --output $(DRAFT_ARTIFACTS) \
	    --fix-shim-arbiter --isolate-main-output; \
	fi

run:
	$(NPU_RUN) $(srcdir)/run.py --target-artifacts "$(TARGET_ARTIFACTS)" \
	  --draft-artifacts "$(DRAFT_ARTIFACTS)" --tokens $(N_TOKENS) \
	  --prefill $(PREFILL) --prefill-context $(PREFILL_CTX) --prefill-artifacts "$(PREFILL_ARTIFACTS)" \
	  $(if $(filter 0,$(STREAM)),--no-stream,) --output "$(BUILD_DIR)/last_run.json"

chat:
	$(NPU_RUN) $(srcdir)/run.py --interactive --target-artifacts "$(TARGET_ARTIFACTS)" \
	  --draft-artifacts "$(DRAFT_ARTIFACTS)" --tokens $(N_TOKENS) \
	  --prefill $(PREFILL) --prefill-context $(PREFILL_CTX) --prefill-artifacts "$(PREFILL_ARTIFACTS)" \
	  --output "$(CHAT_LOG)"

verify-host:
	$(RUN) $(srcdir)/verify.py --host-only
	$(RUN) $(srcdir)/test_streaming.py

verify:
	$(NPU_RUN) $(srcdir)/verify.py --target-artifacts $(TARGET_ARTIFACTS) \
	  --draft-artifacts $(DRAFT_ARTIFACTS) --tokens 32 \
	  --prefill $(PREFILL) --prefill-context $(PREFILL_CTX) --prefill-artifacts "$(PREFILL_ARTIFACTS)" \
	  --output $(BUILD_DIR)/verify.json

benchmark:
	@test -n "$(INPUTS)" || (echo "Set INPUTS=/path/to/prepared_inputs" >&2; exit 2)
	$(NPU_RUN) $(srcdir)/run_benchmark.py --inputs $(INPUTS) --output $(OUTPUT) \
	  --mode $(MODE) --target $(if $(filter baseline,$(MODE)),$(BASELINE_ARTIFACTS),$(TARGET_ARTIFACTS)) \
	  $(if $(filter spec,$(MODE)),--draft $(DRAFT_ARTIFACTS),) \
	  --tokens $(N_TOKENS) --per-dataset $(PER_DATASET) $(if $(filter 1,$(RESUME)),--resume,)
	$(RUN) $(srcdir)/benchmark_summary.py "$(OUTPUT)"

benchmark-report:
	$(RUN) $(srcdir)/benchmark_summary.py "$(OUTPUT)"

clean:
	rm -rf $(BUILD_DIR)

# Reviewer study: original GEMV baseline plus the selected B8 pair.
.PHONY: compile-baseline compile-study benchmark-data benchmark-study study-report
compile-baseline:
	@if [ -f "$(BASELINE_ARTIFACTS)/manifest.json" ]; then \
	  echo "using existing $(BASELINE_ARTIFACTS)"; \
	else \
	  $(NPU_RUN) $(srcdir)/build_target.py --output "$(BASELINE_ARTIFACTS)" \
	    --role target --capacity 4096 --batch 1 --layers 36 --no-taps \
	    --causal --stack-bytes 10240 --l2-pingpong --chain-pingpong \
	    --activation-memtile-col 2 --gather-mode packet --dynamic-context \
	    --vector-attention-norm; \
	fi

compile-study: compile-target compile-draft compile-baseline

benchmark-data:
	$(RUN) $(srcdir)/benchmark_data.py --destination "$(BENCHMARK_BUNDLE)" $(if $(ARCHIVE),--archive "$(ARCHIVE)",)

benchmark-study:
	$(NPU_RUN) $(srcdir)/run_study.py --inputs "$(INPUTS)" --output "$(STUDY_DIR)" \
	  --baseline "$(BASELINE_ARTIFACTS)" --target "$(TARGET_ARTIFACTS)" --draft "$(DRAFT_ARTIFACTS)" \
	  --tokens $(STUDY_TOKENS) --per-dataset $(PER_DATASET) --label "$(STUDY_LABEL)" \
	  $(if $(filter 1,$(RESUME)),--resume,)

study-report:
	$(RUN) $(srcdir)/study_report.py "$(STUDY_DIR)" $(if $(REFERENCE),--reference "$(REFERENCE)",)

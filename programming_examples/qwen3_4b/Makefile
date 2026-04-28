# ============================================================
# Qwen3-4B BF16 Inference on NPU2 (MLIR-AIR)
# ============================================================

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

N_TOKENS ?= 100
PROMPT   ?= The capital of France is
SEQ_LEN  ?= 2048
MAX_SEQ  ?= 256
MODEL    ?= Qwen/Qwen3-4B

PHASE0     := qwen3_4b_reference.py
PHASE1     := qwen3_4b_phase1_test.py
PHASE2     := qwen3_4b_phase2_test.py
PHASE3     := qwen3_4b_phase3_test.py
PHASE4     := qwen3_4b_phase4_test.py
PHASE5     := qwen3_4b_phase5_test.py
INFERENCE  := qwen3_4b_inference.py

.PHONY: help compile run profile verify clean \
        compile-prefill compile-decode \
        run-prefill run-decode run-block run-full run-reference run-decode-only

help:
	@echo "Qwen3-4B Inference on NPU2"
	@echo "  make compile         Compile prefill + decode kernels"
	@echo "  make run             NPU prefill + NPU decode ($(N_TOKENS) tokens)"
	@echo "  make profile         Prefill perf measurement"
	@echo "  make verify          End-to-end NPU vs CPU F32 verification"
	@echo "  make run-block       Phase 2 single-block correctness"
	@echo "  make run-full        Phase 3 full-model"
	@echo "  make run-reference   Phase 0 CPU reference vs HF"
	@echo "Options: N_TOKENS=$(N_TOKENS) PROMPT='$(PROMPT)' SEQ_LEN=$(SEQ_LEN) MODEL=$(MODEL)"

compile:
	python3 $(srcdir)/$(INFERENCE) --compile-only --seq-len $(SEQ_LEN) \
		--cache-dir build/prefill_kernel_cache

compile-prefill:
	python3 $(srcdir)/$(PHASE4) --compile-only --seq-len $(SEQ_LEN)

compile-decode:
	python3 $(srcdir)/$(PHASE5) --compile-only

run:
	python3 $(srcdir)/$(INFERENCE) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) \
		--cache-dir build/prefill_kernel_cache --profile

run-decode-only:
	python3 $(srcdir)/$(PHASE5) --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--max-seq $(MAX_SEQ) --model $(MODEL) --no-cpu-verify

run-prefill:
	python3 $(srcdir)/$(PHASE4) --seq-len $(SEQ_LEN) --model $(MODEL)

run-block:
	python3 $(srcdir)/$(PHASE2) --no-preload --seq-len $(SEQ_LEN) --model $(MODEL)

run-full:
	python3 $(srcdir)/$(PHASE3) --seq-len $(SEQ_LEN) --model $(MODEL)

run-reference:
	python3 $(srcdir)/$(PHASE0) --prompt "$(PROMPT)" --model $(MODEL) --verify

profile: run-prefill

verify:
	python3 $(srcdir)/$(INFERENCE) --n-tokens 1 --prompt "$(PROMPT)" \
		--seq-len $(SEQ_LEN) --model $(MODEL) \
		--cache-dir build/prefill_kernel_cache --verify

clean:
	rm -rf build/ air_project/ __pycache__/ && rm -f air.elf air.mlir air.xclbin air.insts.bin
	@echo "Caches cleared. Run 'make compile' to rebuild."

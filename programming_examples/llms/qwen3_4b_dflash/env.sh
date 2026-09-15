#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -n ${DFLASH_BASE_ENV:-} ]]; then
  # XRT's environment script may reference unset variables.
  set +u
  source "$DFLASH_BASE_ENV"
  set -u
fi

if [[ -n ${DFLASH_AIR_INSTALL:-} ]]; then
  export MLIR_AIR_INSTALL_DIR="$DFLASH_AIR_INSTALL"
fi

for required in MLIR_AIR_INSTALL_DIR MLIR_AIE_INSTALL_DIR PEANO_INSTALL_DIR LLVM_INSTALL_DIR; do
  if [[ -z ${!required:-} ]]; then
    echo "Missing $required. Source utils/env_setup.sh or set DFLASH_BASE_ENV/DFLASH_AIR_INSTALL." >&2
    exit 2
  fi
done

export PATH="$MLIR_AIR_INSTALL_DIR/bin:$MLIR_AIE_INSTALL_DIR/bin:$LLVM_INSTALL_DIR/bin:$PATH"
export PYTHONPATH="$MLIR_AIR_INSTALL_DIR/python:$MLIR_AIE_INSTALL_DIR/python:/opt/xilinx/xrt/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$MLIR_AIR_INSTALL_DIR/lib:$MLIR_AIE_INSTALL_DIR/lib:$LLVM_INSTALL_DIR/lib:/opt/xilinx/xrt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Inline attention must be linked by LLVM <23. Point DFLASH_LLVM_LINK at one
# compatible binary; only llvm-link is shimmed so Peano clang/opt remain first.
if [[ -n ${DFLASH_LLVM_LINK:-} ]]; then
  if [[ ! -x $DFLASH_LLVM_LINK ]]; then
    echo "DFLASH_LLVM_LINK is not executable: $DFLASH_LLVM_LINK" >&2
    exit 2
  fi
  shim="$here/build/toolchain"
  mkdir -p "$shim"
  ln -sfn "$(realpath "$DFLASH_LLVM_LINK")" "$shim/llvm-link"
  export PATH="$shim:$PATH"
fi

# Prevent another example's geometry from leaking into this build/runtime.
while IFS= read -r key; do
  case "$key" in
    DECODE_*|RMS_*|PROJ_*|Q4K_*|UNI_WAVE_*|Q4NX_*) unset "$key" ;;
  esac
done < <(compgen -v)

export W_DUAL_CHAN=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

exec "$@"

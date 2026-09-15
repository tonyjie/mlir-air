# Toolchain and reproducibility

The [reviewer setup](reviewer.md) builds AIR from this branch in a separate
installation. `setup_toolchain.py` records compiler source and binary hashes in
`install/dflash-build.json`; `toolchain.lock.json` pins its dependencies.

The selected implementation uses the MLIR-AIR, MLIR-AIE, Peano, LLVM, and XRT
environment exposed through the repository's standard `utils/env_setup.sh`.
`env.sh` checks that the four install-directory variables are present and adds
their tools, Python modules, and libraries to the execution environment.

## Required compiler detail

The fused attention core is compiled to LLVM IR by Peano and then merged into
the generated core by an external `llvm-link`. Peano's IR uses the sized
`llvm.lifetime` form, so this link step requires LLVM earlier than version 23.
LLVM 23 and newer rewrite that intrinsic and Peano rejects the resulting module.

Point `DFLASH_LLVM_LINK` at a compatible executable:

```bash
export DFLASH_LLVM_LINK=/path/to/llvm-21/bin/llvm-link
```

`env.sh` creates a one-file shim under `build/toolchain` and prepends it to
`PATH`. It does not prepend the complete LLVM installation, because doing so
would shadow Peano's `clang`, `opt`, and `llc`. `make compile` runs the shared
`fused_decode` preflight and stops before routing if this requirement is not met.

The known-good local environment used for this delivery was:

```text
AIR install:  built from this branch; historical install was install_weight_prefetch
MLIR-AIE:     1.4.3.dev55+g10767b5
Peano:        22.0.0.2026090201+a36c62b9
External llvm-link: LLVM 21.0.0
LLVM/MLIR host tools: LLVM 24
XRT:          2.21 runtime family
```

The AIR source branch contains the earlier dynamic-offset and scheduling fixes
used by the selected artifacts. This is not an unmodified upstream-main build. Dependency pins match upstream
`46288593` as checked September 15, but the source base remains `6746658f`.
Model and compiler identities are written into every artifact manifest.

## Fresh-build evidence

The delivery was compiled from an empty `build/` directory. Both routed
artifacts have the exact selected configuration and ABI. Their transaction
headers are byte-identical to the preserved selected artifacts, and their
xclbin containers have the same size. Xclbin hashes differ because a fresh
container is packaged during each build; device behavior is checked separately.

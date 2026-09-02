//===- verify_dma_routing.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A memtile BD chain that no flow carries is a dispatch timeout with nothing
// written, and nothing else in the pipeline says so. This runs after
// aie-place-tiles, so the tiles here are physical.

// RUN: air-opt %s -air-verify-dma-routing -split-input-file -verify-diagnostics

// Routed by a circuit flow: no diagnostic.
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %memtile = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  %buf = aie.buffer(%memtile) {sym_name = "buf"} : memref<64xi32, 1 : i32>
  %lock0 = aie.lock(%memtile, 0) {init = 1 : i32}
  %lock1 = aie.lock(%memtile, 1) {init = 0 : i32}
  aie.flow(%shim, DMA : 0, %memtile, DMA : 0)
  aie.flow(%memtile, DMA : 0, %core, DMA : 0)
  %mtdma = aie.memtile_dma(%memtile) {
    %c1_i32 = arith.constant 1 : i32
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^next)
  ^bd0:
    aie.use_lock(%lock0, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock1, Release, %c1_i32)
    aie.next_bd ^bd0
  ^next:
    %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
  ^bd1:
    aie.use_lock(%lock1, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock0, Release, %c1_i32)
    aie.next_bd ^bd1
  ^end:
    aie.end
  }
}

// -----

// The outgoing flow removed: MM2S 0 pushes into a port nothing routes.
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %memtile = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  %buf = aie.buffer(%memtile) {sym_name = "buf"} : memref<64xi32, 1 : i32>
  %lock0 = aie.lock(%memtile, 0) {init = 1 : i32}
  %lock1 = aie.lock(%memtile, 1) {init = 0 : i32}
  aie.flow(%shim, DMA : 0, %memtile, DMA : 0)
  %mtdma = aie.memtile_dma(%memtile) {
    %c1_i32 = arith.constant 1 : i32
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^next)
  ^bd0:
    aie.use_lock(%lock0, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock1, Release, %c1_i32)
    aie.next_bd ^bd0
  ^next:
    // expected-error@+1 {{no flow carries MM2S channel 0 of memtile (0, 1)}}
    %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
  ^bd1:
    aie.use_lock(%lock1, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock0, Release, %c1_i32)
    aie.next_bd ^bd1
  ^end:
    aie.end
  }
}

// -----

// A packet_flow carries it just as a circuit flow does.
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %memtile = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  %buf = aie.buffer(%memtile) {sym_name = "buf"} : memref<64xi32, 1 : i32>
  %lock0 = aie.lock(%memtile, 0) {init = 1 : i32}
  %lock1 = aie.lock(%memtile, 1) {init = 0 : i32}
  aie.flow(%shim, DMA : 0, %memtile, DMA : 0)
  aie.packet_flow(8) {
    aie.packet_source<%memtile, DMA : 0>
    aie.packet_dest<%core, DMA : 0>
  }
  %mtdma = aie.memtile_dma(%memtile) {
    %c1_i32 = arith.constant 1 : i32
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^next)
  ^bd0:
    aie.use_lock(%lock0, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock1, Release, %c1_i32)
    aie.next_bd ^bd0
  ^next:
    %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
  ^bd1:
    aie.use_lock(%lock1, AcquireGreaterEqual, %c1_i32)
    aie.dma_bd(%buf : memref<64xi32, 1 : i32> offset = 0 len = 64)
    aie.use_lock(%lock0, Release, %c1_i32)
    aie.next_bd ^bd1
  ^end:
    aie.end
  }
}

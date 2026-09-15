// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Replay and ordinary packet outputs share one memory tile, but independent
// streams need independent DMA reader rings. Exercise both allocation orders:
// replay before ordinary, then another replay after ordinary. Same-flow
// collapse is covered separately by memtile_packet_same_flow_collapse.mlir.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[R0:[a-z0-9]+]], ^{{[a-z0-9]+}})
// CHECK: ^[[R0]]:
// CHECK: aie.dma_bd
// CHECK: aie.next_bd ^[[R0]]
// CHECK: aie.dma_start(MM2S, 1, ^[[R1:[a-z0-9]+]], ^{{[a-z0-9]+}})
// CHECK: ^[[R1]]:
// CHECK: aie.dma_bd
// CHECK: aie.next_bd ^[[R1]]
// CHECK: aie.dma_start(MM2S, 2, ^[[R2:[a-z0-9]+]], ^{{[a-z0-9]+}})
// CHECK: ^[[R2]]:
// CHECK: aie.dma_bd
// CHECK: aie.next_bd ^[[R2]]
// CHECK-NOT: aie.dma_start(MM2S, 3

air.channel @w0 [1, 1]
air.channel @r0 [1, 1] {channel_type = "npu_dma_packet"}
air.channel @w1 [1, 1]
air.channel @r1 [1, 1] {channel_type = "npu_dma_packet"}
air.channel @w2 [1, 1]
air.channel @r2 [1, 1] {channel_type = "npu_dma_packet"}
func.func @memtile_refeed_packet_isolation() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %one = arith.constant 1 : index
      %t0, %l2_0 = air.execute -> (memref<32xbf16, 1>) {
        %buf = memref.alloc() {air.memtile_col = 2 : i32, air.no_split, air.refeed_count = 2 : i32} : memref<32xbf16, 1>
        air.execute_terminator %buf : memref<32xbf16, 1>
      }
      air.channel.get @w0[] (%l2_0[] [] []) : (memref<32xbf16, 1>)
      air.channel.put @r0[] (%l2_0[] [] []) : (memref<32xbf16, 1>)
      %t1, %l2_1 = air.execute -> (memref<32xbf16, 1>) {
        %buf = memref.alloc() {air.memtile_col = 2 : i32, air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %buf : memref<32xbf16, 1>
      }
      air.channel.get @w1[] (%l2_1[] [] []) : (memref<32xbf16, 1>)
      air.channel.put @r1[] (%l2_1[] [] []) : (memref<32xbf16, 1>)
      %t2, %l2_2 = air.execute -> (memref<32xbf16, 1>) {
        %buf = memref.alloc() {air.memtile_col = 2 : i32, air.no_split, air.refeed_count = 3 : i32} : memref<32xbf16, 1>
        air.execute_terminator %buf : memref<32xbf16, 1>
      }
      air.channel.get @w2[] (%l2_2[] [] []) : (memref<32xbf16, 1>)
      air.channel.put @r2[] (%l2_2[] [] []) : (memref<32xbf16, 1>)
      air.herd @producer tile (%x, %y) in (%sx=%one, %sy=%one)
          attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %prod0, %l1_0 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.put @w0[] (%l1_0[] [] []) : (memref<32xbf16, 2>)
        %prod1, %l1_1 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.put @w1[] (%l1_1[] [] []) : (memref<32xbf16, 2>)
        %prod2, %l1_2 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.put @w2[] (%l1_2[] [] []) : (memref<32xbf16, 2>)
      }
      air.herd @reader0 tile (%x, %y) in (%sx=%one, %sy=%one)
          attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %t, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<32xbf16, 2>)
      }
      air.herd @reader1 tile (%x, %y) in (%sx=%one, %sy=%one)
          attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %t, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<32xbf16, 2>)
      }
      air.herd @reader2 tile (%x, %y) in (%sx=%one, %sy=%one)
          attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %t, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %buf = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %buf : memref<32xbf16, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<32xbf16, 2>)
      }
    }
  }
  return
}

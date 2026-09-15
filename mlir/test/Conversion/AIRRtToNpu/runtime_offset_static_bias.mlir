// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Stride legalization splits a two-group KV append into two linear BDs. The
// second group's static displacement must be added to the runtime token base.
// Both pieces must also retain their completion waits.
// CHECK-LABEL: aie.runtime_sequence @split_append
// CHECK-SAME: %{{.*}}: memref<268435456xbf16>, %[[OFF:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[BIAS:.*]] = arith.constant 2097152 : i64
// CHECK-DAG: %[[O0:.*]] = arith.trunci %[[OFF]] : i64 to i32
// CHECK-DAG: %[[SUM:.*]] = arith.addi %[[OFF]], %[[BIAS]] : i64
// CHECK-DAG: %[[O1:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @append
// CHECK: aie.dma_bd(%{{.*}} offset = %[[O0]] len = 512 sizes = [512] strides = [1])
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @append
// CHECK: aie.dma_bd(%{{.*}} offset = %[[O1]] len = 512 sizes = [512] strides = [1])
// CHECK-DAG: aiex.dma_await_task(%[[T0]])
// CHECK-DAG: aiex.dma_await_task(%[[T1]])
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @append(%shim, S2MM, 0)
  } {sym_name = "segment0"}
  func.func @split_append(%arg0: memref<268435456xbf16>, %off: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c512 = arith.constant 512 : i64
    %stride = arith.constant 2097152 : i64
    %id = arith.constant 4 : i32
    %p = airrt.segment_load "segment0" : i64
    %event = airrt.dma_memcpy_nd(%id, %c0, %c0, %arg0[%c0, %c0, %c0, %off], [%c1, %c1, %c2, %c512], [%c0, %c0, %stride, %c1]) {metadata = @append} : (i32, i64, i64, memref<268435456xbf16>) : !airrt.event
    airrt.wait_all %event
    return
  }
}

// -----

// A legal strided transfer and a runtime-length transfer need the same bias.
// Scale the static dimension by its stride before combining the two offsets.
// CHECK-LABEL: aie.runtime_sequence @mixed_offsets
// CHECK-SAME: %{{.*}}: memref<1048576xbf16>, %[[OFF:[a-zA-Z0-9_]+]]: i64, %[[N:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[BIAS:.*]] = arith.constant 2048 : i64
// CHECK-DAG: %[[SUM:.*]] = arith.addi %[[OFF]], %[[BIAS]] : i64
// CHECK: %[[O:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK: aie.dma_bd(%{{.*}} offset = %[[O]] len = 1024 sizes = [2, 512] strides = [1024, 1])
// CHECK: aie.dma_bd(%{{.*}} offset = %[[O]] len = %{{.*}})
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @read(%shim, MM2S, 0)
  } {sym_name = "segment1"}
  func.func @mixed_offsets(%arg0: memref<1048576xbf16>, %off: i64, %n: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %id = arith.constant 2 : i32
    %p = airrt.segment_load "segment1" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %arg0[%c0, %c0, %c2, %off], [%c1, %c1, %c2, %c512], [%c0, %c0, %c1024, %c1]) {metadata = @read} : (i32, i64, i64, memref<1048576xbf16>)
    airrt.dma_memcpy_nd(%id, %c0, %c0, %arg0[%c0, %c0, %c2, %off], [%c1, %c1, %c1, %n], [%c0, %c0, %c1024, %c1]) {metadata = @read} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

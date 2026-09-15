// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s
// A branch yields a scalar channel index and a completion token. Tracing the
// token must preserve its branch producers without treating scalar constants
// as asynchronous graph nodes or selecting the branch's first result.
// CHECK-LABEL: func.func @index_switch_mixed_results
// CHECK: %[[SW:[a-zA-Z0-9_]+]]:2 = scf.index_switch
// CHECK: air.channel.get async [{{.*}}%[[SW]]#1{{.*}}]

module {
  air.channel @chan_in [2]
  air.channel @chan_out_0 [1, 1, 1]
  air.channel @chan_out_1 [1, 1, 1]
  air.channel @chan_out_2 [1, 1, 1]
  func.func @index_switch_mixed_results(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg0, %arg6=%arg1, %arg7=%arg2) : memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16> {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      // Matching channel.put for chan_in (feed data into segment)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      // Matching channel.get for each arm's output
      air.channel.get  @chan_out_0[%c0, %c0, %c0] (%arg6[] [] []) : (memref<64x64xbf16>)
      air.channel.get  @chan_out_1[%c0, %c0, %c0] (%arg7[] [] []) : (memref<64x64xbf16>)
      air.channel.get  @chan_out_2[%c0, %c0, %c0] (%arg7[] [] []) : (memref<64x64xbf16>)
      %1 = air.segment @seg async  unroll(%arg8) in (%arg9=%c2) {
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2_2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c512 = arith.constant 512 : index
        %c64_3 = arith.constant 64 : index
        %2 = scf.for %iv = %c0_1 to %c2_2 step %c1_1 iter_args(%dep = %async_token) -> (!air.async.token) {
          %3 = air.channel.get async [%dep]  @chan_in[%arg8] (%results[] [] []) {id = 2 : i32} : (memref<64x64xbf16, 1 : i32>)
          %4:2 = scf.index_switch %arg8 -> index, !air.async.token
          case 0 {
            %7 = air.channel.put async [%3]  @chan_out_0[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 3 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %c0_1, %7 : index, !air.async.token
          }
          case 1 {
            %7 = air.channel.put async [%3]  @chan_out_1[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 4 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %c0_1, %7 : index, !air.async.token
          }
          default {
            %7 = air.channel.put async [%3]  @chan_out_2[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 5 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %c0_1, %7 : index, !air.async.token
          }
          %5 = air.channel.get async [%dep, %4#1]  @chan_in[%4#0] (%results[] [] []) {id = 6 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %5 : !air.async.token
        }
        %async_token_4 = air.execute [%2] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 7 : i32}
      }
    }
    return
  }
}

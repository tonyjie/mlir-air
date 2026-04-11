//===- mv_q4.cc - Q4 GEMV kernel for AIE2P ----------------------*- C++ -*-===//
//
// Matrix-vector multiplication with on-the-fly Q4_1 dequantization.
// Single interleaved weight buffer: each block is [q4_packed | scale | min].
//
// C[M] = dequant(A_q4[M,K]) @ B[K]
//
// Interleaved block layout (per block of 32 values):
//   [16 bytes packed Q4] [2 bytes scale (bf16)] [2 bytes min (bf16)]
//   = 20 bytes per block
//
// Row size = K/32 blocks * 20 bytes = K * 20/32 = K * 5/8 bytes
// (vs K*2 bytes for BF16 = 3.2x reduction)
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

#define Q4_BLOCK_SIZE 32
#define Q4_PACKED_PER_BLOCK 16 // 32 values / 2 per byte
#define Q4_BLOCK_BYTES 20      // 16 packed + 2 scale + 2 min

// Q4 matrix-vector multiplication with inline dequantization.
//
// a_q4:  interleaved Q4 weight data, row-major
//        each row = K/32 blocks, each block = 20 bytes
//        [packed(16B) | scale(2B bf16) | min(2B bf16)] × (K/32)
//        total row bytes = K * 20 / 32 = K * 5 / 8
// b:     input vector, [k] bfloat16
// c:     output vector, [m] bfloat16
void q4_matvec(uint32_t m, uint32_t k, const uint8_t *__restrict a_q4,
               const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  const int blocks_per_row = k / Q4_BLOCK_SIZE;
  const int row_bytes = blocks_per_row * Q4_BLOCK_BYTES;

  for (uint32_t row = 0; row < m; row++) {
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();

    const uint8_t *row_data = a_q4 + row * row_bytes;

    for (int blk = 0; blk < blocks_per_row; blk++) {
      const uint8_t *blk_ptr = row_data + blk * Q4_BLOCK_BYTES;

      // Extract scale and min from the end of the block
      const bfloat16 *meta =
          reinterpret_cast<const bfloat16 *>(blk_ptr + Q4_PACKED_PER_BLOCK);
      float s = static_cast<float>(meta[0]);  // scale
      float mn = static_cast<float>(meta[1]); // min

      // Dequantize 32 values from 16 packed bytes
      aie::vector<bfloat16, 32> a_vec;
      for (int i = 0; i < Q4_PACKED_PER_BLOCK; i++) {
        uint8_t byte = blk_ptr[i];
        uint8_t lo = byte & 0x0F;
        uint8_t hi = (byte >> 4) & 0x0F;
        a_vec[2 * i] = static_cast<bfloat16>(mn + lo * s);
        a_vec[2 * i + 1] = static_cast<bfloat16>(mn + hi * s);
      }

      // MAC with input vector
      const bfloat16 *blk_b = b + blk * Q4_BLOCK_SIZE;
      aie::vector<bfloat16, 32> b_vec = aie::load_v<32>(blk_b);
      acc = aie::mac(acc, a_vec, b_vec);
    }

    c[row] =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
}

extern "C" {

#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 2048
#endif

// Single interleaved weight buffer version.
// a_q4: [m × (k*5/8)] uint8, interleaved blocks
void q4_matvec_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                    const uint8_t *__restrict a_q4,
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out) {
  c_out += row_offset;
  q4_matvec(m, k, a_q4, b_in, c_out);
}

void q4_linalg_fill_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M_OUTPUT, 1>(c_out);
}

} // extern "C"

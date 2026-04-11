//===- mv_q4.cc - Q4 GEMV kernel for AIE2P ----------------------*- C++ -*-===//
//
// Matrix-vector multiplication with on-the-fly Q4_1 dequantization.
// Reads Q4-packed weights (2 values per byte) + per-block BF16 scale/min,
// dequantizes to BF16 in the inner loop, then computes dot product.
//
// C[M] = dequant(A_q4[M,K]) @ B[K]
//
// Q4_1 format (per block of BLOCK_SIZE=32 values):
//   dequant(q) = min + q * scale,  q in [0, 15]
//   packed as uint8 (2 values per byte, low nibble first)
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

// Dequantize a block of 32 Q4 values from 16 packed bytes.
// Low nibble = first value, high nibble = second value.
// dequant(q) = min_val + q * scale
template <int N>
inline aie::vector<bfloat16, N>
dequant_q4_block(const uint8_t *__restrict packed, bfloat16 scale,
                 bfloat16 min_val, int n) {
  aie::vector<bfloat16, N> result;
  float s = static_cast<float>(scale);
  float m = static_cast<float>(min_val);

  for (int i = 0; i < n / 2; i++) {
    uint8_t byte = packed[i];
    uint8_t lo = byte & 0x0F;
    uint8_t hi = (byte >> 4) & 0x0F;
    result[2 * i] = static_cast<bfloat16>(m + lo * s);
    result[2 * i + 1] = static_cast<bfloat16>(m + hi * s);
  }
  return result;
}

// Q4 matrix-vector multiplication with inline dequantization.
//
// a_q4:   packed Q4 weight data, row-major, [m × k/2] bytes
// scales: per-block scale, [m × k/BLOCK_SIZE] bfloat16
// mins:   per-block min,   [m × k/BLOCK_SIZE] bfloat16
// b:      input vector, [k] bfloat16
// c:      output vector, [m] bfloat16
void q4_matvec(uint32_t m, uint32_t k, const uint8_t *__restrict a_q4,
               const bfloat16 *__restrict scales,
               const bfloat16 *__restrict mins, const bfloat16 *__restrict b,
               bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  const int blocks_per_row = k / Q4_BLOCK_SIZE;
  const int packed_bytes_per_row = k / 2;

  for (uint32_t row = 0; row < m; row++) {
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();

    const uint8_t *row_q4 = a_q4 + row * packed_bytes_per_row;
    const bfloat16 *row_scales = scales + row * blocks_per_row;
    const bfloat16 *row_mins = mins + row * blocks_per_row;

    for (int blk = 0; blk < blocks_per_row; blk++) {
      // Dequantize one block of 32 values
      bfloat16 s = row_scales[blk];
      bfloat16 mn = row_mins[blk];
      const uint8_t *blk_data = row_q4 + blk * (Q4_BLOCK_SIZE / 2);
      const bfloat16 *blk_b = b + blk * Q4_BLOCK_SIZE;

      aie::vector<bfloat16, 32> a_vec =
          dequant_q4_block<32>(blk_data, s, mn, Q4_BLOCK_SIZE);
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

void q4_matvec_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                    const uint8_t *__restrict a_q4,
                    const bfloat16 *__restrict scales,
                    const bfloat16 *__restrict mins,
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out) {
  c_out += row_offset;
  q4_matvec(m, k, a_q4, scales, mins, b_in, c_out);
}

void q4_linalg_fill_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M_OUTPUT, 1>(c_out);
}

} // extern "C"

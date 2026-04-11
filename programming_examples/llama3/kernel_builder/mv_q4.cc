//===- mv_q4.cc - Q4 GEMV kernel for AIE2P ----------------------*- C++ -*-===//
//
// Q4 GEMV with reformulated math. Interleaved packed format:
// Block = [16B packed Q4 | 2B scale | 2B min] = 20 bytes per 32 values.
//
//   sum(dequant(q) * b) = min * sum(b) + scale * dot(q, b)
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
#define Q4_PACKED_PER_BLOCK 16
#define Q4_BLOCK_BYTES 20

static const bfloat16 Q4_LUT[16] = {
    static_cast<bfloat16>(0.0f),  static_cast<bfloat16>(1.0f),
    static_cast<bfloat16>(2.0f),  static_cast<bfloat16>(3.0f),
    static_cast<bfloat16>(4.0f),  static_cast<bfloat16>(5.0f),
    static_cast<bfloat16>(6.0f),  static_cast<bfloat16>(7.0f),
    static_cast<bfloat16>(8.0f),  static_cast<bfloat16>(9.0f),
    static_cast<bfloat16>(10.0f), static_cast<bfloat16>(11.0f),
    static_cast<bfloat16>(12.0f), static_cast<bfloat16>(13.0f),
    static_cast<bfloat16>(14.0f), static_cast<bfloat16>(15.0f),
};

void q4_matvec(uint32_t m, uint32_t k,
               const uint8_t *__restrict a_q4,
               const bfloat16 *__restrict b,
               bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const int blocks_per_row = k / Q4_BLOCK_SIZE;
  const int row_bytes = blocks_per_row * Q4_BLOCK_BYTES;
  alignas(64) bfloat16 q_buf[Q4_BLOCK_SIZE];
  aie::vector<bfloat16, 16> ones =
      aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f));

  for (uint32_t row = 0; row < m; row++) {
    float min_sum = 0.0f;
    float scale_dot = 0.0f;
    const uint8_t *row_data = a_q4 + row * row_bytes;

    for (int blk = 0; blk < blocks_per_row; blk++) {
      const uint8_t *blk_ptr = row_data + blk * Q4_BLOCK_BYTES;
      const bfloat16 *meta = reinterpret_cast<const bfloat16 *>(
          blk_ptr + Q4_PACKED_PER_BLOCK);
      float scale = static_cast<float>(meta[0]);
      float min_val = static_cast<float>(meta[1]);

      for (int i = 0; i < Q4_PACKED_PER_BLOCK; i++) {
        uint8_t byte = blk_ptr[i];
        q_buf[2*i] = Q4_LUT[byte & 0x0F];
        q_buf[2*i+1] = Q4_LUT[(byte >> 4) & 0x0F];
      }

      const bfloat16 *blk_b = b + blk * Q4_BLOCK_SIZE;
      auto q_lo = aie::load_v<16>(q_buf);
      auto q_hi = aie::load_v<16>(q_buf + 16);
      auto b_lo = aie::load_v<16>(blk_b);
      auto b_hi = aie::load_v<16>(blk_b + 16);

      aie::accum<accfloat, 16> blk_dot = aie::mul(q_lo, b_lo);
      blk_dot = aie::mac(blk_dot, q_hi, b_hi);
      float dot_qb = aie::reduce_add(blk_dot.template to_vector<float>());

      aie::accum<accfloat, 16> blk_sum = aie::mul(b_lo, ones);
      blk_sum = aie::mac(blk_sum, b_hi, ones);
      float sum_b = aie::reduce_add(blk_sum.template to_vector<float>());

      min_sum += min_val * sum_b;
      scale_dot += scale * dot_qb;
    }
    c[row] = static_cast<bfloat16>(min_sum + scale_dot);
  }
}

extern "C" {
#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 2048
#endif
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

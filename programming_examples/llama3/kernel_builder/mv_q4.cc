//===- mv_q4.cc - Q4 GEMV kernel for AIE2P ----------------------*- C++ -*-===//
//
// Matrix-vector multiplication with on-the-fly Q4_1 dequantization.
// Single interleaved weight buffer: each block is [q4_packed | scale | min].
//
// C[M] = dequant(A_q4[M,K]) @ B[K]
//
// Uses per-block 16-entry LUT for dequant: lut[i] = min + i * scale.
// Then dequant is 2 LUT lookups per byte (no float arithmetic per element).
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

void q4_matvec(uint32_t m, uint32_t k, const uint8_t *__restrict a_q4,
               const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  const int blocks_per_row = k / Q4_BLOCK_SIZE;
  const int row_bytes = blocks_per_row * Q4_BLOCK_BYTES;
  alignas(64) bfloat16 dequant_buf[Q4_BLOCK_SIZE];

  for (uint32_t row = 0; row < m; row++) {
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
    const uint8_t *row_data = a_q4 + row * row_bytes;

    for (int blk = 0; blk < blocks_per_row; blk++) {
      const uint8_t *blk_ptr = row_data + blk * Q4_BLOCK_BYTES;

      const bfloat16 *meta =
          reinterpret_cast<const bfloat16 *>(blk_ptr + Q4_PACKED_PER_BLOCK);
      float s = static_cast<float>(meta[0]);
      float mn = static_cast<float>(meta[1]);

      // Precompute 16-entry LUT (16 float→bf16 conversions per block)
      bfloat16 lut[16];
      for (int i = 0; i < 16; i++) {
        lut[i] = static_cast<bfloat16>(mn + i * s);
      }

      // Dequant via LUT (just table lookups, no float math)
      for (int i = 0; i < Q4_PACKED_PER_BLOCK; i++) {
        uint8_t byte = blk_ptr[i];
        dequant_buf[2 * i] = lut[byte & 0x0F];
        dequant_buf[2 * i + 1] = lut[(byte >> 4) & 0x0F];
      }

      // Vector MAC
      aie::vector<bfloat16, 32> a_vec = aie::load_v<32>(dequant_buf);
      aie::vector<bfloat16, 32> b_vec =
          aie::load_v<32>(b + blk * Q4_BLOCK_SIZE);
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
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out) {
  c_out += row_offset;
  q4_matvec(m, k, a_q4, b_in, c_out);
}

void q4_linalg_fill_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M_OUTPUT, 1>(c_out);
}

} // extern "C"

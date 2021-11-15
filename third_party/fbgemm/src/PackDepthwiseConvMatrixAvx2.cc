/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"

#include <immintrin.h>

#include "./MaskAvx2.h"
#include "fbgemm/UtilsAvx2.h"


namespace fbgemm {

PackedDepthWiseConvMatrix::PackedDepthWiseConvMatrix(
    int OC,
    int kernel_prod,
    const int8_t* smat)
    : OC_(OC), kernel_prod_(kernel_prod) {
  // The input is in OC T R S layout.
  // Transpose the input matrix to make packing faster.
  int8_t* smat_transposed = static_cast<int8_t*>(
      fbgemmAlignedAlloc(64, OC * kernel_prod * sizeof(int8_t)));
  for (int i = 0; i < kernel_prod; ++i) {
    for (int j = 0; j < OC; ++j) {
      smat_transposed[i * OC + j] = smat[i + j * kernel_prod];
    }
  }

  // Allocate packed arrays
  int kernel_prod_aligned = (kernel_prod + 1) / 2 * 2;
  pmat_ = static_cast<int8_t*>(fbgemmAlignedAlloc(
      64, ((OC + 31) / 32) * kernel_prod_aligned * 32 * sizeof(int8_t)));

  // Pack input matrix
  // The layout is optimized to use vpmaddubsw efficiently (see
  // genMaddEpi16xNPacked function).
  // For a group of 32 channels, we have 10 32B SIMD registers.
  // Denote ith channel jth filter as (i, j)
  // 0th SIMD register:
  // (0, 0), (0, 1), (0, 2), (0, 3), ..., (3, 0), (3, 1), (3, 2), (3, 3)
  // (16, 0), (16, 1), (16, 2), (16, 3), ..., (19, 0), (19, 1), (19, 2), (19, 3)
  // 1st SIMD register:
  // (4, 0), (4, 1), (4, 2), (4, 3), ..., (7, 0), (7, 1), (7, 2), (7, 3)
  // (20, 0), (20, 1), (20, 2), (20, 3), ..., (23, 0), (23, 1), (23, 2), (23, 3)
  // 2nd SIMD register:
  // (8, 0), (8, 1), (8, 2), (8, 3), ..., (11, 0), (11, 1), (11, 2), (11, 3)
  // (24, 0), (24, 1), (24, 2), (24, 3), ..., (27, 0), (27, 1), (27, 2), (27, 3)
  // 3rd SIMD register:
  // (12, 0), (12, 1), (12, 2), (12, 3), ..., (15, 0), (15, 1), (15, 2), (15, 3)
  // (28, 0), (28, 1), (28, 2), (28, 3), ..., (31, 0), (31, 1), (31, 2), (31, 3)
  // 4-7th SIMD register: same as the previous 4 registers but for 4-7th filter
  // coefficients
  // ...
  //
  // REMAINDER
  // If kernel_prod % 4 == 1 for example when kernel_prod == 9
  // 8th SIMD register:
  // (0, 8), zero, ..., (7, 8), zero
  // (16, 8), zero, ..., (23, 8), zero
  // 9th SIMD register:
  // (8, 8), zero, ..., (15, 8), zero
  // (24, 8), zero, ..., (31, 8), zero
  // We use madd_epi16_packed for this case
  //
  // If kernel_prod % 4 == 2 for example when kernel_prod == 10
  // 8th SIMD register:
  // (0, 8), (0, 9), ..., (7, 8), (7, 9)
  // (16, 8), (16, 9), ..., (23, 8), (23, 9)
  // 9th SIMD register:
  // (8, 8), (8, 9), ..., (15, 8), (15, 9)
  // (24, 8), (24, 9), ..., (31, 8), (31, 9)
  //
  // If kernel_prod % 4 == 3 for example when kernel_prod == 11
  // 8th SIMD register:
  // (0, 8), (0, 9), (0, 10), zero, ..., (3, 8), (3, 9), (3, 10), zero
  // (16, 8), (16, 9), (16, 10), zero, ..., (19, 8), (19, 9), (19, 10), zero
  // 9th SIMD register:
  // (4, 8), (4, 9), (4, 10), zero, ..., (7, 8), (7, 9), (7, 10), zero
  // (20, 8), (20, 9), (20, 10), zero, ..., (23, 8), (23, 9), (23, 10), zero
  // 10th SIMD register:
  // (8, 8), (8, 9), (8, 10), zero, ..., (11, 8), (11, 9), (11, 10), zero
  // (24, 8), (24, 9), (24, 10), zero, ..., (27, 8), (27, 9), (27, 10), zero
  // 11th SIMD register:
  // (12, 8), (12, 9), (12, 10), zero, ..., (15, 8), (15, 9), (15, 10), zero
  // (28, 8), (28, 9), (28, 10), zero, ..., (31, 8), (31, 9), (31, 10), zero

  // Allocate buffers
  auto b_v = static_cast<__m256i*>(
      fbgemmAlignedAlloc(64, kernel_prod * sizeof(__m256i)));
  auto b_interleaved_epi16 = static_cast<__m256i*>(
      fbgemmAlignedAlloc(64, kernel_prod_aligned * sizeof(__m256i)));
  auto b_interleaved_epi32 = static_cast<__m256i*>(
      fbgemmAlignedAlloc(64, kernel_prod_aligned * sizeof(__m256i)));
  for (int k1 = 0; k1 < OC; k1 += 32) {
    int remainder = OC - k1;
    if (remainder < 32) {
      __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          internal::avx2_ps_or_epi32_masks[remainder / 4]));
      for (int i = 0; i < kernel_prod; ++i) {
        b_v[i] = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(smat_transposed + i * OC + k1),
            mask_v);
      }
    } else {
      for (int i = 0; i < kernel_prod; ++i) {
        b_v[i] = _mm256_lddqu_si256(
            reinterpret_cast<const __m256i*>(smat_transposed + i * OC + k1));
      }
    }

    // Interleave 2 SIMD registers
    __m256i zero_v = _mm256_setzero_si256();
    for (int i = 0; i < kernel_prod_aligned / 2; ++i) {
      if (2 * i + 1 >= kernel_prod) {
        b_interleaved_epi16[2 * i] = _mm256_unpacklo_epi8(b_v[2 * i], zero_v);
        b_interleaved_epi16[2 * i + 1] =
            _mm256_unpackhi_epi8(b_v[2 * i], zero_v);
      } else {
        b_interleaved_epi16[2 * i] =
            _mm256_unpacklo_epi8(b_v[2 * i], b_v[2 * i + 1]);
        b_interleaved_epi16[2 * i + 1] =
            _mm256_unpackhi_epi8(b_v[2 * i], b_v[2 * i + 1]);
      }
    }

    // Interleave 4 SIMD registers
    for (int i = 0; i < kernel_prod_aligned / 4; ++i) {
      b_interleaved_epi32[4 * i] = _mm256_unpacklo_epi16(
          b_interleaved_epi16[4 * i], b_interleaved_epi16[4 * i + 2]);
      b_interleaved_epi32[4 * i + 1] = _mm256_unpackhi_epi16(
          b_interleaved_epi16[4 * i], b_interleaved_epi16[4 * i + 2]);
      b_interleaved_epi32[4 * i + 2] = _mm256_unpacklo_epi16(
          b_interleaved_epi16[4 * i + 1], b_interleaved_epi16[4 * i + 3]);
      b_interleaved_epi32[4 * i + 3] = _mm256_unpackhi_epi16(
          b_interleaved_epi16[4 * i + 1], b_interleaved_epi16[4 * i + 3]);
    }
    for (int i = kernel_prod_aligned / 4 * 4; i < kernel_prod_aligned; ++i) {
      b_interleaved_epi32[i] = b_interleaved_epi16[i];
    }

    for (int i = 0; i < kernel_prod_aligned; ++i) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(
              &pmat_[((k1 / 32) * kernel_prod_aligned + i) * 32]),
          b_interleaved_epi32[i]);
    }
  }
  fbgemmAlignedFree(b_v);
  fbgemmAlignedFree(b_interleaved_epi16);
  fbgemmAlignedFree(b_interleaved_epi32);
  fbgemmAlignedFree(smat_transposed);
}

int PackedDepthWiseConvMatrix::addr(int r, int c) {
  int kernel_prod_aligned = (kernel_prod_ + 1) / 2 * 2;
  if (c >= kernel_prod_ / 4 * 4 &&
      (kernel_prod_ % 4 == 1 || kernel_prod_ % 4 == 2)) {
    int kBlock = r / 32;
    int reg_idx = (r % 16) / 8 + c / 4 * 4;

    int blk_idx = kBlock * kernel_prod_aligned + reg_idx;

    int r_ = r % 8;
    int c_ = c % 4;

    int in_blk_idx = (r % 32) / 16 * 16 + 2 * r_ + c_;
    return blk_idx * 32 + in_blk_idx;

  } else {
    int kBlock = r / 32;
    int reg_idx = (r % 16) / 4 + c / 4 * 4;

    int blk_idx = kBlock * kernel_prod_aligned + reg_idx;

    int r_ = r % 4;
    int c_ = c % 4;

    int in_blk_idx = (r % 32) / 16 * 16 + 4 * r_ + c_;
    return blk_idx * 32 + in_blk_idx;
  }
}

void PackedDepthWiseConvMatrix::unpack(int8_t* unpacked_data) {
  for (int r = 0; r < OC_; ++r) {
    for (int c = 0; c < kernel_prod_; ++c) {
      unpacked_data[r * kernel_prod_ + c] = pmat_[addr(r, c)];
    }
  }
}

PackedDepthWiseConvMatrix::~PackedDepthWiseConvMatrix() {
  fbgemmAlignedFree(pmat_);
}

} // namespace fbgemm

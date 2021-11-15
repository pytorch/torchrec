/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef _MSC_VER
#include <immintrin.h>
#include "./FbgemmFP16UKernelsAvx512_256.h"

namespace fbgemm {

// Intrinsic kernel for MSVC
void gemmkernel_Avx512_256_fp16_fA0fB0fC0(
    GemmParamsFP16* gp,
    const size_t kernel_nrows) {
  // register buffer
  __m256 ymmSum[28];
  size_t idxA = 0, idxB = 0, idxC = 0;
  // ldc in float size
  size_t ldc_floatsize = gp->ldc / sizeof(float);
  // load beta
  __m256 ymmBeta;
  if (gp->beta != 0)
    ymmBeta = _mm256_broadcast_ss(&gp->beta);

  // outer loop - block columns
  for (uint64_t ii = 0; ii < gp->b_block_cols; ii++) {
    // reset index
    idxA = 0;
    // inner loop - k
    for (uint64_t kk = 0; kk < gp->k; kk++) {
      // load B
      __m256 ymmB0 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)(gp->B + idxB)));
      __m256 ymmB1 =
          _mm256_cvtph_ps(_mm_load_si128((__m128i*)(gp->B + idxB + 8)));
      idxB += 16;

      // first element
      if (kk == 0) {
        if (gp->beta != 0) { // accumulate
          for (size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m256 ymmA = _mm256_broadcastss_ps(
                _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B + beta * C
            ymmSum[2 * jj] = _mm256_fmadd_ps(
                ymmA,
                ymmB0,
                _mm256_mul_ps(
                    ymmBeta,
                    _mm256_loadu_ps(gp->C + idxC + jj * ldc_floatsize)));
            ymmSum[2 * jj + 1] = _mm256_fmadd_ps(
                ymmA,
                ymmB1,
                _mm256_mul_ps(
                    ymmBeta,
                    _mm256_loadu_ps(gp->C + idxC + 8 + jj * ldc_floatsize)));
          }
          idxA += kernel_nrows;
        } else { // set zero
          for (size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m256 ymmA = _mm256_broadcastss_ps(
                _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B
            ymmSum[2 * jj] = _mm256_mul_ps(ymmA, ymmB0);
            ymmSum[2 * jj + 1] = _mm256_mul_ps(ymmA, ymmB1);
          }
          idxA += kernel_nrows;
        }
      } else {
        for (size_t jj = 0; jj < kernel_nrows; jj++) {
          // load A
          __m256 ymmA = _mm256_broadcastss_ps(
              _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
          // C = A * B + C
          ymmSum[2 * jj] = _mm256_fmadd_ps(ymmA, ymmB0, ymmSum[2 * jj]);
          ymmSum[2 * jj + 1] = _mm256_fmadd_ps(ymmA, ymmB1, ymmSum[2 * jj + 1]);
        }
        idxA += kernel_nrows;
      }
    }
    // store C
    for (size_t jj = 0; jj < kernel_nrows; jj++) {
      _mm256_storeu_ps(gp->C + idxC + jj * ldc_floatsize, ymmSum[2 * jj]);
      _mm256_storeu_ps(
          gp->C + idxC + 8 + jj * ldc_floatsize, ymmSum[2 * jj + 1]);
    }
    idxC += 16;
  }
}

void NOINLINE gemmkernel_7x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 7);
}
void NOINLINE gemmkernel_8x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 8);
}
void NOINLINE gemmkernel_9x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 9);
}
void NOINLINE gemmkernel_10x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 10);
}
void NOINLINE gemmkernel_11x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 11);
}
void NOINLINE gemmkernel_12x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 12);
}
void NOINLINE gemmkernel_13x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 13);
}
void NOINLINE gemmkernel_14x2_Avx512_256_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_256_fp16_fA0fB0fC0(gp, 14);
}

} // namespace fbgemm
#endif // _MSC_VER

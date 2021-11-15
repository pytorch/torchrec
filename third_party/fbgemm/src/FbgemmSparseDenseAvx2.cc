/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"

#include <immintrin.h>
#include "./MaskAvx2.h"

namespace fbgemm {
namespace internal {

void SparseDenseMMAvx2(
    int M,
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    bool accum) {
  // Calcualtes accum ? C += A * B : C = A * B
  // size of values is equal to number of non-zeros (nnzs)
  // size of row_ptr is equal to M + 1
  // size of col_idx is equal to nnzs

  constexpr int VLEN = 8;
  int j = 0;
  int effective_N = (N + VLEN - 1) / (2 * VLEN) * (2 * VLEN);
  for (; j < effective_N; j += (2 * VLEN)) {
    // r1 is for j:j+VLEN
    // r2 is for j+VLEN:j+2*VLEN
    // r2_rem is used to calculate the mask for r2
    int r2_rem = N - VLEN - j;
    r2_rem = (r2_rem <= VLEN) ? r2_rem : VLEN;
    r2_rem = (r2_rem < 0) ? 0 : r2_rem;
    __m256i mask_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(&avx2_ps_or_epi32_masks[r2_rem]));
    for (int i = 0; i < M; ++i) {
      __m256 c_v_r1;
      __m256 c_v_r2;
      if (accum) {
        c_v_r1 = _mm256_loadu_ps(C + i * ldc + j);
        c_v_r2 = _mm256_maskload_ps(C + i * ldc + j + VLEN, mask_v);
      } else {
        c_v_r1 = _mm256_set1_ps(0.0f);
        c_v_r2 = _mm256_set1_ps(0.0f);
      }
      int r = row_ptr[i];
      // unrolled by 4
      for (; r < row_ptr[i + 1] - 4; r += 4) {
        int acbr_0 = col_idx[r + 0];
        int acbr_1 = col_idx[r + 1];
        int acbr_2 = col_idx[r + 2];
        int acbr_3 = col_idx[r + 3];
        __m256 a_v0 = _mm256_set1_ps(values[r + 0]);
        __m256 a_v1 = _mm256_set1_ps(values[r + 1]);
        __m256 a_v2 = _mm256_set1_ps(values[r + 2]);
        __m256 a_v3 = _mm256_set1_ps(values[r + 3]);
        __m256 br_v_0_r1 = _mm256_loadu_ps(B + acbr_0 * ldb + j);
        __m256 br_v_1_r1 = _mm256_loadu_ps(B + acbr_1 * ldb + j);
        __m256 br_v_2_r1 = _mm256_loadu_ps(B + acbr_2 * ldb + j);
        __m256 br_v_3_r1 = _mm256_loadu_ps(B + acbr_3 * ldb + j);
        __m256 br_v_0_r2 = _mm256_loadu_ps(B + acbr_0 * ldb + j + VLEN);
        __m256 br_v_1_r2 = _mm256_loadu_ps(B + acbr_1 * ldb + j + VLEN);
        __m256 br_v_2_r2 = _mm256_loadu_ps(B + acbr_2 * ldb + j + VLEN);
        __m256 br_v_3_r2 = _mm256_loadu_ps(B + acbr_3 * ldb + j + VLEN);
        c_v_r1 = _mm256_fmadd_ps(a_v0, br_v_0_r1, c_v_r1);
        c_v_r1 = _mm256_fmadd_ps(a_v1, br_v_1_r1, c_v_r1);
        c_v_r1 = _mm256_fmadd_ps(a_v2, br_v_2_r1, c_v_r1);
        c_v_r1 = _mm256_fmadd_ps(a_v3, br_v_3_r1, c_v_r1);
        c_v_r2 = _mm256_fmadd_ps(a_v0, br_v_0_r2, c_v_r2);
        c_v_r2 = _mm256_fmadd_ps(a_v1, br_v_1_r2, c_v_r2);
        c_v_r2 = _mm256_fmadd_ps(a_v2, br_v_2_r2, c_v_r2);
        c_v_r2 = _mm256_fmadd_ps(a_v3, br_v_3_r2, c_v_r2);
      }
      for (; r < row_ptr[i + 1]; ++r) {
        int acbr = col_idx[r];
        __m256 a_v = _mm256_set1_ps(values[r]);
        __m256 br_v_r1 = _mm256_loadu_ps(B + acbr * ldb + j);
        __m256 br_v_r2 = _mm256_maskload_ps(B + acbr * ldb + j + VLEN, mask_v);
        c_v_r1 = _mm256_fmadd_ps(a_v, br_v_r1, c_v_r1);
        c_v_r2 = _mm256_fmadd_ps(a_v, br_v_r2, c_v_r2);
      }
      _mm256_storeu_ps(C + i * ldc + j, c_v_r1);
      _mm256_maskstore_ps(C + i * ldc + j + VLEN, mask_v, c_v_r2);
    } // i loop
  }
  // Handle remainder j loop
  int rem = N - j;
  if (rem > 0) {
    for (int i = 0; i < M; ++i) {
      __m256 c_v_r;
      __m256i mask_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(&avx2_ps_or_epi32_masks[rem]));
      if (accum) {
        c_v_r = _mm256_maskload_ps(C + i * ldc + j, mask_v);
      } else {
        c_v_r = _mm256_set1_ps(0.0f);
      }
      int r = row_ptr[i];
      for (; r < row_ptr[i + 1] - 4; r += 4) {
        int acbr_0 = col_idx[r + 0];
        int acbr_1 = col_idx[r + 1];
        int acbr_2 = col_idx[r + 2];
        int acbr_3 = col_idx[r + 3];
        __m256 a_v0 = _mm256_set1_ps(values[r + 0]);
        __m256 a_v1 = _mm256_set1_ps(values[r + 1]);
        __m256 a_v2 = _mm256_set1_ps(values[r + 2]);
        __m256 a_v3 = _mm256_set1_ps(values[r + 3]);
        __m256 br_v_r0 = _mm256_maskload_ps(B + acbr_0 * ldb + j, mask_v);
        __m256 br_v_r1 = _mm256_maskload_ps(B + acbr_1 * ldb + j, mask_v);
        __m256 br_v_r2 = _mm256_maskload_ps(B + acbr_2 * ldb + j, mask_v);
        __m256 br_v_r3 = _mm256_maskload_ps(B + acbr_3 * ldb + j, mask_v);
        c_v_r = _mm256_fmadd_ps(a_v0, br_v_r0, c_v_r);
        c_v_r = _mm256_fmadd_ps(a_v1, br_v_r1, c_v_r);
        c_v_r = _mm256_fmadd_ps(a_v2, br_v_r2, c_v_r);
        c_v_r = _mm256_fmadd_ps(a_v3, br_v_r3, c_v_r);
      }
      // Handle remainder r loop
      for (; r < row_ptr[i + 1]; ++r) {
        int acbr = col_idx[r];
        __m256 a_v = _mm256_set1_ps(values[r]);
        __m256 br_v_r = _mm256_maskload_ps(B + acbr * ldb + j, mask_v);
        c_v_r = _mm256_fmadd_ps(a_v, br_v_r, c_v_r);
      }
      _mm256_maskstore_ps(C + i * ldc + j, mask_v, c_v_r);
    }
  }
}
} // namespace internal
} // namespace fbgemm

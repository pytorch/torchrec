/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"

#include <immintrin.h>

namespace fbgemm {
namespace internal {

void SparseDenseMMAvx512(
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
  constexpr int VLEN = 16;
  int j = 0;
  const int effective_N = ((int)((N + VLEN - 1) / (2 * VLEN))) * (2 * VLEN);
  for (; j < effective_N; j += 2 * VLEN) {
    // r1 is for j:j+VLEN
    // r2 is for j+VLEN:j+2*VLEN
    // r2_rem is used to calculate the mask for r2
    int r2_rem = N - VLEN - j;
    r2_rem = (r2_rem <= VLEN) ? r2_rem : (VLEN);
    r2_rem = (r2_rem < 0) ? 0 : r2_rem;
    __mmask16 mask_v = (((long long) 1) << r2_rem) - 1;
    for (int i = 0; i < M; ++i) {
      __m512 c_v_r1;
      __m512 c_v_r2;
      if (accum) {
        c_v_r1 = _mm512_loadu_ps(C + i * ldc + j);
        c_v_r2 = _mm512_maskz_loadu_ps(mask_v, C + i * ldc + j + VLEN);
      } else {
        c_v_r1 = _mm512_set1_ps(0.0f);
        c_v_r2 = _mm512_set1_ps(0.0f);
      }
      int r = row_ptr[i];
      int r_end_aligned = row_ptr[i] + (row_ptr[i + 1] - row_ptr[i]) / 3 * 3;
      // unrolled by 3
      for (; r < r_end_aligned; r += 3) {
        int acbr_0 = col_idx[r + 0];
        int acbr_1 = col_idx[r + 1];
        int acbr_2 = col_idx[r + 2];
        __m512 a_v_0 = _mm512_set1_ps(values[r + 0]);
        __m512 a_v_1 = _mm512_set1_ps(values[r + 1]);
        __m512 a_v_2 = _mm512_set1_ps(values[r + 2]);
        __m512 br_v_0_r1 = _mm512_loadu_ps(B + acbr_0 * ldb + j);
        __m512 br_v_1_r1 = _mm512_loadu_ps(B + acbr_1 * ldb + j);
        __m512 br_v_2_r1 = _mm512_loadu_ps(B + acbr_2 * ldb + j);
        c_v_r1 = _mm512_fmadd_ps(a_v_0, br_v_0_r1, c_v_r1);
        c_v_r1 = _mm512_fmadd_ps(a_v_1, br_v_1_r1, c_v_r1);
        c_v_r1 = _mm512_fmadd_ps(a_v_2, br_v_2_r1, c_v_r1);
        __m512 br_v_0_r2 = _mm512_maskz_loadu_ps(mask_v, B + acbr_0 * ldb + j + VLEN);
        __m512 br_v_1_r2 = _mm512_maskz_loadu_ps(mask_v, B + acbr_1 * ldb + j + VLEN);
        __m512 br_v_2_r2 = _mm512_maskz_loadu_ps(mask_v, B + acbr_2 * ldb + j + VLEN);
        c_v_r2 = _mm512_fmadd_ps(a_v_0, br_v_0_r2, c_v_r2);
        c_v_r2 = _mm512_fmadd_ps(a_v_1, br_v_1_r2, c_v_r2);
        c_v_r2 = _mm512_fmadd_ps(a_v_2, br_v_2_r2, c_v_r2);
      }
      for (; r < row_ptr[i + 1]; ++r) {
        int acbr = col_idx[r];
        __m512 a_v = _mm512_set1_ps(values[r]);
        __m512 br_v_r1 = _mm512_loadu_ps(B + acbr * ldb + j);
        c_v_r1 = _mm512_fmadd_ps(a_v, br_v_r1, c_v_r1);
        __m512 br_v_r2 = _mm512_maskz_loadu_ps(mask_v, B + acbr * ldb + j + VLEN);
        c_v_r2 = _mm512_fmadd_ps(a_v, br_v_r2, c_v_r2);
      }
      _mm512_storeu_ps(C + i * ldc + j, c_v_r1);
      _mm512_mask_storeu_ps(C + i * ldc + j + VLEN, mask_v, c_v_r2);
    } // i loop
  }
  // Handle remainder j loop
  int rem = N - j;
  if (rem > 0) {
    for (int i = 0; i < M; ++i) {
      __m512 c_v;
      __mmask16 mask_v = (((long long) 1) << rem) - 1;
      if (accum) {
        c_v = _mm512_maskz_loadu_ps(mask_v, C + i * ldc + j);
      } else {
        c_v = _mm512_set1_ps(0.0f);
      }
      int r = row_ptr[i];
      int r_end_aligned = row_ptr[i] + (row_ptr[i + 1] - row_ptr[i]) / 8 * 8;
      // unrolled by 8
      for (; r < r_end_aligned; r += 8) {
        int acbr_0 = col_idx[r + 0];
        int acbr_1 = col_idx[r + 1];
        int acbr_2 = col_idx[r + 2];
        int acbr_3 = col_idx[r + 3];
        int acbr_4 = col_idx[r + 4];
        int acbr_5 = col_idx[r + 5];
        int acbr_6 = col_idx[r + 6];
        int acbr_7 = col_idx[r + 7];
        __m512 a_v_0 = _mm512_set1_ps(values[r + 0]);
        __m512 a_v_1 = _mm512_set1_ps(values[r + 1]);
        __m512 a_v_2 = _mm512_set1_ps(values[r + 2]);
        __m512 a_v_3 = _mm512_set1_ps(values[r + 3]);
        __m512 a_v_4 = _mm512_set1_ps(values[r + 4]);
        __m512 a_v_5 = _mm512_set1_ps(values[r + 5]);
        __m512 a_v_6 = _mm512_set1_ps(values[r + 6]);
        __m512 a_v_7 = _mm512_set1_ps(values[r + 7]);
        __m512 br_v_0 = _mm512_maskz_loadu_ps(mask_v, B + acbr_0 * ldb + j);
        __m512 br_v_1 = _mm512_maskz_loadu_ps(mask_v, B + acbr_1 * ldb + j);
        __m512 br_v_2 = _mm512_maskz_loadu_ps(mask_v, B + acbr_2 * ldb + j);
        __m512 br_v_3 = _mm512_maskz_loadu_ps(mask_v, B + acbr_3 * ldb + j);
        __m512 br_v_4 = _mm512_maskz_loadu_ps(mask_v, B + acbr_4 * ldb + j);
        __m512 br_v_5 = _mm512_maskz_loadu_ps(mask_v, B + acbr_5 * ldb + j);
        __m512 br_v_6 = _mm512_maskz_loadu_ps(mask_v, B + acbr_6 * ldb + j);
        __m512 br_v_7 = _mm512_maskz_loadu_ps(mask_v, B + acbr_7 * ldb + j);
        c_v = _mm512_fmadd_ps(a_v_0, br_v_0, c_v);
        c_v = _mm512_fmadd_ps(a_v_1, br_v_1, c_v);
        c_v = _mm512_fmadd_ps(a_v_2, br_v_2, c_v);
        c_v = _mm512_fmadd_ps(a_v_3, br_v_3, c_v);
        c_v = _mm512_fmadd_ps(a_v_4, br_v_4, c_v);
        c_v = _mm512_fmadd_ps(a_v_5, br_v_5, c_v);
        c_v = _mm512_fmadd_ps(a_v_6, br_v_6, c_v);
        c_v = _mm512_fmadd_ps(a_v_7, br_v_7, c_v);
      }
      // Handle remainder r loop
      for (; r < row_ptr[i + 1]; ++r) {
        int acbr = col_idx[r];
        __m512 a_v = _mm512_set1_ps(values[r]);
        __m512 br_v = _mm512_maskz_loadu_ps(mask_v, B + acbr * ldb + j);
        c_v = _mm512_fmadd_ps(a_v, br_v, c_v);
      }
      _mm512_mask_storeu_ps(C + i * ldc + j, mask_v, c_v);
    }
  }
}
} // namespace internal
} // namespace fbgemm

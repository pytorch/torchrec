/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8Spmdm.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include "./OptimizedKernelsAvx2.h"

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double spmdm_initial_time = 0.0;
double spmdm_transpose_uint8_time = 0.0;
double spmdm_transpose_32xN_time = 0.0;
double spmdm_compute_time = 0.0;
double spmdm_transpose_Nx32_time = 0.0;
double spmdm_run_time = 0.0;
double sconv_run_time = 0.0;
#endif

using namespace std;

namespace fbgemm {

CompressedSparseColumn::CompressedSparseColumn(int num_of_rows, int num_of_cols)
    : num_rows_(num_of_rows),
      colptr_(num_of_cols + 1),
      hyper_sparse_(false),
      old_nnz_(-1) {}

double CompressedSparseColumn::Density() const {
  return static_cast<double>(NumOfNonZeros()) / (NumOfRows() * NumOfCols());
}

bool CompressedSparseColumn::IsHyperSparse() const {
  if (NumOfNonZeros() != old_nnz_) {
    old_nnz_ = NumOfNonZeros();
    // The number of non-zero per row is very small.
    hyper_sparse_ = static_cast<double>(old_nnz_) / NumOfRows() < 0.3;
  }

  return hyper_sparse_;
}

// TODO: fallback when AVX2 is not available
void CompressedSparseColumn::SpMDM(
    const block_type_t& block,
    const uint8_t* A,
    int lda,
    bool accumulation,
    int32_t* C,
    int ldc) const {
  int K = NumOfRows();
  int N = block.col_size;

  if (K == 0 || N == 0 || block.row_size == 0) {
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_very_start,
      t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

// Note: These (and others below) cause a ~2-3% overall performance drop in
// resnet/resnext so we are keeping arrays with dynamic size for gcc/clang and
// dynamically allocated memory for MSVC even though dynamically allocated
// memory works for all compilers.
#ifdef _MSC_VER
  uint8_t* A_buffer =
      static_cast<uint8_t*>(fbgemmAlignedAlloc(64, K * 32 * sizeof(uint8_t)));
  int32_t* C_buffer =
      static_cast<int32_t*>(fbgemmAlignedAlloc(64, N * 32 * sizeof(int32_t)));
#else
  alignas(64) uint8_t A_buffer[K * 32];
  alignas(64) int32_t C_buffer[N * 32];
#endif

  // If we compute C = C + A * B, where B is a sparse matrix in CSC format, for
  // each non-zero in B, we'd need to access the corresponding column in A.
  // This results in strided access, which we want to avoid.
  // Instead, we pre-transpose A and C, and compute C = (C^T + B^T * A^T)^T

  if (IsHyperSparse()) {
    // The cost of transpose is O(K*N) and we do O(NNZ*N) multiplications.
    // If NNZ/K is small, it's not worth doing transpose so we just use this
    // scalar loop.
#ifdef _MSC_VER
    int32_t* C_temp = static_cast<int32_t*>(
        fbgemmAlignedAlloc(64, block.row_size * sizeof(int32_t)));
#else
    int32_t C_temp[block.row_size];
#endif
    if (accumulation) {
      for (int j = 0; j < block.col_size; ++j) {
        int k = colptr_[block.col_start + j];
        int k_end = colptr_[block.col_start + j + 1];
        if (k_end == k) {
        } else if (k_end == k + 1) {
          int row = rowidx_[k];
          int w = values_[k];
          for (int i = 0; i < block.row_size; ++i) {
            C[i * ldc + j] += A[(block.row_start + i) * lda + row] * w;
          }
        } else {
          for (int i = 0; i < block.row_size; ++i) {
            C_temp[i] = C[i * ldc + j];
          }
          for (; k < k_end; ++k) {
            int row = rowidx_[k];
            int w = values_[k];
            for (int i = 0; i < block.row_size; ++i) {
              C_temp[i] += A[(block.row_start + i) * lda + row] * w;
            }
          }
          for (int i = 0; i < block.row_size; ++i) {
            C[i * ldc + j] = C_temp[i];
          }
        }
      } // for each column of B
    } else {
      for (int j = 0; j < block.col_size; ++j) {
        int k = colptr_[block.col_start + j];
        int k_end = colptr_[block.col_start + j + 1];
        if (k_end == k) {
          for (int i = 0; i < block.row_size; ++i) {
            C[i * ldc + j] = 0;
          }
        } else if (k_end == k + 1) {
          int row = rowidx_[k];
          int w = values_[k];
          for (int i = 0; i < block.row_size; ++i) {
            C[i * ldc + j] = A[(block.row_start + i) * lda + row] * w;
          }
        } else {
          for (int i = 0; i < block.row_size; ++i) {
            C_temp[i] = 0;
          }
          for (; k < k_end; ++k) {
            int row = rowidx_[k];
            int w = values_[k];
            for (int i = 0; i < block.row_size; ++i) {
              C_temp[i] += A[(block.row_start + i) * lda + row] * w;
            }
          }
          for (int i = 0; i < block.row_size; ++i) {
            C[i * ldc + j] = C_temp[i];
          }
        }
      } // for each column of B
    }
#ifdef _MSC_VER
    fbgemmAlignedFree(A_buffer);
    fbgemmAlignedFree(C_buffer);
    fbgemmAlignedFree(C_temp);
#endif
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  spmdm_initial_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // Take 32 rows at a time
  int i_end = block.row_start + block.row_size;
  for (int i1 = block.row_start; i1 < i_end; i1 += 32) {
    // Transpose 32 x K submatrix of A
    if (i_end - i1 < 32) {
#ifdef _MSC_VER
      uint8_t* A_temp_buffer = static_cast<uint8_t*>(
          fbgemmAlignedAlloc(64, K * 32 * sizeof(uint8_t)));
#else
      alignas(64) uint8_t A_temp_buffer[K * 32];
#endif
      for (int i2 = 0; i2 < (i_end - i1) / 8 * 8; i2 += 8) {
        transpose_8rows(K, A + (i1 + i2) * lda, lda, A_buffer + i2, 32);
      }

      for (int i2 = (i_end - i1) / 8 * 8; i2 < i_end - i1; ++i2) {
        memcpy(
            A_temp_buffer + i2 * K, A + (i1 + i2) * lda, K * sizeof(uint8_t));
      }
      memset(
          A_temp_buffer + (i_end - i1) * K,
          0,
          (32 - (i_end - i1)) * K * sizeof(uint8_t));
      for (int i2 = (i_end - i1) / 8 * 8; i2 < 32; i2 += 8) {
        transpose_8rows(K, A_temp_buffer + i2 * K, K, A_buffer + i2, 32);
      }
#ifdef _MSC_VER
      fbgemmAlignedFree(A_temp_buffer);
#endif
    } else {
      for (int i2 = 0; i2 < 32; i2 += 8) {
        transpose_8rows(K, A + (i1 + i2) * lda, lda, A_buffer + i2, 32);
      }
    }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_uint8_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    if (accumulation) {
      // Transpose 32 x N submatrix of C to fill N x 32 C_buffer
      transpose_simd(
          std::min(32, i_end - i1),
          N,
          reinterpret_cast<const float*>(C + (i1 - block.row_start) * ldc),
          ldc,
          reinterpret_cast<float*>(C_buffer),
          32);
    } else {
      memset(C_buffer, 0, N * 32 * sizeof(int32_t));
    }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_32xN_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    spmdmKernelAvx2(
        block.col_size,
        A_buffer,
        colptr_.data() + block.col_start,
        values_.data(),
        rowidx_.data(),
        C_buffer);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_compute_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    // Transpose N x 32 C_buffer to fill 32 x N submatrix of C
    transpose_simd(
        N,
        std::min(32, i_end - i1),
        reinterpret_cast<const float*>(C_buffer),
        32,
        reinterpret_cast<float*>(C + (i1 - block.row_start) * ldc),
        ldc);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_Nx32_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  spmdm_run_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif
#ifdef _MSC_VER
  fbgemmAlignedFree(A_buffer);
  fbgemmAlignedFree(C_buffer);
#endif
}

void CompressedSparseColumn::SparseConv(
    const conv_param_t<>& conv_p,
    const block_type_t& block,
    const uint8_t* A,
    int32_t A_zero_point,
    bool accumulation,
    int32_t* C,
    int ldc) const {
  int K = NumOfRows();
  int N = block.col_size;

  if (K == 0 || N == 0) {
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // TODO: if not hyper sparse, transpose a block of A matrix as in SpMDM.
  if (!accumulation) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        C[(i - block.row_start) * ldc + j - block.col_start] = 0;
      }
    }
  }
  for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
    for (int k = colptr_[j]; k < colptr_[j + 1]; ++k) {
      int v = values_[k];
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        int ow = i % conv_p.OUT_DIM[1];
        int oh = i / conv_p.OUT_DIM[1] % conv_p.OUT_DIM[0];
        int n = i / conv_p.OUT_DIM[1] / conv_p.OUT_DIM[0];
        assert(n < conv_p.MB);
        int iw = -conv_p.pad[1] + ow * conv_p.stride[1] + kw_[k];
        int ih = -conv_p.pad[0] + oh * conv_p.stride[0] + kh_[k];

        if (ih >= 0 && ih < conv_p.IN_DIM[0] && iw >= 0 &&
            iw < conv_p.IN_DIM[1]) {
          C[(i - block.row_start) * ldc + j - block.col_start] +=
              A[((n * conv_p.IN_DIM[0] + ih) * conv_p.IN_DIM[1] + iw) *
                    conv_p.IC +
                ic_[k]] *
              v;
        } else {
          C[(i - block.row_start) * ldc + j - block.col_start] +=
              A_zero_point * v;
        }
      }
    }
  } // for each column of B

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  sconv_run_time += (dt);
#endif
}

} // namespace fbgemm

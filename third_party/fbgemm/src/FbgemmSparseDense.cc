/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <array>

#include "fbgemm/Utils.h"
#include "fbgemm/spmmUtils.h"

using namespace std;

namespace fbgemm {

template <typename T>
FBGEMM_API std::unique_ptr<CSRMatrix<T>>
fbgemmDenseToCSR(int R, int C, const T* inp, int ld) {
  unique_ptr<CSRMatrix<T>> csr(new CSRMatrix<T>());
  csr->rowPtr.push_back(0);
  int nnz = 0;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      if (inp[i * ld + j] != 0) {
        csr->values.push_back(inp[i * ld + j]);
        csr->colIdx.push_back(j);
        nnz++;
      }
    }
    csr->rowPtr.push_back(nnz);
  }
  return csr;
}

template <typename T>
std::unique_ptr<CSRMatrix<T>> fbgemmDenseToCSR(int R, int C, const T* inp) {
  return fbgemmDenseToCSR<T>(R, C, inp, C);
}

template FBGEMM_API std::unique_ptr<CSRMatrix<int8_t>>
fbgemmDenseToCSR(int R, int C, const int8_t* inp);
template FBGEMM_API std::unique_ptr<CSRMatrix<float>>
fbgemmDenseToCSR(int R, int C, const float* inp);

template FBGEMM_API std::unique_ptr<CSRMatrix<int8_t>>
fbgemmDenseToCSR(int R, int C, const int8_t* inp, int ld);
template FBGEMM_API std::unique_ptr<CSRMatrix<float>>
fbgemmDenseToCSR(int R, int C, const float* inp, int ld);

template <typename T, int RB, int CB>
FBGEMM_API std::unique_ptr<BCSRMatrix<T, RB, CB>>
fbgemmDenseToBCSR(int R, int C, const T* inp, int ld) {
  unique_ptr<BCSRMatrix<T, RB, CB>> bcsr(new BCSRMatrix<T, RB, CB>(R, C));
  bcsr->pack(inp, ld);
  return bcsr;
}

template <typename T, int RB, int CB>
FBGEMM_API std::unique_ptr<BCSRMatrix<T, RB, CB>>
fbgemmDenseToBCSR(int R, int C, const T* inp) {
  return fbgemmDenseToBCSR<T, RB, CB>(R, C, inp, C);
}

template <typename T, int RB, int CB>
constexpr int BCSRMatrix<T, RB, CB>::RB;

template <typename T, int RB, int CB>
constexpr int BCSRMatrix<T, RB, CB>::CB;

template <typename T, int RB, int CB>
constexpr int BCSRMatrix<T, RB, CB>::COLTILE;

template <typename T, int RB, int CB>
void BCSRMatrix<T, RB, CB>::pack(const DTYPE* src, size_t ld) {
  rowBPtr.push_back(0);
  int nnzb = 0;
  int numCOLTILEs = (C + COLTILE - 1) / COLTILE;
  int rowBlocks = (R + RB - 1) / RB;
  for (int jt = 0; jt < numCOLTILEs; ++jt) {
    for (int i = 0; i < rowBlocks; ++i) {
      int curCols = min(C - jt * COLTILE, COLTILE);
      int curColBlocks = (curCols + CB - 1) / CB;
      std::array<int32_t, RB> rowSum = {0};
      for (int j = 0; j < curColBlocks; ++j) {
        // is the whole block zero?
        bool isCurrentBlockNonZero = false;
        for (int ib = 0; ib < RB; ++ib) {
          // break if already found a non-zero element or
          // out of bounds
          if (isCurrentBlockNonZero || (i * RB + ib) >= R) {
            break;
          }
          for (int jb = 0; jb < CB; ++jb) {
            // within bound?
            if ((jt * COLTILE + j * CB + jb) >= C) {
              continue;
            } else {
              if (src[(i * RB + ib) * ld + jt * COLTILE + j * CB + jb] != 0) {
                isCurrentBlockNonZero = true;
                break;
              }
            }
          }
        }
        if (isCurrentBlockNonZero) {
          for (int ib = 0; ib < RB; ++ib) {
            for (int jb = 0; jb < CB; ++jb) {
              if ((i * RB + ib) >= R || (jt * COLTILE + j * CB + jb) >= C) {
                // zero fill
                values.push_back(0);
              } else {
                DTYPE val =
                    src[(i * RB + ib) * ld + jt * COLTILE + j * CB + jb];
                values.push_back(val);
                rowSum[ib] += static_cast<int32_t>(val);

              }
            }
          }
          colBIdx.push_back(j);
          nnzb++;
        }
      }
      rowBPtr.push_back(nnzb);
      // Note: in row_offsets we don't need to subtract the constant term
      // weight_zero_point * C because it's 0 as weight_zero_point is always 0
      // for sparse kernels.
      for (int ib = 0; ib < RB; ++ib) {
        if (jt) {
          row_offsets[i * RB + ib] += rowSum[ib];
        } else {
          row_offsets[i * RB + ib] = rowSum[ib];
        }
      }
    }
  }
}

template <typename T, int RB, int CB>
void BCSRMatrix<T, RB, CB>::pack(const DTYPE* src) {
  pack(src, C);
}

template <typename T, int RB, int CB>
void BCSRMatrix<T, RB, CB>::unpack(T* dst, size_t ld) {
  // zero out destination
  memset(dst, 0, R * C * sizeof(T));

  int numCOLTILEs = (C + COLTILE - 1) / COLTILE;
  int rowBlocks = (R + RB - 1) / RB;
  for (int jt = 0; jt < numCOLTILEs; ++jt) {
    for (int i = 0; i < rowBlocks; ++i) {
      // For the current tile, rowBPtr starts from currentTileIdx (i.e., jt) * R
      for (int r = rowBPtr[jt * R + i]; r < rowBPtr[jt * R + i + 1]; ++r) {
        int curColIdx = colBIdx[r];
        for (int ib = 0; ib < RB; ++ib) {
          for (int jb = 0; jb < CB; ++jb) {
            // Are we within bounds of destination matrix?
            if ((i * RB + ib) < R && (jt * COLTILE + curColIdx * CB + jb) < C) {
              dst[(i * RB + ib) * ld + jt * COLTILE + curColIdx * CB + jb] =
                  values[r * RB * CB + ib * CB + jb];
            }
          }
        }
      }
    }
  }
}

template <typename T, int RB, int CB>
void BCSRMatrix<T, RB, CB>::unpack(T* dst) {
  unpack(dst, C);
}

template struct BCSRMatrix<int8_t, 1, 4>;

template struct CSRMatrix<int8_t>;
template struct CSRMatrix<float>;

template FBGEMM_API std::unique_ptr<BCSRMatrix<int8_t, 1, 4>>
fbgemmDenseToBCSR(int R, int C, const int8_t* inp);

template FBGEMM_API std::unique_ptr<BCSRMatrix<int8_t, 1, 4>>
fbgemmDenseToBCSR(int R, int C, const int8_t* inp, int ld);

void SparseDenseMM(
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
  static const auto iset = fbgemmInstructionSet();
  // Run time CPU detection
  if (isZmm(iset)) {
    internal::SparseDenseMMAvx512(
        M, N, row_ptr, col_idx, values, B, ldb, C, ldc, accum);
  } else if (isYmm(iset)) {
    internal::SparseDenseMMAvx2(
        M, N, row_ptr, col_idx, values, B, ldb, C, ldc, accum);
  } else {
    sparseDenseMMRef(M, N, row_ptr, col_idx, values, B, ldb, C, ldc, accum);
  }
}

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
FBGEMM_API void fbgemmSparseDenseInt8MM(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum,
    int thread_id,
    int num_threads) {
  static const auto iset = fbgemmInstructionSet();
  // No parallelization currently
  // All work is done by thread 0
  if (thread_id > 0) {
    return;
  }

  // Run time CPU detection
  if (isZmm(iset)) {
    internal::SparseDenseInt8MMAvx512<FUSE_RELU, Q_GRAN>(
        N,
        bcsr,
        B,
        ldb,
        C_i32,
        C_u8,
        ldc,
        rParams,
        accum,
        thread_id,
        num_threads);
  } else if (isYmm(iset)) {
    internal::SparseDenseInt8MMAvx2<FUSE_RELU, Q_GRAN>(
        N,
        bcsr,
        B,
        ldb,
        C_i32,
        C_u8,
        ldc,
        rParams,
        accum,
        thread_id,
        num_threads);
  } else {
    sparseDenseInt8MMRef<FUSE_RELU, Q_GRAN>(
        N,
        bcsr,
        B,
        ldb,
        C_i32,
        C_u8,
        ldc,
        rParams,
        accum,
        thread_id,
        num_threads);
  }
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)                             \
  template FBGEMM_API void fbgemmSparseDenseInt8MM<FUSE_RELU, QGRAN>( \
      int N,                                                          \
      const std::unique_ptr<BCSRMatrix<>>& bcsr,                      \
      const uint8_t* B,                                               \
      int ldb,                                                        \
      int32_t* C_i32,                                                 \
      uint8_t* C_u8,                                                  \
      int ldc,                                                        \
      trRequantizationParams_t& rParams,                              \
      bool accum,                                                     \
      int thread_id,                                                  \
      int num_threads);
CREATE_INSTANCE(true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

} // namespace fbgemm

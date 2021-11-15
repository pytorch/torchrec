/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"
#include "fbgemm/spmmUtilsAvx2.h"

namespace fbgemm {

template <typename T>
struct FBGEMM_API CSRMatrix {
  std::vector<int> rowPtr;
  std::vector<int> colIdx;
  std::vector<T> values;
};

/**
 * Tiled block CSR format
 * Partial blocks are zero-filled
 *
 */
template <typename T = std::int8_t, int ROW_BLOCK = 1, int COL_BLOCK = 4>
struct FBGEMM_API BCSRMatrix {
  using DTYPE = T;
  static constexpr int RB = ROW_BLOCK; // Block size for rows
  static constexpr int CB = COL_BLOCK; // Block size for cols
  // We only tile in column dimension currently
  // COLTILE must be a multiple of COL_BLOCK
  static constexpr int COLTILE = 4000;
  std::vector<int> rowBPtr; // rowPtr for blocks
  std::vector<int> colBIdx; // colIdx for blocks
  std::vector<DTYPE> values;
  // Sum of all elements in a row
  std::vector<int32_t> row_offsets;
  int R;
  int C;

  BCSRMatrix(int Rows, int Cols) {
    R = Rows;
    C = Cols;
    row_offsets.resize(R, 0);
  }

  /**
   * @brief pack from dense to tiled block CSR format
   * @param R   number of rows in the matrix
   * @param C   number of columns in the matrix
   * @param src is the source matrix with data type DTYPE
   * @param ld is the leading dimension
   */
  void pack(const DTYPE* src, size_t ld);

  /**
   * @brief pack from dense to tiled block CSR format
   * @param R   number of rows in the matrix
   * @param C   number of columns in the matrix
   * @param src is the source matrix with data type DTYPE
   *
   * leading dim of the matrix is assumed to be equal to C
   */
  void pack(const DTYPE* src);

  /**
   * @brief unpack from tiled block CSR to dense
   * @param dst should be able to hold R*C elements of type DTYPE
   * @param ld is the leading dimension
   */
  void unpack(DTYPE* dst, size_t ld);

  /*
   * @brief unpack from tiled block CSR to dense
   * @param dst should be able to hold R*C elements of type DTYPE
   *
   * leading dimension of the matrix is assumed to be equal to C
   */
  void unpack(DTYPE* dst);
};

template <typename T>
FBGEMM_API std::unique_ptr<CSRMatrix<T>>
fbgemmDenseToCSR(int R, int C, const T* inp, int ld);

template <typename T>
FBGEMM_API std::unique_ptr<CSRMatrix<T>>
fbgemmDenseToCSR(int R, int C, const T* inp);

template <typename T = std::int8_t, int RB = 1, int CB = 4>
FBGEMM_API std::unique_ptr<BCSRMatrix<T, RB, CB>>
fbgemmDenseToBCSR(int R, int C, const T* inp, int ld);

template <typename T = std::int8_t, int RB = 1, int CB = 4>
FBGEMM_API std::unique_ptr<BCSRMatrix<T, RB, CB>>
fbgemmDenseToBCSR(int R, int C, const T* inp);

/**
 * @param accum       Controls accumulation.
 *                    1 means we're accumulating to the C Matrix.
 *
 * Note on matrix order and layout:
 *   Unlike other fbgemm functions that follow PyTorch convention where A
 * matrix is activation (so in uint8_t for quantized FC/Conv or fp32) and B
 * matrix is weight (so in int8_t for quantized FC/Conv or fp32), here A is
 * weight matrix. This is because we mostly target sparsity in weights and for
 * row-major layout it's more efficient to have A as a sparse matrix: for each
 * non-zero of A at ith row and kth column, we can access kth row of B, whose
 * elements are contiguous in memory. If B matrix was sparse, for each non-zero
 * of B at kth row and jth column, we would've needed to access kth column of A,
 * whose elements are not contiguous in memory with C/C++'s row-major layout.
 *   Alternatively, we can call this function as if we're computing
 * C^T = B^T * A^T while maintaining PyTorch's convention that the lefthand
 * side matrix B is activation. If B matrix is in column-major layout, we don't
 * need to do an extra transposition. The C matrix will be output in
 * column-major layout, so if we have a back-to-back Sparse-Dense matrix-matrix
 * multiplications, B matrices of subsequent matrices will be already in
 * column-major layout. Refer to SparseDenseMMFP32Benchmark.cc for an example.
 *
 */
FBGEMM_API void SparseDenseMM(
    int M,
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    bool accum = false);

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
    bool accum = false,
    int thread_id = 0,
    int num_threads = 1);

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
    bool accum = false);

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
    bool accum = false);

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MMAvx2(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum = false,
    int thread_id = 0,
    int num_threads = 1);

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MMAvx512(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum = false,
    int thread_id = 0,
    int num_threads = 1);

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MVAvx512(
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    trRequantizationParams_t& rParams,
    bool accum = false,
    int thread_id = 0,
    int num_threads = 1);

} // namespace internal

} // namespace fbgemm

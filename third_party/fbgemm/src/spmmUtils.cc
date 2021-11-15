/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <iostream>
#include <cstring>
#include <cassert>
#include "fbgemm/spmmUtils.h"

using namespace std;

namespace fbgemm {

void sparseDenseMMRef(
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
  for (int i = 0; i < M; ++i) {
    if (!accum) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] = 0;
      }
    }
    for (int r = row_ptr[i]; r < row_ptr[i + 1]; ++r) {
      int acbr = col_idx[r];
      float v = values[r];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += v * B[acbr * ldb + j];
      }
    }
  }
}

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
FBGEMM_API void trRequantizeRef(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const trRequantizationParams_t& r) {
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
      int32_t raw = inp[(i - block.row_start) * ld_in + (j - block.col_start)];
      if (r.act_zero_point) {
        raw -= r.act_zero_point * r.weight_row_offsets[i];
      }
      int weight_zeropoint_idx;
      if (Q_GRAN == QuantizationGranularity::TENSOR) {
        weight_zeropoint_idx = 0;
      } else {
        // Q_GRAN == QuantizationGranularity::OUT_CHANNEL
        weight_zeropoint_idx = i;
      }
      if (r.act_col_offsets) {
        raw -= r.act_col_offsets[j - block.col_start] *
            r.weight_zero_points[weight_zeropoint_idx];
      }
      float raw_f = raw;
      if (r.bias) {
        raw_f += r.bias[i] / r.act_times_w_scale[weight_zeropoint_idx];
      }

      float ab = raw_f * r.act_times_w_scale[weight_zeropoint_idx] / r.C_scale;
      int rounded = std::rintf(ab) + r.C_zero_point;
      out[i * ld_out + j] = std::max(
          FUSE_RELU ? static_cast<int>(r.C_zero_point) : 0,
          std::min(255, rounded));
    }
  }
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)                     \
  template FBGEMM_API void trRequantizeRef<FUSE_RELU, QGRAN>( \
      uint8_t * out,                                          \
      const int32_t* inp,                                     \
      const block_type_t& block,                              \
      int ld_out,                                             \
      int ld_in,                                              \
      const trRequantizationParams_t& r);
CREATE_INSTANCE(true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

vector<vector<int>> getSparseMatrixShapes() {
  // clang-format off
  // {M, N, K}
  vector<vector<int>> shapes = {
    {1,128,160},
    {1,16,128},
    {1,256,160},
    {168,15,197},
    {168,8,197},
    {176,15,197},
    {176,8,197},
    {21,1,1027},
    {21,120,512},
    {21,125,300},
    {21,128,120},
    {21,128,176},
    {21,16,128},
    {21,256,5018},
    {21,256,512},
    {21,2955,512},
    {21,5018,256},
    {21,512,128},
    {21,512,2125},
    {21,512,256},
    {21,512,3851},
    {21,512,4085},
    {21,8,512},
    {22,1,1027},
    {22,120,512},
    {22,125,300},
    {22,128,120},
    {22,128,176},
    {22,16,128},
    {22,256,5018},
    {22,256,512},
    {22,2955,512},
    {22,5018,256},
    {22,512,128},
    {22,512,2125},
    {22,512,256},
    {22,512,3851},
    {22,512,4085},
    {22,8,512},
    {128,128,128},
    {256,256,256},
    {512,512,512},
  };

  // RoBERTa shapes
  const char* include_roberta = std::getenv("INCLUDE_ROBERTA");
  if(include_roberta && (strcmp(include_roberta, "1") == 0)) {
    vector<vector<int>> roberta_shapes = {
      // average input length = 25
      {25, 2304,  768},
      {25,  768,  768},
      {25, 3072,  768},
      {25,  768, 3072},
      {25, 3072, 1024},
      {25, 1024, 1024},
      {25, 4096, 1024},
      {25, 1024, 4096},
      // high input length = 51
      {51, 2304,  768},
      {51,  768,  768},
      {51, 3072,  768},
      {51,  768, 3072},
      {51, 3072, 1024},
      {51, 1024, 1024},
      {51, 4096, 1024},
      {51, 1024, 4096},
    };
    shapes.insert(shapes.end(), roberta_shapes.begin(), roberta_shapes.end() );
    cout << "RoBERTa shapes included." << endl;
  }
  else {
    cout << "RoBERTa shapes not included. " <<
      "To include, add \"INCLUDE_ROBERTA=1\" as an env variable." << endl;
  }

  // LSTM shapes
  const char* include_lstm = std::getenv("INCLUDE_LSTM");
  if(include_lstm && (strcmp(include_lstm, "1") == 0)) {
    vector<vector<int>> lstm_shapes = {
      { 1, 2560, 640},
      {16, 2560, 640},
      {18, 2560, 640},
      { 1, 2560, 720},
      {16, 2560, 720},
      {18, 2560, 720},
    };
    shapes.insert(shapes.end(), lstm_shapes.begin(), lstm_shapes.end() );
    cout << "LSTM shapes included." << endl;
  }
  else {
    cout << "LSTM shapes not included. " <<
      "To include, add \"INCLUDE_LSTM=1\" as an env variable." << endl;
  }

  // RNNT shapes
  const char* include_rnnt = std::getenv("INCLUDE_RNNT");
  if(include_rnnt && (strcmp(include_rnnt, "1") == 0)) {
    vector<vector<int>> rnnt_shapes = {
      {1, 4096, 640},
      {1, 640, 1024},
      {5, 4096, 640},
      {20, 4096, 640},
      {4, 4096, 1024},
      {3, 4096, 1024},
      {1, 4096, 1024},
      {2, 4096, 1024},
      {5, 1024, 640},
      {5, 4096, 1280},
      {20, 4096, 880},
      {10, 4096, 640},
      {10, 4096, 1280},
      {5, 4096, 1024},
      {1, 1024, 640},
      {6, 4096, 1024},
      {1, 640, 256},
      {1, 1024, 256},
      {7, 4096, 1024},
      {8, 4096, 1024},
      {9, 4096, 1024},
      {7, 4096, 640},
      {4, 4096, 640},
      {28, 4096, 640},
      {16, 4096, 640},
      {10, 4096, 1024},
      {8, 4096, 640},
      {8, 4096, 1280},
      {7, 1024, 640},
      {7, 4096, 1280},
      {4, 1024, 640},
      {4, 4096, 1280},
      {28, 4096, 880},
      {16, 4096, 880},
      {14, 4096, 640},
      {14, 4096, 1280},
      {1, 256, 5000},
      {2, 256, 4500},
      {64, 256, 4500},
    };
    shapes.insert(shapes.end(), rnnt_shapes.begin(), rnnt_shapes.end() );
    cout << "rnnt shapes included." << endl;
  }
  else {
    cout << "RNNT shapes not included. " <<
      "To include, add \"INCLUDE_RNNT=1\" as an env variable." << endl;
  }
  // clang-format on
  return shapes;
}

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void sparseDenseInt8MMRef(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_i8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum,
    int /*thread_id*/,
    int /*num_threads*/) {
  // Calcualtes accum ? C += A * B : C = A * B
  constexpr int rowBlockSize = BCSRMatrix<>::RB;
  constexpr int colBlockSize = BCSRMatrix<>::CB;
  constexpr int colTileSize = BCSRMatrix<>::COLTILE;
  int M = bcsr->R;
  int K = bcsr->C;
  int kTiles = (K + colTileSize - 1) / colTileSize;
  assert(
      M % rowBlockSize == 0 &&
      "Number of rows is not a multiple of rowBlockSize size");

  for (int j = 0; j < N; ++j) {
    for (int kt = 0; kt < kTiles; ++kt) {
      int* rowBPtr_start = bcsr->rowBPtr.data() + kt * M;
      for (int i = 0; i < M / rowBlockSize; i += rowBlockSize) {
        // only initialize to 0 for the first ktile
        if (!accum && !kt) {
          C_i32[i * ldc + j] = 0;
        }
        for (int r = rowBPtr_start[i]; r < rowBPtr_start[i + 1]; ++r) {
          int acbr_block = bcsr->colBIdx[r];
          const int8_t* blockValues =
              bcsr->values.data() + r * rowBlockSize * colBlockSize;
          for (int i_b = 0; i_b < rowBlockSize; ++i_b) {
            for (int k_b = 0; k_b < colBlockSize; ++k_b) {
              C_i32[(i * rowBlockSize + i_b) * ldc + j] +=
                  static_cast<int32_t>(blockValues[i_b * colBlockSize + k_b]) *
                  static_cast<int32_t>(
                      B[(acbr_block * colBlockSize + k_b + kt * colTileSize) *
                            ldb +
                        j]);
            }
          }
        }
      }
    }
  }
  block_type_t block{0, M, 0, N};
  trRequantizeRef<FUSE_RELU, Q_GRAN>(C_i8, C_i32, block, ldc, ldc, rParams);
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)               \
  template void sparseDenseInt8MMRef<FUSE_RELU, QGRAN>( \
      int N,                                            \
      const std::unique_ptr<BCSRMatrix<>>& bcsr,        \
      const uint8_t* B,                                 \
      int ldb,                                          \
      int32_t* C_i32,                                   \
      uint8_t* C_u8,                                    \
      int ldc,                                          \
      trRequantizationParams_t& rParams,                \
      bool accum,                                       \
      int thread_id,                                    \
      int num_threads);
CREATE_INSTANCE(true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

} // namespace fbgemm

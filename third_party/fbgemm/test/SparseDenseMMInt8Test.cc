/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <iostream>

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/Utils.h"
#include "fbgemm/spmmUtils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::OUT_CHANNEL};

// tuple represents M, N, K, fnz, fuse_relu and QuantizationGranularity
class SPMMInt8Test : public testing::TestWithParam<tuple<int, int, int, float, bool, QuantizationGranularity>> {};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    SPMMInt8Test,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 3, 4, 7, 13, 16, 20, 32}), // M
        ::testing::ValuesIn({1, 2, 3, 4, 7, 13, 16, 20, 32}), // N
        ::testing::ValuesIn(
            {1, 2, 3, 4, 7, 8, 14, 24, 4000, 4001, 4096, 5000}), // K
        ::testing::ValuesIn({0.1f, 0.2f, 0.3f}), // fnz
        ::testing::Bool(), // fuse relu
        ::testing::ValuesIn(qGranularityVals))); // QuantizationGranularity

/**
 * Test for sparse-dense matrix-matrix multiplication (int8)
 */
TEST_P(SPMMInt8Test, spInt8) {
  int M, N, K;
  float fnz;
  bool fuse_relu;
  QuantizationGranularity qGran;
  tie(M, N, K, fnz, fuse_relu, qGran) = GetParam();

  auto aData = getRandomBlockSparseMatrix<uint8_t>(
      M, K, 1.0, 1 /* rowBlockSize */, 1 /* colBlockSize */);
  auto bData = getRandomBlockSparseMatrix<int8_t>(K, N, fnz);
  auto cData = getRandomBlockSparseMatrix<int32_t>(
      M, N, 1.0, 1 /* rowBlockSize */, 1 /* colBlockSize */);

  aligned_vector<uint8_t> atData(K * M);
  aligned_vector<int8_t> btData(N * K);
  aligned_vector<int32_t> ctDataRef(N * M, 5);
  aligned_vector<uint8_t> ctDataRef_u8(N * M, 7);
  aligned_vector<int32_t> ctDataIntrin_i32(N * M, 9);
  aligned_vector<uint8_t> ctDataIntrin_u8(N * M, 11);

  transpose_matrix(M, K, aData.data(), K, atData.data(), M);
  transpose_matrix(K, N, bData.data(), N, btData.data(), K);

  unique_ptr<BCSRMatrix<>> bcsr = fbgemmDenseToBCSR(N, K, btData.data());

  // output scale and zero point
  float scale = 128.0f;
  int32_t zero_point = 2;

  int32_t act_zero_point = 2;

  // symmetric quant for weights
  aligned_vector<int32_t> weight_zero_point(N);
  randFill<int32_t>(weight_zero_point, 0, 0);

  // Each row of weight matrix has it's own scale
  // The following is a multiplication activation scale with
  // weight scales.
  aligned_vector<float> act_times_w_scale(N);
  randFill<float>(act_times_w_scale, -8.0f, 8.0f);

  trRequantizationParams_t reqParams = {
      act_zero_point,
      weight_zero_point.data(),
      zero_point,
      scale,
      bcsr->row_offsets.data(),
      nullptr,
      nullptr,
      act_times_w_scale.data()};

  int ldat = M;
  int ldct = M;

  if (fuse_relu) {
    if (qGran == QuantizationGranularity::TENSOR) {
      fbgemmSparseDenseInt8MM<true, QuantizationGranularity::TENSOR>(
          M,
          bcsr,
          atData.data(),
          ldat,
          ctDataIntrin_i32.data(),
          ctDataIntrin_u8.data(),
          ldct,
          reqParams);
    } else {
      fbgemmSparseDenseInt8MM<true, QuantizationGranularity::OUT_CHANNEL>(
          M,
          bcsr,
          atData.data(),
          ldat,
          ctDataIntrin_i32.data(),
          ctDataIntrin_u8.data(),
          ldct,
          reqParams);
    }
  } else {
    if (qGran == QuantizationGranularity::TENSOR) {
      fbgemmSparseDenseInt8MM<false, QuantizationGranularity::TENSOR>(
          M,
          bcsr,
          atData.data(),
          ldat,
          ctDataIntrin_i32.data(),
          ctDataIntrin_u8.data(),
          ldct,
          reqParams);
    } else {
      fbgemmSparseDenseInt8MM<false, QuantizationGranularity::OUT_CHANNEL>(
          M,
          bcsr,
          atData.data(),
          ldat,
          ctDataIntrin_i32.data(),
          ctDataIntrin_u8.data(),
          ldct,
          reqParams);
    }
  }

  matmul_u8i8acc32_ref(
      M,
      N,
      K,
      K, // lda
      N, // ldb
      N, // ldc
      aData.data(),
      bData.data(),
      cData.data());
  transpose_matrix(M, N, cData.data(), N, ctDataRef.data(), M);

  // ctDataRef is nxm
  block_type_t block{0, N, 0, M};
  if (fuse_relu) {
    if (qGran == QuantizationGranularity::TENSOR) {
      trRequantizeRef<true, QuantizationGranularity::TENSOR>(
          ctDataRef_u8.data(), ctDataRef.data(), block, M, M, reqParams);
    } else {
      trRequantizeRef<true, QuantizationGranularity::OUT_CHANNEL>(
          ctDataRef_u8.data(), ctDataRef.data(), block, M, M, reqParams);
    }
  } else {
    if (qGran == QuantizationGranularity::TENSOR) {
      trRequantizeRef<false, QuantizationGranularity::TENSOR>(
          ctDataRef_u8.data(), ctDataRef.data(), block, M, M, reqParams);
    } else {
      trRequantizeRef<false, QuantizationGranularity::OUT_CHANNEL>(
          ctDataRef_u8.data(), ctDataRef.data(), block, M, M, reqParams);
    }
  }
  // printMatrix(matrix_op_t::NoTranspose, ctDataRef_u8.data(), n, m, m,
  // "ctDataRef_u8");
  // printMatrix(matrix_op_t::NoTranspose, ctDataIntrin_u8.data(), n, m, m,
  // "ctDataIntrin_u8");
  //
  // Compare results
  for (auto i = 0; i < ctDataRef.size(); i++) {
    EXPECT_EQ(ctDataRef_u8[i], ctDataIntrin_u8[i])
        << "Results differ ref " << ctDataRef_u8[i] << " and test "
        << ctDataIntrin_u8[i] << " at " << i;
  }
}

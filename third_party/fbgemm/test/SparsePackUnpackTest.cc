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
#include "fbgemm/spmmUtils.h"

using namespace std;
using namespace fbgemm;

// tuple represents N and K
class packUnpackTest : public testing::TestWithParam<tuple<int, int, float>> {};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    packUnpackTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 3, 4, 7, 13, 16, 20, 32}), // N
        ::testing::ValuesIn(
            {1, 2, 3, 4, 7, 8, 14, 24, 4000, 4001, 4096, 5000}), // K
        ::testing::ValuesIn({0.01f, 0.02f, 0.3f}))); // fnz

/**
 * Test for packing/unpacking
 */
TEST_P(packUnpackTest, sparseUnpackTest) {
  int N, K;
  float fnz;
  tie(N, K, fnz) = GetParam();

  // wData is dense
  auto wData = getRandomBlockSparseMatrix<int8_t>(N, K, fnz, 1, 4);
  // printMatrix(matrix_op_t::NoTranspose, wData.data(), N, K, K, "original");

  // bcsr is tiled block sparse
  unique_ptr<BCSRMatrix<>> bcsr = fbgemmDenseToBCSR(N, K, wData.data());

  // wUnpackedData is unpacked from bcsr

  vector<int8_t> wUnpackedData(N * K, 0);

  // unpack
  bcsr->unpack(wUnpackedData.data());
  // printMatrix(matrix_op_t::NoTranspose, wUnpackedData.data(), N, K, K,
  // "unpacked");

  // compare results with original dense
  for (size_t j = 0; j < N; ++j) {
    for (size_t k = 0; k < K; ++k) {
      ASSERT_EQ(wData[j * K + k], wUnpackedData[j * K + k])
          << "Original and unpacked data elements are not the same at idx ["
          << j << ", " << k << "]: "
          << "original: " << static_cast<int>(wData[j * K + k])
          << " , unpacked: " << static_cast<int>(wUnpackedData[j * K + k]);
    }
  }
}

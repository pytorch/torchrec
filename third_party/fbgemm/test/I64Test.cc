/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include <limits>
#include <random>

#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmI64.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
class Int64GemmTest : public testing::Test {
 protected:
  vector<array<int, 3>> GenParams() {
    vector<array<int, 3>> shapes;
    default_random_engine generator;
    uniform_int_distribution<int> dist_dim(1, 128);
    for (int i = 0; i < 256; ++i) {
      shapes.push_back(
          {dist_dim(generator), dist_dim(generator), dist_dim(generator)});
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(Int64GemmTest, test) {
  const auto shapes = GenParams();
  for (const auto s : shapes) {
    const int m = s[0];
    const int n = s[1];
    const int k = s[2];

    aligned_vector<int64_t> A(m * k);
    aligned_vector<int64_t> B(k * n);

    for (matrix_op_t transa :
         {matrix_op_t::NoTranspose, matrix_op_t::Transpose}) {
      const int lda = transa == matrix_op_t::Transpose ? m : k;
      for (matrix_op_t transb :
           {matrix_op_t::NoTranspose, matrix_op_t::Transpose}) {
        const int ldb = transb == matrix_op_t::Transpose ? k : n;

        aligned_vector<int64_t> C(m * n);
        aligned_vector<int64_t> C_ref = C;

        randFill(
            A,
            numeric_limits<int64_t>::lowest(),
            numeric_limits<int64_t>::max());
        randFill(
            B,
            numeric_limits<int64_t>::lowest(),
            numeric_limits<int64_t>::max());

        for (const bool accumulate : {false, true}) {
          cblas_gemm_i64_i64acc(
              transa,
              transb,
              m,
              n,
              k,
              A.data(),
              lda,
              B.data(),
              ldb,
              accumulate,
              C.data(),
              n);

          cblas_gemm_i64_i64acc_ref(
              transa,
              transb,
              m,
              n,
              k,
              A.data(),
              lda,
              B.data(),
              ldb,
              accumulate,
              C_ref.data(),
              n);

          compare_validate_buffers<int64_t>(
              C_ref.data(), C.data(), m, n, n, 0L);
        }
      } // transb
    } // transa
  } // for each shape
}

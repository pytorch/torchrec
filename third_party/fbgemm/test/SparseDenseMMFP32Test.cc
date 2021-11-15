/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include <random>

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/spmmUtils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
uniform_int_distribution<int> dist_dim(1, 256);
default_random_engine generator;

class SparseDenseTest : public testing::Test {
 protected:
  vector<tuple<int, int, int, float>> GenParams() {
    vector<tuple<int, int, int, float>> shapes;

    uniform_real_distribution<float> dist_fnz(0, 1.0);
    for (int i = 0; i < 256; ++i) {
      shapes.push_back(make_tuple(
          dist_dim(generator),
          dist_dim(generator),
          dist_dim(generator),
          dist_fnz(generator)));
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SparseDenseTest, fp32) {
  auto shapes = GenParams();
  int m, n, k;
  float fnz;
  for (auto s : shapes) {
    tie(m, n, k, fnz) = s;
    auto aData = getRandomSparseVector(m * k);
    auto bData = getRandomSparseVector(k * n, fnz);
    auto cDataNaive = getRandomSparseVector(m * n);

    cblas_sgemm_ref(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        m,
        n,
        k,
        1.0f,
        aData.data(),
        k,
        bData.data(),
        n,
        0.0f,
        cDataNaive.data(),
        n);

    // run fast version
    // Pick arbitrary leading dimensions that are not same as m or k for
    // testing purpose
    int ldat = 2 * m;
    int ldbt = 2 * k;
    int ldct = 2 * m;

    // To compute A*B where B is sparse matrix, we need to do
    // (B^T*A^T)^T
    aligned_vector<float> atData(k * ldat);
    transpose_matrix(m, k, aData.data(), k, atData.data(), ldat);
    aligned_vector<float> btData(n * ldbt);
    transpose_matrix(k, n, bData.data(), n, btData.data(), ldbt);
    auto cData = getRandomSparseVector(n * ldct);

    unique_ptr<CSRMatrix<float>> csr =
        fbgemmDenseToCSR(n, k, btData.data(), ldbt);
    SparseDenseMM(
        n,
        m,
        csr->rowPtr.data(),
        csr->colIdx.data(),
        csr->values.data(),
        atData.data(),
        ldat,
        cData.data(),
        ldct);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = cDataNaive[i * n + j];
        float actual = cData[i + j * ldct];
        EXPECT_NEAR(expected, actual, 1e-6 * std::abs(expected) + 1e-7)
            << "Results differ at (" << i << ", " << j << ")";
      }
    }
  } // for each shape
}

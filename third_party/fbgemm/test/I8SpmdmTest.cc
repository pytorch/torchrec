/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include <gtest/gtest.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmI8Spmdm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

std::vector<float> densities{0.0001f, 0.001f, 0.01f, 0.1f, 1.0f};

namespace {
class fbgemmSPMDMTest
    : public testing::TestWithParam<std::tuple<float, bool, bool>> {};
} // namespace

INSTANTIATE_TEST_CASE_P(
    Instance0,
    fbgemmSPMDMTest,
    ::testing::Combine(
        ::testing::ValuesIn(densities),
        ::testing::Bool(),
        ::testing::Bool()));

TEST_P(fbgemmSPMDMTest, TestsSpMDM) {
  const vector<array<int, 3>> shapes = {
      //   M,    N,    K
      {1024, 1024, 1024},
      {511, 512, 512},
      {111, 111, 111},
      {14 * 14 * 2, 4, 2},
  };

  float density;
  bool accumulation, test_ld;
  tie(density, accumulation, test_ld) = GetParam();

  for (const auto& shape : shapes) {
    int M = shape[0];
    int N = shape[1];
    int K = shape[2];
    int N_adjusted = N;
    int K_adjusted = K;
    if (test_ld) {
      // When test_ld is true, we multiply with the bottom-right quadrant of B
      N_adjusted = std::max(N / 2, 1);
      K_adjusted = std::max(K / 2, 1);
    }

    aligned_vector<uint8_t> A(M * K);
    randFill<uint8_t>(A, 0, 255);

    CompressedSparseColumn B_csc(K_adjusted, N_adjusted);
    vector<int32_t> C(M * N);
    vector<int32_t> C_ref(C.size());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C_ref[i * N + j] = i + j;
      }
    }

    // deterministic random number
    default_random_engine eng;
    binomial_distribution<> per_col_nnz_dist(K_adjusted, density);
    uniform_int_distribution<> value_dist(
        numeric_limits<int8_t>::min() / 2, numeric_limits<int8_t>::max() / 2);

    vector<int> row_indices(K_adjusted);

    int total_nnz = 0;
    for (int j = 0; j < N_adjusted; ++j) {
      B_csc.ColPtr()[j] = total_nnz;

      int nnz_of_j = per_col_nnz_dist(eng);
      total_nnz += nnz_of_j;

      iota(row_indices.begin(), row_indices.end(), 0);
      shuffle(row_indices.begin(), row_indices.end(), eng);
      sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

      for (int k = 0; k < nnz_of_j; ++k) {
        B_csc.RowIdx().push_back(row_indices[k]);
        B_csc.Values().push_back(value_dist(eng));
      }
    }
    B_csc.ColPtr()[N_adjusted] = total_nnz;

    spmdm_ref(
        M,
        A.data() + (test_ld ? K_adjusted : 0),
        K,
        B_csc,
        accumulation,
        C_ref.data() + (test_ld ? N_adjusted : 0),
        N);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (accumulation) {
          C[i * N + j] = i + j;
        } else {
          C[i * N + j] = i + j + 1;
        }
      }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();
      int i_per_thread = (M + num_threads - 1) / num_threads;
      int i_begin = std::min(tid * i_per_thread, M);
      int i_end = std::min(i_begin + i_per_thread, M);

      block_type_t block = {i_begin, i_end - i_begin, 0, N_adjusted};
      B_csc.SpMDM(
          block,
          A.data() + (test_ld ? K_adjusted : 0),
          K,
          accumulation,
          C.data() + i_begin * N + (test_ld ? N_adjusted : 0),
          N);
    }

    compare_validate_buffers(
        C_ref.data() + (test_ld ? N_adjusted : 0),
        C.data() + (test_ld ? N_adjusted : 0),
        M,
        N_adjusted,
        N,
        static_cast<int32_t>(0));
  } // for each shape
}

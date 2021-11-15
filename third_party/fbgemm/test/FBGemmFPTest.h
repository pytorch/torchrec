/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <random>
#include <gtest/gtest.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "./TestUtils.h"
#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmPackMatrixB.h"
#include "src/RefImplementations.h"

#ifdef USE_IACA
#include "iacaMarks.h"
#endif

namespace fbgemm {
/*
 * @brief Abstract of the GEMM FP test
 * The template parameter is transpose of A and B
 */
template<typename T>
class FBGemmFPTest
    : public testing::TestWithParam<std::pair<fbgemm::matrix_op_t, fbgemm::matrix_op_t>> {
 protected:
  std::vector<std::vector<int>> GenShapes() const {
    std::vector<std::vector<int>> shapes;
    std::random_device r;
    std::default_random_engine generator(r());
    std::uniform_int_distribution<int> dm(1, 256);
    std::uniform_int_distribution<int> dnk(1, 1024);
    for (int i = 0; i < 10; i++) {
      int m = dm(generator);
      int n = dnk(generator);
      int k = dnk(generator);
      shapes.push_back({m, n, k});
    }
    return shapes;
  }

  void TestRun() {
    auto shapes = GenShapes();
    float alpha = 1.f, beta = 0.f;
    matrix_op_t atrans, btrans;
    std::tie(atrans, btrans) = GetParam();

    for (auto s : shapes) {
      int m = s[0];
      int n = s[1];
      int k = s[2];

      std::cerr << "m = " << m << " n = " << n << " k = " << k;
      if (atrans == matrix_op_t::Transpose) {
        std::cerr << " A_transposed";
      }
      if (btrans == matrix_op_t::Transpose) {
        std::cerr << " B_transposed";
      }
      std::cerr << std::endl;

      // initialize with small numbers
      aligned_vector<int> Aint(m * k);
      aligned_vector<int> Bint(k * n);
      randFill(Aint, 0, 4);
      randFill(Bint, 0, 4);
      aligned_vector<float> A(Aint.begin(), Aint.end());
      aligned_vector<float> B(Bint.begin(), Bint.end());

      aligned_vector<float> C(m * n, NAN);

      aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

      // Gold via reference sgemm
      cblas_sgemm_ref(
          atrans,
          btrans,
          m,
          n,
          k,
          1.0f,
          A_ref.data(),
          atrans == matrix_op_t::Transpose ? m : k,
          B_ref.data(),
          btrans == matrix_op_t::Transpose ? k : n,
          0.0f,
          C_ref.data(),
          n);

      PackedGemmMatrixB<T> Bp(btrans, k, n, alpha, B.data());
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        cblas_gemm_compute(
            atrans, m, A.data(), Bp, beta, C.data(), tid, num_threads);
      }

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = C_ref[i * n + j];
          float actual = C[i * n + j];
          EXPECT_EQ(actual, expected)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
        }
      }
    }
  }

  void UnpackTestRun() {
    auto shapes = GenShapes();
    float alpha = 1.f, beta = 0.f;
    matrix_op_t atrans, btrans;
    std::tie(atrans, btrans) = GetParam();

    for (auto s : shapes) {
      int m = s[0];
      int n = s[1];
      int k = s[2];

      std::cerr << "m = " << m << " n = " << n << " k = " << k;
      if (atrans == matrix_op_t::Transpose) {
        std::cerr << " A_transposed";
      }
      if (btrans == matrix_op_t::Transpose) {
        std::cerr << " B_transposed";
      }
      std::cerr << std::endl;

      // initialize with small numbers
      aligned_vector<int> Aint(m * k);
      aligned_vector<int> Bint(k * n);
      randFill(Aint, 0, 4);
      randFill(Bint, 0, 4);
      aligned_vector<float> A(Aint.begin(), Aint.end());
      aligned_vector<float> B(Bint.begin(), Bint.end());

      aligned_vector<float> C(m * n, NAN);

      aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

      // Gold via reference sgemm
      cblas_sgemm_ref(
          atrans,
          btrans,
          m,
          n,
          k,
          1.0f,
          A_ref.data(),
          atrans == matrix_op_t::Transpose ? m : k,
          B_ref.data(),
          btrans == matrix_op_t::Transpose ? k : n,
          0.0f,
          C_ref.data(),
          n);

      // fbgemm fp16
      PackedGemmMatrixB<T> Bp(btrans, k, n, alpha, B.data());
      EXPECT_TRUE(Bp.packed());

      // Test unpack
      aligned_vector<T> tmp(Bp.matSize());
      memcpy(tmp.data(), Bp.pmat(), Bp.matSize() * sizeof(T));
      Bp.unpackFromSrc(btrans, tmp.data());
      EXPECT_FALSE(Bp.packed());
      memcpy(tmp.data(), Bp.pmat(), Bp.matSize() * sizeof(T));
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
          EXPECT_EQ(
              sizeof(T) == sizeof(float16)
                  ? cpu_half2float(tmp[i * n + j])
                  : tmp[i * n + j],
              B[i * n + j]);
        }
      }

      // Pack it back
      Bp.packFromSrc(btrans, tmp.data());
      EXPECT_TRUE(Bp.packed());

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
      {
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        cblas_gemm_compute(
            atrans, m, A.data(), Bp, beta, C.data(), tid, num_threads);
      }

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = C_ref[i * n + j];
          float actual = C[i * n + j];
          EXPECT_EQ(actual, expected)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
        }
      }
    }
  }
};

} // namespace fbgemm

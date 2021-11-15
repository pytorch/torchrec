/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <array>
#include <iostream>
#include <limits>
#include <random>

#include "./BenchUtils.h"
#include "fbgemm/FbgemmI64.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

int main() {
  vector<char> llc(128 * 1024 * 1024);
  // clang-format off
  const vector<array<int, 3>> shapes = {
      {1024, 1024, 1024}
  };
  // clang-format on

  for (const auto s : shapes) {
    const int m = s[0];
    const int n = s[1];
    const int k = s[2];
    cout << "m " << m << " n " << n << " k " << k << endl;

    aligned_vector<int64_t> A(m * k);
    aligned_vector<int64_t> B(k * n);
    aligned_vector<int64_t> C(m * n);
    aligned_vector<int64_t> C_ref = C;

    randFill(
        A, numeric_limits<int64_t>::lowest(), numeric_limits<int64_t>::max());
    randFill(
        B, numeric_limits<int64_t>::lowest(), numeric_limits<int64_t>::max());

    cblas_gemm_i64_i64acc_ref(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        m,
        n,
        k,
        A.data(),
        k,
        B.data(),
        n,
        false, /* accumulation*/
        C_ref.data(),
        n);

    constexpr int NWARMUP = 4;
    constexpr int NITER = 16;
    double ttot = measureWithWarmup(
        [&]() {
          cblas_gemm_i64_i64acc(
              matrix_op_t::NoTranspose,
              matrix_op_t::NoTranspose,
              m,
              n,
              k,
              A.data(),
              k,
              B.data(),
              n,
              false, /* accumulation*/
              C.data(),
              n);
        },
        NWARMUP,
        NITER,
        [&]() { llc_flush(llc); });

    const double ops = 2.0 * m * n * k;
    cout << "Gops/s = " << ops / ttot / 1e9 << endl;
    compare_buffers(C_ref.data(), C.data(), m, n, n, 5);

    cblas_gemm_i64_i64acc_ref(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        m,
        n,
        k,
        A.data(),
        k,
        B.data(),
        n,
        true, /* accumulation*/
        C_ref.data(),
        n);

    aligned_vector<int64_t> C_acc = C;
    cblas_gemm_i64_i64acc(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        m,
        n,
        k,
        A.data(),
        k,
        B.data(),
        n,
        true, /* accumulation*/
        C_acc.data(),
        n);

    ttot = measureWithWarmup(
        [&]() {
          cblas_gemm_i64_i64acc(
              matrix_op_t::NoTranspose,
              matrix_op_t::NoTranspose,
              m,
              n,
              k,
              A.data(),
              k,
              B.data(),
              n,
              true, /* accumulation*/
              C.data(),
              n);
        },
        NWARMUP,
        NITER,
        [&]() { llc_flush(llc); });

    cout << "Gops/s = " << ops / ttot / 1e9 << endl;
    compare_buffers(C_ref.data(), C_acc.data(), m, n, n, 5);
  }
}

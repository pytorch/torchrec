/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  normal_distribution<float> dist;
  default_random_engine engine;

  cout << setw(4) << "M"
       << " elements_per_sec_ref"
       << " elements_per_sec_simd" << endl;

  array<int, 8> dims{1, 10, 32, 40, 129, 256, 1024, 8000};

  for (int M : dims) {
    vector<float> a(M);
    vector<float> b(M), b_ref(M);
    vector<float16> t(M);

    generate(a.begin(), a.end(), [&dist, &engine] { return dist(engine); });

    double duration_ref = measureWithWarmup(
        [&]() {
          FloatToFloat16_ref(a.data(), t.data(), M);
          Float16ToFloat_ref(t.data(), b_ref.data(), M);
        },
        NWARMUP,
        NITER);
    duration_ref *= 1e9; // convert to ns

    double duration_simd = measureWithWarmup(
        [&]() {
          FloatToFloat16_simd(a.data(), t.data(), M);
          Float16ToFloat_simd(t.data(), b.data(), M);
        },
        NWARMUP,
        NITER);
    duration_simd *= 1e9; // convert to ns

    cout << setw(4) << M << setw(10) << setprecision(3) << M / duration_ref
         << setw(10) << setprecision(3) << M / duration_simd << endl;

    compare_buffers(b_ref.data(), b.data(), M, 1, 1, 5);
  } // M
} // performance_test

int main() {
  performance_test();
  return 0;
}

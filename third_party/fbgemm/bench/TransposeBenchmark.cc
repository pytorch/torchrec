/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Utils.h"
#include "src/TransposeUtils.h"

using namespace std;
using namespace fbgemm;

template <typename T>
void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  uniform_int_distribution<int> dist(0, 10);
  default_random_engine engine;

  string runType;
  if (is_same<T, float>::value) {
    runType = "float";
  } else {
    runType = "i8";
  }

  cout << setw(8) << "dtype" << setw(4) << "M" << setw(4) << "N"
       << " B_elements_per_sec" << endl;

  int dims[] = {1,  2,  3,  4,  5,  6,  8,   9,   10,  15,  16,
                17, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256};
  for (int M : dims) {
    for (int N : dims) {
      vector<T> a(M * N);
      vector<T> b(N * M), b_ref(N * M);

      generate(a.begin(), a.end(), [&dist, &engine] { return dist(engine); });
      transpose_ref(M, N, a.data(), N, b_ref.data(), M);

      double duration = measureWithWarmup(
          [&]() { transpose_simd(M, N, a.data(), N, b.data(), M); },
          NWARMUP,
          NITER);
      duration *= 1e9; // convert to ns

      cout << setw(8) << runType << setw(4) << M << setw(4) << N << setw(10)
           << setprecision(3) << (M * N) / duration << endl;

      compare_buffers(b_ref.data(), b.data(), M, N, N, 5);
    } // N
  } // M
} // performance_test

int main() {
  performance_test<float>();
  performance_test<uint8_t>();
  return 0;
}

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <initializer_list>
#include <iomanip>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/OptimizedKernelsAvx2.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  cout << setw(4) << "len"
       << ", B_elements_per_sec" << endl;

  for (int len : {1,  2,  3,  4,  5,  7,  8,   9,   15,  16,  17,
                  31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256}) {
    aligned_vector<uint8_t> a(len);

    double duration =
        measureWithWarmup([&]() { reduceAvx2(a.data(), len); }, NWARMUP, NITER);
    duration *= 1e9; // convert to ns

    cout << setw(4) << len << ", " << setw(10) << setprecision(3)
         << len / duration << endl;
  } // for each length
} // performance_test

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test();
  return 0;
}

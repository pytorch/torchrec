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
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/QuantUtils.h"
#include "fbgemm/Types.h"

using namespace std;
using namespace fbgemm;

// T is the type of scale and bias
template <typename T>
void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  if (is_same<T, float16>::value) {
    cout << "With scale and bias as float16" << endl;
  } else {
    cout << "With scale and bias as float" << endl;
  }
  cout << setw(8) << "bit_rate"
       << ", " << setw(6) << "rows"
       << "," << setw(6) << "cols"
       << "," << setw(16) << "elems_per_usec"
       << "," << setw(10) << "GB/Sec" << endl;
  std::vector<int> bit_rates;
  if (is_same<T, float16>::value) {
    bit_rates = {2, 4, 8};
  } else {
    // float
    bit_rates = {8};
  }
  for (int bit_rate : bit_rates) {
    for (int rowSize : {100, 120, 1000}) {
      for (int colSize : {16, 64, 128, 256, 512, 1024, 2048}) {
        aligned_vector<float> inpVec(rowSize * colSize);
        randFill<float>(inpVec, -10.0f, 10.0f);

        int out_emb_cols = colSize;

        if (is_same<T, float16>::value) {
          int elements_per_byte = 8 / bit_rate;
          out_emb_cols = (colSize + elements_per_byte - 1) / elements_per_byte;
        }
        int outVecSize = rowSize * (out_emb_cols + 2 * sizeof(float16));
        aligned_vector<uint8_t> outVec(outVecSize);

        double duration = 0.0f;

        duration = measureWithWarmup(
            [&]() {
              is_same<T, float16>::value
                  ? FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
                        bit_rate,
                        inpVec.data(),
                        rowSize,
                        colSize,
                        outVec.data())
                  : FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
                        inpVec.data(), rowSize, colSize, outVec.data());
            },
            NWARMUP,
            NITER,
            [&]() {
              cache_evict(inpVec);
              cache_evict(outVec);
            });

        float elements_per_usec = rowSize * colSize / (duration * 1e6);

        duration *= 1e9; // convert to ns
        long bytes_read = rowSize * colSize * sizeof(float);
        float gigabyes_per_sec = bytes_read / duration;

        cout << setw(8) << bit_rate << "," << setw(6) << rowSize << ", "
             << setw(6) << colSize << ",";
        cout << setw(16) << std::fixed << std::setprecision(2)
             << elements_per_usec << ", ";
        cout << setw(10) << std::fixed << std::setprecision(2)
             << gigabyes_per_sec << endl;
      } // for each cols
    } // for each rows
  } // for each bit_rate
} // performance_test

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test<float16>();
  performance_test<float>();
  return 0;
}

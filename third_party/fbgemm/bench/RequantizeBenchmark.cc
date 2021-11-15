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

using namespace std;
using namespace fbgemm;

enum class BenchmarkType {
  BARE_BONE,
  BIAS,
  A_ASYMMETRIC,
  B_ASYMMETRIC,
  PER_CHANNEL,
};

void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  cout << setw(4) << "len"
       << ", " << setw(10) << "Type"
       << ", B_elements_per_sec" << endl;

  for (int len : {1,  2,  3,  4,  5,  7,  8,   9,   15,  16,  17,
                  31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256}) {
    aligned_vector<float> C_multiplier(len);
    randFill<float>(C_multiplier, -8, 8);

    aligned_vector<int32_t> Bint8_zero_point(len), row_offset_buf(len),
        col_offsets(len), bias_vector(len), input(len);
    randFill<int32_t>(Bint8_zero_point, -8, 8);
    randFill<int32_t>(row_offset_buf, -8, 8);
    randFill<int32_t>(col_offsets, -8, 8);
    randFill<int32_t>(bias_vector, -8, 8);
    randFill<int32_t>(input, -8, 8);

    int32_t C_zero_point = -3;

    block_type_t block{0, 1, 0, len};

    aligned_vector<uint8_t> output(len);

    for (BenchmarkType bench_type : {BenchmarkType::BARE_BONE,
                                     BenchmarkType::BIAS,
                                     BenchmarkType::A_ASYMMETRIC,
                                     BenchmarkType::B_ASYMMETRIC,
                                     BenchmarkType::PER_CHANNEL}) {
      int32_t Aint8_zero_point =
          bench_type < BenchmarkType::A_ASYMMETRIC ? 0 : -3;
      if (bench_type < BenchmarkType::B_ASYMMETRIC) {
        Bint8_zero_point[0] = 0;
      }
      const int32_t* bias =
          bench_type == BenchmarkType::BARE_BONE ? nullptr : bias_vector.data();

      double duration = 0.0;

      DoNothing<> doNothingObj{};
      if (bench_type == BenchmarkType::PER_CHANNEL) {
        ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL> reqObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_point,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            row_offset_buf.data(),
            col_offsets.data(),
            bias,
            len);

        duration = measureWithWarmup(
            [&]() {
              reqObj.f<inst_set_t::avx2>(
                  output.data(), input.data(), block, len, len);
            },
            NWARMUP,
            NITER);
      } else {
        ReQuantizeOutput<false> reqObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_point,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            row_offset_buf.data(),
            col_offsets.data(),
            bias,
            len);

        duration = measureWithWarmup(
            [&]() {
              reqObj.f<inst_set_t::avx2>(
                  output.data(), input.data(), block, len, len);
            },
            NWARMUP,
            NITER);
      }

      duration *= 1e9; // convert to ns

      cout << setw(4) << len << ", ";
      switch (bench_type) {
        case BenchmarkType::BARE_BONE:
          cout << setw(10) << "bare_bone";
          break;
        case BenchmarkType::BIAS:
          cout << setw(10) << "bias";
          break;
        case BenchmarkType::A_ASYMMETRIC:
          cout << setw(10) << "a_asymmetric";
          break;
        case BenchmarkType::B_ASYMMETRIC:
          cout << setw(10) << "b_asymmetric";
          break;
        case BenchmarkType::PER_CHANNEL:
          cout << setw(10) << "per_channel";
          break;
      }
      cout << ", " << setw(10) << setprecision(3) << len / duration << endl;
    } // for each bench_type
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

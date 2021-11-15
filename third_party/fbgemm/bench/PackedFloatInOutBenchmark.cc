/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/QuantizationHelpers.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  // clang-format off
  vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {1, 128, 512},
    {1, 1024, 256},
    {1, 2048, 512},
    {1, 4096, 1024},

    {6, 256, 1024},
    {6, 256, 2048},
    {6, 512, 512},
    {6, 1024, 256},
    {6, 2048, 256},
    {6, 2048, 512},
    {6, 4096, 256},
    {6, 4096, 1024},
    {6, 4096, 2048},

    {10, 2048, 256},
    {10, 4096, 1024},

    {20, 2048, 256},
    {20, 4096, 1024},

    {102, 1024, 512},
    {102, 2323, 256},
    {102, 512, 256},

    {1, 800, 3200},
    {1, 800, 8000},

    {16, 256, 1500},
    {16, 256, 1567},
    {1, 128, 2876},
    {16, 128, 1567},
    {1, 128, 2722},
    {16, 256, 512},
  };
  // clang-format on
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(22)
       << "Packing (ms), " << setw(22) << "Kernel (ms), " << setw(22)
       << "Postprocessing (ms), " << setw(22) << "Total (ms), " << setw(22)
       << "Type, " << setw(5) << "GOPs" << endl;
#else
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(22)
       << "Type, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> start, end;
  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    float alpha = 1.f, beta = 0.f;
    aligned_vector<float> Afp32(m * k);
    aligned_vector<uint8_t> Aint8(Afp32.size());

    aligned_vector<float> Bfp32(k * n);
    aligned_vector<int8_t> Bint8(Bfp32.size());

    aligned_vector<float> Cfp32_mkl(m * n);
    aligned_vector<float> Cfp32_fb(Cfp32_mkl.size());

    aligned_vector<uint8_t> Cint8_fb(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_buffer(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 255);
    float Aint8_scale = 0.11;
    int32_t Aint8_zero_point = 43;
    for (auto i = 0; i < Afp32.size(); ++i) {
      Afp32[i] = Aint8_scale * (Aint8[i] - Aint8_zero_point);
    }

    randFill<int8_t>(Bint8, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    float Bint8_scale = 0.49;
    int32_t Bint8_zero_point = -30;
    for (auto i = 0; i < Bfp32.size(); ++i) {
      Bfp32[i] = Bint8_scale * (Bint8[i] - Bint8_zero_point);
    }

    // computing column offset
    vector<int32_t> col_offsets(n);
    col_offsets_with_zero_pt_s8acc32_ref(
        k, n, n, Bint8.data(), &Bint8_zero_point, col_offsets.data(), n);

    double ttot = 0;
    std::string type;
    double nops = 2.0 * m * n * k;
#ifdef USE_MKL
    type = "MKL_FP32";
    ttot = measureWithWarmup(
        [&]() {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m,
              n,
              k,
              alpha,
              Afp32.data(),
              k,
              Bfp32.data(),
              n,
              beta,
              Cfp32_mkl.data(),
              n);
        },
        NWARMUP,
        NITER,
        [&]() {
          if (flush) {
            llc_flush(llc);
          }
        });
    ttot *= 1e9; // convert to ns

    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k
         << ", ";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << setw(20) << fixed << setprecision(3) << 0.0f << ", " << setw(20)
         << 0.0f << ", " << setw(20) << 0.0f << ", " << setw(20) << 0.0f
         << ", ";
#endif
    cout << setw(20) << type << ", " << setw(5) << fixed << setprecision(1)
         << nops / ttot << endl;
#endif

    int32_t C_multiplier = 16544;
    int32_t C_right_shift = 35;
    int32_t C_zero_pt = 5;

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cfp32_mkl.data(),
    // m, n, n, "C mkl fp32");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_local.data(), m, n, n, "C requantized");
    // printMatrix(matrix_op_t::NoTranspose, col_offsets.data(), 1, n, n, "col
    // offsets before");

    vector<int32_t> row_offset_buf(
        PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithQuantRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k,
        Afp32.data(),
        k,
        nullptr, /*buffer for packed matrix*/
        Aint8_scale,
        Aint8_zero_point,
        1, /*groups*/
        row_offset_buf.data());

    PackBMatrix<int8_t> packedBN(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    DoNothing<float, float> doNothingObj{};
    ReQuantizeForFloat<false> outputProcObj(
        doNothingObj,
        Aint8_scale,
        &Bint8_scale,
        Aint8_zero_point,
        &Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr,
        n);

    ttot = 0;
    type = "FBGEMM_i8_acc32";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double total_packing_time = 0.0;
    double total_computing_time = 0.0;
    double total_kernel_time = 0.0;
    double total_postprocessing_time = 0.0;
    double total_run_time = 0.0;
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k
      << ", ";

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif

      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      fbgemmPacked(
          packAN,
          packedBN,
          Cfp32_fb.data(),
          (int32_t*)Cfp32_fb.data(),
          n,
          outputProcObj,
          0,
          1);
      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_packing_time += packing_time;
        total_computing_time += computing_time;
        total_kernel_time += kernel_time;
        total_postprocessing_time += postprocessing_time;
        total_run_time += run_time;
#endif
      }
    }
    ((volatile char*)(llc.data()));
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint8_local.data(),
    // m, n, n, "C requantized after");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");
    // printMatrix(matrix_op_t::NoTranspose,
    // col_offsets.data(), 1, n, n, "col offsets after");
    // compare_buffers(row_offsets.data(), row_offset_buf.data(),
    // row_offsets.size(), 5);
    // printMatrix(matrix_op_t::NoTranspose, Cfp32_fb.data(),
    // m, n, n, "C fb fp32");
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << setprecision(3) << setw(20)
         << total_packing_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_kernel_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_postprocessing_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_run_time / (double)NITER / 1e6 << ", ";
#endif
    cout << setw(20) << type << ", " << setw(5) << fixed << setprecision(1)
         << NITER * nops / ttot << endl;
    cout << endl;
    // cout << "total time: " << ttot << " ns" << endl;

    float maximum = *max_element(Cfp32_mkl.begin(), Cfp32_mkl.end());
    float minimum = *min_element(Cfp32_mkl.begin(), Cfp32_mkl.end());
    float atol = (maximum - minimum) / 255 / 1.9;

#ifdef USE_MKL
    // correctness check
    compare_buffers(Cfp32_mkl.data(), Cfp32_fb.data(), m, n, n, 5, atol);
#endif
  }
}

int main(int /* unused */, char** /* unused */) {
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

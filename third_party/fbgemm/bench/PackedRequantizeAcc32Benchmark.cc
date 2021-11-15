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
    // m, n, k
    {156800, 4, 36},
    {156800, 8, 36},
    {156800, 16, 36},
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
    aligned_vector<uint8_t> Aint8(m * k);

    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb(Cfp32_mkl.size());
    aligned_vector<uint8_t> Cint8_fb(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_local(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_buffer(Cfp32_mkl.size());
    aligned_vector<uint8_t> Cint8_local(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 255);
    // float Aint8_scale = 0.11;
    int32_t Aint8_zero_point = 43;
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    // float Bint8_scale = 0.49;
    int32_t Bint8_zero_point = -30;
    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    // computing column offset
    vector<int32_t> col_offsets(n);
    col_offsets_with_zero_pt_s8acc32_ref(
        k, n, n, Bint8.data(), &Bint8_zero_point, col_offsets.data(), n);

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;
#ifdef USE_MKL
    runType = "MKL_fp32";
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
    cout << setw(20) << runType << ", " << setw(5) << fixed << setprecision(1)
         << nops / ttot << endl;
#endif

    vector<int32_t> row_offsets(m);

    float C_multiplier = 0.1234;
    int32_t C_zero_pt = 5;

    matmul_u8i8acc32_ref(
        m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_local.data());

    row_offsets_u8acc32_ref(m, k, k, Aint8.data(), row_offsets.data());

    requantize_u8acc32_ref(
        m,
        n,
        n,
        Cint32_local.data(),
        Cint8_local.data(),
        &C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        &Bint8_zero_point,
        row_offsets.data(),
        col_offsets.data(),
        nullptr, // bias
        n); // ncols per quant group
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_local.data(),
    // m, n, n, "C int32");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_local.data(), m, n, n, "C requantized");
    // printMatrix(matrix_op_t::NoTranspose, col_offsets.data(), 1, n, n, "col
    // offsets before");

    PackBMatrix<int8_t> packedBN(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";
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

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        vector<int32_t> row_offset_buf(
            PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

        PackAWithRowOffset<uint8_t> packAN(
            matrix_op_t::NoTranspose,
            m,
            k,
            Aint8.data(),
            k,
            nullptr,
            1,
            row_offset_buf.data());

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj,
            &C_multiplier,
            C_zero_pt,
            Aint8_zero_point,
            &Bint8_zero_point,
            packAN.getRowOffsetBuffer(),
            col_offsets.data(),
            nullptr,
            n);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packAN,
            packedBN,
            Cint8_fb.data(),
            Cint32_buffer.data(),
            n,
            outputProcObj,
            tid,
            num_threads);
      }

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

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << setprecision(3) << setw(20)
         << total_packing_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_kernel_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_postprocessing_time / (double)NITER / 1e6 << ", " << setw(20)
         << total_run_time / (double)NITER / 1e6 << ", ";
#endif
    cout << setw(20) << runType << ", " << setw(5) << fixed << setprecision(1)
         << NITER * nops / ttot << endl;
    cout << endl;

#ifdef USE_MKL
    compare_buffers(Cint8_local.data(), Cint8_fb.data(), m, n, n, 5);
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

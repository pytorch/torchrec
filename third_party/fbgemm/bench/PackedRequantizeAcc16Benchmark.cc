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
#include <numeric>
#include <random>
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

using namespace std;
using namespace fbgemm;

enum class BenchmarkType {
  BARE_BONE, // no row-offset in input packing, and no output processing
  REQUANTIZATION, // no row-offset in input packing, and requantization
  ROW_OFFSET_AND_REQUANTIZATION, // row-offset in input packing, and
                                 // requantization
  EVERYTHING, // row-offset in input packing, and requantization + spmdm
};

void performance_test() {
  // clang-format off
  vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    {64, 68, 17},
    {60, 128, 64},

    {25088, 256, 64},
    {25088, 64, 64},
    {25088, 64, 576},
    {25088, 64, 256},

    {6272, 512, 256},
    {6272, 128, 256},
    {6272, 128, 1152},
    {6272, 512, 128},
    {6272, 128, 512},

    {1568, 1024, 512},
    {1568, 256, 512},
    {1568, 256, 2304},
    {1568, 1024, 256},
    {1568, 256, 1024},

    {392, 2048, 1024},
    {392, 512, 1024},
    {392, 512, 4608},
    {392, 2048, 512},
    {392, 512, 2048},
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
  cout << "M, "
       << "N, "
       << "K, "
       << "Output Processing, "
       << "Packing (ms), "
       << "Kernel (ms), "
       << "Postprocessing (ms), "
       << "Total (ms), "
       << "GOPS" << endl;
#else
  cout << setw(7) << "M, " << setw(7) << "N, " << setw(7) << "K, " << setw(32)
       << "Output Processing, " << setw(18) << "Type, " << setw(5) << "GOPS"
       << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> begin, end;
  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    float alpha = 1.0f, beta = 0.0f;
    aligned_vector<uint8_t> Aint8(m * k);
    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    // just used for result comparisons
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    // requantize results
    aligned_vector<uint8_t> Cint8_mkl(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb(Cfp32_mkl.size());
    aligned_vector<uint8_t> Cint8_fb(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 50);
    int32_t Aint8_zero_point = 43;
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -8, 8);
    aligned_vector<int8_t> Bint8_copy(Bint8);
    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;

#ifdef USE_MKL
    ttot = 0.0;
    runType = "MKL_fp32";
    cout << setw(5) << m << ", " << setw(5) << n << ", " << setw(5) << k
         << ", ";
    cout << setw(30) << "NA";
    cout << ", ";

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
    cout << setw(16) << runType << ", " << fixed << setw(5) << setprecision(1)
         << nops / ttot << endl;

    Cint32_mkl.assign(Cfp32_mkl.begin(), Cfp32_mkl.end());
#endif

    for (BenchmarkType bench_type :
         {BenchmarkType::BARE_BONE,
          BenchmarkType::REQUANTIZATION,
          BenchmarkType::ROW_OFFSET_AND_REQUANTIZATION,
          BenchmarkType::EVERYTHING}) {
      // When we don't compute row_offset in fbgemm, we set B_zero_point to 0
      // to get the same result as the reference.
      int32_t Bint8_zero_point = (bench_type == BenchmarkType::BARE_BONE ||
                                  bench_type == BenchmarkType::REQUANTIZATION)
          ? 0
          : -30;

      // computing column offset
      vector<int32_t> col_offsets(n);
      Bint8 = Bint8_copy;
      col_offsets_with_zero_pt_s8acc32_ref(
          k, n, n, Bint8.data(), &Bint8_zero_point, col_offsets.data(), n);

      vector<int32_t> row_offsets(m);

      row_offsets_u8acc32_ref(m, k, k, Aint8.data(), row_offsets.data());

      float C_multiplier =
          (bench_type == BenchmarkType::BARE_BONE) ? 1.0f : 0.1234f;
      int32_t C_zero_pt = (bench_type == BenchmarkType::BARE_BONE) ? 0 : 5;

      // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
      // "A unpacked");
      // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n,
      // "B unpacked");
      // packedB.printPackedMatrix("B Packed");
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      double total_packing_time = 0.0;
      double total_computing_time = 0.0;
      double total_kernel_time = 0.0;
      double total_postprocessing_time = 0.0;
      double total_run_time = 0.0;
#endif

      cout << setw(5) << m << ", " << setw(5) << n << ", " << setw(5) << k
           << ", ";
      switch (bench_type) {
        case BenchmarkType::BARE_BONE:
          cout << setw(30) << "bare_bone";
          break;
        case BenchmarkType::REQUANTIZATION:
          cout << setw(30) << "requantization";
          break;
        case BenchmarkType::ROW_OFFSET_AND_REQUANTIZATION:
          cout << setw(30) << "row_offset_and_requantization";
          break;
        case BenchmarkType::EVERYTHING:
          cout << setw(30) << "everything";
          break;
      };
      cout << ", ";

      requantize_u8acc32_ref(
          m,
          n,
          n,
          Cint32_mkl.data(),
          Cint8_mkl.data(),
          &C_multiplier,
          C_zero_pt,
          Aint8_zero_point,
          &Bint8_zero_point,
          row_offsets.data(),
          col_offsets.data(),
          nullptr, // bias
          n); // ncols per quant group

      CompressedSparseColumn B_csc(k, n);

      float density = 0.001f;

      // deterministic random number
      default_random_engine eng;
      binomial_distribution<> per_col_nnz_dist(k, density);

      if (bench_type == BenchmarkType::EVERYTHING) {
        vector<int> row_indices(k);

        int total_nnz = 0;
        for (int j = 0; j < n; ++j) {
          B_csc.ColPtr()[j] = total_nnz;

          int nnz_of_j = per_col_nnz_dist(eng);
          total_nnz += nnz_of_j;

          iota(row_indices.begin(), row_indices.end(), 0);
          shuffle(row_indices.begin(), row_indices.end(), eng);
          sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

          for (int kidx = 0; kidx < nnz_of_j; ++kidx) {
            B_csc.RowIdx().push_back(row_indices[kidx]);
            // put the current B value
            B_csc.Values().push_back(Bint8[row_indices[kidx] * n + j]);
            // make current B value zero
            Bint8[row_indices[kidx] * n + j] = 0;
            // std::cout << "(" << row_indices[kidx] << ", " << j << ")" <<
            // endl;
          }
        }
        B_csc.ColPtr()[n] = total_nnz;
      }

      PackBMatrix<int8_t, int16_t> packedB(
          matrix_op_t::NoTranspose, k, n, Bint8.data(), n);

      // printMatrix(matrix_op_t::NoTranspose,
      // Cint32_mkl.data(), m, n, n, "C mkl");
      ttot = 0;
      runType = "FBGEMM_i8_acc16";
      for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        packing_time = 0.0;
        computing_time = 0.0;
        kernel_time = 0.0;
        postprocessing_time = 0.0;
        run_time = 0.0;
#endif
        llc_flush(llc);
        begin = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          vector<int32_t> row_offset_buf(
              PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

          PackAMatrix<uint8_t, int16_t> packA(
              matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);
          PackAWithRowOffset<uint8_t, int16_t> packAWithRowOffset(
              matrix_op_t::NoTranspose,
              m,
              k,
              Aint8.data(),
              k,
              nullptr,
              1,
              row_offset_buf.data());

          // no-op output process objects
          DoNothing<int32_t, int32_t> doNothing32BitObj;
          memCopy<> memcopyObj(doNothing32BitObj);

          // spmdm -> requantization -> nothing
          // construct an output processing pipeline in reverse order
          // i.e. last output operation first
          // Last operation should always be DoNothing with
          // correct input and output type.
          DoNothing<> doNothingObj{};
          // Requantization back to int8
          ReQuantizeOutput<false> reqObj(
              doNothingObj,
              &C_multiplier,
              C_zero_pt,
              Aint8_zero_point,
              &Bint8_zero_point,
              bench_type == BenchmarkType::REQUANTIZATION
                  ? nullptr
                  : packAWithRowOffset.getRowOffsetBuffer(),
              col_offsets.data(),
              nullptr,
              n);

          // the top most (first) operation in the output processing
          // pipeline is spmdm
          // outType = final output type after fullly processing through
          // pipeline; inType = initial input type at the first call to the
          // whole pipeline
          DoSpmdmOnInpBuffer<
              ReQuantizeOutput<false>::outType,
              int32_t,
              ReQuantizeOutput<false>>
              spmdmObj(reqObj, Aint8.data(), k, B_csc);

          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();
          // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
          switch (bench_type) {
            case BenchmarkType::BARE_BONE:
              fbgemmPacked(
                  packA,
                  packedB,
                  Cint32_fb.data(),
                  Cint32_fb.data(),
                  n,
                  memcopyObj,
                  tid,
                  num_threads);
              break;
            case BenchmarkType::REQUANTIZATION:
              fbgemmPacked(
                  packA,
                  packedB,
                  Cint8_fb.data(),
                  Cint32_fb.data(),
                  n,
                  reqObj,
                  tid,
                  num_threads);
              break;
            case BenchmarkType::ROW_OFFSET_AND_REQUANTIZATION:
              fbgemmPacked(
                  packAWithRowOffset,
                  packedB,
                  Cint8_fb.data(),
                  Cint32_fb.data(),
                  n,
                  reqObj,
                  tid,
                  num_threads);
              break;
            case BenchmarkType::EVERYTHING:
              fbgemmPacked(
                  packAWithRowOffset,
                  packedB,
                  Cint8_fb.data(),
                  Cint32_fb.data(),
                  n,
                  spmdmObj,
                  tid,
                  num_threads);
              break;
          };
        }

        end = chrono::high_resolution_clock::now();

        if (i >= NWARMUP) {
          auto dur = chrono::duration_cast<chrono::nanoseconds>(end - begin);
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
      cout << fixed << total_packing_time / (double)NITER / 1e6 << ", "
           << total_kernel_time / (double)NITER / 1e6 << ", "
           << total_postprocessing_time / (double)NITER / 1e6 << ", "
           << total_run_time / (double)NITER / 1e6 << ", ";
#endif
      cout << setw(16) << runType << ", " << fixed << setw(5) << setprecision(1)
           << NITER * nops / ttot << endl;

#ifdef USE_MKL
      if (bench_type == BenchmarkType::BARE_BONE) {
        compare_buffers(Cint32_mkl.data(), Cint32_fb.data(), m, n, n, 5);
      } else {
        compare_buffers(Cint8_mkl.data(), Cint8_fb.data(), m, n, n, 5);
      }
#endif
    } // test_outlier
    cout << endl;
  } // shapes
}

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

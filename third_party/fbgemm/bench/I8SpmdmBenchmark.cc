/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/FbgemmI8Spmdm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif

  const vector<array<int, 3>> shapes = {
      //   M,    N,    K
      {1024, 1024, 1024},
      {511, 512, 512},
  };

  // SpMDM is often memory BW bound so we want to flush LLC.
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 16;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << "M, "
       << "N, "
       << "K, "
       << "Density, "
       << "Accumulation, "
       << "Initialize (ms), "
       << "Transpose uint8 (ms), "
       << "Transpose 32xN (ms), "
       << "Compute (ms), "
       << "Transpose 32xN (ms), "
       << "Total (ms), "
       << "GB/s, "
       << "GOPs" << endl;
#else
  cout << "M, "
       << "N, "
       << "K, "
       << "Density, "
       << "Accumulation, "
       << "GB/s, "
       << "GOPs" << endl;
#endif

  for (const auto& shape : shapes) {
    for (float density : {0.00001f, 0.0001f, 0.001f, 0.01f, 0.1f, 1.0f}) {
      for (bool accumulation : {false, true}) {
        int M = shape[0];
        int N = shape[1];
        int K = shape[2];

        cout << M << ", " << N << ", " << K << ", ";

        aligned_vector<uint8_t> A(M * K);
        randFill<uint8_t>(A, 0, 255);

        fbgemm::CompressedSparseColumn B_csc(K, N);
        vector<int32_t> C(M * N);
        vector<int32_t> C_ref(C.size());

        for (int i = 0; i < M; ++i) {
          for (int j = 0; j < N; ++j) {
            C_ref[i * N + j] = i + j;
          }
        }

        // deterministic random number
        std::default_random_engine eng;
        binomial_distribution<> per_col_nnz_dist(K, density);
        uniform_int_distribution<> value_dist(
            numeric_limits<int8_t>::min() / 2,
            numeric_limits<int8_t>::max() / 2);

        vector<int> row_indices(K);

        int total_nnz = 0;
        for (int j = 0; j < N; ++j) {
          B_csc.ColPtr()[j] = total_nnz;

          int nnz_of_j = per_col_nnz_dist(eng);
          total_nnz += nnz_of_j;

          iota(row_indices.begin(), row_indices.end(), 0);
          shuffle(row_indices.begin(), row_indices.end(), eng);
          sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

          for (int k = 0; k < nnz_of_j; ++k) {
            B_csc.RowIdx().push_back(row_indices[k]);
            B_csc.Values().push_back(value_dist(eng));
          }
        }
        B_csc.ColPtr()[N] = total_nnz;

        double ttot = 0;
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        double total_initial_time = 0.0;
        double total_transpose_uint8_time = 0.0;
        double total_transpose_32xN_time = 0.0;
        double total_compute_time = 0.0;
        double total_transpose_Nx32_time = 0.0;
        double total_run_time = 0.0;
#endif
        double ops = double(NITER) * B_csc.NumOfNonZeros() * M * 2;
        double bytes = double(NITER) *
            (M * N * sizeof(int32_t) + M * K +
             B_csc.NumOfNonZeros() * (sizeof(int16_t) + sizeof(int8_t)) +
             B_csc.ColPtr().size() * sizeof(int32_t));

        spmdm_ref(M, A.data(), K, B_csc, accumulation, C_ref.data(), N);

        chrono::time_point<chrono::system_clock> t_begin, t_end;
        for (int iter = 0; iter < NWARMUP + NITER; ++iter) {
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
              C[i * N + j] = i + j;
            }
          }
          llc_flush(llc);

          t_begin = chrono::system_clock::now();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
          spmdm_initial_time = 0.0;
          spmdm_transpose_uint8_time = 0.0;
          spmdm_transpose_32xN_time = 0.0;
          spmdm_compute_time = 0.0;
          spmdm_transpose_Nx32_time = 0.0;
          spmdm_run_time = 0.0;
#endif

#ifndef FBGEMM_MEASURE_TIME_BREAKDOWN
#pragma omp parallel
#endif
          {
            int num_threads = fbgemm_get_num_threads();
            int tid = fbgemm_get_thread_num();
            int i_per_thread =
                ((M + 31) / 32 + num_threads - 1) / num_threads * 32;
            int i_begin = std::min(tid * i_per_thread, M);
            int i_end = std::min(i_begin + i_per_thread, M);

            block_type_t block = {i_begin, i_end - i_begin, 0, N};
            B_csc.SpMDM(
                block, A.data(), K, accumulation, C.data() + i_begin * N, N);
          }
          t_end = chrono::system_clock::now();
          if (iter >= NWARMUP) {
            double dt = chrono::duration<double>(t_end - t_begin).count();
            // double dt = chrono::duration_cast<chrono::nanoseconds>(t_end -
            // t_begin).count();
            ttot += dt;
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
            total_initial_time += spmdm_initial_time;
            total_transpose_uint8_time += spmdm_transpose_uint8_time;
            total_transpose_32xN_time += spmdm_transpose_32xN_time;
            total_compute_time += spmdm_compute_time;
            total_transpose_Nx32_time += spmdm_transpose_Nx32_time;
            total_run_time += spmdm_run_time;
#endif
          }
        }

        compare_buffers(C_ref.data(), C.data(), M, N, N, 5, 0);

        cout << fixed << B_csc.Density() << ", " << accumulation << ", ";

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        cout << fixed << total_initial_time / (double)NITER / 1e6 << ", "
             << total_transpose_uint8_time / (double)NITER / 1e6 << ", "
             << total_transpose_32xN_time / (double)NITER / 1e6 << ", "
             << total_compute_time / (double)NITER / 1e6 << ", "
             << total_transpose_Nx32_time / (double)NITER / 1e6 << ", "
             << total_run_time / (double)NITER / 1e6 << ", ";
#endif
        // Report performance
        cout << fixed << bytes / ttot / 1e9 << ", " << ops / ttot / 1e9 << endl;

      } // accumulation
    } // for each density
  } // for each shape

  return 0;
}

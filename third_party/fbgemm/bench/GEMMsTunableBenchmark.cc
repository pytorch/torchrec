/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/QuantizationHelpers.h"

using namespace std;
using namespace fbgemm;

void performance_test(
    const BlockingFactors* tuning_params,
    set<vector<int>>& incorrect_configs,
    const vector<int>& shape,
    array<int, 6>& best_config,
    float& giga_ops) {
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
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(18) << "Packing (us), " << setw(18)
       << "Kernel (us), " << setw(18) << "Postproc (us), " << setw(18)
       << "Total (us), " << setw(5) << "GOPs" << endl;
#else
#endif

  chrono::time_point<chrono::high_resolution_clock> start, end;

  int m = shape[0];
  int n = shape[1];
  int k = shape[2];

  aligned_vector<uint8_t> Aint8(m * k);
  aligned_vector<int8_t> Bint8(k * n);
  aligned_vector<int32_t> Cint32_ref(m * n);
  aligned_vector<int32_t> Cint32_fb_acc32(Cint32_ref.size());
  aligned_vector<int32_t> Cint32_fb_acc16(Cint32_ref.size());

  // A matrix
  randFill<uint8_t>(Aint8, 0, 5);
  aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

  randFill<int8_t>(Bint8, -4, 4);
  avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

  aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

  double nops = 2.0 * static_cast<double>(NITER) * m * n * k;
  double ttot = 0.0;
  string runType;

  vector<int32_t> row_offsets(m);

  matmul_u8i8acc32_ref(
      m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_ref.data());

  PackBMatrix<int8_t> packedB_int32(
      matrix_op_t::NoTranspose,
      k,
      n,
      Bint8.data(),
      n,
      nullptr,
      1,
      tuning_params);

  ttot = 0.0;
  runType = "FBGEMM_i8_acc32";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  double total_packing_time = 0.0;
  double total_computing_time = 0.0;
  double total_kernel_time = 0.0;
  double total_postprocessing_time = 0.0;
  double total_run_time = 0.0;
#endif

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
      PackAMatrix<uint8_t> packA_int32(
          matrix_op_t::NoTranspose,
          m,
          k,
          Aint8.data(),
          k,
          nullptr,
          1,
          tuning_params);

      DoNothing<int32_t, int32_t> doNothing32BitObj;
      memCopy<> memcopyObj(doNothing32BitObj);
      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();
      // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
      fbgemmPacked(
          packA_int32,
          packedB_int32,
          Cint32_fb_acc32.data(),
          Cint32_fb_acc32.data(),
          n,
          memcopyObj,
          tid,
          num_threads,
          tuning_params);
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

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3 << ", "
       << setw(16) << total_kernel_time / (double)NITER / 1e3 << ", "
       << setw(16) << total_postprocessing_time / (double)NITER / 1e3 << ", "
       << setw(16) << total_run_time / (double)NITER / 1e3;
#endif

  if (compare_buffers(Cint32_ref.data(), Cint32_fb_acc32.data(), m, n, n, 5)) {
    vector<int> config = {tuning_params->MCB,
                          tuning_params->NCB,
                          tuning_params->KCB,
                          tuning_params->MR,
                          tuning_params->NR,
                          tuning_params->ROW_INTERLEAVE};
    incorrect_configs.insert(config);
  } else {
    cout << setw(5) << "MCB, " << setw(5) << "NCB, " << setw(5) << "KCB, "
         << setw(5) << "MR, " << setw(5) << "NR, " << setw(5) << "ROW INT."
         << endl;
    cout << setw(5) << tuning_params->MCB << setw(5) << tuning_params->NCB
         << setw(5) << tuning_params->KCB << setw(5) << tuning_params->MR
         << setw(5) << tuning_params->NR << setw(5)
         << tuning_params->ROW_INTERLEAVE << endl;

    cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
         << "Type, " << setw(5) << "GOPS" << endl;
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;
    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << nops / ttot << endl;
    if ((nops / ttot) > giga_ops) {
      giga_ops = nops / ttot;
      best_config = {tuning_params->MCB,
                     tuning_params->NCB,
                     tuning_params->KCB,
                     tuning_params->MR,
                     tuning_params->NR,
                     tuning_params->ROW_INTERLEAVE};
    }
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

  // clang-format off
  vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    //warning these take time to run!
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
    {64, 800, 320},
    {64, 768, 512},
    {16, 256, 512},
    {128, 128, 128},
    {256, 512, 256},
    {1024, 1024, 1024},
  };
  // clang-format on

  vector<int> MCBs;
  vector<int> NCBs;
  vector<int> KCBs;
  vector<int> MRs;
  int NR = 16;
  int NR_MIN = 16;
  int ROW_INTERLEAVE = 4; // do 32-bit accumulation for now

  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      NR = 16;
      MCBs.insert(MCBs.end(), {48, 96, 144, 192, 240});
      NCBs.insert(NCBs.end(), {16, 32, 64, 128, 48, 98, 192, 384});
      KCBs.insert(
          KCBs.end(),
          {256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024});
      MRs.insert(MRs.end(), {24, 12, 6, 3, 8, 4, 2, 1});
    } else if (fbgemmHasAvx2Support()) {
      assert(0 && "Benchmark will be extended for this architecture");
    } else {
      assert(0 && "architecture not supported");
      return 0;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  set<vector<int>> incorrect_configs;
  float giga_ops = 0.0;
  array<int, 6> best_config = {0, 0, 0, 0, 0, 0};
  BlockingFactors params;
  for (auto const& shape : shapes) {
    for (auto const& mcb : MCBs) {
      for (auto const& ncb : NCBs) {
        for (auto const& kcb : KCBs) {
          for (auto const& mr : MRs) {
            params.MCB = mcb;
            params.NCB = ncb;
            params.KCB = kcb;
            params.MR = mr;
            params.NR = NR;
            params.ROW_INTERLEAVE = ROW_INTERLEAVE;
            params.NR_MIN = NR_MIN;
            if (isValidBlockingFactor<int32_t>(&params)) {
              performance_test(
                  &params, incorrect_configs, shape, best_config, giga_ops);
            }
          }
        }
      }
    }
    cout << endl << "This is the Best Config!" << endl;
    cout << setw(5) << "MCB, " << setw(5) << "NCB, " << setw(5) << "KCB, "
         << setw(5) << "MR, " << setw(5) << "NR, " << setw(5) << "ROW INT."
         << setw(5) << "GOPS" << endl;
    cout << setw(5) << best_config[0] << setw(5) << best_config[1] << setw(5)
         << best_config[2] << setw(5) << best_config[3] << setw(5)
         << best_config[4] << setw(5) << best_config[5] << giga_ops << endl;
  } // end shapes

  cout << endl << "Warning there are configs that didn't work!" << endl;
  for (auto const& entry : incorrect_configs) {
    cout << setw(5) << "MCB, " << setw(5) << "NCB, " << setw(5) << "KCB, "
         << setw(5) << "MR, " << setw(5) << "NR, " << setw(5) << "ROW INT."
         << endl;
    cout << setw(5) << entry[0] << setw(5) << entry[1] << setw(5) << entry[2]
         << setw(5) << entry[3] << setw(5) << entry[4] << setw(5) << entry[5]
         << endl;
  }
  return 0;
}

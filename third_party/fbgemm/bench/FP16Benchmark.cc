/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <cmath>
#include <memory>
#include <random>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "fbgemm/FbgemmFP16.h"
#include "bench/BenchUtils.h"

using namespace fbgemm;

int main(int argc, const char* argv[]) {
  int num_instances = 1;
#ifdef _OPENMP
  const char* inst = getenv("GEMMBENCH_NUM_INSTANCES");
  if (inst != nullptr && *inst) {
    num_instances = std::max(atoi(inst), num_instances);
  }
  num_instances =
      parseArgumentInt(argc, argv, "--inst=", num_instances, num_instances);
  printf("Running %d instances\n", num_instances);
  if (num_instances > 1) {
    // Set-up execution for multi-instance mode
    // Number of threads in OpenMP parallel region is explicitly
    // set to the number of instances to be executed.
    omp_set_num_threads(num_instances);
#ifdef USE_MKL
    // each instance should be run with a single thread
    mkl_set_num_threads(1);
#endif
  } else {
    // When running single instance use OMP_NUM_THREADS to determine
    // parallelism. Default behaviour is using a single thread.
    int num_threads = parseArgumentInt(argc, argv, "--num_threads=", 1, 1);
    const char* val = getenv("OMP_NUM_THREADS");
    if (val == nullptr || !*val) {
      omp_set_num_threads(num_threads);
    }
  }

#endif

  int repetitions = parseArgumentInt(argc, argv, "--repit=", 1, 1);
  bool no_flush = parseArgumentBool(argc, argv, "--no-flush", false);
  bool no_mkl = parseArgumentBool(argc, argv, "--no-mkl", false);
  bool enable_avx512_ymm = parseArgumentBool(argc, argv, "--avx512-256", false);
  fbgemmEnableAvx512Ymm(enable_avx512_ymm);
  performance_test<float16>(num_instances, !no_flush, repetitions, !no_mkl);
}

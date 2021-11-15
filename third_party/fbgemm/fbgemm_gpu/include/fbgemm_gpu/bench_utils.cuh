/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <vector>
#include "./cuda_utils.cuh"

__global__ void flush_gpu(char* d_flush, char* d_flush2, bool do_write) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  char val = d_flush[idx];
  if (do_write * val) {
    d_flush2[idx] = val;
  }
}

void flush_cache(int cache_size_mb = 40, bool do_write = false) {
  const int cache_size = cache_size_mb * 1024 * 1024; // A100 40MB L2 cache
  std::vector<char> flush(cache_size, (char)255);
  char* d_flush;
  char* d_flush2;
  CUDA_CHECK(cudaMalloc(&d_flush, cache_size));
  CUDA_CHECK(cudaMalloc(&d_flush2, cache_size));
  CUDA_CHECK(
      cudaMemcpy(d_flush, flush.data(), cache_size, cudaMemcpyHostToDevice));
  flush_gpu<<<cache_size / 512, 512>>>(d_flush, d_flush2, do_write);
  cudaFree(d_flush);
  cudaFree(d_flush2);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}

void generate_random_table(float* d_f32_table, unsigned size) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Random seed
  curandGenerateUniform(gen, d_f32_table, size);
  CUDA_CHECK(cudaGetLastError());
  curandDestroyGenerator(gen);
}

template <typename Lambda>
float benchmark_function(int iters, Lambda&& f) {
  float elapsed = 0;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < iters; i++) {
    float local_elapsed = 0;
    flush_cache(40); // A100 40MB L2 cache
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(start, 0));

    // kernel launch here
    f();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&local_elapsed, start, stop));
    CUDA_CHECK(cudaGetLastError());
    elapsed += local_elapsed;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / iters;
}

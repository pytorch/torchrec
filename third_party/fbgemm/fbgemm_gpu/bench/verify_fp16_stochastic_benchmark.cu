/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <unistd.h>
#include <vector>

__device__ half float_to_sto_half_direct(float w) {
  curandState_t state;
  curand_init((unsigned long long)(w * 100), 0, 0, &state);
  half up = __float2half_ru(w);
  half down = __float2half_rd(w);
  const float up_f32 = __half2float(up);
  const float down_f32 = __half2float(down);
  // 1 - (w - w_down) / (w_up - w_down) = (w_up - w) / (w_up - w_down) = n / m
  const float m = (up_f32 - down_f32);
  const float rand = curand_uniform(&state);
  if (__float_as_uint(m) == 0) {
    return up;
  }
  const float n = (up_f32 - w);
  return rand > n / m ? up : down;
}

__device__ float two_to_e(float X) {
  const float Y = 16777216 * X; // 2^24
  const float U = ((Y + X) - Y) * 0.5;
  return U == 0 ? X : U;
}

__device__ half float_to_sto_half_bitcarry(float w) {
  curandState_t state;
  curand_init((unsigned long long)(w * 100), 0, 0, &state);
  float rand = curand_uniform(&state);
  float rand_match_w = two_to_e(w) * rand * 0.0009765625; // 2^(-10)
  float Z = w + rand_match_w;
  return __float2half_rz(Z);
}

__device__ half float_to_sto_half_shortrand(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned w_new = w_int + (rand << 5);
  return __float2half_rz(__uint_as_float(w_new));
}

__device__ half float_to_sto_half_assemblefloat(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned assmebles = (w_int & 0xff800000) | (rand << 5);
  const unsigned subtract = (w_int & 0xff800000);
  const float assmeble_float = __uint_as_float(assmebles) - __uint_as_float(subtract);
  return __float2half_rz(w + assmeble_float);
}

__global__ void convert_float_to_half_direct(half* dst, float* src, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_direct(src[idx]);
  }
}

__global__ void
convert_float_to_half_bitcarry(half* dst, float* src, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_bitcarry(src[idx]);
  }
}

__global__ void
convert_float_to_half_shortrand(half* dst, float* src, uint8_t* r, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_shortrand(src[idx], r[idx]);
  }
}

__global__ void convert_float_to_half_assemblefloat(
    half* dst,
    float* src,
    uint8_t* r,
    int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_assemblefloat(src[idx], r[idx]);
  }
}

void gen_data(float* d_f32_array, int test_size) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Random seed
  curandGenerateUniform(gen, d_f32_array, test_size);
  curandDestroyGenerator(gen);
  cudaDeviceSynchronize();
}

// generate 64bit random number and then copy back to 8bit memory
void gen_8bit_random(uint8_t* d_random_number, int test_size) {
  curandGenerator_t gen;
  unsigned* d_random_number_f32;
  cudaMalloc(
      &d_random_number_f32,
      (test_size / sizeof(unsigned) + 1) * sizeof(unsigned));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, 5678ULL); // Random seed
  curandGenerate(gen, d_random_number_f32, (test_size / sizeof(unsigned) + 1));
  cudaMemcpy(
      d_random_number,
      d_random_number_f32,
      test_size * sizeof(uint8_t),
      cudaMemcpyDeviceToDevice);
  curandDestroyGenerator(gen);
  cudaFree(d_random_number_f32);
}

__global__ void flush_gpu(char* d_flush, char* d_flush2, bool do_write) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const char val = d_flush[idx];
  if (do_write * val) {
    d_flush2[idx] = val;
  }
}

void flush_cache(
    std::vector<char> flush,
    char* d_flush,
    char* d_flush2,
    int cache_size,
    bool do_write = false) {
  cudaMemcpy(d_flush, flush.data(), cache_size, cudaMemcpyHostToDevice);
  const unsigned num_blocks = cache_size / 512;
  flush_gpu<<<num_blocks, 512>>>(d_flush, d_flush2, do_write);
  cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
  std::vector<float> f32_array;
  std::vector<half> f16_direct_array;
  std::vector<half> f16_bitcarry_array;
  std::vector<half> f16_shortrand_array;
  std::vector<half> f16_assemblefloat_array;
  float* d_f32_array;
  half* d_f16_direct_array;
  half* d_f16_bitcarry_array;
  half* d_f16_shortrand_array;
  half* d_f16_assemblefloat_array;
  uint8_t* d_random_number;

  std::vector<char> flush;
  char* d_flush;
  char* d_flush2;

  int test_size = 10;
  bool verbose = false;
  int opt;
  while ((opt = getopt(argc, argv, "n:v")) != -1) {
    switch (opt) {
      case 'n':
        test_size = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
    }
  }

  std::cout << "Start stochastic algorithm tests with test_size = " << test_size
            << std::endl;
  constexpr int cache_size = 40 * 1024 * 1024; // A100 40MB L2 cache

  f32_array.reserve(test_size);
  f16_direct_array.reserve(test_size);
  f16_bitcarry_array.reserve(test_size);
  f16_shortrand_array.reserve(test_size);
  f16_assemblefloat_array.reserve(test_size);
  cudaMalloc(&d_f32_array, test_size * sizeof(float));
  cudaMalloc(&d_f16_direct_array, test_size * sizeof(half));
  cudaMalloc(&d_f16_bitcarry_array, test_size * sizeof(half));
  cudaMalloc(&d_f16_shortrand_array, test_size * sizeof(half));
  cudaMalloc(&d_f16_assemblefloat_array, test_size * sizeof(half));
  cudaMalloc(&d_random_number, test_size * sizeof(uint8_t));

  flush.assign(cache_size, 255);
  cudaMalloc(&d_flush, cache_size * sizeof(char));
  cudaMalloc(&d_flush2, cache_size * sizeof(char));

  gen_data(d_f32_array, test_size);
  gen_8bit_random(d_random_number, test_size);

  constexpr int block_size = 128;
  const int num_blocks = (test_size + block_size - 1) / block_size;

  flush_cache(flush, d_flush, d_flush2, cache_size);
  std::cout << "Starting algorithm direct..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  convert_float_to_half_direct<<<num_blocks, block_size>>>(
      d_f16_direct_array, d_f32_array, test_size);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "Cuda failure: " << cudaGetErrorString(e) << std::endl;
    exit(-1);
  }
  std::chrono::duration<double> time = end - start;
  std::cout << "Direct stochastic algorithm runs: " << time.count() << " sec "
            << std::endl;

  flush_cache(flush, d_flush, d_flush2, cache_size);
  std::cout << "Starting algorithm bitcarry..." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  convert_float_to_half_bitcarry<<<num_blocks, block_size>>>(
      d_f16_bitcarry_array, d_f32_array, test_size);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "Cuda failure: " << cudaGetErrorString(e) << std::endl;
    exit(-1);
  }
  time = end - start;
  std::cout << "Bitcarry stochastic algorithm runs: " << time.count() << " sec"
            << std::endl;

  flush_cache(flush, d_flush, d_flush2, cache_size);
  std::cout << "Starting algorithm shortrand..." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  convert_float_to_half_shortrand<<<num_blocks, block_size>>>(
      d_f16_shortrand_array, d_f32_array, d_random_number, test_size);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "Cuda failure: " << cudaGetErrorString(e) << std::endl;
    exit(-1);
  }
  time = end - start;
  std::cout << "Shortrand stochastic algorithm runs: " << time.count() << " sec"
            << std::endl;

  flush_cache(flush, d_flush, d_flush2, cache_size);
  std::cout << "Starting algorithm assemblefloat..." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  convert_float_to_half_assemblefloat<<<num_blocks, block_size>>>(
      d_f16_assemblefloat_array, d_f32_array, d_random_number, test_size);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "Cuda failure: " << cudaGetErrorString(e) << std::endl;
    exit(-1);
  }
  time = end - start;
  std::cout << "Assemblefloat stochastic algorithm runs: " << time.count()
            << " sec" << std::endl;

  if (verbose) {
    cudaMemcpy(
        f32_array.data(),
        d_f32_array,
        test_size * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        f16_direct_array.data(),
        d_f16_direct_array,
        test_size * sizeof(half),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        f16_bitcarry_array.data(),
        d_f16_bitcarry_array,
        test_size * sizeof(half),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        f16_shortrand_array.data(),
        d_f16_shortrand_array,
        test_size * sizeof(half),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        f16_assemblefloat_array.data(),
        d_f16_assemblefloat_array,
        test_size * sizeof(half),
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < test_size; i++) {
      std::cout << std::hexfloat << f32_array[i] << ":\t(up:" << std::hexfloat
                << __half2float(__float2half_ru(f32_array[i]))
                << "\tdown:" << std::hexfloat
                << __half2float(__float2half_rd(f32_array[i]))
                << ") \tdirect: " << std::hexfloat
                << __half2float(f16_direct_array[i])
                << "\tbitcarry: " << std::hexfloat
                << __half2float(f16_bitcarry_array[i])
                << " \tshortrand: " << std::hexfloat
                << __half2float(f16_shortrand_array[i])
                << " \tassemblefloat: " << std::hexfloat
                << __half2float(f16_assemblefloat_array[i]) << std::endl;
    }
  }

  cudaFree(d_f32_array);
  cudaFree(d_f16_direct_array);
  cudaFree(d_f16_bitcarry_array);
  cudaFree(d_f16_shortrand_array);
  cudaFree(d_f16_assemblefloat_array);

  return 0;
}

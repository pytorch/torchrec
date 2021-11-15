/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./BenchUtils.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fbgemm {

std::default_random_engine eng;

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::true_type) {
  std::uniform_int_distribution<int> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::false_type) {
  std::uniform_real_distribution<T> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high) {
  randFill(vec, low, high, std::is_integral<T>());
}

template void
randFill<float>(aligned_vector<float>& vec, float low, float high);
template void
randFill<uint8_t>(aligned_vector<uint8_t>& vec, uint8_t low, uint8_t high);
template void
randFill<int8_t>(aligned_vector<int8_t>& vec, int8_t low, int8_t high);
template void randFill<int>(aligned_vector<int>& vec, int low, int high);
// template void
// randFill<int64_t>(aligned_vector<int64_t>& vec, int64_t low, int64_t high);
template <>
void randFill(aligned_vector<int64_t>& vec, int64_t low, int64_t high) {
  std::uniform_int_distribution<int64_t> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (auto i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}

int fbgemm_get_max_threads() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 1;
#else
  return omp_get_max_threads();
#endif
}

int fbgemm_get_num_threads() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 1;
#else
  return omp_get_num_threads();
#endif
}

int fbgemm_get_thread_num() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 0;
#else
  return omp_get_thread_num();
#endif
}

int parseArgumentInt(
    int argc,
    const char* argv[],
    const char* arg,
    int non_exist_val,
    int def_val) {
  int val = non_exist_val;
  int arg_len = strlen(arg);
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      int res = atoi(ptr + arg_len);
      val = (*(ptr + arg_len - 1) == '=') ? res : def_val;
      break;
    }
  }
  return val;
}

bool parseArgumentBool(
    int argc,
    const char* argv[],
    const char* arg,
    bool def_val) {
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      return true;
    }
  }
  return def_val;
}

#if defined(USE_MKL)
void test_xerbla(char* srname, const int* info, int) {
  // srname - name of the function that called xerbla
  // info - position of the invalid parameter in the parameter list
  // len - length of the name in bytes
  printf("\nXERBLA(MKL Error) is called :%s: %d\n", srname, *info);
}
#endif

aligned_vector<float> getRandomSparseVector(
    unsigned size,
    float fractionNonZeros) {
  aligned_vector<float> res(size);

  std::mt19937 gen(345);

  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (auto& f : res) {
    f = dis(gen);
  }

  // Create exactly fractionNonZeros in result
  aligned_vector<float> sorted_res(res);
  std::sort(sorted_res.begin(), sorted_res.end());
  int32_t numZeros =
      size - static_cast<int32_t>(std::round(size * fractionNonZeros));
  float thr;
  if (numZeros) {
    thr = sorted_res[numZeros - 1];

    for (auto& f : res) {
      if (f <= thr) {
        f = 0.0f;
      }
    }
  }

  return res;
}

template <typename T>
aligned_vector<T> getRandomBlockSparseMatrix(
    int Rows,
    int Cols,
    float fractionNonZerosBlocks,
    int RowBlockSize,
    int ColBlockSize,
    T low,
    T high) {
  aligned_vector<T> res(Rows * Cols, 0);

  std::mt19937 gen(345);

  std::uniform_int_distribution<int> dis(low, high);
  std::bernoulli_distribution bernDis{fractionNonZerosBlocks};

  int rowBlocks = (Rows + RowBlockSize - 1) / RowBlockSize;
  int colBlocks = (Cols + ColBlockSize - 1) / ColBlockSize;

  int fnzb = 0;

  for (int i = 0; i < rowBlocks; ++i) {
    for (int j = 0; j < colBlocks; ++j) {
      if (bernDis(gen)) {
        // fill in this block
        for (int i_b = 0; i_b < std::min(RowBlockSize, Rows - i * RowBlockSize);
             ++i_b) {
          for (int j_b = 0;
               j_b < std::min(ColBlockSize, Cols - j * ColBlockSize);
               ++j_b) {
            res[(i * RowBlockSize + i_b) * Cols + j * ColBlockSize + j_b] =
                dis(gen);
          }
        }
        fnzb++;
      }
    }
  }

  // std::cout << "Requested non-zero fraction: " << fractionNonZerosBlocks
  // << " , generated non-zero fraction: "
  // << static_cast<float>(fnzb) / rowBlocks / colBlocks << std::endl;
  // std::cout << "Requested non-zero blocks: "
  // << rowBlocks * colBlocks * fractionNonZerosBlocks
  // << ", generated non-zero blocks: " << fnzb << std::endl;

  return res;
}

template aligned_vector<uint8_t> getRandomBlockSparseMatrix(
    int Rows,
    int Cols,
    float fractionNonZerosBlocks,
    int RowBlockSize,
    int ColBlockSize,
    uint8_t low,
    uint8_t high);
template aligned_vector<int8_t> getRandomBlockSparseMatrix(
    int Rows,
    int Cols,
    float fractionNonZerosBlocks,
    int RowBlockSize,
    int ColBlockSize,
    int8_t low,
    int8_t high);
template aligned_vector<int32_t> getRandomBlockSparseMatrix(
    int Rows,
    int Cols,
    float fractionNonZerosBlocks,
    int RowBlockSize,
    int ColBlockSize,
    int32_t low,
    int32_t high);


} // namespace fbgemm

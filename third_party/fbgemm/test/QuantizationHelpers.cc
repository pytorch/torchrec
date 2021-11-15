/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./QuantizationHelpers.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using namespace std;

namespace fbgemm {
/*
 * @brief Make sure we won't have overflows from vpmaddubsw instruction.
 */
template <typename T>
void avoidOverflow(
    int m,
    int n,
    int k,
    const uint8_t* Aint8,
    int lda,
    T* B,
    int ldb) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int kk = 0; kk < k / 2 * 2; kk += 2) {
        int a0 = Aint8[i * lda + kk], a1 = Aint8[i * lda + kk + 1];
        int b0 = B[kk * ldb + j], b1 = B[(kk + 1) * ldb + j];
        int sum_pair = a0 * b0 + a1 * b1;
        if (sum_pair < numeric_limits<int16_t>::lowest()) {
          int b1_adjusted =
              ceil((numeric_limits<int16_t>::lowest() - a0 * b0) / a1);
          b1_adjusted = std::min(std::max(b1_adjusted, -128), 127);

          int new_sum_pair = a0 * b0 + a1 * b1_adjusted;
          assert(
              new_sum_pair >= numeric_limits<int16_t>::lowest() &&
              new_sum_pair <= numeric_limits<int16_t>::max());
          B[(kk + 1) * n + j] = b1_adjusted;
        } else if (sum_pair > numeric_limits<int16_t>::max()) {
          int b1_adjusted =
              floor((numeric_limits<int16_t>::max() - a0 * b0) / a1);
          b1_adjusted = std::min(std::max(b1_adjusted, -128), 127);

          int new_sum_pair = a0 * b0 + a1 * b1_adjusted;
          assert(
              new_sum_pair >= numeric_limits<int16_t>::lowest() &&
              new_sum_pair <= numeric_limits<int16_t>::max());
          B[(kk + 1) * ldb + j] = b1_adjusted;
        }
      }
    } // for each j
  } // for each i
}

template <typename T>
void avoidOverflow(int m, int n, int k, const uint8_t* Aint8, T* B) {
  return avoidOverflow(m, n, k, Aint8, k, B, n);
}

template void avoidOverflow(
    int m,
    int n,
    int k,
    const uint8_t* Aint8,
    int lda,
    int8_t* B,
    int ldb);
template void avoidOverflow(
    int m,
    int n,
    int k,
    const uint8_t* Aint8,
    int lda,
    float* B,
    int ldb);

template void
avoidOverflow(int m, int n, int k, const uint8_t* Aint8, int8_t* B);
template void
avoidOverflow(int m, int n, int k, const uint8_t* Aint8, float* B);
} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "./TransposeUtils.h"
#include "fbgemm/Utils.h"
#include <cstring>

namespace fbgemm {

template <typename T>
void transpose_ref(unsigned M, unsigned N, const T* src, unsigned ld_src, T* dst, unsigned ld_dst) {
  for (unsigned j = 0; j < N; j++) {
    for (unsigned i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

template <typename T>
void transpose_simd(
    unsigned M,
    unsigned N,
    const T* src,
    unsigned ld_src,
    T* dst,
    unsigned ld_dst) {
  if ((M == 1 && ld_dst == 1) || (N == 1 && ld_src == 1)) {
    if (dst != src) {
      // sizeof must be first operand force dims promotion to OS-bitness type
      memcpy(dst, src, sizeof(T) * M * N);
    }
    return;
  }
  static const auto iset = fbgemmInstructionSet();
  // Run time CPU detection
  if (isZmm(iset)) {
    internal::transpose_avx512<T>(M, N, src, ld_src, dst, ld_dst);
  } else if (isYmm(iset)) {
    internal::transpose_avx2<T>(M, N, src, ld_src, dst, ld_dst);
  } else {
    transpose_ref<T>(M, N, src, ld_src, dst, ld_dst);
  }
}

template void transpose_ref<float>(
    unsigned M,
    unsigned N,
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst);

template void transpose_ref<uint8_t>(
    unsigned M,
    unsigned N,
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst);

template FBGEMM_API void transpose_simd<float>(
    unsigned M,
    unsigned N,
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst);

template FBGEMM_API void transpose_simd<uint8_t>(
    unsigned M,
    unsigned N,
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst);

} // namespace fbgemm

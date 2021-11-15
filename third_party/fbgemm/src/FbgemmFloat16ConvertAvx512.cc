/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

namespace {

inline void FloatToFloat16KernelAvx512(const float* src, float16* dst) {
  __m512 float_vector = _mm512_loadu_ps(src);
  __m256i half_vector = _mm512_cvtps_ph(
      float_vector, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm256_storeu_si256((__m256i*)dst, half_vector);
}

inline void FloatToFloat16KernelAvx512WithClip(const float* src, float16* dst) {
  constexpr float FP16_MAX = 65504.f;
  __m512 neg_fp16_max_vector = _mm512_set1_ps(-FP16_MAX);
  __m512 pos_fp16_max_vector = _mm512_set1_ps(FP16_MAX);

  __m512 float_vector = _mm512_loadu_ps(src);

  // Do the clipping.
  float_vector = _mm512_max_ps(
      neg_fp16_max_vector, _mm512_min_ps(float_vector, pos_fp16_max_vector));

  __m256i half_vector = _mm512_cvtps_ph(
      float_vector, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm256_storeu_si256((__m256i*)dst, half_vector);
}

inline void Float16ToFloatKernelAvx512(const float16* src, float* dst) {
  __m256i half_vector = _mm256_loadu_si256((__m256i*)src);
  __m512 float_vector = _mm512_cvtph_ps(half_vector);
  _mm512_storeu_ps(dst, float_vector);
}

} // namespace

void FloatToFloat16_avx512(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  if (do_clip) {
    size_t i = 0;
    for (i = 0; i + 16 <= size; i += 16) {
      FloatToFloat16KernelAvx512WithClip(src + i, dst + i);
    }
    FloatToFloat16_avx2(src + i, dst + i, size - i, do_clip);
  } else {
    size_t i = 0;
    for (i = 0; i + 16 <= size; i += 16) {
      FloatToFloat16KernelAvx512(src + i, dst + i);
    }
    FloatToFloat16_avx2(src + i, dst + i, size - i);
  }
}

void Float16ToFloat_avx512(const float16* src, float* dst, size_t size) {
  size_t i = 0;
  for (i = 0; i + 16 <= size; i += 16) {
    Float16ToFloatKernelAvx512(src + i, dst + i);
  }
  Float16ToFloat_avx2(src + i, dst + i, size - i);
}

} // namespace fbgemm

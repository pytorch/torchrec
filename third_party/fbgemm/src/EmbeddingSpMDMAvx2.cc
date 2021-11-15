/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmEmbedding.h"

#include <cassert>
#include <cmath>

#include "fbgemm/Types.h"

namespace fbgemm {
namespace internal {

template <typename InType, typename IndexType, typename OffsetType>
bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    out[m] = 0;
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    int i = 0;

    // The following code doesn't speedup
#if 0
    constexpr int VLEN = std::is_same<IndexType, std::int64_t>::value ? 4 : 8;
    for (; i < lengths[m] / VLEN * VLEN; i += VLEN) {
      if (std::is_same<IndexType, std::int64_t>::value) {
        __m256i idx_v = _mm256_lddqu_si256(
            reinterpret_cast<const __m256i*>(indices + current));
        // Should be none true
        int mask1 = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(_mm256_setzero_si256(), idx_v)));
        // Should be all true
        int mask2 = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(data_size), idx_v)));
        if (mask1 || mask2 != 0x0f) {
          return false;
        }

        __m128 in_v = _mm256_i64gather_ps(input, idx_v, 4);
        alignas(64) float in_buf[VLEN];
        _mm_store_ps(in_buf, in_v);
        for (int j = 0; j < VLEN; ++j) {
          if (weights) {
            out[m] = std::fma(
                weights[is_weight_positional ? i + j : current + j],
                in_buf[j],
                out[m]);
          } else {
            out[m] += in_buf[j];
          }
        }
      } else {
        __m256i idx_v = _mm256_lddqu_si256(
            reinterpret_cast<const __m256i*>(indices + current));
        // Should be none true
        int mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), idx_v)));
        // Should be all true
        int mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(_mm256_set1_epi32(data_size), idx_v)));
        if (mask1 || mask2 != 0x00ff) {
          return false;
        }

        __m256 in_v = _mm256_i32gather_ps(input, idx_v, 4);
        alignas(64) float in_buf[VLEN];
        _mm256_store_ps(in_buf, in_v);
        for (int j = 0; j < VLEN; ++j) {
          if (weights) {
            out[m] = std::fma(
                weights[is_weight_positional ? i + j : current + j],
                in_buf[j],
                out[m]);
          } else {
            out[m] += in_buf[j];
          }
        }
      }

      current += VLEN;
    }
#endif

    for (; i < len; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      float w = 1.f;
      if (weights) {
        w = weights[is_weight_positional ? i : current];
      }

      const InType* inptr = input + indices[current];
      out[m] = std::fma(
          w,
          std::is_same<InType, float16>::value ? cpu_half2float(*inptr)
                                               : *inptr,
          out[m]);

      ++current;
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      out[m] *= scale;
    }
  }
  return current == index_size;
}

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE) \
  template bool EmbeddingSpMDMBlockSize1_(                       \
      const std::int64_t output_size,                            \
      const std::int64_t index_size,                             \
      const std::int64_t data_size,                              \
      const IN_TYPE* input,                                      \
      const INDEX_TYPE* indices,                                 \
      const OFFSET_TYPE* offsets_or_lengths,                     \
      const float* weights,                                      \
      bool normalize_by_lengths,                                 \
      float* out,                                                \
      bool is_weight_positional,                                 \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE)     \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)          \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(std::uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_BASE

} // namespace internal
} // namespace fbgemm

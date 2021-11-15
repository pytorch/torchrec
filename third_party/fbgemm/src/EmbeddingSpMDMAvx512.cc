/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmEmbedding.h"

#include <immintrin.h>
#include <type_traits>

namespace fbgemm {
namespace internal {

template <typename T>
struct reg_t;

template <>
struct reg_t<int32_t> {
  using w_reg_t = __m512;
  using mask_reg_t = __mmask16;
};

template <>
struct reg_t<int64_t> {
  using w_reg_t = __m256;
  using mask_reg_t = __mmask8;
};

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static constexpr int get_vlen() {
  return 16;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static constexpr int get_vlen() {
  return 8;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i load(void const* addr) {
  return _mm512_loadu_si512(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i load(void const* addr) {
  return _mm512_loadu_si512(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512 load_weights(void const* addr) {
  return _mm512_loadu_ps(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m256 load_weights(float const* addr) {
  return _mm256_loadu_ps(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512
mask_load_weights(__m512i src, __mmask16 mask_rem_v, void const* addr) {
  return _mm512_mask_loadu_ps(_mm512_castsi512_ps(src), mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m256
mask_load_weights(__m512i src, __mmask8 mask_rem_v, void const* addr) {
  return _mm256_mask_loadu_ps(
      _mm256_castsi256_ps(_mm512_castsi512_si256(src)), mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void mask_compress_and_store_weights(
    void* addr,
    __m512i zero_v,
    __mmask16 compress_mask_v,
    __mmask16 store_mask_v,
    __m512 src) {
  __m512 out_weights_v = _mm512_mask_compress_ps(
      _mm512_castsi512_ps(zero_v), compress_mask_v, src);
  _mm512_mask_storeu_ps(addr, store_mask_v, out_weights_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void mask_compress_and_store_weights(
    void* addr,
    __m512i zero_v,
    __mmask8 compress_mask_v,
    __mmask8 store_mask_v,
    __m256 src) {
  __m256 out_weights_v = _mm256_mask_compress_ps(
      _mm256_castsi256_ps(_mm512_castsi512_si256(zero_v)),
      compress_mask_v,
      src);
  _mm256_mask_storeu_ps(addr, store_mask_v, out_weights_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __mmask16 mask_from_rem(int rem) {
  __mmask16 mask_rem_v = (((long long)1) << rem) - 1;
  return mask_rem_v;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __mmask8 mask_from_rem(int rem) {
  __mmask8 mask_rem_v = (((long long)1) << rem) - 1;
  return mask_rem_v;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i
mask_load(__m512i zero_v, __mmask16 mask_rem_v, void const* addr) {
  return _mm512_mask_loadu_epi32(zero_v, mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i
mask_load(__m512i zero_v, __mmask8 mask_rem_v, void const* addr) {
  return _mm512_mask_loadu_epi64(zero_v, mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i maskz_load(__mmask16 mask_rem_v, void const* addr) {
  return _mm512_maskz_loadu_epi32(mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i maskz_load(__mmask8 mask_rem_v, void const* addr) {
  return _mm512_maskz_loadu_epi64(mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i mask_mov(__m512i src, __mmask16 mask_rem_v, __m512i a) {
  return _mm512_mask_mov_epi32(src, mask_rem_v, a);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i mask_mov(__m512i src, __mmask8 mask_rem_v, __m512i a) {
  return _mm512_mask_mov_epi64(src, mask_rem_v, a);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i gather(__m512i indices, const int32_t* addr) {
  return _mm512_i32gather_epi32(indices, addr, 4);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i gather(__m512i indices, const int32_t* addr) {
  // ToDo: Change this _mm512_i64gather_epi64 once mapping table is 64-bit
  __m256i res_32 = _mm512_i64gather_epi32(indices, addr, 4);
  return _mm512_cvtepi32_epi64(res_32);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i mask_gather(
    __m512i src,
    __mmask16 mask_rem_v,
    __m512i indices,
    const int32_t* addr) {
  return _mm512_mask_i32gather_epi32(src, mask_rem_v, indices, addr, 4);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i mask_gather(
    __m512i src,
    __mmask8 mask_rem_v,
    __m512i indices,
    const int32_t* addr) {
  // ToDo: Change this _mm512_mask_i64gather_epi64 once mapping table is 64-bit
  __m256i res_32 = _mm512_mask_i64gather_epi32(
      _mm512_castsi512_si256(src), mask_rem_v, indices, addr, 4);
  return _mm512_cvtepi32_epi64(res_32);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __mmask16 gen_mask(__m512i indices, __m512i zero_v) {
  return _mm512_cmpge_epi32_mask(indices, zero_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __mmask8 gen_mask(__m512i indices, __m512i zero_v) {
  return _mm512_cmpge_epi64_mask(indices, zero_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void compress_store(void* addr, __mmask16 mask, __m512i src_v) {
  _mm512_mask_compressstoreu_ps(addr, mask, _mm512_castsi512_ps(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void compress_store(void* addr, __mmask8 mask, __m512i src_v) {
  _mm512_mask_compressstoreu_pd(addr, mask, _mm512_castsi512_pd(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void
compress_store_weights(void* addr, __mmask16 mask, __m512 src_v) {
  _mm512_mask_compressstoreu_ps(addr, mask, src_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void
compress_store_weights(void* addr, __mmask8 mask, __m256 src_v) {
  _mm256_mask_compressstoreu_ps(addr, mask, src_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512 compress(__m512i zero_v, __mmask16 mask, __m512i src_v) {
  return _mm512_mask_compress_ps(
      _mm512_castsi512_ps(zero_v), mask, _mm512_castsi512_ps(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512d compress(__m512i zero_v, __mmask8 mask, __m512i src_v) {
  return _mm512_mask_compress_pd(
      _mm512_castsi512_pd(zero_v), mask, _mm512_castsi512_pd(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void mask_store(void* addr, __mmask16 mask, __m512 src_v) {
  _mm512_mask_storeu_epi32(addr, mask, _mm512_castps_si512(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void mask_store(void* addr, __mmask8 mask, __m512d src_v) {
  _mm512_mask_storeu_epi64(addr, mask, _mm512_castpd_si512(src_v));
}

// copy len bytes from src to dest
static inline void mymemcpy(char* src, char* dest, int len) {
  constexpr int VLEN = 64;
  int i = 0;
  for (; i < len / VLEN * VLEN; i += VLEN) {
    auto src_v = _mm512_loadu_si512(src + i);
    _mm512_storeu_si512(dest + i, src_v);
  }
  int rem = len - i;
  if (rem > 0) {
    __mmask64 mask_rem_v = (((long long)1) << rem) - 1;
    auto src_v = _mm512_maskz_loadu_epi8(mask_rem_v, src + i);
    _mm512_mask_storeu_epi8(dest + i, mask_rem_v, src_v);
  }
}

template <
    typename IndexType,
    bool HAS_WEIGHTS,
    int UNROLL = 8,
    bool USE_MASK = false>
static inline void compressed_indices_remap_avx512_helper(
    __m512i zero_v,
    __m512i minus1_v,
    const IndexType* offsets,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const float* weights,
    IndexType* out_indices,
    float* out_weights,
    IndexType* count_indices,
    const int32_t* rem,
    const int32_t* ind_w_start_offsets) {
  typename reg_t<IndexType>::mask_reg_t mask_rem_v[UNROLL];
  for (int i = 0; i < UNROLL; ++i) {
    mask_rem_v[i] = mask_from_rem<IndexType>(rem[i]);
  }
  for (int i = 0; i < UNROLL; ++i) {
    __m512i indices_v;
    if (USE_MASK) {
      indices_v = mask_load<IndexType>(
          zero_v,
          mask_rem_v[i],
          reinterpret_cast<void const*>(
              indices + offsets[i] + ind_w_start_offsets[i]));
    } else {
      indices_v = load<IndexType>(reinterpret_cast<void const*>(
          indices + offsets[i] + ind_w_start_offsets[i]));
    }

    // gather remapped indices from the mapping table
    __m512i remapped_indices_v;
    if (USE_MASK) {
      remapped_indices_v = mask_gather<IndexType>(
          zero_v, mask_rem_v[i], indices_v, compressed_indices_mapping);
      // mov -1 to not used places in the vector
      remapped_indices_v =
          mask_mov<IndexType>(minus1_v, mask_rem_v[i], remapped_indices_v);

    } else {
      remapped_indices_v =
          gather<IndexType>(indices_v, compressed_indices_mapping);
    }

    typename reg_t<IndexType>::w_reg_t weights_v;
    if (HAS_WEIGHTS) {
      if (USE_MASK) {
        weights_v = mask_load_weights<IndexType>(
            zero_v,
            mask_rem_v[i],
            reinterpret_cast<void const*>(
                weights + offsets[i] + ind_w_start_offsets[i]));
      } else {
        weights_v = load_weights<IndexType>(
            weights + offsets[i] + ind_w_start_offsets[i]);
      }
    }

    // Now remove -1 from the remapped indices
    auto mask_indices_v = gen_mask<IndexType>(remapped_indices_v, zero_v);

    if (USE_MASK) {
      auto out_indices_v =
          compress<IndexType>(zero_v, mask_indices_v, remapped_indices_v);

      mask_store<IndexType>(
          reinterpret_cast<void*>(out_indices + offsets[i] + count_indices[i]),
          mask_rem_v[i],
          out_indices_v);
    } else {
      compress_store<IndexType>(
          reinterpret_cast<void*>(out_indices + offsets[i] + count_indices[i]),
          mask_indices_v,
          remapped_indices_v);
    }

    if (HAS_WEIGHTS) {
      if (USE_MASK) {
        mask_compress_and_store_weights<IndexType>(
            reinterpret_cast<void*>(
                out_weights + offsets[i] + count_indices[i]),
            zero_v,
            mask_indices_v,
            mask_rem_v[i],
            weights_v);
      } else {
        compress_store_weights<IndexType>(
            reinterpret_cast<void*>(
                out_weights + offsets[i] + count_indices[i]),
            mask_indices_v,
            weights_v);
      }
    }

    count_indices[i] += _mm_popcnt_u32(mask_indices_v);
  }
}

template <typename IndexType, bool HAS_WEIGHTS>
void compressed_indices_remap_avx512(
    std::int32_t offsets_len,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights) {
  __m512i zero_v = _mm512_set1_epi32(0);
  __m512i minus1_v = _mm512_set1_epi32(-1);
  out_offsets[0] = offsets[0];
  constexpr int UNROLL = 8;
  constexpr int VLEN = get_vlen<IndexType>();
  int k = 1;
  for (; k < (offsets_len - 1) / UNROLL * UNROLL; k += UNROLL) {
    int32_t len[UNROLL];
    int32_t rem[UNROLL];
    for (int l = 0; l < UNROLL; ++l) {
      len[l] = offsets[k + l] - offsets[k + l - 1];
    }
    // count of non-pruned indices
    IndexType count_indices[UNROLL] = {0};
    // read indices/weights starting at these offsets
    int32_t ind_w_start_offsets[UNROLL] = {0};
    __m256i vec_len_v = _mm256_set1_epi32(VLEN);
    __m256i len_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(len));
    __mmask8 cmp_res_v = _mm256_cmpge_epi32_mask(len_v, vec_len_v);
    len_v = _mm256_mask_sub_epi32(len_v, cmp_res_v, len_v, vec_len_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(len), len_v);
    __m256i rem_v = _mm256_maskz_mov_epi32(cmp_res_v, vec_len_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rem), rem_v);
    int active_unrolls = _mm_popcnt_u32(cmp_res_v);

    // if we have any at least 1 full vector length work
    // take vector path
    while (active_unrolls > 0) {
      compressed_indices_remap_avx512_helper<
          IndexType,
          HAS_WEIGHTS,
          UNROLL,
          true>(
          zero_v,
          minus1_v,
          offsets + k - 1,
          indices,
          compressed_indices_mapping,
          weights,
          out_indices,
          out_weights,
          count_indices,
          rem,
          ind_w_start_offsets);

      __m256i start_offsets_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(ind_w_start_offsets));
      start_offsets_v = _mm256_mask_add_epi32(
          start_offsets_v, cmp_res_v, start_offsets_v, vec_len_v);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(ind_w_start_offsets), start_offsets_v);

      len_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(len));
      cmp_res_v = _mm256_cmpge_epi32_mask(len_v, vec_len_v);
      len_v = _mm256_mask_sub_epi32(len_v, cmp_res_v, len_v, vec_len_v);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(len), len_v);
      rem_v = _mm256_maskz_mov_epi32(cmp_res_v, vec_len_v);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(rem), rem_v);
      active_unrolls = _mm_popcnt_u32(cmp_res_v);
    }

    // Now work on all the remainders
    __m256i len_rem_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(len));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rem), len_rem_v);
    compressed_indices_remap_avx512_helper<
        IndexType,
        HAS_WEIGHTS,
        UNROLL,
        true>(
        zero_v,
        minus1_v,
        offsets + k - 1,
        indices,
        compressed_indices_mapping,
        weights,
        out_indices,
        out_weights,
        count_indices,
        rem,
        ind_w_start_offsets);

    // update output offsets
    for (int l = 0; l < UNROLL; ++l) {
      out_offsets[k + l] = out_offsets[k + l - 1] + count_indices[l];
    }
  }

  // work on remaining offsets_len serially
  constexpr int UNROLL_REM = 1;
  for (; k < offsets_len; ++k) {
    int32_t len[UNROLL_REM];
    int32_t rem[UNROLL_REM] = {0};
    for (int l = 0; l < UNROLL_REM; ++l) {
      len[l] = offsets[k + l] - offsets[k + l - 1];
    }
    IndexType count_indices[UNROLL_REM] = {0};
    int32_t ind_w_start_offsets[UNROLL_REM] = {0};
    int i = 0;
    for (; i < len[0] / VLEN * VLEN; i += VLEN) {
      compressed_indices_remap_avx512_helper<
          IndexType,
          HAS_WEIGHTS,
          UNROLL_REM,
          false>(
          zero_v,
          minus1_v,
          offsets + k - 1,
          indices,
          compressed_indices_mapping,
          weights,
          out_indices,
          out_weights,
          count_indices,
          rem,
          ind_w_start_offsets);
      ind_w_start_offsets[0] += VLEN;
    }
    // remainder
    rem[0] = len[0] - i;
    if (rem[0] > 0) {
      compressed_indices_remap_avx512_helper<
          IndexType,
          HAS_WEIGHTS,
          UNROLL_REM,
          true>(
          zero_v,
          minus1_v,
          offsets + k - 1,
          indices,
          compressed_indices_mapping,
          weights,
          out_indices,
          out_weights,
          count_indices,
          rem,
          ind_w_start_offsets);
    }

    for (int l = 0; l < UNROLL_REM; ++l) {
      out_offsets[k + l] = out_offsets[k + l - 1] + count_indices[l];
    }
  }

  // Results are stored at input offsets in output variables
  // copy results to right output locations
  for (int i = 1; i < offsets_len; ++i) {
    int out_len = out_offsets[i] - out_offsets[i - 1];
    mymemcpy(
        reinterpret_cast<char*>(out_indices + offsets[i - 1]),
        reinterpret_cast<char*>(out_indices + out_offsets[i - 1]),
        out_len * sizeof(IndexType));
    if (HAS_WEIGHTS) {
      mymemcpy(
          reinterpret_cast<char*>(out_weights + offsets[i - 1]),
          reinterpret_cast<char*>(out_weights + out_offsets[i - 1]),
          out_len * sizeof(float));
    }
  }
}

#define INSTANTIATE_REMAP_BASE(INDEX_TYPE, HAS_WEIGHTS)                   \
  template void compressed_indices_remap_avx512<INDEX_TYPE, HAS_WEIGHTS>( \
      std::int32_t offsets_numel,                                         \
      const INDEX_TYPE* indices,                                          \
      const int32_t* compressed_indices_mapping,                          \
      const INDEX_TYPE* offsets,                                          \
      const float* weights,                                               \
      INDEX_TYPE* out_indices,                                            \
      INDEX_TYPE* out_offsets,                                            \
      float* out_weights);

INSTANTIATE_REMAP_BASE(int32_t, true)
INSTANTIATE_REMAP_BASE(int32_t, false)
INSTANTIATE_REMAP_BASE(int64_t, true)
INSTANTIATE_REMAP_BASE(int64_t, false)

#undef INSTANTIATE_REMAP_BASE

} // namespace internal
} // namespace fbgemm

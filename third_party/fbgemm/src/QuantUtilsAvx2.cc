/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/QuantUtilsAvx2.h"
#include <immintrin.h>
#include <algorithm> //for std::min/std::max
#include <cassert> //for assert
#include <cfloat> // for FLT_MAX
#include <cmath> //for nearbyint
#include <cstring> //for memcpy
#include <limits> //for numeric_limits
#include "./MaskAvx2.h"
#include "fbgemm/Types.h"

namespace fbgemm {

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Utility functions

template <typename T, bool LEGACY>
void QuantizeAvx2(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams) {
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
  constexpr int VLEN = 8;
  constexpr int32_t min_val = std::numeric_limits<T>::min();
  constexpr int32_t max_val = std::numeric_limits<T>::max();
  // This is the largest int32 value less than int32_max
  // that is exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  float inverse_scale = 1.f / qparams.scale;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  // clang-format off
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00);
  // clang-format on
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256 src_v = _mm256_loadu_ps(src + i);
    __m256 transformed_v;
    if (LEGACY) { // static if
      transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
    } else {
      transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    }
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    if (!LEGACY) {
      rounded_v =
          _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    }
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // An instruction sequence to save 8 32-bit integers as 8 8-bit integers
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + i), _mm256_castsi256_si128(clipped_v));
  }

  // Handle remainder using mask instructions so that
  // the main loop and remainder loop have the same behavior
  int rem = len - i;
  if (rem > 0) {
    __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[rem]));
    // __m128i store_mask_v = _mm_load_si128(
    // reinterpret_cast<const __m128i*>(internal::sse_epi8_masks[rem]));
    __m256 src_v = _mm256_maskload_ps(src + i, mask_v);
    __m256 transformed_v;
    if (LEGACY) {
      transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
    } else {
      transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    }
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    if (!LEGACY) {
      rounded_v =
          _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    }
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // An instruction sequence to save "rem" number of 32-bit integers
    // as "rem" number of 8-bit integers
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    // do not use _mm_maskmoveu_si128 instead of memcpy.
    // asan has false positives for _mm_maskmoveu_si128 and this instruction
    // sometimes causes segfault (root cause is unknown).
    memcpy(dst + i, reinterpret_cast<void*>(&clipped_v), rem * sizeof(T));
    // _mm_maskmoveu_si128(
    // _mm256_castsi256_si128(clipped_v),
    // store_mask_v,
    // reinterpret_cast<char*>(dst + i));
  }
#endif
}

uint32_t Xor128(void) {
  /* library-local */ static uint32_t x = 123456789;
  /* library-local */ static uint32_t y = 362436069;
  /* library-local */ static uint32_t z = 521288629;
  /* library-local */ static uint32_t w = 88675123;
  uint32_t t;
  t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

// Instantiate QuantizeAvx2 for known datatypes
#define SPECIALIZE_QUANTIZEAVX2(T, LEGACY) \
  template void QuantizeAvx2<T, LEGACY>(   \
      const float* src,                    \
      T* dst,                              \
      int len,                             \
      const TensorQuantizationParams& qparams);
SPECIALIZE_QUANTIZEAVX2(uint8_t, true)
SPECIALIZE_QUANTIZEAVX2(int8_t, true)
SPECIALIZE_QUANTIZEAVX2(uint8_t, false)
SPECIALIZE_QUANTIZEAVX2(int8_t, false)
#undef SPECIALIZE_QUANTIZEAVX2

template <typename T>
void NO_SANITIZE("address") FusedQuantizeDequantizeAvx2(
    const float* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams,
    float noise_ratio) {
  float inverse_scale = 1.f / qparams.scale;
  constexpr int32_t min_val = std::numeric_limits<T>::min();
  constexpr int32_t max_val = std::numeric_limits<T>::max();
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))

  constexpr int VLEN = 8;
  // This is the largest int32 value less than int32_max
  // that is exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  uint32_t rand;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  __m256 scale_v = _mm256_set1_ps(qparams.scale);
  __m256 zp_v = _mm256_set1_ps(qparams.zero_point);

  for (; i < len / VLEN * VLEN; i += VLEN) {
    // prefetch src and dst
    _mm_prefetch(reinterpret_cast<const char*>(src + i + VLEN), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(dst + i + VLEN), _MM_HINT_T0);

    __m256 src_v = _mm256_loadu_ps(src + i);
    __m256 transformed_v;
    if (noise_ratio > 0) {
      rand = Xor128() % 10;
      if (rand < noise_ratio * 10) {
        _mm256_storeu_ps(dst + i, src_v);
        continue;
      }
    }

    transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    rounded_v =
        _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // convert int32 to float32
    __m256 fp32_clipped_v = _mm256_cvtepi32_ps(clipped_v);
    // minus zero point, multiply by scale
    __m256 fp32_dq_sub = _mm256_sub_ps(fp32_clipped_v, zp_v);
    __m256 fp32_dq = _mm256_mul_ps(fp32_dq_sub, scale_v);

    // save fusued quantize-dequantize fp32 values into dst
    _mm256_storeu_ps(dst + i, fp32_dq);
  }

  // Handle remainder using mask instructions so that
  // the main loop and remainder loop have the same behavior
  int rem = len - i;
  if (rem > 0) {
    __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[rem]));

    __m256 src_v = _mm256_maskload_ps(src + i, mask_v);
    __m256 transformed_v;

    if (noise_ratio > 0) {
      rand = Xor128() % 10;
      if (rand < noise_ratio * 10) {
        _mm256_storeu_ps(dst + i, src_v);
        return;
      }
    }

    transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    rounded_v =
        _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));

    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // convert int32 to float32
    __m256 fp32_clipped_v = _mm256_cvtepi32_ps(clipped_v);
    // minus zero point, multiply by scale
    __m256 fp32_dq_sub =
        _mm256_sub_ps(fp32_clipped_v, _mm256_set1_ps(qparams.zero_point));
    __m256 fp32_dq = _mm256_mul_ps(fp32_dq_sub, _mm256_set1_ps(qparams.scale));

    // store fp32 values with mask
    _mm256_maskstore_ps(dst + i, mask_v, fp32_dq);
  }
#endif
}

// Instantiate QuantizeAvx2 for known datatypes
#define SPECIALIZE_FUSEDDQAVX2(T)               \
  template void FusedQuantizeDequantizeAvx2<T>( \
      const float* src,                         \
      float* dst,                               \
      int len,                                  \
      const TensorQuantizationParams& qparams,  \
      float noise_ratio);
SPECIALIZE_FUSEDDQAVX2(uint8_t)
SPECIALIZE_FUSEDDQAVX2(int8_t)

#undef SPECIALIZE_FUSEDDQAVX2

void FindMinMax(const float* a, float* min, float* max, int len) {
  if (len <= 0) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  float temp_min = *a, temp_max = *a;
  int i = 0;

#ifdef __AVX__
  __m256 min_v = _mm256_set1_ps(*a), max_v = _mm256_set1_ps(*a);
  constexpr int VLEN = 8;
  if (len >= VLEN) {
    for (; i < len / VLEN * VLEN; i += VLEN) {
      min_v = _mm256_min_ps(min_v, _mm256_loadu_ps(a + i));
      max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(a + i));
    }

    float min_buf[VLEN], max_buf[VLEN];
    _mm256_storeu_ps(min_buf, min_v);
    _mm256_storeu_ps(max_buf, max_v);
    for (int j = 0; j < VLEN; ++j) {
      temp_min = std::min(temp_min, min_buf[j]);
      temp_max = std::max(temp_max, max_buf[j]);
    }
  }
#endif

  for (; i < len; i++) {
    temp_min = std::min(temp_min, a[i]);
    temp_max = std::max(temp_max, a[i]);
  }
  *min = temp_min;
  *max = temp_max;
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

#ifdef __AVX2__
void RequantizeAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  int32_t Bq_zero_point[] = {0};

  requantizationParams_t<> reqObj = {
      0, // Aq_zero_point
      Bq_zero_point,
      params.target_qparams.zero_point,
      &params.real_multiplier,
      nullptr, // row_offsets
      nullptr, // col_offsets
      nullptr, // bias
      static_cast<std::uint32_t>(len), // ncols
      1, // groups
      nullptr}; // act_times_w_scale
  requantizeOutputProcessingAvx2<
      true, // A_SYMMETRIC
      true, // B_SYMMETRIC
      QuantizationGranularity::TENSOR,
      false, // HAS_BIAS
      false // FUSE_RELU
      >(dst, src, {0, 1, 0, len}, len, len, reqObj);
}

void RequantizeFixedPointAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  constexpr int VLEN = 8;

  __m256i b = _mm256_set1_epi32(params.multiplier);

  // AVX2 doesn't support arithmetic right shift.
  // As a work around, we convert 64-bit multiplied results to uint64_t by
  // adding 0x8000000000000000ULL, logical right shift, and subtract by
  // (0x8000000000000000ULL >> right_shift).
  __m256i pre_shift_nudge = _mm256_set1_epi64x(
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL);
  __m256i post_shift_nudge = _mm256_set1_epi64x(
      params.target_qparams.zero_point -
      (0x8000000000000000ULL >> params.right_shift));

  __m256i min_v = _mm256_set1_epi32(numeric_limits<uint8_t>::min());
  __m256i max_v = _mm256_set1_epi32(numeric_limits<uint8_t>::max());

  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int i = 0;
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256i a_v = _mm256_loadu_si256((const __m256i*)(src + i));

    // a = a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7
    // b = b0 | b1 | b3 | b3 | b4 | b5 | b6 | b7
    __m256i a_even_v = a_v;
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);

    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);

    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, pre_shift_nudge);
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, pre_shift_nudge);

    __m256i even_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(even_rounded_v, params.right_shift),
        post_shift_nudge);
    __m256i odd_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(odd_rounded_v, params.right_shift), post_shift_nudge);
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);

    // even_result_v has numbers we want in its even 32-bit SIMD lanes, and
    // odd_result_v has numbers we want in its odd 32-bit SIMD lanes.
    // Use blend to combine them.
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa);
    __m256i clipped_v =
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));

    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    *(int64_t*)(dst + i) = _mm256_extract_epi64(clipped_v, 0);
  }

  for (; i < len; ++i) {
    int64_t ab_64 =
        static_cast<int64_t>(src[i]) * static_cast<int64_t>(params.multiplier);
    int64_t nudge = 1ll << std::max(0, params.right_shift - 1);
    int64_t quantized_down = params.target_qparams.zero_point +
        ((ab_64 + nudge) >> params.right_shift);
    dst[i] = std::min<int64_t>(std::max<int64_t>(quantized_down, 0l), 255l);
  }
}
#endif

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    typename BIAS_TYPE>
void requantizeOutputProcessingAvx2(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);

  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v;
  if (!(Q_GRAN == QuantizationGranularity::OUT_CHANNEL)) {
    if (is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v =
          _mm256_set1_ps(1.0 / r.act_times_w_scale[quant_param_idx]);
    }
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

  assert(
      (A_SYMMETRIC == (r.A_zero_point == 0)) &&
      "A_SYMMETRIC == true if and only if A_zero_point == 0");
  assert(
      (B_SYMMETRIC ==
       ((Q_GRAN == QuantizationGranularity::TENSOR && r.B_zero_point[0] == 0) ||
        r.row_offsets == nullptr)) &&
      "B_SYMMETRIC == true if and only if B_zero_point == 0 "
      "or r.row_offsets == nullptr");
  assert(
      (HAS_BIAS == (r.bias != nullptr)) &&
      "HAS_BIAS == true if and only if bias != nullptr");

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(r.C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(r.C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    // Scale row_offset with Bq_zero_point
    int32_t row_offset = 0;
    if (B_SYMMETRIC) {
      row_offset = 0;
    } else if (
        Q_GRAN == QuantizationGranularity::TENSOR ||
        Q_GRAN == QuantizationGranularity::GROUP) {
      row_offset =
          r.row_offsets[i - block.row_start] * r.B_zero_point[quant_param_idx];
    } else {
      assert(
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL &&
          "unknown quantization granularity");
    }
    __m256i row_offset_v = _mm256_set1_epi32(row_offset);

    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / (VLEN * 4) * (VLEN * 4));
         j += (VLEN * 4)) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));
      __m256i y_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          1 * VLEN));
      __m256i z_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          2 * VLEN));
      __m256i w_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          3 * VLEN));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j + VLEN)));
        y_v = _mm256_sub_epi32(y_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.col_offsets + j + 2 * VLEN)));
        z_v = _mm256_sub_epi32(z_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.col_offsets + j + 3 * VLEN)));
        w_v = _mm256_sub_epi32(w_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j + VLEN)));
        }
        y_v = _mm256_sub_epi32(y_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  r.B_zero_point + j + 2 * VLEN)));
        }
        z_v = _mm256_sub_epi32(z_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  r.B_zero_point + j + 3 * VLEN)));
        }
        w_v = _mm256_sub_epi32(w_v, row_offset_v);
      }
      __m256 xf_v, yf_v, zf_v, wf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v, y_bias_v, z_bias_v, w_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 0 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 0 * VLEN));
            y_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 1 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 1 * VLEN));
            z_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 2 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 2 * VLEN));
            w_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 3 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 3 * VLEN));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 0 * VLEN)),
                act_times_w_rcp_v);
            y_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 1 * VLEN)),
                act_times_w_rcp_v);
            z_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 2 * VLEN)),
                act_times_w_rcp_v);
            w_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 3 * VLEN)),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
          yf_v = _mm256_add_ps(_mm256_cvtepi32_ps(y_v), y_bias_v);
          zf_v = _mm256_add_ps(_mm256_cvtepi32_ps(z_v), z_bias_v);
          wf_v = _mm256_add_ps(_mm256_cvtepi32_ps(w_v), w_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 0 * VLEN)));
          y_v = _mm256_add_epi32(
              y_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 1 * VLEN)));
          z_v = _mm256_add_epi32(
              z_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 2 * VLEN)));
          w_v = _mm256_add_epi32(
              w_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 3 * VLEN)));
          xf_v = _mm256_cvtepi32_ps(x_v);
          yf_v = _mm256_cvtepi32_ps(y_v);
          zf_v = _mm256_cvtepi32_ps(z_v);
          wf_v = _mm256_cvtepi32_ps(w_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
        yf_v = _mm256_cvtepi32_ps(y_v);
        zf_v = _mm256_cvtepi32_ps(z_v);
        wf_v = _mm256_cvtepi32_ps(w_v);
      }

      /*
       * Convert int32_t input to FP32 and multiply by FP32 scale.
       * Both operations involve statistically unbiased roundings (with
       * default MXCSR rounding mode):
       * - Large int32_t values can't be exactly represented as FP32.
       * CVTDQ2PS instruction on x86 would round it according to nearest
       * FP32 value with ties to even (assuming default MXCSR rounding
       * mode).
       * - Product of two FP32 values is generally not exactly
       * representation as an FP32 value, and will be rounded to nearest
       * FP32 value with ties to even with default MXCSR rounding mode.
       */
      __m256 x_scaled_v, y_scaled_v, z_scaled_v, w_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v =
            _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j + 0 * VLEN));
        y_scaled_v =
            _mm256_mul_ps(yf_v, _mm256_loadu_ps(r.C_multiplier + j + 1 * VLEN));
        z_scaled_v =
            _mm256_mul_ps(zf_v, _mm256_loadu_ps(r.C_multiplier + j + 2 * VLEN));
        w_scaled_v =
            _mm256_mul_ps(wf_v, _mm256_loadu_ps(r.C_multiplier + j + 3 * VLEN));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
        y_scaled_v = _mm256_mul_ps(yf_v, multiplier_v);
        z_scaled_v = _mm256_mul_ps(zf_v, multiplier_v);
        w_scaled_v = _mm256_mul_ps(wf_v, multiplier_v);
      }

      /*
       * Convert scaled FP32 result to int32_t using CVTPS2DQ instruction.
       * CVTPS2DQ instruction rounds result according to nearest FP32 value
       * with ties to even (assuming default MXCSR rounding mode). However,
       * when conversion overflows, it produces INT32_MIN as a result. For
       * large positive inputs the result of conversion can become negative,
       * which affects the final requantization result. Note that on x86
       * SSE2 we have e.g. int32_t(float(INT32_MAX)) == INT32_MIN! This
       * happens because float(INT32_MAX) rounds to 2**31, which overflows
       * int32_t when it is converted back to integer.
       *
       * Thankfully, we can prove that overflow never happens in this
       * requantization scheme. The largest positive input is INT32_MAX
       * (2**31 - 1), which turns into 2**31 when converted to float. The
       * largest scale value is 0x1.FFFFFEp-1. When multiplied together, the
       * result is 2147483520 (compare to INT32_MAX = 2147483647), which
       * fits into int32_t without overflow.
       */
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
      __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
      __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
      __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

      /*
       * Standard final sequence on x86 AVX2:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m256i xy_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
      __m256i zw_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
      __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
      __m256i xyzw_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(xyzw_packed_v, max_v));

      /*
       * xyzw_clamped_v has results in the following layout so we need to
       * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
       */
      xyzw_clamped_v =
          _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);

      /*
       * 4x CVTDQ2PS
       * 4x MULPS
       * 4x CVTPS2DQ
       * 2x PACKSSDW
       * 1x PACKUSWB
       * 2x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 20 instructions total
       */
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(out + i * ld_out + j), xyzw_clamped_v);
    } // j loop vectorized and unrolled 4x

    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j)),
                _mm256_loadu_ps(r.act_times_w_scale + j));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j)),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r.bias + j)));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      }
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

      /*
       * 1x CVTDQ2PS
       * 1x MULPS
       * 1x CVTPS2DQ
       * 1x PACKSSDW
       * 1x PACKUSWB
       * 1x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 9 instructions total
       */
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(out + i * ld_out + j),
          _mm256_castsi256_si128(x_clamped_v));
    } // j loop vectorized

    int remainder = block.col_start + block.col_size - j;
    if (remainder > 0) {
      __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          internal::avx2_ps_or_epi32_masks[remainder]));

      __m256i x_v = _mm256_maskload_epi32(
          inp + (i - block.row_start) * ld_in + (j - block.col_start), mask_v);

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v, _mm256_maskload_epi32(r.col_offsets + j, mask_v));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_maskload_epi32(r.B_zero_point + j, mask_v));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_maskload_ps(
                    reinterpret_cast<const float*>(r.bias + j), mask_v),
                _mm256_maskload_ps(r.act_times_w_scale + j, mask_v));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_maskload_ps(
                    reinterpret_cast<const float*>(r.bias + j), mask_v),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_maskload_epi32(
                  reinterpret_cast<const int*>(r.bias + j), mask_v));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v =
            _mm256_mul_ps(xf_v, _mm256_maskload_ps(r.C_multiplier + j, mask_v));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      }
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

      /*
       * 1x CVTDQ2PS
       * 1x MULPS
       * 1x CVTPS2DQ
       * 1x PACKSSDW
       * 1x PACKUSWB
       * 1x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 9 instructions total
       */
      alignas(64) uint8_t x_clamped_buffer[32];
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(x_clamped_buffer), x_clamped_v);
      for (int k = 0; k < remainder; ++k) {
        out[i * ld_out + j + k] = x_clamped_buffer[k];
      }
    } // j loop remainder
  } // i loop
}

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU>
void requantizeForFloatAvx2(
    float* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationForFloatParams_t& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.A_scale * r.B_scale[quant_param_idx]);

  assert(
      (A_SYMMETRIC == (r.A_zero_point == 0)) &&
      "A_SYMMETRIC == true if and only if A_zero_point == 0");
  assert(
      (B_SYMMETRIC ==
       ((Q_GRAN == QuantizationGranularity::TENSOR && r.B_zero_point[0] == 0) ||
        r.row_offsets == nullptr)) &&
      "B_SYMMETRIC == true if and only if B_zero_point == 0 "
      "or r.row_offsets == nullptr");
  assert(
      (HAS_BIAS == (r.bias != nullptr)) &&
      "HAS_BIAS == true if and only if bias != nullptr");

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    // Scale row_offset with Bq_zero_point
    int32_t row_offset = 0;
    if (B_SYMMETRIC) {
      row_offset = 0;
    } else if (
        Q_GRAN == QuantizationGranularity::TENSOR ||
        Q_GRAN == QuantizationGranularity::GROUP) {
      row_offset =
          r.row_offsets[i - block.row_start] * r.B_zero_point[quant_param_idx];
    } else {
      assert(
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL &&
          "unknown quantization granularity");
    }
    __m256i row_offset_v = _mm256_set1_epi32(row_offset);

    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(x_v),
            _mm256_mul_ps(
                _mm256_set1_ps(r.A_scale), _mm256_loadu_ps(r.B_scale + j)));
      } else {
        x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
      }

      if (HAS_BIAS) {
        x_scaled_v = _mm256_add_ps(x_scaled_v, _mm256_loadu_ps(r.bias + j));
      }
      if (FUSE_RELU) {
        x_scaled_v = _mm256_max_ps(_mm256_setzero_ps(), x_scaled_v);
      }

      _mm256_storeu_ps(out + i * ld_out + j, x_scaled_v);
    } // j loop vectorized

    int remainder = block.col_start + block.col_size - j;
    if (remainder > 0) {
      __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          internal::avx2_ps_or_epi32_masks[remainder]));

      __m256i x_v = _mm256_maskload_epi32(
          inp + (i - block.row_start) * ld_in + (j - block.col_start), mask_v);

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v, _mm256_maskload_epi32(r.col_offsets + j, mask_v));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_maskload_epi32(r.B_zero_point + j, mask_v));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(x_v),
            _mm256_mul_ps(
                _mm256_set1_ps(r.A_scale),
                _mm256_maskload_ps(r.B_scale + j, mask_v)));
      } else {
        x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
      }

      if (HAS_BIAS) {
        x_scaled_v =
            _mm256_add_ps(x_scaled_v, _mm256_maskload_ps(r.bias + j, mask_v));
      }
      if (FUSE_RELU) {
        x_scaled_v = _mm256_max_ps(_mm256_setzero_ps(), x_scaled_v);
      }

      _mm256_maskstore_ps(out + i * ld_out + j, mask_v, x_scaled_v);
    } // j loop remainder
  } // i loop
}

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G,
    typename BIAS_TYPE>
void requantizeOutputProcessingGConvAvx2(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);

  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v;
  if (!(Q_GRAN == QuantizationGranularity::OUT_CHANNEL)) {
    if (is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v =
          _mm256_set1_ps(1.0 / r.act_times_w_scale[quant_param_idx]);
    }
  }
  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

  assert(
      (A_SYMMETRIC == (r.A_zero_point == 0)) &&
      "A_SYMMETRIC == true if and only if A_zero_point == 0");
  assert(
      (B_SYMMETRIC ==
       ((Q_GRAN == QuantizationGranularity::TENSOR && r.B_zero_point[0] == 0) ||
        r.row_offsets == nullptr)) &&
      "B_SYMMETRIC == true if and only if B_zero_point == 0 "
      "or r.row_offsets == nullptr");
  assert(
      (HAS_BIAS == (r.bias != nullptr)) &&
      "HAS_BIAS == true if and only if bias != nullptr");

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(r.C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(r.C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        __m256i row_offset_v;

        if (C_PER_G == 2) {
          // When C_PER_G == 2, we need to handle 4 groups at a time to fully
          // utilize 32B AVX2 vector register (C_PER_G * 4 * sizeof(int32_t) ==
          // 32B)
          // Load row_offsets for 4 groups and broadcast by 2 times.
          row_offset_v =
              _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(reinterpret_cast<const float*>(
                          r.row_offsets + (i - block.row_start) * 4))),
                  permute_mask_v)));

        }
        // When C_PER_G == 4, we need to handle 2 groups at a time to fully
        // utilize 32B AVX2 vector register (C_PER_G * 2 * sizeof(int32_t) ==
        // 32B)
        // When C_PER_G == 8, we just need 1 group at a time on the other hand.

        // Groups 0 and 1 when C_PER_G == 4
        // Group 0 when C_PER_G == 8
        else if (C_PER_G == 4) {
          // Load row_offsets for 2 groups and broadcast by 4 times each because
          // we have 4 channels per group.
          // groups 0 and 1
          row_offset_v = _mm256_insertf128_si256(
              _mm256_castsi128_si256(
                  _mm_set1_epi32(r.row_offsets[(i - block.row_start) * 2 + 0])),
              _mm_set1_epi32(r.row_offsets[(i - block.row_start) * 2 + 1]),
              1);
        } else if (C_PER_G == 8) {
          row_offset_v =
              _mm256_set1_epi32(r.row_offsets[(i - block.row_start)]);
        } else {
          assert(C_PER_G == 16);
          row_offset_v =
              _mm256_set1_epi32(r.row_offsets[(i - block.row_start)]);
        }

        __m256i B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[0]);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(r.B_zero_point + j));
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 2) {
            B_zero_point_v =
                _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                    _mm256_castps128_ps256(
                        _mm_loadu_ps(reinterpret_cast<const float*>(
                            r.B_zero_point + quant_param_idx))),
                    permute_mask_v)));
          } else if (C_PER_G == 4) {
            B_zero_point_v = _mm256_insertf128_si256(
                _mm256_castsi128_si256(
                    _mm_set1_epi32(r.B_zero_point[quant_param_idx])),
                _mm_set1_epi32(r.B_zero_point[quant_param_idx + 1]),
                1);
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[quant_param_idx]);
          } else {
            B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[quant_param_idx]);
          }
        }
        row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v =
              _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j));
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                x_bias_v, _mm256_loadu_ps(r.act_times_w_scale + j));
          } else if (Q_GRAN == QuantizationGranularity::GROUP) {
            __m256 diviser_v;
            if (C_PER_G == 2) {
              diviser_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(r.act_times_w_scale + quant_param_idx)),
                  permute_mask_v));
            } else if (C_PER_G == 4) {
              diviser_v = _mm256_insertf128_ps(
                  _mm256_castps128_ps256(
                      _mm_set1_ps(r.act_times_w_scale[quant_param_idx + 0])),
                  _mm_set1_ps(r.act_times_w_scale[quant_param_idx + 1]),
                  1);
            } else if (C_PER_G == 8) {
              diviser_v = _mm256_set1_ps(r.act_times_w_scale[quant_param_idx]);
            } else {
              assert(C_PER_G == 16);
              diviser_v = _mm256_set1_ps(r.act_times_w_scale[quant_param_idx]);
            }
            x_bias_v = _mm256_div_ps(x_bias_v, diviser_v);
          } else {
            x_bias_v = _mm256_mul_ps(x_bias_v, act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r.bias + j)));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      /*
       * Convert int32_t input to FP32 and multiply by FP32 scale.
       * Both operations involve statistically unbiased roundings (with
       * default MXCSR rounding mode):
       * - Large int32_t values can't be exactly represented as FP32.
       * CVTDQ2PS instruction on x86 would round it according to nearest
       * FP32 value with ties to even (assuming default MXCSR rounding
       * mode).
       * - Product of two FP32 values is generally not exactly
       * representation as an FP32 value, and will be rounded to nearest
       * FP32 value with ties to even with default MXCSR rounding mode.
       */
      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        if (C_PER_G == 2) {
          multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
              _mm256_castps128_ps256(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx)),
              permute_mask_v));
        } else if (C_PER_G == 4) {
          multiplier_v = _mm256_insertf128_ps(
              _mm256_castps128_ps256(
                  _mm_set1_ps(r.C_multiplier[quant_param_idx])),
              _mm_set1_ps(r.C_multiplier[quant_param_idx + 1]),
              1);
        } else if (C_PER_G == 8) {
          multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);
        } else {
          multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);
        }
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      }

      /*
       * Convert scaled FP32 result to int32_t using CVTPS2DQ instruction.
       * CVTPS2DQ instruction rounds result according to nearest FP32 value
       * with ties to even (assuming default MXCSR rounding mode). However,
       * when conversion overflows, it produces INT32_MIN as a result. For
       * large positive inputs the result of conversion can become negative,
       * which affects the final requantization result. Note that on x86
       * SSE2 we have e.g. int32_t(float(INT32_MAX)) == INT32_MIN! This
       * happens because float(INT32_MAX) rounds to 2**31, which overflows
       * int32_t when it is converted back to integer.
       *
       * Thankfully, we can prove that overflow never happens in this
       * requantization scheme. The largest positive input is INT32_MAX
       * (2**31 - 1), which turns into 2**31 when converted to float. The
       * largest scale value is 0x1.FFFFFEp-1. When multiplied together, the
       * result is 2147483520 (compare to INT32_MAX = 2147483647), which
       * fits into int32_t without overflow.
       */
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      /*
       * Standard final sequence on x86 AVX2:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

      /*
       * 1x CVTDQ2PS
       * 1x MULPS
       * 1x CVTPS2DQ
       * 1x PACKSSDW
       * 1x PACKUSWB
       * 1x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 9 instructions total
       */

      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(out + i * ld_out + j),
          _mm256_castsi256_si128(x_clamped_v));
    } // j loop vectorized

    const int remainder = block.col_start + block.col_size - j;
    (void)remainder; // Suppress unused variable warning
    assert(remainder == 0);
  } // i loop
}

#define INSTANTIATE_REQUANTIZE_BIAS_TYPE(                                      \
    A_SYM, B_SYM, Q_GRAN, BIAS, RELU, BIAS_TYPE)                               \
  template void FBGEMM_API                                                     \
  requantizeOutputProcessingAvx2<A_SYM, B_SYM, Q_GRAN, BIAS, RELU, BIAS_TYPE>( \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      2,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      4,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      8,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      16,                                                                      \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);

#define INSTANTIATE_REQUANTIZE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU)              \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, float)   \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, int32_t) \
  template void requantizeForFloatAvx2<A_SYM, B_SYM, Q_GRAN, BIAS, RELU>(     \
      float* out,                                                             \
      const int32_t* inp,                                                     \
      const block_type_t& block,                                              \
      int ld_out,                                                             \
      int ld_in,                                                              \
      const requantizationForFloatParams_t& r);

#define INSTANTIATE_A_SYM(B_SYM, Q_GRAN, BIAS, RELU)      \
  INSTANTIATE_REQUANTIZE(true, B_SYM, Q_GRAN, BIAS, RELU) \
  INSTANTIATE_REQUANTIZE(false, B_SYM, Q_GRAN, BIAS, RELU)

#define INSTANTIATE_B_SYM(Q_GRAN, BIAS, RELU) \
  INSTANTIATE_A_SYM(true, Q_GRAN, BIAS, RELU) \
  INSTANTIATE_A_SYM(false, Q_GRAN, BIAS, RELU)

#define INSTANTIATE_Q_GRANS(BIAS, RELU)                          \
  INSTANTIATE_B_SYM(QuantizationGranularity::TENSOR, BIAS, RELU) \
  INSTANTIATE_B_SYM(QuantizationGranularity::GROUP, BIAS, RELU)  \
  INSTANTIATE_B_SYM(QuantizationGranularity::OUT_CHANNEL, BIAS, RELU)

#define INSTANTIATE_BIAS(RELU)    \
  INSTANTIATE_Q_GRANS(true, RELU) \
  INSTANTIATE_Q_GRANS(false, RELU)

INSTANTIATE_BIAS(true)
INSTANTIATE_BIAS(false)

#undef INSTANTIATE_A_SYM
#undef INSTANTIATE_B_SYM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BIAS

static inline uint16_t floatToHalf(float val) {
#ifdef _MSC_VER
  // Use _mm256_cvtps_ph/_mm256_cvtph_ps because _cvtsh_ss/_cvtss_sh don't
  // exist in MSVC.
  __m256 val_v = _mm256_set1_ps(val);
  __m128i val_half_v =
      _mm256_cvtps_ph(val_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return static_cast<std::uint16_t>(_mm_cvtsi128_si32(val_half_v));
#else
  return _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
}
static inline float halfToFloat(uint16_t val) {
#ifdef _MSC_VER
  return _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_cvtsi32_si128(val)));
#else
  return _cvtsh_ss(val);
#endif
}

template <typename InputType, int BIT_RATE>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  static_assert(
      std::is_same<InputType, float>() || std::is_same<InputType, float16>(),
      "Only float and float16 types are allowed.");
  constexpr int VLEN = 8;
  constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  int output_columns =
      (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
      2 * sizeof(std::uint16_t);

  float* input_row_float_for_fp16;
  if (std::is_same<InputType, float16>()) {
    input_row_float_for_fp16 = static_cast<float*>(
        fbgemmAlignedAlloc(64, input_columns * sizeof(float)));
  }

  for (int row = 0; row < input_rows; ++row) {
    const InputType* input_row = input + row * input_columns;
    const float* input_row_float;
    if (std::is_same<InputType, float>()) {
      // NOTE: this reinterpret_cast is only to workaround c++
      // type requirements -- it is not for fp16 case and `input_row` HAS to be
      // float* type. Remove it and use constexpr when pytorch allows C++17.
      input_row_float = reinterpret_cast<const float*>(input_row);
    } else {
      input_row_float = input_row_float_for_fp16;
    }

    std::uint8_t* output_row = output + row * output_columns;
    std::uint16_t* output_row_scale_bias = reinterpret_cast<std::uint16_t*>(
        output_row +
        (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

    float minimum_element = FLT_MAX;
    float maximum_element = -FLT_MAX;
    __m256 min_v = _mm256_set1_ps(minimum_element);
    __m256 max_v = _mm256_set1_ps(maximum_element);

    int col;
    for (col = 0; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v;
      if (std::is_same<InputType, float>()) {
        in_v = _mm256_loadu_ps(input_row_float + col);
      } else {
        __m128i in_half_v =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + col));
        in_v = _mm256_cvtph_ps(in_half_v);
        _mm256_store_ps(input_row_float_for_fp16 + col, in_v);
      }

      min_v = _mm256_min_ps(min_v, in_v);
      max_v = _mm256_max_ps(max_v, in_v);
    }
    alignas(64) float min_buf[VLEN], max_buf[VLEN];
    _mm256_store_ps(min_buf, min_v);
    _mm256_store_ps(max_buf, max_v);
    for (int i = 0; i < VLEN; ++i) {
      minimum_element = std::min(minimum_element, min_buf[i]);
      maximum_element = std::max(maximum_element, max_buf[i]);
    }

    for (; col < input_columns; ++col) {
      if (std::is_same<InputType, float>()) {
        minimum_element = std::min(minimum_element, input_row_float[col]);
        maximum_element = std::max(maximum_element, input_row_float[col]);
      } else {
        float element = halfToFloat(input_row[col]);
        input_row_float_for_fp16[col] = element;
        minimum_element = std::min(minimum_element, element);
        maximum_element = std::max(maximum_element, element);
      }
    }

    output_row_scale_bias[1] = floatToHalf(minimum_element);
    minimum_element = halfToFloat(output_row_scale_bias[1]);
    const float range = maximum_element - minimum_element;

    float scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
    std::uint16_t scale_fp16 = floatToHalf(scale);
    scale = halfToFloat(scale_fp16);
    if (scale == 0) {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because maximum_element - minimum_element will be
      // 0 for all X
      scale = 1.0f;
    }
    float inverse_scale = 1.0f / scale;
    if (std::isinf(inverse_scale)) {
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    output_row_scale_bias[0] = floatToHalf(scale);

    col = 0;
    if (BIT_RATE == 2 || BIT_RATE == 4) {
      __m256i permute_mask1_v =
          _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
      __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
      min_v = _mm256_set1_ps(minimum_element);

      for (; col + 4 * VLEN <= input_columns; col += 4 * VLEN) {
        __m256i x_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row_float + col), min_v),
            inverse_scale_v));
        __m256i y_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row_float + col + VLEN), min_v),
            inverse_scale_v));
        __m256i z_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(
                _mm256_loadu_ps(input_row_float + col + 2 * VLEN), min_v),
            inverse_scale_v));
        __m256i w_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(
                _mm256_loadu_ps(input_row_float + col + 3 * VLEN), min_v),
            inverse_scale_v));

        // An instruction sequence to save 32 32-bit integers as 8-bit integers
        __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
        __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
        __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
        xyzw_packed_v =
            _mm256_permutevar8x32_epi32(xyzw_packed_v, permute_mask1_v);

        // saturate to BIT_RATE
        xyzw_packed_v = _mm256_min_epu8(
            xyzw_packed_v,
            _mm256_set1_epi8(static_cast<char>((1 << BIT_RATE) - 1)));

        if (BIT_RATE == 4) {
          // pack into lower 8-bit of each 16-bit
          xyzw_packed_v = _mm256_and_si256(
              _mm256_or_si256(
                  xyzw_packed_v, _mm256_srli_epi16(xyzw_packed_v, 4)),
              _mm256_set1_epi16(0x00ff));
        } else {
          // pack into lower 8-bit of each 32-bit
          xyzw_packed_v = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      xyzw_packed_v, _mm256_srli_epi32(xyzw_packed_v, 6)),
                  _mm256_or_si256(
                      _mm256_srli_epi32(xyzw_packed_v, 8 + 4),
                      _mm256_srli_epi32(xyzw_packed_v, 2 * 8 + 2))),
              _mm256_set1_epi32(0x00ff));
        }

        __m128i out_v;
        if (BIT_RATE == 4) {
          // avx2 doesn't have _mm256_cvtepi16_epi8
          out_v = _mm_packus_epi16(
              _mm256_castsi256_si128(xyzw_packed_v),
              _mm256_extractf128_si256(xyzw_packed_v, 1));
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col / NUM_ELEM_PER_BYTE),
              out_v);
        } else {
          // avx2 doesn't have _mm256_cvtepi32_epi8
          out_v = _mm_packus_epi32(
              _mm256_castsi256_si128(xyzw_packed_v),
              _mm256_extractf128_si256(xyzw_packed_v, 1));
          out_v = _mm_packus_epi16(out_v, out_v);
          _mm_storel_epi64(
              reinterpret_cast<__m128i*>(output_row + col / NUM_ELEM_PER_BYTE),
              out_v);
        }
      }
    }

    for (; col < input_columns; ++col) {
      float X = input_row_float[col];
      std::uint8_t quantized = std::max(
          0,
          std::min<int>(
              std::lrintf((X - minimum_element) * inverse_scale),
              (1 << BIT_RATE) - 1));
      if (col % NUM_ELEM_PER_BYTE == 0) {
        output_row[col / NUM_ELEM_PER_BYTE] = quantized;
      } else {
        output_row[col / NUM_ELEM_PER_BYTE] |=
            (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
      }
    }
  }

  if (std::is_same<InputType, float16>()) {
    fbgemmAlignedFree(input_row_float_for_fp16);
  }
}

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatAvx2(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr int VLEN = 8;
  constexpr float kEpsilon = 1e-8f;

  __m256i permute_mask1_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  // clang-format off
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00);
  // clang-format on

  __m256i permute_mask2_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int output_columns = input_columns + 2 * sizeof(float);
  float* input_row_float_for_fp16;
  if (std::is_same<InputType, float16>()) {
    input_row_float_for_fp16 = static_cast<float*>(
        fbgemmAlignedAlloc(64, input_columns * sizeof(float)));
  }
  for (int row = 0; row < input_rows; ++row) {
    const InputType* input_row = input + row * input_columns;
    const float* input_row_float;
    if (std::is_same<InputType, float>()) {
      // NOTE: this reinterpret_cast is only to workaround c++
      // type requirements -- it is not for fp16 case and `input_row` HAS to be
      // float* type. Remove it and use constexpr when pytorch allows C++17.
      input_row_float = reinterpret_cast<const float*>(input_row);
    } else {
      input_row_float = input_row_float_for_fp16;
    }
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + input_columns);

    float minimum_element = FLT_MAX;
    float maximum_element = -FLT_MAX;
    __m256 min_v = _mm256_set1_ps(minimum_element);
    __m256 max_v = _mm256_set1_ps(maximum_element);
    int col;
    for (col = 0; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v;
      if (std::is_same<InputType, float>()) {
        in_v = _mm256_loadu_ps(input_row_float + col);
      } else {
        __m128i in_half_v =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + col));
        in_v = _mm256_cvtph_ps(in_half_v);
        _mm256_store_ps(input_row_float_for_fp16 + col, in_v);
      }
      min_v = _mm256_min_ps(min_v, in_v);
      max_v = _mm256_max_ps(max_v, in_v);
    }
    alignas(64) float min_buf[VLEN], max_buf[VLEN];
    _mm256_store_ps(min_buf, min_v);
    _mm256_store_ps(max_buf, max_v);
    for (int i = 0; i < VLEN; ++i) {
      minimum_element = std::min(minimum_element, min_buf[i]);
      maximum_element = std::max(maximum_element, max_buf[i]);
    }

    for (; col < input_columns; ++col) {
      if (std::is_same<InputType, float>()) {
        minimum_element = std::min(minimum_element, input_row_float[col]);
        maximum_element = std::max(maximum_element, input_row_float[col]);
      } else {
        float element = halfToFloat(input_row[col]);
        input_row_float_for_fp16[col] = element;
        minimum_element = std::min(minimum_element, element);
        maximum_element = std::max(maximum_element, element);
      }
    }

    float range = maximum_element - minimum_element;
    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    min_v = _mm256_set1_ps(minimum_element);
    __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);

    for (col = 0; col < input_columns / (4 * VLEN) * (4 * VLEN);
         col += 4 * VLEN) {
      __m256i x_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row_float + col), min_v),
          inverse_scale_v));
      __m256i y_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row_float + col + VLEN), min_v),
          inverse_scale_v));
      __m256i z_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(
              _mm256_loadu_ps(input_row_float + col + 2 * VLEN), min_v),
          inverse_scale_v));
      __m256i w_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(
              _mm256_loadu_ps(input_row_float + col + 3 * VLEN), min_v),
          inverse_scale_v));

      // An instruction sequence to save 32 32-bit integers as 8-bit integers
      __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
      __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
      __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
      xyzw_packed_v =
          _mm256_permutevar8x32_epi32(xyzw_packed_v, permute_mask1_v);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(output_row + col), xyzw_packed_v);
    }
    for (; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256i rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row_float + col), min_v),
          inverse_scale_v));

      // An instruction sequence to save 8 32-bit integers as 8-bit integers
      rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v);
      rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask2_v);
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(output_row + col),
          _mm256_castsi256_si128(rounded_v));
    }
    for (; col < input_columns; ++col) {
      output_row[col] =
          std::lrintf((input_row_float[col] - minimum_element) * inverse_scale);
    }
  }
  if (std::is_same<InputType, float16>()) {
    fbgemmAlignedFree(input_row_float_for_fp16);
  }
}

template <typename OutputType, int BIT_RATE>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  static_assert(
      std::is_same<OutputType, float>() || std::is_same<OutputType, float16>(),
      "Only float and float16 types are allowed.");
  constexpr int VLEN = 8;
  constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  int output_columns =
      (input_columns - 2 * sizeof(uint16_t)) * NUM_ELEM_PER_BYTE;

  // Compute a remainder for vector load
  // Since every row is followed by 2 fp16 (scale and bias), luckily
  // we don't need mask at bit-rate granularity but just at 32-bit
  // granularity.
  constexpr int NUM_ELEM_PER_32BIT = 32 / BIT_RATE;
  // multiply by 4 because we're handling 4 vlen per iteration
  constexpr int NUM_OF_32BIT_PER_VLOAD = VLEN * 4 / NUM_ELEM_PER_32BIT;

  int remainder_32bit_granularity, remainder;
  __m128i vmask_load;
  __m256i vmask_store0, vmask_store1, vmask_store2, vmask_store3;
  if (BIT_RATE == 4 || BIT_RATE == 2) {
    remainder_32bit_granularity = (output_columns + NUM_ELEM_PER_32BIT - 1) /
        NUM_ELEM_PER_32BIT % NUM_OF_32BIT_PER_VLOAD;
    vmask_load = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(
        internal::avx2_ps_or_epi32_combined_mask + NUM_OF_32BIT_PER_VLOAD +
        (NUM_OF_32BIT_PER_VLOAD - remainder_32bit_granularity) %
            NUM_OF_32BIT_PER_VLOAD));
    remainder = output_columns % (4 * VLEN);
    int remainder_ratio = 1;
    if (std::is_same<OutputType, float16>()) {
      // For fp16 we only need half of the mask.
      //
      // For instance, if reminder is 2, for FP32 the masks are
      // {-1, -1, 0, ..., 0}, {0, ..., 0}, {0, ..., 0}, {0, ..., 0}
      // (8 32-bit integers for each mask)
      // for FP16 we only need
      // {-1, 0, 0, 0}, {0, ..., 0}, {0, ..., 0}, {0, ..., 0}
      // (4 32-bit integers for each mask)
      // since we reinterpret 2 FP16 numbers as one 32-bit number.
      // NOTE: for bit_rate 4 or 2, reminders are always multiple of 2 or 4,
      // so we do have to worry about odd number of FP16 numbers.
      //
      // Or, if reminder is 30, for FP32 the masks are
      // {-1, ..., -1}, {-1, ..., -1}, {-1, ..., -1}, {-1, .., -1, 0, 0}
      // for FP16 we only need
      // {-1, ..., -1}, {-1, ..., -1}, {-1, ..., -1}, {-1, -1, -1, 0}
      remainder_ratio = 2;
    }
    vmask_store0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_combined_mask +
        (VLEN - std::min(remainder, VLEN) / remainder_ratio % (VLEN + 1))));
    vmask_store1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_combined_mask +
        (VLEN -
         std::max(0, std::min(remainder - VLEN, VLEN) / remainder_ratio) %
             (VLEN + 1))));
    vmask_store2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_combined_mask +
        (VLEN -
         std::max(0, std::min(remainder - 2 * VLEN, VLEN) / remainder_ratio) %
             (VLEN + 1))));
    vmask_store3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_combined_mask +
        (VLEN -
         std::max(0, std::min(remainder - 3 * VLEN, VLEN) / remainder_ratio) %
             (VLEN + 1))));
  }

  for (int row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const uint16_t* input_row_scale_bias = reinterpret_cast<const uint16_t*>(
        input_row +
        (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = halfToFloat(input_row_scale_bias[0]);
    float bias = halfToFloat(input_row_scale_bias[1]);
    OutputType* output_row = output + row * output_columns;
    float* output_row_float;
    if (std::is_same<OutputType, float>()) {
      // NOTE: this reinterpret_cast is only to workaround c++
      // type requirements -- it is not for fp16 case and `output_row` HAS to be
      // float* type. Remove it and use constexpr when pytorch allows C++17.
      output_row_float = reinterpret_cast<float*>(output_row);
    }

    int col = 0;
    if (BIT_RATE == 4 || BIT_RATE == 2) {
      __m256 vscale = _mm256_set1_ps(scale);
      __m256 vbias = _mm256_set1_ps(bias);
      for (; col + 4 * VLEN <= output_columns; col += 4 * VLEN) {
        __m256i vinq;
        // unpack to 8-bit integers
        if (BIT_RATE == 4) {
          vinq = _mm256_cvtepu8_epi16(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                  input_row + col / NUM_ELEM_PER_BYTE)));
          vinq = _mm256_and_si256(
              _mm256_or_si256(vinq, _mm256_slli_epi32(vinq, 4)),
              _mm256_set1_epi16(0x0f0f));
        } else {
          vinq = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                  input_row + col / NUM_ELEM_PER_BYTE)));
          vinq = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      _mm256_slli_epi32(vinq, 2 * 8 + 2),
                      _mm256_slli_epi32(vinq, 8 + 4)),
                  _mm256_or_si256(_mm256_slli_epi32(vinq, 6), vinq)),
              _mm256_set1_epi32(0x03030303));
        }
        __m256 vinq0 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm256_castsi256_si128(vinq)));
        __m256 vinq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 1))));
        __m256 vinq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 2))));
        __m256 vinq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 3))));
        vinq0 = _mm256_fmadd_ps(vscale, vinq0, vbias);
        vinq1 = _mm256_fmadd_ps(vscale, vinq1, vbias);
        vinq2 = _mm256_fmadd_ps(vscale, vinq2, vbias);
        vinq3 = _mm256_fmadd_ps(vscale, vinq3, vbias);

        if (std::is_same<OutputType, float>()) {
          _mm256_storeu_ps(output_row_float + col, vinq0);
          _mm256_storeu_ps(output_row_float + col + VLEN, vinq1);
          _mm256_storeu_ps(output_row_float + col + 2 * VLEN, vinq2);
          _mm256_storeu_ps(output_row_float + col + 3 * VLEN, vinq3);
        } else {
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col),
              _mm256_cvtps_ph(
                  vinq0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col + VLEN),
              _mm256_cvtps_ph(
                  vinq1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col + 2 * VLEN),
              _mm256_cvtps_ph(
                  vinq2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col + 3 * VLEN),
              _mm256_cvtps_ph(
                  vinq3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
      }

      if (remainder) {
        __m256i vinq;
        if (BIT_RATE == 4) {
          vinq = _mm256_cvtepu8_epi16(_mm_maskload_epi32(
              reinterpret_cast<const int*>(input_row + col / NUM_ELEM_PER_BYTE),
              vmask_load));
          vinq = _mm256_and_si256(
              _mm256_or_si256(vinq, _mm256_slli_epi32(vinq, 4)),
              _mm256_set1_epi16(0x0f0f));
        } else {
          vinq = _mm256_cvtepu8_epi32(_mm_maskload_epi32(
              reinterpret_cast<const int*>(input_row + col / NUM_ELEM_PER_BYTE),
              vmask_load));
          vinq = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      _mm256_slli_epi32(vinq, 2 * 8 + 2),
                      _mm256_slli_epi32(vinq, 8 + 4)),
                  _mm256_or_si256(_mm256_slli_epi32(vinq, 6), vinq)),
              _mm256_set1_epi32(0x03030303));
        }

        __m256 vinq0 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm256_castsi256_si128(vinq)));
        __m256 vinq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 1))));
        __m256 vinq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 2))));
        __m256 vinq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 3))));

        vinq0 = _mm256_fmadd_ps(vscale, vinq0, vbias);
        vinq1 = _mm256_fmadd_ps(vscale, vinq1, vbias);
        vinq2 = _mm256_fmadd_ps(vscale, vinq2, vbias);
        vinq3 = _mm256_fmadd_ps(vscale, vinq3, vbias);

        if (std::is_same<OutputType, float>()) {
          _mm256_maskstore_ps(output_row_float + col, vmask_store0, vinq0);
          _mm256_maskstore_ps(
              output_row_float + col + VLEN, vmask_store1, vinq1);
          _mm256_maskstore_ps(
              output_row_float + col + 2 * VLEN, vmask_store2, vinq2);
          _mm256_maskstore_ps(
              output_row_float + col + 3 * VLEN, vmask_store3, vinq3);
        } else {
          _mm_maskstore_epi32(
              reinterpret_cast<int*>(output_row + col),
              _mm256_castsi256_si128(vmask_store0),
              _mm256_cvtps_ph(
                  vinq0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_maskstore_epi32(
              reinterpret_cast<int*>(output_row + col + VLEN),
              _mm256_castsi256_si128(vmask_store1),
              _mm256_cvtps_ph(
                  vinq1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_maskstore_epi32(
              reinterpret_cast<int*>(output_row + col + 2 * VLEN),
              _mm256_castsi256_si128(vmask_store2),
              _mm256_cvtps_ph(
                  vinq2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          _mm_maskstore_epi32(
              reinterpret_cast<int*>(output_row + col + 3 * VLEN),
              _mm256_castsi256_si128(vmask_store3),
              _mm256_cvtps_ph(
                  vinq3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
      }
    } else {
      for (; col < output_columns; ++col) {
        std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
        quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
        quantized &= (1 << BIT_RATE) - 1;
        float output_value = scale * quantized + bias;
        if (std::is_same<OutputType, float>()) {
          output_row[col] = output_value;
        } else {
          output_row[col] = cpu_float2half_rn(output_value);
        }
      }
    }
  }
}

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfAvx2(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  constexpr int VLEN = 8;
  int output_columns = input_columns - 2 * sizeof(float);

  for (int row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    OutputType* output_row = output + row * output_columns;

    __m256 scale_v = _mm256_set1_ps(input_row_scale_bias[0]);
    __m256 bias_v = _mm256_set1_ps(input_row_scale_bias[1]);

    int col;
    for (col = 0; col < output_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_row + col))));
      __m256 dequantzed_v = _mm256_add_ps(_mm256_mul_ps(in_v, scale_v), bias_v);
      if (std::is_same<OutputType, float>()) {
        float* output_row_float = reinterpret_cast<float*>(output_row);
        _mm256_storeu_ps(output_row_float + col, dequantzed_v);
      } else {
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(output_row + col),
            _mm256_cvtps_ph(
                dequantzed_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      }
    }

    for (; col < output_columns; ++col) {
      float output_value =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
      if (std::is_same<OutputType, float>()) {
        output_row[col] = output_value;
      } else {
        output_row[col] = cpu_float2half_rn(output_value);
      }
    }
  }
}

#define INSTANTIATE_QuantizationAvx2FunctionsNBits(type, bit_rate)  \
  template void                                                     \
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2<type, bit_rate>( \
      const type* input,                                            \
      int input_rows,                                               \
      int input_columns,                                            \
      std::uint8_t* output);                                        \
  template void                                                     \
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2<type, bit_rate>( \
      const std::uint8_t* input,                                    \
      int input_rows,                                               \
      int input_columns,                                            \
      type* output);

// clang-format off
INSTANTIATE_QuantizationAvx2FunctionsNBits(float, 2)
INSTANTIATE_QuantizationAvx2FunctionsNBits(float, 4)
INSTANTIATE_QuantizationAvx2FunctionsNBits(float, 8)
INSTANTIATE_QuantizationAvx2FunctionsNBits(float16, 2)
INSTANTIATE_QuantizationAvx2FunctionsNBits(float16, 4)
INSTANTIATE_QuantizationAvx2FunctionsNBits(float16, 8)
// clang-format on
#undef INSTANTIATE_QuantizationAvx2FunctionsNBits

#define INSTANTIATE_QuantizationAvx2Functions8Bits(type)                 \
  template void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatAvx2<type>( \
      const type* input,                                                 \
      int input_rows,                                                    \
      int input_columns,                                                 \
      std::uint8_t* output);                                             \
  template void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfAvx2<type>( \
      const std::uint8_t* input,                                         \
      int input_rows,                                                    \
      int input_columns,                                                 \
      type* output);

// clang-format off
INSTANTIATE_QuantizationAvx2Functions8Bits(float)
INSTANTIATE_QuantizationAvx2Functions8Bits(float16)
// clang-format on
#undef INSTANTIATE_QuantizationAvx2Functions8Bits

} // namespace fbgemm

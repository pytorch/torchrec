/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <immintrin.h>
#include <cassert> //for assert
#include "fbgemm/spmmUtilsAvx2.h"
#include "./MaskAvx2.h"

namespace fbgemm {

template <
    bool FUSE_RELU,
    bool ACT_SYMMETRIC,
    bool WEIGHT_SYMMETRIC,
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN>
FBGEMM_API void trRequantizeOpt(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const trRequantizationParams_t& r) {
  assert (
      (Q_GRAN != QuantizationGranularity::GROUP) &&
      "GROUP Granularity is not supported");

  // Broadcasted act_times_w_scale / C_scale
  __m256 act_times_w_div_c_v;
  if (Q_GRAN != QuantizationGranularity::OUT_CHANNEL) {
    act_times_w_div_c_v = _mm256_set1_ps(r.act_times_w_scale[0] / r.C_scale);
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

  assert(
      (ACT_SYMMETRIC == (r.act_zero_point == 0)) &&
      "ACT_SYMMETRIC == true if and only if act_zero_point == 0");
  assert(
      (WEIGHT_SYMMETRIC ==
       ((Q_GRAN == QuantizationGranularity::TENSOR &&
         r.weight_zero_points[0] == 0) ||
        r.act_col_offsets == nullptr)) &&
      "WEIGHT_SYMMETRIC == true if and only if weight_zero_point == 0 "
      "or r.act_col_offsets == nullptr");
  assert(
      (HAS_BIAS == (r.bias != nullptr)) &&
      "HAS_BIAS == true if and only if bias != nullptr");

  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(r.C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(r.C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    // Scale weight_row_offset with act_zero_point
    int32_t row_offset = 0;
    if (!ACT_SYMMETRIC) {
      row_offset = r.act_zero_point * r.weight_row_offsets[i];
    }

    __m256i row_offset_v = _mm256_set1_epi32(row_offset);

    int weight_zeropoint_idx = 0;
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
      weight_zeropoint_idx = i;
    }
    __m256 bias_v;
    if (HAS_BIAS) {
      float bias = r.bias[i] / r.act_times_w_scale[weight_zeropoint_idx];
      bias_v = _mm256_set1_ps(bias);
    }

    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
      float act_times_w_div_c = r.act_times_w_scale[weight_zeropoint_idx] /
          r.C_scale;
      act_times_w_div_c_v = _mm256_set1_ps(act_times_w_div_c);
    }

    __m256i weight_zeropoint_v = _mm256_set1_epi32(
        r.weight_zero_points[weight_zeropoint_idx]);

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

      if (!ACT_SYMMETRIC) {
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
        y_v = _mm256_sub_epi32(y_v, row_offset_v);
        z_v = _mm256_sub_epi32(z_v, row_offset_v);
        w_v = _mm256_sub_epi32(w_v, row_offset_v);
      }
      if (!WEIGHT_SYMMETRIC) {
        __m256i col_offset_v = _mm256_mullo_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.act_col_offsets + j - block.col_start)),
            weight_zeropoint_v);
        x_v = _mm256_sub_epi32(x_v, col_offset_v);

        col_offset_v = _mm256_mullo_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.act_col_offsets + VLEN + j - block.col_start)),
            weight_zeropoint_v);
        y_v = _mm256_sub_epi32(y_v, col_offset_v);

        col_offset_v = _mm256_mullo_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.act_col_offsets + 2 * VLEN + j - block.col_start)),
            weight_zeropoint_v);
        z_v = _mm256_sub_epi32(z_v, col_offset_v);

        col_offset_v = _mm256_mullo_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.act_col_offsets + 3 * VLEN + j - block.col_start)),
            weight_zeropoint_v);
        w_v = _mm256_sub_epi32(w_v, col_offset_v);
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
      __m256 xf_v, yf_v, zf_v, wf_v;
      if (HAS_BIAS) {
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), bias_v);
        yf_v = _mm256_add_ps(_mm256_cvtepi32_ps(y_v), bias_v);
        zf_v = _mm256_add_ps(_mm256_cvtepi32_ps(z_v), bias_v);
        wf_v = _mm256_add_ps(_mm256_cvtepi32_ps(w_v), bias_v);
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
        yf_v = _mm256_cvtepi32_ps(y_v);
        zf_v = _mm256_cvtepi32_ps(z_v);
        wf_v = _mm256_cvtepi32_ps(w_v);
      }

      __m256 x_scaled_v, y_scaled_v, z_scaled_v, w_scaled_v;

      x_scaled_v = _mm256_mul_ps(xf_v, act_times_w_div_c_v);
      y_scaled_v = _mm256_mul_ps(yf_v, act_times_w_div_c_v);
      z_scaled_v = _mm256_mul_ps(zf_v, act_times_w_div_c_v);
      w_scaled_v = _mm256_mul_ps(wf_v, act_times_w_div_c_v);

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

      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(out + i * ld_out + j), xyzw_clamped_v);
    } // j loop vectorized and unrolled 4x

    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!ACT_SYMMETRIC) {
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      if (!WEIGHT_SYMMETRIC) {
        __m256i col_offset_v = _mm256_mullo_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.act_col_offsets + j - block.col_start)),
            weight_zeropoint_v);
        x_v = _mm256_sub_epi32(x_v, col_offset_v);
      }
      __m256 xf_v;
      if (HAS_BIAS) {
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), bias_v);
      }
      else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      __m256 x_scaled_v = _mm256_mul_ps(xf_v, act_times_w_div_c_v);
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

      if (!ACT_SYMMETRIC) {
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      if (!WEIGHT_SYMMETRIC) {
        __m256i col_offset_v = _mm256_mullo_epi32(
            _mm256_maskload_epi32(
                r.act_col_offsets + j - block.col_start,
                mask_v),
            weight_zeropoint_v);
        x_v = _mm256_sub_epi32(x_v, col_offset_v);
      }

      __m256 xf_v;
      if (HAS_BIAS) {
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), bias_v);
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }
      __m256 x_scaled_v = _mm256_mul_ps(xf_v, act_times_w_div_c_v);
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

      alignas(64) uint8_t x_clamped_buffer[32];
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(x_clamped_buffer), x_clamped_v);
      for (int k = 0; k < remainder; ++k) {
        out[i * ld_out + j + k] = x_clamped_buffer[k];
      }
    } // j loop remainder
  } // i loop
}

#define CREATE_INSTANCE(                                         \
    FUSE_RELU, ACT_SYMMETRIC, WEIGHT_SYMMETRIC, HAS_BIAS, QGRAN) \
  template FBGEMM_API void trRequantizeOpt<                      \
      FUSE_RELU,                                                 \
      ACT_SYMMETRIC,                                             \
      WEIGHT_SYMMETRIC,                                          \
      HAS_BIAS,                                                  \
      QGRAN>(                                                    \
      uint8_t * out,                                             \
      const int32_t* inp,                                        \
      const block_type_t& block,                                 \
      int ld_out,                                                \
      int ld_in,                                                 \
      const trRequantizationParams_t& r);
CREATE_INSTANCE(true, true, true, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, true, true, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, true, false, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, true, false, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, false, true, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, false, true, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, false, false, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, false, false, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, true, true, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, true, true, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, true, false, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, true, false, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, false, true, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, false, true, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, false, false, true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, false, false, false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, true, true, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, true, true, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, true, false, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, true, false, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, false, true, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, false, true, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, false, false, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(true, false, false, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, true, true, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, true, true, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, true, false, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, true, false, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, false, true, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, false, true, false, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, false, false, true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(
    false,
    false,
    false,
    false,
    QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

}  // namespace fbgemm

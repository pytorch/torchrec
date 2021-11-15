/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm> // for min and max
#include <cassert>
#include <cmath> // for lrintf and sqrt
#include <cstdint>
#include <type_traits> // for is_same

#include <immintrin.h>

namespace fbgemm {

// Almost same as ReQuantizeOutput in OutputProcessing-inh.h but different
// row_offsets for each row because of depth-wise convolution
template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    int K_PER_G,
    typename BIAS_TYPE>
static ALWAYS_INLINE void requantize_(
    std::int32_t A_zero_point,
    const std::int32_t* B_zero_point,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    const std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    int n,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale = nullptr) {
  __m256 multiplier_v = _mm256_setzero_ps();
  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v = _mm256_setzero_ps();
  __m256i B_zero_point_v = _mm256_setzero_si256();
  if (Q_GRAN == QuantizationGranularity::TENSOR) {
    multiplier_v = _mm256_set1_ps(*C_multiplier);
    if (std::is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v = _mm256_set1_ps(1.0 / (*act_times_w_scale));
    }
    B_zero_point_v = _mm256_set1_epi32(B_zero_point[0]);
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<std::uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<std::uint8_t>(255));

  if (A_SYMMETRIC) {
    assert(A_zero_point == 0 || col_offsets == nullptr);
  }
  __m256i A_zero_point_v = _mm256_set1_epi32(A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  int j = 0;
  for (; j < n / (VLEN * 4) * (VLEN * 4); j += (VLEN * 4)) {
    __m256i x_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C_int32 + j));
    __m256i y_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + VLEN));
    __m256i z_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + 2 * VLEN));
    __m256i w_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + 3 * VLEN));

    __m256i row_offset_v;
    if (!B_SYMMETRIC) {
      if (K_PER_G == 1) {
        row_offset_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row_offsets + j));
      } else {
        assert(K_PER_G == 2);
        // Load row_offsets for 4 groups and broadcast by 2 times.
        row_offset_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(
                    reinterpret_cast<const float*>(row_offsets + j / 2))),
                permute_mask_v)));
      }
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + j));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        assert(K_PER_G == 2);
        B_zero_point_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(
                    reinterpret_cast<const float*>(B_zero_point + j / 2))),
                permute_mask_v)));
      }
      row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
      x_v = _mm256_sub_epi32(x_v, row_offset_v);
    }
    __m256i col_off_v;
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j)));
      x_v = _mm256_sub_epi32(x_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      if (K_PER_G == 1) {
        row_offset_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row_offsets + j + VLEN));
      } else {
        row_offset_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        row_offsets + (j + VLEN) / 2))),
                permute_mask_v)));
      }
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + j + VLEN));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        B_zero_point + (j + VLEN) / 2))),
                permute_mask_v)));
      }
      row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
      y_v = _mm256_sub_epi32(y_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + VLEN)));
      y_v = _mm256_sub_epi32(y_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      if (K_PER_G == 1) {
        row_offset_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row_offsets + j + 2 * VLEN));
      } else {
        row_offset_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        row_offsets + (j + 2 * VLEN) / 2))),
                permute_mask_v)));
      }
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + j + 2 * VLEN));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        B_zero_point + (j + 2 * VLEN) / 2))),
                permute_mask_v)));
      }
      row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
      z_v = _mm256_sub_epi32(z_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + 2 * VLEN)));
      z_v = _mm256_sub_epi32(z_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      if (K_PER_G == 1) {
        row_offset_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row_offsets + j + 3 * VLEN));
      } else {
        row_offset_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        row_offsets + (j + 3 * VLEN) / 2))),
                permute_mask_v)));
      }
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + j + 3 * VLEN));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        B_zero_point + (j + 3 * VLEN) / 2))),
                permute_mask_v)));
      }
      row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
      w_v = _mm256_sub_epi32(w_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + 3 * VLEN)));
      w_v = _mm256_sub_epi32(w_v, col_off_v);
    }

    // convert to float
    __m256 xf_v, yf_v, zf_v, wf_v;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        __m256 x_bias_v, y_bias_v, z_bias_v, w_bias_v;
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
            (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 0 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 0 * VLEN));
          y_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 1 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 1 * VLEN));
          z_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 2 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 2 * VLEN));
          w_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 3 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 3 * VLEN));
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          assert(K_PER_G == 2);
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 0 * VLEN)),
              _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(act_times_w_scale + j / 2)),
                  permute_mask_v)));
          y_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 1 * VLEN)),
              _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(act_times_w_scale + (j + VLEN) / 2)),
                  permute_mask_v)));
          z_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 2 * VLEN)),
              _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(act_times_w_scale + (j + 2 * VLEN) / 2)),
                  permute_mask_v)));
          w_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 3 * VLEN)),
              _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(act_times_w_scale + (j + 3 * VLEN) / 2)),
                  permute_mask_v)));
        } else {
          x_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 0 * VLEN)),
              act_times_w_rcp_v);
          y_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 1 * VLEN)),
              act_times_w_rcp_v);
          z_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 2 * VLEN)),
              act_times_w_rcp_v);
          w_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 3 * VLEN)),
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
                reinterpret_cast<const __m256i*>(bias + j + 0 * VLEN)));
        y_v = _mm256_add_epi32(
            y_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 1 * VLEN)));
        z_v = _mm256_add_epi32(
            z_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 2 * VLEN)));
        w_v = _mm256_add_epi32(
            w_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 3 * VLEN)));
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

    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 0 * VLEN);
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(C_multiplier + j / 2)),
          permute_mask_v));
    }
    __m256 x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 1 * VLEN);
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(C_multiplier + (j + VLEN) / 2)),
          permute_mask_v));
    }
    __m256 y_scaled_v = _mm256_mul_ps(yf_v, multiplier_v);
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 2 * VLEN);
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
          _mm256_castps128_ps256(
              _mm_loadu_ps(C_multiplier + (j + 2 * VLEN) / 2)),
          permute_mask_v));
    }
    __m256 z_scaled_v = _mm256_mul_ps(zf_v, multiplier_v);
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 3 * VLEN);
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
          _mm256_castps128_ps256(
              _mm_loadu_ps(C_multiplier + (j + 3 * VLEN) / 2)),
          permute_mask_v));
    }
    __m256 w_scaled_v = _mm256_mul_ps(wf_v, multiplier_v);

    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

    __m256i xy_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
    __m256i zw_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
    __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
    __m256i xyzw_clamped_v = _mm256_max_epu8(
        FUSE_RELU ? C_zero_point_epi8_v : min_v,
        _mm256_min_epu8(xyzw_packed_v, max_v));

    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);

    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(C_uint8 + j), xyzw_clamped_v);
  } // j loop vectorized and unrolled 4x

  for (; j < n / VLEN * VLEN; j += VLEN) {
    __m256i x_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C_int32 + j));

    if (!B_SYMMETRIC) {
      __m256i row_offset_v;
      if (K_PER_G == 1) {
        row_offset_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row_offsets + j));
      } else {
        assert(K_PER_G == 2);
        // Load row_offsets for 4 groups and broadcast by 2 times.
        row_offset_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(
                    reinterpret_cast<const float*>(row_offsets + j / 2))),
                permute_mask_v)));
      }
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + j));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        assert(K_PER_G == 2);
        B_zero_point_v =
            _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(
                    reinterpret_cast<const float*>(B_zero_point + j / 2))),
                permute_mask_v)));
      }
      row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
      x_v = _mm256_sub_epi32(x_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      __m256i col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j)));
      x_v = _mm256_sub_epi32(x_v, col_off_v);
    }

    // Convert to float
    __m256 xf_v;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        __m256 x_bias_v;
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
            (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(reinterpret_cast<const float*>(bias + j)),
              _mm256_loadu_ps(act_times_w_scale + j));
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(reinterpret_cast<const float*>(bias + j)),
              _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(act_times_w_scale + j / 2)),
                  permute_mask_v)));
        } else {
          x_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(reinterpret_cast<const float*>(bias + j)),
              act_times_w_rcp_v);
        }
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
      } else {
        x_v = _mm256_add_epi32(
            x_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + j)));
        xf_v = _mm256_cvtepi32_ps(x_v);
      }
    } else {
      xf_v = _mm256_cvtepi32_ps(x_v);
    }

    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j);
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(C_multiplier + j / 2)),
          permute_mask_v));
    }
    __m256 x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

    __m256i x_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
        C_zero_point_epi16_v);
    x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
    __m256i x_clamped_v = _mm256_max_epu8(
        FUSE_RELU ? C_zero_point_epi8_v : min_v,
        _mm256_min_epu8(x_packed_v, max_v));

    x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(C_uint8 + j),
        _mm256_castsi256_si128(x_clamped_v));
  } // j loop vectorized

  for (; j < n; ++j) {
    std::int32_t raw = C_int32[j];
    int quant_param_idx = 0;
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      quant_param_idx = j;
    } else if (Q_GRAN == QuantizationGranularity::GROUP) {
      quant_param_idx = j / 2;
    }
    if (!B_SYMMETRIC) {
      raw -= B_zero_point[quant_param_idx] * row_offsets[j / K_PER_G];
    }
    if (!A_SYMMETRIC) {
      raw -= A_zero_point * col_offsets[j];
    }
    float raw_f;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        raw_f = raw;
        raw_f += bias[j] / act_times_w_scale[quant_param_idx];
      } else {
        raw += bias[j];
        raw_f = raw;
      }
    } else {
      raw_f = raw;
    }

    float ab = raw_f * C_multiplier[quant_param_idx];
    long rounded = lrintf(ab) + C_zero_point;

    C_uint8[j] = std::max(
        FUSE_RELU ? static_cast<long>(C_zero_point) : 0l,
        std::min(255l, rounded));
  }
}

static inline std::pair<int, int> closest_factors_(int n) {
  int a = static_cast<int>(std::sqrt(n));
  while (n % a != 0) {
    a--;
  }
  return {a, n / a}; // a <= n / a
}

} // namespace fbgemm

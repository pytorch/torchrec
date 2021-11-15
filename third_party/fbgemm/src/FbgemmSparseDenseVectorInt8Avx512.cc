/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/Utils.h"
#include "fbgemm/spmmUtilsAvx2.h"

#include <immintrin.h>
#include <cassert>

namespace fbgemm {

namespace internal {

static inline int32_t horizontal_add(__m256i a) {
  __m256i t1 = _mm256_hadd_epi32(a, a);
  __m256i t2 = _mm256_hadd_epi32(t1, t1);
  __m128i t3 = _mm256_extracti128_si256(t2, 1);
  __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(t2), t3);
  return _mm_cvtsi128_si32(t4);
}

template <
    bool FUSE_RELU,
    bool ACT_ZP_0, // is activation zero point 0?
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN>
static inline void requantizeForMV(
    uint8_t* dst,
    int32_t* src,
    int len,
    trRequantizationParams_t& rParams) {
  constexpr int VLEN_INT32 = 16;
  __m512i C_zero_point_epi8_v = _mm512_set1_epi8(rParams.C_zero_point);
  __m512i C_zero_point_epi32_v = _mm512_set1_epi32(rParams.C_zero_point);
  // clang-format off
  __m512i permute_mask_v = _mm512_set_epi32(
      0x0F, 0x0B, 0x07, 0x03,
      0x0E, 0x0A, 0x06, 0x02,
      0x0D, 0x09, 0x05, 0x01,
      0x0C, 0x08, 0x04, 0x00);
  // clang-format on
  int i = 0;
  for (; i < len / VLEN_INT32 * VLEN_INT32; i += VLEN_INT32) {
    __m512i x_v = _mm512_loadu_si512(src + i);
    if (!ACT_ZP_0) {
      __m512i weight_row_offset_v =
          _mm512_loadu_si512(rParams.weight_row_offsets + i);
      __m512i act_zero_point_v = _mm512_set1_epi32(rParams.act_zero_point);
      x_v = _mm512_sub_epi32(
          x_v, _mm512_mullo_epi32(act_zero_point_v, weight_row_offset_v));
    }
    __m512 act_times_w_scale_v;
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
      act_times_w_scale_v = _mm512_loadu_ps(rParams.act_times_w_scale + i);
    } else {
      act_times_w_scale_v = _mm512_set1_ps(rParams.act_times_w_scale[0]);
    }
    __m512 c_scale_v = _mm512_set1_ps(rParams.C_scale);
    __m512 act_times_w_div_c_v = _mm512_div_ps(act_times_w_scale_v, c_scale_v);

    __m512 xf_v;
    if (HAS_BIAS) {
      __m512 bias_v = _mm512_loadu_ps(rParams.bias + i);
      bias_v = _mm512_div_ps(bias_v, act_times_w_scale_v);
      xf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x_v), bias_v);
    } else {
      xf_v = _mm512_cvtepi32_ps(x_v);
    }

    __m512 x_scaled_v = _mm512_mul_ps(xf_v, act_times_w_div_c_v);
    __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);
    __m512i x_added_v = _mm512_add_epi32(x_rounded_v, C_zero_point_epi32_v);

    __m512i x_clamped_v = _mm512_packs_epi32(x_added_v, _mm512_setzero_si512());
    x_clamped_v = _mm512_packus_epi16(x_clamped_v, _mm512_setzero_si512());
    if (FUSE_RELU) {
      x_clamped_v = _mm512_max_epu8(C_zero_point_epi8_v, x_clamped_v);
    }
    x_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, x_clamped_v);

    _mm_store_si128(
        reinterpret_cast<__m128i*>(dst + i),
        _mm512_castsi512_si128(x_clamped_v));
  }
  int rem_int32 = len - i;
  if (rem_int32 > 0) {
    __mmask64 mask_int8_v = (((long long)1) << rem_int32) - 1;
    __mmask16 mask_int32_v = (((long long)1) << rem_int32) - 1;
    __m512i x_v = _mm512_maskz_loadu_epi32(mask_int32_v, src + i);

    if (!ACT_ZP_0) {
      __m512i weight_row_offset_v =
          _mm512_maskz_loadu_epi32(mask_int32_v, rParams.weight_row_offsets + i);
      __m512i act_zero_point_v = _mm512_set1_epi32(rParams.act_zero_point);
      x_v = _mm512_sub_epi32(
          x_v, _mm512_mullo_epi32(act_zero_point_v, weight_row_offset_v));
    }
    __m512 act_times_w_scale_v;
    if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
      act_times_w_scale_v =
          _mm512_maskz_loadu_ps(mask_int32_v, rParams.act_times_w_scale + i);
    } else {
      act_times_w_scale_v = _mm512_set1_ps(rParams.act_times_w_scale[0]);
    }
    __m512 c_scale_v = _mm512_set1_ps(rParams.C_scale);
    __m512 act_times_w_div_c_v = _mm512_div_ps(act_times_w_scale_v, c_scale_v);

    __m512 xf_v;
    if (HAS_BIAS) {
      __m512 bias_v = _mm512_maskz_loadu_ps(mask_int32_v, rParams.bias + i);
      bias_v = _mm512_div_ps(bias_v, act_times_w_scale_v);
      xf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x_v), bias_v);
    } else {
      xf_v = _mm512_cvtepi32_ps(x_v);
    }

    __m512 x_scaled_v = _mm512_mul_ps(xf_v, act_times_w_div_c_v);
    __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);
    __m512i x_added_v = _mm512_add_epi32(x_rounded_v, C_zero_point_epi32_v);

    __m512i x_clamped_v = _mm512_packs_epi32(x_added_v, _mm512_setzero_si512());
    x_clamped_v = _mm512_packus_epi16(x_clamped_v, _mm512_setzero_si512());
    if (FUSE_RELU) {
      x_clamped_v = _mm512_max_epu8(C_zero_point_epi8_v, x_clamped_v);
    }
    x_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, x_clamped_v);

    _mm512_mask_storeu_epi8(dst + i, mask_int8_v, x_clamped_v);
  }
}

// matrix-vector product
// i.e., produces same results as SparseDenseInt8MMAvx512 with N == 1
template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MVAvx512(
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    trRequantizationParams_t& rParams,
    bool accum,
    int thread_id,
    int num_threads) {
  // Calcualtes accum ? C += A * B : C = A * B
  constexpr int VLEN_INT32 = 16;

  constexpr int block_size = BCSRMatrix<>::CB;
  constexpr int colTileSize = BCSRMatrix<>::COLTILE;

  // all work is done by thread 0 for now
  assert(num_threads > 0 && "Numbers of threads should be > 0");
  if (thread_id > 0) {
    return;
  }

  assert(ldb == 1 && "ldb should be 1");
  __m512i one_16bit_v = _mm512_set1_epi16(1);
  // Number of columns in the sparse matrix A
  int K = bcsr->C;
  int M = bcsr->R;
  assert(K % 4 == 0 && "K should be multiple of 4");
  assert((K > 0) && "K needs to be positive");
  int kTiles = (K + colTileSize - 1) / colTileSize;
  const int* row_ptr = bcsr->rowBPtr.data();
  const int* col_idx = bcsr->colBIdx.data();
  const int8_t* values = bcsr->values.data();
  for (int kt = 0; kt < kTiles; ++kt) {
    const int* cur_row_ptr = row_ptr + kt * M;
    const uint8_t* cur_B = B + kt * colTileSize * ldb;
    // TODO: unroll this loop?
    for (int i = 0; i < M; ++i) {
      __m512i res = _mm512_set1_epi32(0);
      int r = cur_row_ptr[i];
      int r_end_aligned = cur_row_ptr[i] +
          (cur_row_ptr[i + 1] - cur_row_ptr[i]) / VLEN_INT32 * VLEN_INT32;
      for (; r < r_end_aligned; r += VLEN_INT32) {
        __m512i a_v = _mm512_loadu_si512(values + r * block_size);
        __m512i b_idx = _mm512_loadu_si512(col_idx + r);
        __m512i b_v = _mm512_i32gather_epi32(
            b_idx, reinterpret_cast<const int32_t*>(cur_B), block_size);
        __m512i c_i16_v = _mm512_maddubs_epi16(b_v, a_v);
        __m512i c_i32_v = _mm512_madd_epi16(one_16bit_v, c_i16_v);
        res = _mm512_add_epi32(res, c_i32_v);
      }

      int rem = cur_row_ptr[i + 1] - r;
      if (rem > 0) {
        __mmask16 mask_int32_v = (((long long)1) << (rem)) - 1;
        __m512i a_v =
            _mm512_maskz_loadu_epi32(mask_int32_v, values + r * block_size);
        __m512i b_idx = _mm512_maskz_loadu_epi32(mask_int32_v, col_idx + r);
        __m512i b_v = _mm512_i32gather_epi32(
            b_idx, reinterpret_cast<const int32_t*>(cur_B), block_size);
        __m512i c_i16_v = _mm512_maddubs_epi16(b_v, a_v);
        __m512i c_i32_v = _mm512_madd_epi16(one_16bit_v, c_i16_v);
        res = _mm512_add_epi32(res, c_i32_v);
      }
      // Horizontal reduce
      // _mm512_reduce_add_epi32 is only available for gcc version > 7
#if __GNUC__ >= 7
      int32_t res_i32 = _mm512_reduce_add_epi32(res);
#else
      __m256i low = _mm512_castsi512_si256(res);
      __m256i high = _mm512_extracti64x4_epi64(res, 1);
      int32_t res_i32 = horizontal_add(_mm256_add_epi32(low, high));
#endif

      // store the results
      if (accum || kt > 0) {
        C_i32[i] += res_i32;
      } else {
        C_i32[i] = res_i32;
      }
    }
  }
  if (rParams.bias == nullptr) {
    if (rParams.act_zero_point) {
      requantizeForMV<FUSE_RELU, false, false, Q_GRAN>(C_u8, C_i32, M, rParams);
    } else {
      requantizeForMV<FUSE_RELU, true, false, Q_GRAN>(C_u8, C_i32, M, rParams);
    }
  } else {
    if (rParams.act_zero_point) {
      requantizeForMV<FUSE_RELU, false, true, Q_GRAN>(C_u8, C_i32, M, rParams);
    } else {
      requantizeForMV<FUSE_RELU, true, true, Q_GRAN>(C_u8, C_i32, M, rParams);
    }
  }
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)                  \
  template void SparseDenseInt8MVAvx512<FUSE_RELU, QGRAN>( \
      const std::unique_ptr<BCSRMatrix<>>& bcsr,           \
      const uint8_t* B,                                    \
      int ldb,                                             \
      int32_t* C_i32,                                      \
      uint8_t* C_u8,                                       \
      trRequantizationParams_t& rParams,                   \
      bool accum,                                          \
      int thread_id,                                       \
      int num_threads);
CREATE_INSTANCE(true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

} // namespace internal
} // namespace fbgemm


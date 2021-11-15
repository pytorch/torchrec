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
#include <algorithm> // for min and max
#include <cassert>

namespace fbgemm {

namespace internal {

template <
    bool FUSE_RELU,
    bool ACT_ZP_0, // is activation zero point 0?
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN>
static inline __m512i
requantizeForMM(__m512i x[], int rowIdx, trRequantizationParams_t& rParams) {
  __m512i C_zero_point_epi8_v = _mm512_set1_epi8(rParams.C_zero_point);
  __m512i C_zero_point_epi16_v = _mm512_set1_epi16(rParams.C_zero_point);
  // clang-format off
  __m512i permute_mask_v = _mm512_set_epi32(
      0x0F, 0x0B, 0x07, 0x03,
      0x0E, 0x0A, 0x06, 0x02,
      0x0D, 0x09, 0x05, 0x01,
      0x0C, 0x08, 0x04, 0x00);
  // clang-format on
  int32_t row_offset = 0;
  if (!ACT_ZP_0) {
    row_offset = rParams.act_zero_point * rParams.weight_row_offsets[rowIdx];
  }
  __m512i row_offset_v = _mm512_set1_epi32(row_offset);

  int weight_zeropoint_idx = 0;
  if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
    weight_zeropoint_idx = rowIdx;
  }

  __m512 bias_v;
  if (HAS_BIAS) {
    float bias =
        rParams.bias[rowIdx] / rParams.act_times_w_scale[weight_zeropoint_idx];
    bias_v = _mm512_set1_ps(bias);
  }

  __m512 act_times_w_div_c_v;
  if (Q_GRAN == QuantizationGranularity::TENSOR) {
    act_times_w_div_c_v =
        _mm512_set1_ps(rParams.act_times_w_scale[0] / rParams.C_scale);
  } else {
    act_times_w_div_c_v = _mm512_set1_ps(
        rParams.act_times_w_scale[weight_zeropoint_idx] / rParams.C_scale);
  }
  if (!ACT_ZP_0) {
    x[0] = _mm512_sub_epi32(x[0], row_offset_v);
    x[1] = _mm512_sub_epi32(x[1], row_offset_v);
    x[2] = _mm512_sub_epi32(x[2], row_offset_v);
    x[3] = _mm512_sub_epi32(x[3], row_offset_v);
  }

  __m512 xf_v, yf_v, zf_v, wf_v;
  if (HAS_BIAS) {
    xf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x[0]), bias_v);
    yf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x[1]), bias_v);
    zf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x[2]), bias_v);
    wf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x[3]), bias_v);
  } else {
    xf_v = _mm512_cvtepi32_ps(x[0]);
    yf_v = _mm512_cvtepi32_ps(x[1]);
    zf_v = _mm512_cvtepi32_ps(x[2]);
    wf_v = _mm512_cvtepi32_ps(x[3]);
  }

  __m512 x_scaled_v, y_scaled_v, z_scaled_v, w_scaled_v;

  x_scaled_v = _mm512_mul_ps(xf_v, act_times_w_div_c_v);
  y_scaled_v = _mm512_mul_ps(yf_v, act_times_w_div_c_v);
  z_scaled_v = _mm512_mul_ps(zf_v, act_times_w_div_c_v);
  w_scaled_v = _mm512_mul_ps(wf_v, act_times_w_div_c_v);

  __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);
  __m512i y_rounded_v = _mm512_cvtps_epi32(y_scaled_v);
  __m512i z_rounded_v = _mm512_cvtps_epi32(z_scaled_v);
  __m512i w_rounded_v = _mm512_cvtps_epi32(w_scaled_v);

  __m512i xy_packed_v = _mm512_adds_epi16(
      _mm512_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
  __m512i zw_packed_v = _mm512_adds_epi16(
      _mm512_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
  // _mm512_packus_epi16 takes care of saturating to uint8 range
  __m512i xyzw_clamped_v = _mm512_packus_epi16(xy_packed_v, zw_packed_v);
  if (FUSE_RELU) {
    xyzw_clamped_v = _mm512_max_epu8(C_zero_point_epi8_v, xyzw_clamped_v);
  }

  xyzw_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
  return xyzw_clamped_v;
}

static inline __m512i permute_row(__m512i row) {
  // clang-format off
  __m256i shuffle_256v = _mm256_set_epi8(
      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  // clang-format on
  __m512i shuffle_512v = _mm512_castsi256_si512(shuffle_256v);
  row = _mm512_shuffle_epi8(
      row, _mm512_inserti64x4(shuffle_512v, shuffle_256v, 1));
  return row;
}

static inline void interleave_4rows(__m512i data[]) {
  __m512i __t0 = _mm512_unpacklo_epi32(data[0], data[1]);
  __m512i __t1 = _mm512_unpackhi_epi32(data[0], data[1]);
  __m512i __t2 = _mm512_unpacklo_epi32(data[2], data[3]);
  __m512i __t3 = _mm512_unpackhi_epi32(data[2], data[3]);
  __m512i __tt0 = _mm512_unpacklo_epi64(__t0, __t2);
  __m512i __tt1 = _mm512_unpacklo_epi64(__t1, __t3);
  __m512i __tt2 = _mm512_unpackhi_epi64(__t0, __t2);
  __m512i __tt3 = _mm512_unpackhi_epi64(__t1, __t3);
  __m512i row0 = _mm512_permutex2var_epi64(
      __tt0, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), __tt2);
  __m512i row1 = _mm512_permutex2var_epi64(
      __tt1, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), __tt3);
  __m512i row2 = _mm512_permutex2var_epi64(
      __tt0, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), __tt2);
  __m512i row3 = _mm512_permutex2var_epi64(
      __tt1, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), __tt3);
  __m512i row0i = _mm512_shuffle_i64x2(row0, row1, 0x44);
  __m512i row1i = _mm512_shuffle_i64x2(row0, row1, 0xEE);
  __m512i row2i = _mm512_shuffle_i64x2(row2, row3, 0x44);
  __m512i row3i = _mm512_shuffle_i64x2(row2, row3, 0xEE);
  // End of int32 transpose
  // Now we only need a simple row permutation to get the right result
  data[0] = permute_row(row0i);
  data[1] = permute_row(row1i);
  data[2] = permute_row(row2i);
  data[3] = permute_row(row3i);
  return;
}

// By default, we proecess 4 column blocks (default value of COLBLOCKS).
// Each block is of size 64 (VLEN_INT8) uint8 values.
// We can change the number of blocks we process by
// using a different value for COLBLOCKS.
// For example, for cases with 16 <= N <= 32 and 4 rows are interleaved,
// we only need to process the first 32 columns, which means
// COLBLOCKS = 32 * 4 (row_interleave) / 64 (VLEN_INT8) = 2.
template <int UNROLL = 3, int COLBLOCKS = 4>
static inline void loopOverReductionDim(
    const int* row_ptr,
    int rowIdx,
    const int* col_idx,
    const int8_t* values,
    const uint8_t* interleave_buffer,
    __m512i one_16bit_v,
    __m512i c_v[]) {
  constexpr int VLEN_INT8 = 64;

  int r = row_ptr[rowIdx];
  int r_end_aligned = row_ptr[rowIdx] +
      (row_ptr[rowIdx + 1] - row_ptr[rowIdx]) / UNROLL * UNROLL;
  for (; r < r_end_aligned; r += UNROLL) {
    __m512i a_v[UNROLL];
    int acbr_block[UNROLL];
    for (int i = 0; i < UNROLL; ++i) {
      acbr_block[i] = col_idx[r + i];
      int32_t v = reinterpret_cast<const int32_t*>(values)[r + i];
      a_v[i] = _mm512_set1_epi32(v);
    }

    __m512i br_v[UNROLL][COLBLOCKS];
    for (int i = 0; i < UNROLL; ++i) {
      for (int idx = 0; idx < COLBLOCKS; ++idx) {
        br_v[i][idx] = _mm512_loadu_si512(
            interleave_buffer + (acbr_block[i] * COLBLOCKS + idx) * VLEN_INT8);
      }
    }

    for (int i = 0; i < UNROLL; ++i) {
      for (int idx = 0; idx < COLBLOCKS; ++idx) {
        __m512i c_i16_v = _mm512_maddubs_epi16(br_v[i][idx], a_v[i]);
        __m512i c_i32_v = _mm512_madd_epi16(one_16bit_v, c_i16_v);
        c_v[idx] = _mm512_add_epi32(c_v[idx], c_i32_v);
      }
    }
  }
  // remainder loop
  for (; r < row_ptr[rowIdx + 1]; ++r) {
    int acbr_block = col_idx[r];
    int32_t v = reinterpret_cast<const int32_t*>(values)[r];
    __m512i a_v = _mm512_set1_epi32(v);

    __m512i br_v[COLBLOCKS];
    for (int idx = 0; idx < COLBLOCKS; ++idx) {
      br_v[idx] = _mm512_loadu_si512(
          interleave_buffer + (acbr_block * COLBLOCKS + idx) * VLEN_INT8);
    }

    for (int idx = 0; idx < COLBLOCKS; ++idx) {
      __m512i c_i16_v = _mm512_maddubs_epi16(br_v[idx], a_v);
      __m512i c_i32_v = _mm512_madd_epi16(one_16bit_v, c_i16_v);
      c_v[idx] = _mm512_add_epi32(c_v[idx], c_i32_v);
    }
  }
}

template <int ROWSIZE = 4, bool MASKLOAD = false>
static inline void loadBRows(
    __m512i br_v[],
    const uint8_t* B_start,
    int ld,
    __mmask64 mask_int8_v = 0) {
  int idx = 0;
  for (; idx < ROWSIZE; ++idx) {
    if (MASKLOAD) {
      br_v[idx] = _mm512_maskz_loadu_epi8(mask_int8_v, B_start + idx * ld);
    } else {
      br_v[idx] = _mm512_loadu_si512(B_start + idx * ld);
    }
  }
  // set rests to 0
  for (; idx < 4; ++idx) {
    br_v[idx] = _mm512_set1_epi32(0);
  }
}

// For COLBLOCKS, see description at loopOverReductionDim
template <int COLBLOCKS = 4>
static inline void
storeToInterleaveBuffer(__m512i br_v[], uint8_t* interleave_start, int ld) {
  for (int idx = 0; idx < COLBLOCKS; ++idx) {
    _mm512_storeu_si512(interleave_start + idx * ld, br_v[idx]);
  }
}

// For COLBLOCKS, see description at loopOverReductionDim
template <int COLBLOCKS = 4>
static inline void interleave4RowsTile(
    int N,
    int kSize,
    const uint8_t* B,
    uint8_t* interleave_buffer,
    int ld,
    const int col_start) {
  constexpr int VLEN_INT8 = 64;
  constexpr int colBlockSize = 4;
  assert(colBlockSize == 4 && "column block size should be 4");
  const int kBlocks = kSize / colBlockSize;
  if (col_start < N / VLEN_INT8 * VLEN_INT8) {
    __m512i br_v[4];
    int i = 0;
    for (; i < kBlocks; ++i) {
      loadBRows<4, false>(br_v, B + i * colBlockSize * ld + col_start, ld);
      interleave_4rows(br_v);
      storeToInterleaveBuffer<4>(
          br_v, interleave_buffer + i * colBlockSize * VLEN_INT8, VLEN_INT8);
    }
    int rem = kSize - i * colBlockSize;
    if (rem > 0) {
      if (rem == 3) {
        loadBRows<3, false>(br_v, B + i * colBlockSize * ld + col_start, ld);
      } else if (rem == 2) {
        loadBRows<2, false>(br_v, B + i * colBlockSize * ld + col_start, ld);
      } else {
        loadBRows<1, false>(br_v, B + i * colBlockSize * ld + col_start, ld);
      }
      interleave_4rows(br_v);
      storeToInterleaveBuffer<4>(
          br_v, interleave_buffer + i * colBlockSize * VLEN_INT8, VLEN_INT8);
    }
  } else {
    int rem_int8 = N - col_start;
    __mmask64 mask_int8_v = (((long long)1) << rem_int8) - 1;
    __m512i br_v[4];
    int i = 0;
    for (; i < kBlocks; ++i) {
      loadBRows<4, true>(
          br_v, B + i * colBlockSize * ld + col_start, ld, mask_int8_v);
      interleave_4rows(br_v);
      storeToInterleaveBuffer<COLBLOCKS>(
          br_v, interleave_buffer + i * COLBLOCKS * VLEN_INT8, VLEN_INT8);
    }
    int rem = kSize - i * colBlockSize;
    if (rem > 0) {
      if (rem == 3) {
        loadBRows<3, true>(
            br_v, B + i * colBlockSize * ld + col_start, ld, mask_int8_v);
      } else if (rem == 2) {
        loadBRows<2, true>(
            br_v, B + i * colBlockSize * ld + col_start, ld, mask_int8_v);
      } else {
        loadBRows<1, true>(
            br_v, B + i * colBlockSize * ld + col_start, ld, mask_int8_v);
      }
      interleave_4rows(br_v);
      storeToInterleaveBuffer<COLBLOCKS>(
          br_v, interleave_buffer + i * COLBLOCKS * VLEN_INT8, VLEN_INT8);
    }
  }
  return;
}

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MMAvx512(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum,
    int thread_id,
    int num_threads) {
  // gemv
  if (N == 1 && ldb == 1 && ldc == 1 && bcsr->C % 4 == 0) {
    return SparseDenseInt8MVAvx512<FUSE_RELU, Q_GRAN>(
        bcsr, B, ldb, C_i32, C_u8, rParams, accum, thread_id, num_threads);
  }

  // Calcualtes accum ? C += A * B : C = A * B
  constexpr int VLEN_INT8 = 64;
  constexpr int VLEN_INT32 = 16;

  constexpr int colTileSize = BCSRMatrix<>::COLTILE;
  // Number of columns in the sparse matrix A
  int K = bcsr->C;
  int M = bcsr->R;
  assert((K > 0) && "K needs to be positive");
  int kTiles = (K + colTileSize - 1) / colTileSize;
  const int* row_ptr = bcsr->rowBPtr.data();
  const int* col_idx = bcsr->colBIdx.data();
  const int8_t* values = bcsr->values.data();

  constexpr int buffer_size = BCSRMatrix<>::COLTILE * VLEN_INT8;
  static thread_local uint8_t* interleave_buffer_ = nullptr;

  if (interleave_buffer_ == nullptr) {
    interleave_buffer_ =
        static_cast<uint8_t*>(fbgemmAlignedAlloc(64, buffer_size));
  }

  assert(
      (interleave_buffer_ != nullptr) &&
      "interleave_buffer_ cannot be nullptr");

  __m512i one_16bit_v = _mm512_set1_epi16(1);
  int j = 0;
  for (; j < N / VLEN_INT8 * VLEN_INT8; j += VLEN_INT8) {
    for (int kt = 0; kt < kTiles; ++kt) {
      int curKSize = std::min(K - kt * colTileSize, colTileSize);
      interleave4RowsTile<4 /*COLBLOCKS*/>(
          N, curKSize, B + kt * colTileSize * ldb, interleave_buffer_, ldb, j);
      for (int i = 0; i < M; ++i) {
        __m512i c_v[4];
        if (accum || kt > 0) {
          for (int idx = 0; idx < 4; ++idx) {
            c_v[idx] = _mm512_loadu_si512(C_i32 + i * ldb + idx * VLEN_INT32);
          }
        } else {
          for (int idx = 0; idx < 4; ++idx) {
            c_v[idx] = _mm512_set1_epi32(0);
          }
        }

        loopOverReductionDim<2 /*UNROLL*/, 4 /*COLBLOCKS*/>(
            row_ptr + kt * M,
            i,
            col_idx,
            values,
            interleave_buffer_,
            one_16bit_v,
            c_v);

        if (kt == kTiles - 1) {
          // Requantize after last ktile
          __m512i res;
          if (rParams.bias == nullptr) {
            if (rParams.act_zero_point) {
              res = requantizeForMM<FUSE_RELU, false, false, Q_GRAN>(
                  c_v, i, rParams);
            } else {
              res = requantizeForMM<FUSE_RELU, true, false, Q_GRAN>(
                  c_v, i, rParams);
            }
          } else {
            if (rParams.act_zero_point) {
              res = requantizeForMM<FUSE_RELU, false, true, Q_GRAN>(
                  c_v, i, rParams);
            } else {
              res = requantizeForMM<FUSE_RELU, true, true, Q_GRAN>(
                  c_v, i, rParams);
            }
          }
          _mm512_storeu_si512(C_u8 + i * ldc + j, res);
        } else {
          // store the results
          for (int idx = 0; idx < 4; ++idx) {
            _mm512_storeu_si512(C_i32 + i * ldb + idx * VLEN_INT32, c_v[idx]);
          }
        }
      }
    }
  }
  // Handle remainder j loop
  int rem_int8 = N - j;
  int rem_int32 = N % VLEN_INT32;
  int colBlocks = (rem_int8 + VLEN_INT32 - 1) / VLEN_INT32;
  if (rem_int8 > 0) {
    for (int kt = 0; kt < kTiles; ++kt) {
      // last k tile may have less than colTileSize columns of A matrix (aka
      // rows of B)
      int curKSize = std::min(K - kt * colTileSize, colTileSize);
      switch (colBlocks) {
        case 1:
          interleave4RowsTile<1>(
              N,
              curKSize,
              B + kt * colTileSize * ldb,
              interleave_buffer_,
              ldb,
              j);
          break;
        case 2:
          interleave4RowsTile<2>(
              N,
              curKSize,
              B + kt * colTileSize * ldb,
              interleave_buffer_,
              ldb,
              j);
          break;
        case 3:
          interleave4RowsTile<3>(
              N,
              curKSize,
              B + kt * colTileSize * ldb,
              interleave_buffer_,
              ldb,
              j);
          break;
        case 4:
          interleave4RowsTile<4>(
              N,
              curKSize,
              B + kt * colTileSize * ldb,
              interleave_buffer_,
              ldb,
              j);
          break;
        default:
          // not reachable
          break;
      }

      __mmask16 mask_int32_v = (((long long)1) << rem_int32) - 1;
      __mmask64 mask_int8_v = (((long long)1) << rem_int8) - 1;
      for (int i = 0; i < M; ++i) {
        __m512i c_v[4] = {};
        if (accum || kt > 0) {
          int idx = 0;
          for (; idx < rem_int8 / VLEN_INT32; ++idx) {
            c_v[idx] = _mm512_loadu_si512(C_i32 + i * ldb + idx * VLEN_INT32);
          }
          c_v[idx] = _mm512_maskz_loadu_epi32(
              mask_int32_v, C_i32 + i * ldb + idx * VLEN_INT32);
        }

        switch (colBlocks) {
          case 1:
            loopOverReductionDim<3 /*UNROLL*/, 1 /*colBlocks*/>(
                row_ptr + M * kt,
                i,
                col_idx,
                values,
                interleave_buffer_,
                one_16bit_v,
                c_v);
            break;
          case 2:
            loopOverReductionDim<3 /*UNROLL*/, 2 /*colBlocks*/>(
                row_ptr + M * kt,
                i,
                col_idx,
                values,
                interleave_buffer_,
                one_16bit_v,
                c_v);
            break;
          case 3:
            loopOverReductionDim<2 /*UNROLL*/, 3 /*colBlocks*/>(
                row_ptr + M * kt,
                i,
                col_idx,
                values,
                interleave_buffer_,
                one_16bit_v,
                c_v);
            break;
          case 4:
            loopOverReductionDim<2 /*UNROLL*/, 4 /*colBlocks*/>(
                row_ptr + M * kt,
                i,
                col_idx,
                values,
                interleave_buffer_,
                one_16bit_v,
                c_v);
            break;
          default:
            // not reachable
            break;
        }

        if (kt == kTiles - 1) {
          // Requantize after last ktile
          __m512i res;
          if (rParams.bias == nullptr) {
            if (rParams.act_zero_point) {
              res = requantizeForMM<FUSE_RELU, false, false, Q_GRAN>(
                  c_v, i, rParams);
            } else {
              res = requantizeForMM<FUSE_RELU, true, false, Q_GRAN>(
                  c_v, i, rParams);
            }
          } else {
            if (rParams.act_zero_point) {
              res = requantizeForMM<FUSE_RELU, false, true, Q_GRAN>(
                  c_v, i, rParams);
            } else {
              res = requantizeForMM<FUSE_RELU, true, true, Q_GRAN>(
                  c_v, i, rParams);
            }
          }
          _mm512_mask_storeu_epi8(C_u8 + i * ldc + j, mask_int8_v, res);
        } else {
          int idx = 0;
          for (; idx < rem_int8 / VLEN_INT32; ++idx) {
            _mm512_storeu_si512(C_i32 + i * ldb + idx * VLEN_INT32, c_v[idx]);
          }
          _mm512_mask_storeu_epi32(
              C_i32 + i * ldb + idx * VLEN_INT32, mask_int32_v, c_v[idx]);
        }
      }
    }
  }
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)                  \
  template void SparseDenseInt8MMAvx512<FUSE_RELU, QGRAN>( \
      int N,                                               \
      const std::unique_ptr<BCSRMatrix<>>& bcsr,           \
      const uint8_t* B,                                    \
      int ldb,                                             \
      int32_t* C_i32,                                      \
      uint8_t* C_u8,                                       \
      int ldc,                                             \
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

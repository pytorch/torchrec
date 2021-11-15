/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/spmmUtilsAvx2.h"

#include <immintrin.h>
#include <cassert>
#include <cstring>
#include <algorithm> // for min and max
#include "./MaskAvx2.h"

namespace fbgemm {
namespace internal {

static inline __m256i permute_row(__m256i row) {
  // clang-format off
  __m256i ret = _mm256_shuffle_epi8(
      row,
      _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0));
  // clang-format on
  return ret;
}

static inline void interleave_4rows(__m256i data[]) {
  __m256i __t0 = _mm256_unpacklo_epi32(data[0], data[1]);
  __m256i __t1 = _mm256_unpackhi_epi32(data[0], data[1]);
  __m256i __t2 = _mm256_unpacklo_epi32(data[2], data[3]);
  __m256i __t3 = _mm256_unpackhi_epi32(data[2], data[3]);
  __m256i __tt0 = _mm256_unpacklo_epi64(__t0, __t2);
  __m256i __tt1 = _mm256_unpacklo_epi64(__t1, __t3);
  __m256i __tt2 = _mm256_unpackhi_epi64(__t0, __t2);
  __m256i __tt3 = _mm256_unpackhi_epi64(__t1, __t3);
  __m256i row0 = _mm256_permute2x128_si256(__tt0, __tt2, 0x20);
  __m256i row1 = _mm256_permute2x128_si256(__tt1, __tt3, 0x20);
  __m256i row2 = _mm256_permute2x128_si256(__tt0, __tt2, 0x31);
  __m256i row3 = _mm256_permute2x128_si256(__tt1, __tt3, 0x31);
  // End of int32 transpose
  // Now we only need a simple row permutation to get the right result
  data[0] = permute_row(row0);
  data[1] = permute_row(row1);
  data[2] = permute_row(row2);
  data[3] = permute_row(row3);
  return;
}

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
void SparseDenseInt8MMAvx2(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum,
    int /*thread_id*/,
    int /*num_threads*/) {
  // Calcualtes accum ? C += A * B : C = A * B
  constexpr int VLEN_INT8 = 32;
  constexpr int VLEN_INT32 = 8;
  constexpr int rowBlockSize = BCSRMatrix<>::RB;
  (void)rowBlockSize; // Suppress unused variable warning
  constexpr int colBlockSize = BCSRMatrix<>::CB;

  constexpr int colTileSize = BCSRMatrix<>::COLTILE;
  int K = bcsr->C;
  int M = bcsr->R;
  int kTiles = (K + colTileSize - 1) / colTileSize;

  for (int i = 0; i < M; ++i) {
    if (!accum) {
      int j = 0;
      __m256i c_v = _mm256_set1_epi32(0);
      for (; j < N / VLEN_INT32 * VLEN_INT32; j += VLEN_INT32) {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(C_i32 + i * ldc + j), c_v);
      }
      // Handle remainder
      int rem = N - j;
      if (rem > 0) {
        __m256i mask_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            &avx2_ps_or_epi32_combined_mask[VLEN_INT32 - rem]));
        _mm256_maskstore_epi32(
            reinterpret_cast<int32_t*>(C_i32 + i * ldc + j), mask_v, c_v);
      }
    }
    for (int kt = 0; kt < kTiles; ++kt) {
      int* row_ptr = bcsr->rowBPtr.data() + kt * M;
      int* col_idx = bcsr->colBIdx.data();
      int8_t* values = bcsr->values.data();
      int curKSize = std::min(K - kt * colTileSize, colTileSize);

      int r = row_ptr[i];
      // int r_end_aligned = row_ptr[i] + (row_ptr[i + 1] - row_ptr[i]) / 4 * 4;
      // unrolled by 1
      for (; r < row_ptr[i + 1]; ++r) {
        // this is needed for correct operation
        assert(rowBlockSize == 1 && "row block size should be 1");
        assert(colBlockSize == 4 && "column block size should be 4");
        int acbr_block = col_idx[r];
        int32_t v = reinterpret_cast<const int32_t*>(values)[r];
        __m256i a_v = _mm256_set1_epi32(v);
        int j = 0;
        for (; j < N / VLEN_INT8 * VLEN_INT8; j += VLEN_INT8) {
          __m256i br_v[4] = {};

          for (int idx = 0;
               idx < std::min(4, curKSize - acbr_block * colBlockSize);
               ++idx) {
            br_v[idx] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                B + (acbr_block * colBlockSize + idx + kt * colTileSize) * ldb +
                j));
          }

          // interleave these 4 rows
          interleave_4rows(br_v);

          __m256i one_16bit_v = _mm256_set1_epi16(1);
          __m256i c_v[4];
          for (int idx = 0; idx < 4; ++idx) {
            c_v[idx] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                C_i32 + i * ldc + j + idx * VLEN_INT32));
            __m256i c_i16_v = _mm256_maddubs_epi16(br_v[idx], a_v);
            __m256i c_i32_v = _mm256_madd_epi16(one_16bit_v, c_i16_v);
            c_v[idx] = _mm256_add_epi32(c_v[idx], c_i32_v);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(
                    C_i32 + i * ldc + j + idx * VLEN_INT32),
                c_v[idx]);
          }
        }
        // Handle remainder j loop
        int rem = N - j;
        if (rem > 0) {
          __m256i br_v[4] = {};
          for (int idx = 0;
               idx < std::min(4, curKSize - acbr_block * colBlockSize);
               ++idx) {
            uint8_t tmpDest[VLEN_INT8] = {};
            std::memcpy(
                tmpDest,
                B + (acbr_block * colBlockSize + idx + kt * colTileSize) * ldb +
                    j,
                rem);
            br_v[idx] =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpDest));
          }
          // interleave these 4 rows
          interleave_4rows(br_v);

          __m256i c_v[4] = {};
          int idx1 = 0;
          for (; idx1 < rem / VLEN_INT32; ++idx1) {
            c_v[idx1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                C_i32 + i * ldc + j + idx1 * 8));
          }
          int rem_int32 = rem - idx1 * VLEN_INT32;
          __m256i mask_int32_v;
          if (rem_int32 > 0) {
            mask_int32_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                &avx2_ps_or_epi32_combined_mask[VLEN_INT32 - rem_int32]));
            c_v[idx1] = _mm256_maskload_epi32(
                reinterpret_cast<const int*>(
                    C_i32 + i * ldc + j + idx1 * VLEN_INT32),
                mask_int32_v);
          }

          __m256i one_16bit_v = _mm256_set1_epi16(1);
          for (int idx = 0; idx < 4; ++idx) {
            __m256i c_i16_v = _mm256_maddubs_epi16(br_v[idx], a_v);
            __m256i c_i32_v = _mm256_madd_epi16(one_16bit_v, c_i16_v);
            c_v[idx] = _mm256_add_epi32(c_v[idx], c_i32_v);
          }

          int idx2 = 0;
          for (; idx2 < rem / VLEN_INT32; ++idx2) {
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(
                    C_i32 + i * ldc + j + idx2 * VLEN_INT32),
                c_v[idx2]);
          }
          if (rem_int32 > 0) {
            _mm256_maskstore_epi32(
                reinterpret_cast<int*>(C_i32 + i * ldc + j + idx2 * VLEN_INT32),
                mask_int32_v,
                c_v[idx2]);
          }
        }
      }
    }
  }

  block_type_t block{0, M, 0, N};
  if (rParams.bias == nullptr) {
    if (rParams.act_zero_point) {
      trRequantizeOpt<
          FUSE_RELU,
          /*ACT_SYMMETRIC*/ false,
          /*WEIGHT_SYMMETRIC*/ true,
          /*HAS_BIAS*/ false,
          Q_GRAN>(C_u8, C_i32, block, ldc, ldc, rParams);
    } else {
      trRequantizeOpt<
          FUSE_RELU,
          /*ACT_SYMMETRIC*/ true,
          /*WEIGHT_SYMMETRIC*/ true,
          /*HAS_BIAS*/ false,
          Q_GRAN>(C_u8, C_i32, block, ldc, ldc, rParams);
    }
  } else {
    if (rParams.act_zero_point) {
      trRequantizeOpt<
          FUSE_RELU,
          /*ACT_SYMMETRIC*/ false,
          /*WEIGHT_SYMMETRIC*/ true,
          /*HAS_BIAS*/ true,
          Q_GRAN>(C_u8, C_i32, block, ldc, ldc, rParams);
    } else {
      trRequantizeOpt<
          FUSE_RELU,
          /*ACT_SYMMETRIC*/ true,
          /*WEIGHT_SYMMETRIC*/ true,
          /*HAS_BIAS*/ true,
          Q_GRAN>(C_u8, C_i32, block, ldc, ldc, rParams);
    }
  }
}

#define CREATE_INSTANCE(FUSE_RELU, QGRAN)                \
  template void SparseDenseInt8MMAvx2<FUSE_RELU, QGRAN>( \
      int N,                                             \
      const std::unique_ptr<BCSRMatrix<>>& bcsr,         \
      const uint8_t* B,                                  \
      int ldb,                                           \
      int32_t* C_i32,                                    \
      uint8_t* C_u8,                                     \
      int ldc,                                           \
      trRequantizationParams_t& rParams,                 \
      bool accum,                                        \
      int thread_id,                                     \
      int num_threads);
CREATE_INSTANCE(true, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(true, QuantizationGranularity::OUT_CHANNEL)
CREATE_INSTANCE(false, QuantizationGranularity::TENSOR)
CREATE_INSTANCE(false, QuantizationGranularity::OUT_CHANNEL)
#undef CREATE_INSTANCE

} // namespace internal
} // namespace fbgemm

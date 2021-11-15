/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./OptimizedKernelsAvx2.h"
#include <immintrin.h>


namespace fbgemm {

int32_t reduceAvx2(const uint8_t* A, int len) {
  int32_t row_sum = 0;
#if defined(__AVX2__)
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);

  int i;
  // vectorized
  for (i = 0; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
        sum_v,
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(src_v, one_epi8_v), one_epi16_v));
  }

  alignas(64) int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (int k = 0; k < 8; ++k) {
    row_sum += temp[k];
  }

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

#else
  for (int i = 0; i < len; ++i) {
    row_sum += A[i];
  }
#endif
  return row_sum;
}

void transpose_8rows(
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst) {
  constexpr int M = 8;
  int j;
  // vectorized loop
  for (j = 0; j < N / 32 * 32; j += 32) {
    // a : a0 a1 ... a31
    // b : b0 b1 ... b31
    // c : c0 c1 ... c31
    // d : d0 d1 ... d31
    __m256i a = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 0 * ld_src));
    __m256i b = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 1 * ld_src));
    __m256i c = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 2 * ld_src));
    __m256i d = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 3 * ld_src));
    __m256i e = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 4 * ld_src));
    __m256i f = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 5 * ld_src));
    __m256i g = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 6 * ld_src));
    __m256i h = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 7 * ld_src));

    // even-odd interleaving
    // ab_lo : a0 b0 a1 b1 ...  a7  b7 | a16 b16 ... a23 b23
    // ab_hi : a8 b8 a9 b9 ... a15 b15 | a24 b24 ... a31 b31
    // cd_lo : c0 d0 c1 d1 ...  c7  d7 | c16 d16 ... c23 d23
    // cd_hi : c8 d8 c9 d9 ... c15 d15 | c24 d24 ... c31 d31
    __m256i ab_lo = _mm256_unpacklo_epi8(a, b);
    __m256i ab_hi = _mm256_unpackhi_epi8(a, b);
    __m256i cd_lo = _mm256_unpacklo_epi8(c, d);
    __m256i cd_hi = _mm256_unpackhi_epi8(c, d);
    __m256i ef_lo = _mm256_unpacklo_epi8(e, f);
    __m256i ef_hi = _mm256_unpackhi_epi8(e, f);
    __m256i gh_lo = _mm256_unpacklo_epi8(g, h);
    __m256i gh_hi = _mm256_unpackhi_epi8(g, h);

    // 4-row interleaving but permuted at 128-bit granularity
    // abcd0 :  a0  b0  c0  d0 ...  a-d3 | a-d16 ... a-d19
    // abcd1 :  a4  b4  c4  d4 ...  a-d7 | a-d20 ... a-d23
    // abcd2 :  a8  b8  c8  d8 ... a-d11 | a-d24 ... a-d27
    // abcd3 : a12 b12 c12 d12 ... a-d15 | a-d28 ... a-d31
    __m256i abcd0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);
    __m256i abcd1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
    __m256i abcd2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);
    __m256i abcd3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);
    __m256i efgh0 = _mm256_unpacklo_epi16(ef_lo, gh_lo);
    __m256i efgh1 = _mm256_unpackhi_epi16(ef_lo, gh_lo);
    __m256i efgh2 = _mm256_unpacklo_epi16(ef_hi, gh_hi);
    __m256i efgh3 = _mm256_unpackhi_epi16(ef_hi, gh_hi);

    // 8-row interleaving
    __m256i y0 = _mm256_unpacklo_epi32(abcd0, efgh0);
    __m256i y1 = _mm256_unpackhi_epi32(abcd0, efgh0);
    __m256i y2 = _mm256_unpacklo_epi32(abcd1, efgh1);
    __m256i y3 = _mm256_unpackhi_epi32(abcd1, efgh1);
    __m256i y4 = _mm256_unpacklo_epi32(abcd2, efgh2);
    __m256i y5 = _mm256_unpackhi_epi32(abcd2, efgh2);
    __m256i y6 = _mm256_unpacklo_epi32(abcd3, efgh3);
    __m256i y7 = _mm256_unpackhi_epi32(abcd3, efgh3);

    // Storing with 128-bit lanes are permuted so that everything is in order
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 0) * ld_dst),
        _mm256_castsi256_si128(y0));
    *reinterpret_cast<int64_t*>(dst + (j + 1) * ld_dst) =
        _mm256_extract_epi64(y0, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 2) * ld_dst),
        _mm256_castsi256_si128(y1));
    *reinterpret_cast<int64_t*>(dst + (j + 3) * ld_dst) =
        _mm256_extract_epi64(y1, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 4) * ld_dst),
        _mm256_castsi256_si128(y2));
    *reinterpret_cast<int64_t*>(dst + (j + 5) * ld_dst) =
        _mm256_extract_epi64(y2, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 6) * ld_dst),
        _mm256_castsi256_si128(y3));
    *reinterpret_cast<int64_t*>(dst + (j + 7) * ld_dst) =
        _mm256_extract_epi64(y3, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 8) * ld_dst),
        _mm256_castsi256_si128(y4));
    *reinterpret_cast<int64_t*>(dst + (j + 9) * ld_dst) =
        _mm256_extract_epi64(y4, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 10) * ld_dst),
        _mm256_castsi256_si128(y5));
    *reinterpret_cast<int64_t*>(dst + (j + 11) * ld_dst) =
        _mm256_extract_epi64(y5, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 12) * ld_dst),
        _mm256_castsi256_si128(y6));
    *reinterpret_cast<int64_t*>(dst + (j + 13) * ld_dst) =
        _mm256_extract_epi64(y6, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 14) * ld_dst),
        _mm256_castsi256_si128(y7));
    *reinterpret_cast<int64_t*>(dst + (j + 15) * ld_dst) =
        _mm256_extract_epi64(y7, 1);
    *reinterpret_cast<int64_t*>(dst + (j + 16) * ld_dst) =
        _mm256_extract_epi64(y0, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 17) * ld_dst) =
        _mm256_extract_epi64(y0, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 18) * ld_dst) =
        _mm256_extract_epi64(y1, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 19) * ld_dst) =
        _mm256_extract_epi64(y1, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 20) * ld_dst) =
        _mm256_extract_epi64(y2, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 21) * ld_dst) =
        _mm256_extract_epi64(y2, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 22) * ld_dst) =
        _mm256_extract_epi64(y3, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 23) * ld_dst) =
        _mm256_extract_epi64(y3, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 24) * ld_dst) =
        _mm256_extract_epi64(y4, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 25) * ld_dst) =
        _mm256_extract_epi64(y4, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 26) * ld_dst) =
        _mm256_extract_epi64(y5, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 27) * ld_dst) =
        _mm256_extract_epi64(y5, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 28) * ld_dst) =
        _mm256_extract_epi64(y6, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 29) * ld_dst) =
        _mm256_extract_epi64(y6, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 30) * ld_dst) =
        _mm256_extract_epi64(y7, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 31) * ld_dst) =
        _mm256_extract_epi64(y7, 3);
  }

  // scalar loop for remainder
  for (; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      dst[j * ld_dst + i] = src[j + i * ld_src];
    }
  }
}

void spmdmKernelAvx2(
    int N,
    const uint8_t* A_buffer,
    const int32_t* colptr,
    const int8_t* values,
    const int16_t* rowidx,
    int32_t* C_buffer) {
  for (int j = 0; j < N; ++j) {
    int k = colptr[j];
    int k_end_aligned = colptr[j] + (colptr[j + 1] - colptr[j]) / 4 * 4;

    for (; k < k_end_aligned; k += 4) {
      __m256i w =
          _mm256_set1_epi32(*(reinterpret_cast<const int32_t*>(&values[k])));
      __m256i a[4];
      a[0] = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(&A_buffer[rowidx[k + 0] * 32]));
      a[1] = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(&A_buffer[rowidx[k + 1] * 32]));
      a[2] = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(&A_buffer[rowidx[k + 2] * 32]));
      a[3] = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(&A_buffer[rowidx[k + 3] * 32]));

      __m256i a01_lo = _mm256_unpacklo_epi8(a[0], a[1]);
      __m256i a01_hi = _mm256_unpackhi_epi8(a[0], a[1]);
      __m256i a23_lo = _mm256_unpacklo_epi8(a[2], a[3]);
      __m256i a23_hi = _mm256_unpackhi_epi8(a[2], a[3]);

      a[0] = _mm256_unpacklo_epi16(a01_lo, a23_lo);
      a[1] = _mm256_unpackhi_epi16(a01_lo, a23_lo);
      a[2] = _mm256_unpacklo_epi16(a01_hi, a23_hi);
      a[3] = _mm256_unpackhi_epi16(a01_hi, a23_hi);

      __m256i ab[4];
      ab[0] = _mm256_maddubs_epi16(a[0], w);
      ab[1] = _mm256_maddubs_epi16(a[1], w);
      ab[2] = _mm256_maddubs_epi16(a[2], w);
      ab[3] = _mm256_maddubs_epi16(a[3], w);

      __m256i one = _mm256_set1_epi16(1);
      ab[0] = _mm256_madd_epi16(ab[0], one);
      ab[1] = _mm256_madd_epi16(ab[1], one);
      ab[2] = _mm256_madd_epi16(ab[2], one);
      ab[3] = _mm256_madd_epi16(ab[3], one);

      __m256i t[4];
      t[0] = _mm256_permute2f128_si256(ab[0], ab[1], 0x20);
      t[1] = _mm256_permute2f128_si256(ab[2], ab[3], 0x20);
      t[2] = _mm256_permute2f128_si256(ab[0], ab[1], 0x31);
      t[3] = _mm256_permute2f128_si256(ab[2], ab[3], 0x31);

      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 0 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 0 * 8])),
              t[0]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 1 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 1 * 8])),
              t[1]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 2 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 2 * 8])),
              t[2]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 3 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 3 * 8])),
              t[3]));
    }

    int remainder = colptr[j + 1] - k;
    if (remainder > 0) {
      int32_t temp_w = 0;
      for (int r = 0; r < remainder; ++r) {
        (reinterpret_cast<int8_t*>(&temp_w))[r] = values[k + r];
      }
      __m256i w = _mm256_set1_epi32(temp_w);
      __m256i a[4];
      a[0] = _mm256_load_si256(
          reinterpret_cast<const __m256i*>(&A_buffer[rowidx[k + 0] * 32]));
      a[1] = remainder > 1 ? _mm256_load_si256(reinterpret_cast<const __m256i*>(
                                 &A_buffer[rowidx[k + 1] * 32]))
                           : _mm256_setzero_si256();
      a[2] = remainder > 2 ? _mm256_load_si256(reinterpret_cast<const __m256i*>(
                                 &A_buffer[rowidx[k + 2] * 32]))
                           : _mm256_setzero_si256();
      a[3] = _mm256_setzero_si256();

      __m256i a01_lo = _mm256_unpacklo_epi8(a[0], a[1]);
      __m256i a01_hi = _mm256_unpackhi_epi8(a[0], a[1]);
      __m256i a23_lo = _mm256_unpacklo_epi8(a[2], a[3]);
      __m256i a23_hi = _mm256_unpackhi_epi8(a[2], a[3]);

      a[0] = _mm256_unpacklo_epi16(a01_lo, a23_lo);
      a[1] = _mm256_unpackhi_epi16(a01_lo, a23_lo);
      a[2] = _mm256_unpacklo_epi16(a01_hi, a23_hi);
      a[3] = _mm256_unpackhi_epi16(a01_hi, a23_hi);

      __m256i ab[4];
      ab[0] = _mm256_maddubs_epi16(a[0], w);
      ab[1] = _mm256_maddubs_epi16(a[1], w);
      ab[2] = _mm256_maddubs_epi16(a[2], w);
      ab[3] = _mm256_maddubs_epi16(a[3], w);

      __m256i one = _mm256_set1_epi16(1);
      ab[0] = _mm256_madd_epi16(ab[0], one);
      ab[1] = _mm256_madd_epi16(ab[1], one);
      ab[2] = _mm256_madd_epi16(ab[2], one);
      ab[3] = _mm256_madd_epi16(ab[3], one);

      __m256i t[4];
      t[0] = _mm256_permute2f128_si256(ab[0], ab[1], 0x20);
      t[1] = _mm256_permute2f128_si256(ab[2], ab[3], 0x20);
      t[2] = _mm256_permute2f128_si256(ab[0], ab[1], 0x31);
      t[3] = _mm256_permute2f128_si256(ab[2], ab[3], 0x31);

      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 0 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 0 * 8])),
              t[0]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 1 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 1 * 8])),
              t[1]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 2 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 2 * 8])),
              t[2]));
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 3 * 8]),
          _mm256_add_epi32(
              _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(&C_buffer[j * 32 + 3 * 8])),
              t[3]));
    }
  } // for each column of B
}

} // namespace fbgemm

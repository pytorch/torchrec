/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <immintrin.h>
#include <cassert>
#include <cstdint>

#include "./MaskAvx2.h"

namespace fbgemm {

namespace internal {

#ifdef __AVX2__
// NOTE: Make sure every function defined in here has static linkage because
// this header file is included by UtilsAvx512.cc compiled with -mavx512f option

// 4 * 4 = 16 instructions
static inline void transpose_kernel_4x4_sse(
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3
  // b : b0 b1 b2 b3
  // c : c0 c1 c2 c3
  // d : d0 d1 d2 d3
  __m128 a = _mm_loadu_ps(&src[0 * ld_src]);
  __m128 b = _mm_loadu_ps(&src[1 * ld_src]);
  __m128 c = _mm_loadu_ps(&src[2 * ld_src]);
  __m128 d = _mm_loadu_ps(&src[3 * ld_src]);

  // transpose the 4x4 matrix formed by 32-bit elements: Macro from SSE
  // a : a0 b0 c0 d0
  // b : a1 b1 c1 d1
  // c : a2 b2 c2 d2
  // d : a3 b3 c3 d3
  _MM_TRANSPOSE4_PS(a, b, c, d);

  // store from registers to dst
  _mm_storeu_ps(&dst[0 * ld_dst], a);
  _mm_storeu_ps(&dst[1 * ld_dst], b);
  _mm_storeu_ps(&dst[2 * ld_dst], c);
  _mm_storeu_ps(&dst[3 * ld_dst], d);
}

// kernel for transpose mxn where m, n <= 4
// M + (M + 1) / 2 * 2 + 2 * N instructions
template <unsigned M>
static void transpose_kernel_mxn_sse(
    unsigned N,
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst) {
  // clang-format off
  alignas(64) static const int masks[5][4] = {
    {  0,  0,  0,  0, },
    { -1,  0,  0,  0, },
    { -1, -1,  0,  0, },
    { -1, -1, -1,  0, },
    { -1, -1, -1, -1, },
  };
  // clang-format on

  // load from src to registers
  __m128i mask_v = _mm_load_si128(reinterpret_cast<const __m128i*>(masks[N]));
  __m128 input[4];
  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 4; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm_setzero_ps();
  }

  __m128 temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = _mm_setzero_ps();
  }

  mask_v = _mm_load_si128(reinterpret_cast<const __m128i*>(masks[M]));
  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] = _mm_movelh_ps(temp[i / 2], temp[2 + i / 2]);
    } else {
      input[i] = _mm_movehl_ps(temp[2 + i / 2], temp[i / 2]);
    }
    _mm_maskstore_ps(&dst[i * ld_dst], mask_v, input[i]);
  }
}

// 8 * 5 = 40 instructions
static inline void transpose_kernel_8x8_avx2(
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7
  // b : b0 b1 b2 b3 b4 b5 b6 b7
  // c : c0 c1 c2 c3 c4 c5 c6 c7
  // d : d0 d1 d2 d3 d4 d5 d6 d7
  // e : e0 e1 e2 e3 e4 e5 e6 e7
  // f : f0 f1 f2 f3 f4 f5 f6 f7
  // g : g0 g1 g2 g3 g4 g5 g6 g7
  // h : h0 h1 h2 h3 h4 h5 h6 h7
  __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);

  __m256 ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  __m256 abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;
  // unpacking and interleaving 32-bit elements
  // ab0145 : a0 b0 a1 b1 a4 b4 a5 b5
  // ab2367 : a2 b2 a3 b3 a6 b6 a7 b7
  // cd0145 : c0 d0 c1 d1 c4 d4 c5 d5
  // cd2367 : c2 d2 c3 d3 c6 d6 c7 d7
  // ef0145 : e0 f0 e1 f1 e4 f4 e5 f5
  // ef2367 : e2 f2 e3 f3 e6 f6 e7 f7
  // gh0145 : g0 h0 g1 h1 g4 h4 g5 h5
  // gh2367 : g2 h2 g3 h3 g6 h6 g7 h7
  ab0145 = _mm256_unpacklo_ps(a, b);
  ab2367 = _mm256_unpackhi_ps(a, b);
  cd0145 = _mm256_unpacklo_ps(c, d);
  cd2367 = _mm256_unpackhi_ps(c, d);
  ef0145 = _mm256_unpacklo_ps(e, f);
  ef2367 = _mm256_unpackhi_ps(e, f);
  gh0145 = _mm256_unpacklo_ps(g, h);
  gh2367 = _mm256_unpackhi_ps(g, h);

  // shuffling the 32-bit elements
  // abcd04 : a0 b0 c0 d0 a4 b4 c4 d4
  // abcd15 : a1 b1 c1 d1 a5 b5 c5 d5
  // efgh04 : e0 f0 g0 h0 e4 f4 g4 h4
  // efgh15 : e1 f1 g1 h1 e5 b5 c5 d5
  // abcd26 : a2 b2 c2 d2 a6 b6 c6 d6
  // abcd37 : a3 b3 c3 d3 a7 b7 c7 d7
  // efgh26 : e2 f2 g2 h2 e6 f6 g6 h6
  // efgh37 : e3 f3 g3 h3 e7 f7 g7 h7
  abcd04 = _mm256_shuffle_ps(ab0145, cd0145, 0x44);
  abcd15 = _mm256_shuffle_ps(ab0145, cd0145, 0xee);
  efgh04 = _mm256_shuffle_ps(ef0145, gh0145, 0x44);
  efgh15 = _mm256_shuffle_ps(ef0145, gh0145, 0xee);
  abcd26 = _mm256_shuffle_ps(ab2367, cd2367, 0x44);
  abcd37 = _mm256_shuffle_ps(ab2367, cd2367, 0xee);
  efgh26 = _mm256_shuffle_ps(ef2367, gh2367, 0x44);
  efgh37 = _mm256_shuffle_ps(ef2367, gh2367, 0xee);

  // shuffling 128-bit elements
  // a : a0 b0 c0 d0 e0 f0 g0 h0
  // b : a1 b1 c1 d1 e1 f1 g1 h1
  // c : a2 b2 c2 d2 e2 f2 g2 h2
  // d : a3 b3 c3 d3 e3 f3 g3 h3
  // e : a4 b4 c4 d4 e4 f4 g4 h4
  // f : a5 b5 c5 d5 e5 f5 g5 h5
  // g : a6 b6 c6 d6 e6 f6 g6 h6
  // h : a7 b7 c7 d7 e7 f7 g7 h7
  a = _mm256_permute2f128_ps(efgh04, abcd04, 0x02);
  b = _mm256_permute2f128_ps(efgh15, abcd15, 0x02);
  c = _mm256_permute2f128_ps(efgh26, abcd26, 0x02);
  d = _mm256_permute2f128_ps(efgh37, abcd37, 0x02);
  e = _mm256_permute2f128_ps(efgh04, abcd04, 0x13);
  f = _mm256_permute2f128_ps(efgh15, abcd15, 0x13);
  g = _mm256_permute2f128_ps(efgh26, abcd26, 0x13);
  h = _mm256_permute2f128_ps(efgh37, abcd37, 0x13);

  // store from registers to dst
  _mm256_storeu_ps(&dst[0 * ld_dst], a);
  _mm256_storeu_ps(&dst[1 * ld_dst], b);
  _mm256_storeu_ps(&dst[2 * ld_dst], c);
  _mm256_storeu_ps(&dst[3 * ld_dst], d);
  _mm256_storeu_ps(&dst[4 * ld_dst], e);
  _mm256_storeu_ps(&dst[5 * ld_dst], f);
  _mm256_storeu_ps(&dst[6 * ld_dst], g);
  _mm256_storeu_ps(&dst[7 * ld_dst], h);
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned M>
static void transpose_kernel_mxn_avx2(
    unsigned N,
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst) {
  // load from src to registers
  __m256i mask_v = _mm256_load_si256(
      reinterpret_cast<const __m256i*>(internal::avx2_ps_or_epi32_masks[N]));
  __m256 input[8];
  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm256_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 8; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm256_setzero_ps();
  }

  // unpacking and interleaving 32-bit elements
  __m256 temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm256_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm256_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = _mm256_setzero_ps();
  }

  // shuffling the 32-bit elements
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0x44);
    input[4 * i + 1] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0xee);
    input[4 * i + 2] =
        _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0x44);
    input[4 * i + 3] =
        _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0xee);
  }

  // shuffling 128-bit elements
  // store from registers to dst
  mask_v = _mm256_load_si256(
      reinterpret_cast<const __m256i*>(internal::avx2_ps_or_epi32_masks[M]));
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] = _mm256_permute2f128_ps(input[4 + i], input[i], 0x02);
    } else {
      temp[i] = _mm256_permute2f128_ps(input[i], input[i - 4], 0x13);
    }
    _mm256_maskstore_ps(&dst[i * ld_dst], mask_v, temp[i]);
  }
}

inline __m256i permute_row(__m256i row) {
  // clang-format off
  row = _mm256_shuffle_epi8(
      row,
      _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0));
  // clang-format on
  return row;
}

// template <>
inline static void transpose_kernel_8x32_avx2(
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7 ... a31
  // b : b0 b1 b2 b3 b4 b5 b6 b7 ... b31
  // c : c0 c1 c2 c3 c4 c5 c6 c7 ... c31
  // d : d0 d1 d2 d3 d4 d5 d6 d7 ... d31
  // e : e0 e1 e2 e3 e4 e5 e6 e7 ... e31
  // f : f0 f1 f2 f3 f4 f5 f6 f7 ... f31
  // g : g0 g1 g2 g3 g4 g5 g6 g7 ... g31
  // h : h0 h1 h2 h3 h4 h5 h6 h7 ... h31

  // load from src
  __m256i a = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (0 * ld_src)));
  __m256i b = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (1 * ld_src)));
  __m256i c = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (2 * ld_src)));
  __m256i d = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (3 * ld_src)));
  __m256i e = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (4 * ld_src)));
  __m256i f = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (5 * ld_src)));
  __m256i g = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (6 * ld_src)));
  __m256i h = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>((src) + (7 * ld_src)));

  // shuffle in stride of one:
  // t0 : a0 -- a3,  b0 -- b3,  a4 -- a7, b4 -- b7,
  // a16 -- a19, b16 -- b19, a20 -- a23, b20 -- b23

  // t1 : a8 -- a11, b8 -- b11, a12 -- a15, b12 -- b15,
  // a24 -- a27, b24 -- b27, a28 -- a31, b28 -- b31

  // t2 : c0 -- c3,  d0 -- d3,  c4 -- c7, d4 -- d7,
  // c16 -- c19, d16 -- d19, c20 -- c23, d20 -- d23

  __m256i __t0 = _mm256_unpacklo_epi32(a, b);
  __m256i __t1 = _mm256_unpackhi_epi32(a, b);
  __m256i __t2 = _mm256_unpacklo_epi32(c, d);
  __m256i __t3 = _mm256_unpackhi_epi32(c, d);
  __m256i __t4 = _mm256_unpacklo_epi32(e, f);
  __m256i __t5 = _mm256_unpackhi_epi32(e, f);
  __m256i __t6 = _mm256_unpacklo_epi32(g, h);
  __m256i __t7 = _mm256_unpackhi_epi32(g, h);

  // shuffle in stride of two:
  // tt0: a0--a3, b0--b3, c0--c3, d0--d3,
  // a16--a19, b16 -- b19, c16 -- c19, d16--d19

  // tt1: a4 -- a7, b4 -- b7, c8--c11, d8--d11,
  // a20--a23, b20--b23, c20--c23, d20--d23

  // tt2: a8 -- a11, b8 -- b11, c8 -- c11, d8 -- d11,
  // a24 -- a27, b24 -- b27, c24 -- c27, d24 -- d27

  // tt3: a12 -- a15, b12 -- b15, c12--c15, d12--d15,
  // a28--a31, b28--b31, c28--c31, d28--d31

  // tt4:  e0--e3, f0--f3, g0--h3, g0--h3,
  // e16--e19, f16--f19, g16--h19, g16--h19
  __m256i __tt0 = _mm256_unpacklo_epi64(__t0, __t2);
  __m256i __tt1 = _mm256_unpackhi_epi64(__t0, __t2);
  __m256i __tt2 = _mm256_unpacklo_epi64(__t1, __t3);
  __m256i __tt3 = _mm256_unpackhi_epi64(__t1, __t3);
  __m256i __tt4 = _mm256_unpacklo_epi64(__t4, __t6);
  __m256i __tt5 = _mm256_unpackhi_epi64(__t4, __t6);
  __m256i __tt6 = _mm256_unpacklo_epi64(__t5, __t7);
  __m256i __tt7 = _mm256_unpackhi_epi64(__t5, __t7);

  // permute: pack consecutive elements(0-3) together
  // ttt0: a0--d0 a1--d1 a2--d2 a3--d3 a16-d16 a17-d17 a18-d18 a18-d19

  // ttt1: a4--d4 a5--d5 a6--d6 a7--d7 a20-d20 a21-d21 a22-d22 a23-d23

  // ttt2: a8--d8 a9--d9 a10--d10 a11--d11 a24-d24 a25-d25 a26-d26 a27-d27
  __m256i __ttt0 = permute_row(__tt0);
  __m256i __ttt1 = permute_row(__tt1);
  __m256i __ttt2 = permute_row(__tt2);
  __m256i __ttt3 = permute_row(__tt3);
  __m256i __ttt4 = permute_row(__tt4);
  __m256i __ttt5 = permute_row(__tt5);
  __m256i __ttt6 = permute_row(__tt6);
  __m256i __ttt7 = permute_row(__tt7);

  //
  // a: a0-h0 a1-h1 a16-h16 a17-h17
  // b: a2-h2 a3-h3 a18-h18 a19-h19

  // c: a4-h4 a6-h6 a20-h20 a22-h22 (a-h)x(4-7)
  // d: a5-h5 a7-h7 a21-h21 a23-h23 (a-h)x(20-23)

  // e: a8-h8 a9-h9 a24-h24 a25-h25 (a-h)x(8-11)
  // f: a10-h10 a11-h11 a26-h26 a27-h27 (a-h)x(24-27)

  // g: (a-h)x(12-15)
  // h: (a-h)x(28-31)
  a = _mm256_unpacklo_epi32(__ttt0, __ttt4);
  b = _mm256_unpackhi_epi32(__ttt0, __ttt4);
  c = _mm256_unpacklo_epi32(__ttt1, __ttt5);
  d = _mm256_unpackhi_epi32(__ttt1, __ttt5);
  e = _mm256_unpacklo_epi32(__ttt2, __ttt6);
  f = _mm256_unpackhi_epi32(__ttt2, __ttt6);
  g = _mm256_unpacklo_epi32(__ttt3, __ttt7);
  h = _mm256_unpackhi_epi32(__ttt3, __ttt7);

  // stores back 32 rows:

  reinterpret_cast<uint64_t*>(dst)[0] = _mm256_extract_epi64(a, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst)[0] = _mm256_extract_epi64(a, 1);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 2)[0] =
      _mm256_extract_epi64(b, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 3)[0] =
      _mm256_extract_epi64(b, 1);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 4)[0] =
      _mm256_extract_epi64(c, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 5)[0] =
      _mm256_extract_epi64(c, 1);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 6)[0] =
      _mm256_extract_epi64(d, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 7)[0] =
      _mm256_extract_epi64(d, 1);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 8)[0] =
      _mm256_extract_epi64(e, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 9)[0] =
      _mm256_extract_epi64(e, 1);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 10)[0] =
      _mm256_extract_epi64(f, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 11)[0] =
      _mm256_extract_epi64(f, 1);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 12)[0] =
      _mm256_extract_epi64(g, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 13)[0] =
      _mm256_extract_epi64(g, 1);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 14)[0] =
      _mm256_extract_epi64(h, 0);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 15)[0] =
      _mm256_extract_epi64(h, 1);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 16)[0] =
      _mm256_extract_epi64(a, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 17)[0] =
      _mm256_extract_epi64(a, 3);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 18)[0] =
      _mm256_extract_epi64(b, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 19)[0] =
      _mm256_extract_epi64(b, 3);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 20)[0] =
      _mm256_extract_epi64(c, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 21)[0] =
      _mm256_extract_epi64(c, 3);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 22)[0] =
      _mm256_extract_epi64(d, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 23)[0] =
      _mm256_extract_epi64(d, 3);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 24)[0] =
      _mm256_extract_epi64(e, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 25)[0] =
      _mm256_extract_epi64(e, 3);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 26)[0] =
      _mm256_extract_epi64(f, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 27)[0] =
      _mm256_extract_epi64(f, 3);

  reinterpret_cast<uint64_t*>((dst) + ld_dst * 28)[0] =
      _mm256_extract_epi64(g, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 29)[0] =
      _mm256_extract_epi64(g, 3);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 30)[0] =
      _mm256_extract_epi64(h, 2);
  reinterpret_cast<uint64_t*>((dst) + ld_dst * 31)[0] =
      _mm256_extract_epi64(h, 3);
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned M>
static void transpose_kernel_mxn_avx2_uint8(
    unsigned N,
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst) {
  // load from src to registers
  // first load masks
  __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
      internal::avx2_ps_or_epi32_masks[N / 4]));

  __m256i input[8];
  unsigned i, j;
  for (i = 0; i < M; ++i) {
    uint8_t local_buffer[32] = {0};

    // first load into local buffer with mask
    input[i] = _mm256_maskload_epi32(
        reinterpret_cast<const int*>(src + i * ld_src), mask_v);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&local_buffer[0]), input[i]);

    // fill in the local buffer with the remainder elements
    for (j = N / 4 * 4; j < N; j++)
      local_buffer[j] = src[i * ld_src + j];

    // from local buffer to input registers
    input[i] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&local_buffer[0]));
  }

  // for (; i < 8; ++i) {
  // input[i] = _mm256_setzero_si256();
  //}

  // interleaving 8-bit elements
  // e.g., temp[0] now becomes: a0 b0 a1 b1 a2 b2 ...
  __m256i temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm256_unpacklo_epi8(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm256_unpackhi_epi8(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = _mm256_setzero_si256();
  }

  // interleaving 16-bit elements
  // e.g., temp[0] now becomes: a0 b0 c0 d0 a1 b1 c1 d1 ...
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm256_unpacklo_epi16(temp[i * 4], temp[i * 4 + 2]);
    input[4 * i + 1] = _mm256_unpackhi_epi16(temp[i * 4], temp[i * 4 + 2]);
    input[4 * i + 2] = _mm256_unpacklo_epi16(temp[i * 4 + 1], temp[i * 4 + 3]);
    input[4 * i + 3] = _mm256_unpackhi_epi16(temp[i * 4 + 1], temp[i * 4 + 3]);
  }

  // interleaving 32-bit elements
  // e.g., temp[0] now becomes a0 b0 c0 d0 e0 f0 g0 h0 ...
  for (i = 0; i < 4 /*(M + 1) / 2*/; ++i) {
    temp[2 * i] = _mm256_unpacklo_epi32(input[i], input[(i + 4)]);
    temp[2 * i + 1] = _mm256_unpackhi_epi32(input[i], input[(i + 4)]);
  }

  // retrieve the final result, extract every 64-bit
  // i.e., take a 256-bit temp[0] for example, that will
  // 0-63 bit: a0 -- h0,
  // 64-127 bit: a1 -- h1,
  // 128-191 bit:  a16 -- h16,
  // 192-255 bit:   a17 -- h17
  uint64_t t;
  mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
      internal::avx2_ps_or_epi32_masks[M / 4]));
  for (i = 0; i < N; ++i) {
    if (i < 16) {
      if (i % 2 == 0)
        t = _mm256_extract_epi64(temp[i / 2], 0);
      else
        t = _mm256_extract_epi64(temp[i / 2], 1);

    } else {
      if (i % 2 == 0)
        t = _mm256_extract_epi64(temp[(i - 16) / 2], 2);
      else
        t = _mm256_extract_epi64(temp[(i - 16) / 2], 3);
    }
    __m256i t_vec = _mm256_set_epi64x(0, 0, 0, t);
    _mm256_maskstore_epi32(
        reinterpret_cast<int*>(dst + i * ld_dst), mask_v, t_vec);
    for (j = M / 4 * 4; j < M; j++) {
      dst[ld_dst * i + j] = ((t >> (8 * j)) & 255);
    }
  }
}

#endif // __AVX2__

} // namespace internal

} // namespace fbgemm

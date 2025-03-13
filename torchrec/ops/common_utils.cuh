/*
 * The MIT License (MIT)
 *
 * Copyright (C) 2016 ExplosionAI GmbH, 2014-2015 Matthew Honnibal, 2016 spaCy
 * GmbH
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#define AT_DISPATCH_INTEGER_TYPES(TYPE, NAME, HINT, ...)                      \
  AT_DISPATCH_SWITCH(                                                         \
      TYPE,                                                                   \
      NAME,                                                                   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Int, HINT, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                                    \
              at::ScalarType::Long, HINT, __VA_ARGS__))

namespace torch::torchrec::turborec {

#if defined(TORBOREC_CUDA)
#define TORBOREC_INLINE __device__ __host__ __inline__
#else
#define TORBOREC_INLINE inline
#endif

// Inspired by
// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash3.cpp#L286
// NOLINTNEXTLINE:
TORBOREC_INLINE uint64_t
murmur_hash3_2x64(const uint64_t x, const uint64_t y, const uint64_t seed) {
  const uint64_t c1 = 0x87c37b91114253d5;
  const uint64_t c2 = 0x4cf5ad432745937f;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  // First 64-bit block
  uint64_t k1 = x;
  k1 *= c1;
  k1 = (k1 << 31) | (k1 >> (64 - 31));
  k1 *= c2;
  h1 ^= k1;
  h1 = (h1 << 27) | (h1 >> (64 - 27));
  h1 += h2;
  h1 = h1 * 5 + 0x52dce729;

  // Second 64-bit block
  uint64_t k2 = y;
  k2 *= c2;
  k2 = (k2 << 33) | (k2 >> (64 - 33));
  k2 *= c1;
  h2 ^= k2;
  h2 = (h2 << 31) | (h2 >> (64 - 31));
  h2 += h1;
  h2 = h2 * 5 + 0x38495ab5;

  // Finalization
  h1 ^= 16;
  h2 ^= 16;
  h1 += h2;
  h2 += h1;
  h1 ^= h1 >> 33;
  h1 *= 0xff51afd7ed558ccd;
  h1 ^= h1 >> 33;
  h1 *= 0xc4ceb9fe1a85ec53;
  h1 ^= h1 >> 33;
  h2 ^= h2 >> 33;
  h2 *= 0xff51afd7ed558ccd;
  h2 ^= h2 >> 33;
  h2 *= 0xc4ceb9fe1a85ec53;
  h2 ^= h2 >> 33;
  h1 += h2;
  h2 += h1;

  return h1 ^ h2;
}

// NOLINTNEXTLINE:
template <bool CIRCULAR_PROBE>
TORBOREC_INLINE int64_t next_output_index(
    int64_t output_index,
    int64_t modulo,
    int64_t& /* max_probe_local */) {
  static_assert(CIRCULAR_PROBE);
  return (output_index + 1) % modulo;
}

// NOLINTNEXTLINE:
template <>
TORBOREC_INLINE int64_t next_output_index<false>(
    int64_t output_index,
    int64_t modulo,
    int64_t& max_probe_local) {
  output_index = (output_index + 1) % modulo;
  if (output_index == 0) {
    // circular, using max_probe_local to control exit.
    max_probe_local = 0;
  }
  return output_index;
}

TORBOREC_INLINE bool is_eviction_enabled(
    bool readonly,
    int eviction_threshold,
    int eviction_policy) {
  return !readonly && (eviction_threshold > 0 || eviction_policy > 0);
}

#undef TORBOREC_INLINE

} // namespace torch::torchrec::turborec

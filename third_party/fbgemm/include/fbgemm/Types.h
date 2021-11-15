/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fbgemm {

using float16 = std::uint16_t;

// Round to nearest even
static inline float16 cpu_float2half_rn(float f) {
  float16 ret;

  static_assert(
      sizeof(unsigned int) == sizeof(float),
      "Programming error sizeof(unsigned int) != sizeof(float)");

  unsigned* xp = reinterpret_cast<unsigned int*>(&f);
  unsigned x = *xp;
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret = 0x7fffU;
    return ret;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret = static_cast<float16>(sign | 0x7c00U);
    return ret;
  }
  if (u < 0x33000001) {
    ret = static_cast<float16>(sign | 0x0000);
    return ret;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  ret = static_cast<float16>(sign | (exponent << 10) | mantissa);

  return ret;
}

// Round to zero
static inline float16 cpu_float2half_rz(float f) {
  float16 ret;

  static_assert(
      sizeof(unsigned int) == sizeof(float),
      "Programming error sizeof(unsigned int) != sizeof(float)");

  unsigned* xp = reinterpret_cast<unsigned int*>(&f);
  unsigned x = *xp;
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret = static_cast<float16>(0x7fffU);
    return ret;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret = static_cast<float16>(sign | 0x7c00U);
    return ret;
  }
  if (u < 0x33000001) {
    ret = static_cast<float16>(sign | 0x0000);
    return ret;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to zero.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;

  ret = static_cast<float16>(sign | (exponent << 10) | mantissa);

  return ret;
}

static inline float cpu_half2float(float16 h) {
  unsigned sign = ((h >> 15) & 1);
  unsigned exponent = ((h >> 10) & 0x1f);
  unsigned mantissa = ((h & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  unsigned i = ((sign << 31) | (exponent << 23) | mantissa);
  float ret;
  memcpy(&ret, &i, sizeof(i));
  return ret;
}

} // namespace fbgemm

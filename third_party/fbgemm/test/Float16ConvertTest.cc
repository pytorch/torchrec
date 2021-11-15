/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
class FBGemmFloat16Test : public testing::TestWithParam<bool> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFloat16Test,
    ::testing::Bool());

TEST_P(FBGemmFloat16Test, Conversion) {
  bool do_clip = GetParam();
  constexpr float FP16_MAX = 65504.f;

  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  if (do_clip) {
    a[3] += 1024 * FP16_MAX;
  }
  float16 b[100]; // float16 type
  float c[100]; // fp32 type
  FloatToFloat16_ref(a, b, 100, do_clip);
  Float16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^10) since float16
    // has 10 bits mantissa.
    float expected = a[i];
    if (do_clip) {
      expected = std::max(-FP16_MAX, std::min(expected, FP16_MAX));
    }
    EXPECT_LE(fabs(expected - c[i]) / expected, 1.0 / 1024);
  }
}

TEST_P(FBGemmFloat16Test, Conversion_simd) {
  bool do_clip = GetParam();
  constexpr float FP16_MAX = 65504.f;

  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  if (do_clip) {
    a[3] += 1024 * FP16_MAX;
  }
  float16 b[100]; // float16 type
  float c[100]; // fp32 type
  FloatToFloat16_simd(a, b, 100, do_clip);
  Float16ToFloat_simd(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^10) since float16
    // has 10 bits mantissa.
    float expected = a[i];
    if (do_clip) {
      expected = std::max(-FP16_MAX, std::min(expected, FP16_MAX));
    }
    EXPECT_LE(fabs(expected - c[i]) / expected, 1.0 / 1024);
  }
}

TEST_P(FBGemmFloat16Test, Conversion_simd2) {
  bool do_clip = GetParam();
  constexpr float FP16_MAX = 65504.f;

  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);
  uniform_int_distribution<int> dn(1, 1024);

  for (int i = 0; i < 10; i++) {
    int m = dm(generator);
    int n = dn(generator);
    shapes.push_back({m, n});
  }

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];

    cerr << "m = " << m << " n = " << n << endl;
    aligned_vector<float> A_fp32_ref(m * n); // fp32 type
    aligned_vector<float16> A_float16(m * n); // float16 type
    aligned_vector<float> A_fp32_final(m * n); // fp32 type
    // randFill(A_fp32_ref, 0.0f, 4.0f);
    for (int i = 0; i < m * n; ++i) {
      A_fp32_ref[i] = (i % 10000) + 1.25;
    }
    if (do_clip) {
      A_fp32_ref[0] += 1024 * FP16_MAX;
    }

    FloatToFloat16_simd(A_fp32_ref.data(), A_float16.data(), m * n, do_clip);
    Float16ToFloat_simd(A_float16.data(), A_fp32_final.data(), m * n);
    for (int i = 0; i < m * n; ++i) {
      // The relative error should be less than 1/(2^10) since float16
      // has 10 bits mantissa.
      // printf( "A_fp32_final[%d]: %f; A_fp32_ref[%d]: %f\n", i,
      // A_fp32_final[i], i, A_fp32_ref[i]);
      float expected = A_fp32_ref[i];
      if (do_clip) {
        expected = std::max(-FP16_MAX, std::min(expected, FP16_MAX));
      }
      EXPECT_LE(fabs(expected - A_fp32_final[i]) / expected, 1.0 / 1024);
    }
  }
}

TEST_P(FBGemmFloat16Test, Conversion_fake_rounding) {
  bool do_clip = GetParam();
  constexpr float FP16_MAX = 65504.f;
  union epsilon_t {
    float f;
    uint32_t i;
  };
  union epsilon_t epsilon;
  epsilon.i = 0x38800000u; // 1 / 16384
  float FP16_MIN = epsilon.f;

  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(2, 1024*256);

  for (int i = 0; i < 10; i++) {
    int m = dm(generator);
    shapes.push_back({m});
  }

  for (auto s : shapes) {
    int m = s[0];

    cerr << "m = " << m << endl;
    aligned_vector<float> A_fp32_ref(m); // fp32 type
    aligned_vector<float16> A_float16(m); // float16 type
    aligned_vector<float> A_fp32_final(m); // fp32 type
    // randFill(A_fp32_ref, 0.0f, 4.0f);
    for (int i = 0; i < m; ++i) {
      A_fp32_ref[i] = (i % 10000) + 1.25;
    }
    if (do_clip) {
      A_fp32_ref[0] += 1024 * FP16_MAX;
      A_fp32_ref[1] = 1e-10;
    }

    RoundToFloat16(
        A_fp32_ref.data(), A_fp32_final.data(), m, do_clip, do_clip);

    for (int i = 0; i < m; ++i) {
      // The relative error should be less than 1/(2^10) since float16
      // has 10 bits mantissa.
      // printf(
      //     "A_fp32_final[%d]: %f; A_fp32_ref[%d]: %f\n",
      //     i,
      //     A_fp32_final[i],
      //     i,
      //     A_fp32_ref[i]);
      float expected = A_fp32_ref[i];
      if (do_clip) {
        expected = std::max(-FP16_MAX, std::min(expected, FP16_MAX));
        if (std::abs(expected) < FP16_MIN) {
          expected = 0.0;
        }
      }
      constexpr float kEpsilon = 1e-8f; // To handle the case where expected == 0.0;
      EXPECT_LE(fabs(expected - A_fp32_final[i]) / (expected + kEpsilon), 1.0 / 1024);
    }
    if (do_clip) {
      EXPECT_EQ(A_fp32_final[1], 0.0);
    }
  }
}

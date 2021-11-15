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

using namespace std;
using namespace fbgemm;

TEST(FBGemmBfloat16Test, Conversion) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_ref(a, b, 100);
  Bfloat16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_simd(a, b, 100);
  Bfloat16ToFloat_simd(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128)
        << "Conversion results differ at (" << i << " ). ref: " << a[i]
        << " conversion: " << c[i];
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd2) {
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
    aligned_vector<bfloat16> A_bfloat16(m * n); // bfloat16 type
    aligned_vector<float> A_fp32_final(m * n); // fp32 type
    // randFill(A_fp32_ref, 0.0f, 4.0f);
    for (int i = 0; i < m * n; ++i) {
      A_fp32_ref[i] = i + 1.25;
    }

    FloatToBfloat16_simd(A_fp32_ref.data(), A_bfloat16.data(), m * n);
    Bfloat16ToFloat_simd(A_bfloat16.data(), A_fp32_final.data(), m * n);
    for (int i = 0; i < m * n; ++i) {
      // The relative error should be less than 1/(2^7) since bfloat16
      // has 7 bits mantissa.
      // printf( "A_fp32_final[%d]: %f; A_fp32_ref[%d]: %f\n", i,
      // A_fp32_final[i], i, A_fp32_ref[i]);
      EXPECT_LE(
          fabs(A_fp32_final[i] - A_fp32_ref[i]) / A_fp32_ref[i], 1.0 / 128);
    }
  }
}

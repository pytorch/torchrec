/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

template <typename T>
::testing::AssertionResult compare_tranpose_results(
    vector<T> expected,
    vector<T> acutal,
    int m,
    int n,
    int ld_src,
    int ld_dst) {
  std::stringstream ss;
  if (is_same<T, float>::value) {
    ss << " float results ";
  } else {
    ss << " i8 results ";
  }
  ss << " mismatch at ";
  bool match = true;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int exp = expected[i * ld_src + j];
      int act = acutal[i + j * ld_dst];
      if (exp != act) {
        ss << "(" << i << ", " << j << "). ref " << exp << " actual " << act;
        match = false;
      }
    }
  }
  if (match)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "results differ: " << ss.str();
}

TEST(TransposeTest, TransposeTest) {
  // Generate shapes to test
  vector<tuple<int, int, int, int>> shapes;
  uniform_int_distribution<int> dist(0, 32);
  default_random_engine generator;
  for (int i = 0; i < 1024; ++i) {
    int m = dist(generator);
    int n = dist(generator);
    int ld_src = n + dist(generator);
    int ld_dst = m + dist(generator);
    shapes.push_back(make_tuple(m, n, ld_src, ld_dst));
  }

  for (const auto& shape : shapes) {
    int m, n, ld_src, ld_dst;
    tie(m, n, ld_src, ld_dst) = shape;

    // float test
    vector<float> a(m * ld_src);
    vector<float> b(n * ld_dst);
    generate(
        a.begin(), a.end(), [&dist, &generator] { return dist(generator); });

    transpose_simd(m, n, a.data(), ld_src, b.data(), ld_dst);

    EXPECT_TRUE(compare_tranpose_results(a, b, m, n, ld_src, ld_dst));

    // i8 test
    vector<uint8_t> a_i8(m * ld_src);
    vector<uint8_t> b_i8(n * ld_dst);
    generate(a_i8.begin(), a_i8.end(), [&dist, &generator] {
      return dist(generator);
    });

    transpose_simd(m, n, a_i8.data(), ld_src, b_i8.data(), ld_dst);
    EXPECT_TRUE(compare_tranpose_results(a_i8, b_i8, m, n, ld_src, ld_dst));
  }
}

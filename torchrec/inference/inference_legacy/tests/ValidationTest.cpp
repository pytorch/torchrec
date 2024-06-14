/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Validation.h"

#include <ATen/ATen.h>
#include <gtest/gtest.h>

TEST(ValidationTest, validateSparseFeatures) {
  auto values = at::tensor({1, 2, 3, 4});
  auto lengths = at::tensor({1, 1, 1, 1});
  auto weights = at::tensor({.1, .2, .3, .4});

  // pass 1D
  EXPECT_TRUE(torchrec::validateSparseFeatures(values, lengths, weights));

  // pass 2D
  lengths.reshape({2, 2});
  EXPECT_TRUE(torchrec::validateSparseFeatures(values, lengths, weights));

  // fail 1D
  auto invalidLengths = at::tensor({1, 2, 1, 1});
  EXPECT_FALSE(
      torchrec::validateSparseFeatures(values, invalidLengths, weights));

  // fail 2D
  invalidLengths.reshape({2, 2});
  EXPECT_FALSE(
      torchrec::validateSparseFeatures(values, invalidLengths, weights));
}

TEST(ValidationTest, validateDenseFeatures) {
  auto values = at::tensor({1, 2, 3, 4});
  EXPECT_TRUE(torchrec::validateDenseFeatures(values, 1));
  EXPECT_TRUE(torchrec::validateDenseFeatures(values, 4));
  EXPECT_FALSE(torchrec::validateDenseFeatures(values, 3));
}

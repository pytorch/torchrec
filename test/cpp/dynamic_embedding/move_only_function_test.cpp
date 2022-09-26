/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/move_only_function.h>

namespace torchrec {

TEST(tde, move_only_function) {
  MoveOnlyFunction<int()> foo = +[] { return 0; };
  ASSERT_EQ(foo(), 0);
  ASSERT_TRUE(foo);
  foo = {};
  ASSERT_FALSE(foo);
}

} // namespace torchrec

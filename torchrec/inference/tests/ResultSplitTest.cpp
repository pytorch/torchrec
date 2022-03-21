/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/ResultSplit.h"

#include <gtest/gtest.h>

template <typename T>
void checkTensor(const at::Tensor& tensor, const std::vector<T>& expected) {
  EXPECT_EQ(tensor.sizes(), at::ArrayRef({(long)expected.size()}));
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(tensor[i].item<T>(), expected[i]) << "pos: " << i;
  }
}

TEST(ResultSplitTest, SplitDictOfTensor) {
  c10::impl::GenericDict pred(c10::StringType::get(), c10::TensorType::get());
  pred.insert("par", at::tensor({0, 1, 2}));
  pred.insert("foo", at::tensor({3, 4, 5}));

  auto splitResult = torchrec::splitDictOfTensor(pred, 0, 2);
  checkTensor<float>(
      splitResult.toGenericDict().at("par").toTensor(), {0., 1.});
  checkTensor<float>(
      splitResult.toGenericDict().at("foo").toTensor(), {3., 4.});
}

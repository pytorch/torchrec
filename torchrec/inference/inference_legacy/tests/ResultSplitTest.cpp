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

  auto splitResult = torchrec::splitDictOfTensor(pred, 0, 2, 3);
  checkTensor<float>(
      splitResult.toGenericDict().at("par").toTensor(), {0., 1.});
  checkTensor<float>(
      splitResult.toGenericDict().at("foo").toTensor(), {3., 4.});
}

TEST(ResultSplitTest, SplitDictOfTensorVariableLength) {
  c10::impl::GenericDict pred(c10::StringType::get(), c10::TensorType::get());
  pred.insert("par", at::tensor({0, 1, 2}));
  pred.insert("foo", at::tensor({3, 4, 5, 6, 7, 8}));

  auto splitResult = torchrec::splitDictOfTensor(pred, 0, 2, 3);
  checkTensor<float>(
      splitResult.toGenericDict().at("par").toTensor(), {0., 1.});
  checkTensor<float>(
      splitResult.toGenericDict().at("foo").toTensor(), {3., 4., 5., 6.});
}

TEST(ResultSplitTest, SplitDictOfTensors) {
  c10::impl::GenericDict pred(
      c10::StringType::get(),
      c10::TupleType::create(
          {c10::TensorType::get(),
           c10::TensorType::get(),
           c10::TensorType::get()}));
  pred.insert(
      "par",
      c10::ivalue::Tuple::create(
          {at::tensor({0, 1, 2, 3}), at::tensor({4, 5}), at::tensor({6, 7})}));
  pred.insert(
      "foo",
      c10::ivalue::Tuple::create(
          {at::tensor({8, 9}),
           at::tensor({10, 11}),
           at::tensor({12, 13, 14, 15})}));

  auto splitResult = torchrec::splitDictOfTensors(pred, 1, 1, 2);
  {
    auto tuple = splitResult.toGenericDict().at("par").toTuple();
    checkTensor<float>(tuple->elements()[0].toTensor(), {2., 3.});
    checkTensor<float>(tuple->elements()[1].toTensor(), {5.});
    checkTensor<float>(tuple->elements()[2].toTensor(), {7.});
  }
  {
    auto tuple = splitResult.toGenericDict().at("foo").toTuple();
    checkTensor<float>(tuple->elements()[0].toTensor(), {9.});
    checkTensor<float>(tuple->elements()[1].toTensor(), {11.});
    checkTensor<float>(tuple->elements()[2].toTensor(), {14., 15.});
  }
}

TEST(ResultSplitTest, SplitDictWithMaskTensor) {
  c10::impl::GenericDict pred(
      c10::StringType::get(),
      c10::TupleType::create({c10::TensorType::get(), c10::TensorType::get()}));
  pred.insert(
      "par",
      c10::ivalue::Tuple::create(at::tensor({0, 1, 2}), at::tensor({3, 4, 5})));
  pred.insert(
      "foo",
      c10::ivalue::Tuple::create(at::tensor({2, 1, 0}), at::tensor({5, 4, 3})));

  auto splitResult = torchrec::splitDictWithMaskTensor(pred, 0, 2);
  checkTensor<float>(
      splitResult.toGenericDict()
          .at("par")
          .toTupleRef()
          .elements()[0]
          .toTensor(),
      {0., 1.});
  checkTensor<float>(
      splitResult.toGenericDict()
          .at("par")
          .toTupleRef()
          .elements()[1]
          .toTensor(),
      {3., 4., 5.});
  checkTensor<float>(
      splitResult.toGenericDict()
          .at("foo")
          .toTupleRef()
          .elements()[0]
          .toTensor(),
      {2., 1.});
  checkTensor<float>(
      splitResult.toGenericDict()
          .at("foo")
          .toTupleRef()
          .elements()[1]
          .toTensor(),
      {5., 4., 3.});
}

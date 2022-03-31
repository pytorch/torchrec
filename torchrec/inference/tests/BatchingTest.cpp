/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Batching.h"

#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Functions.h> // @manual
#include <c10/core/ScalarType.h>
#include <folly/io/IOBuf.h>
#include <gtest/gtest.h>

#include "torchrec/inference/Types.h"
#include "torchrec/inference/Utils.h"

namespace torchrec {

namespace {

template <typename T>
void checkTensor(at::Tensor& tensor, const std::vector<T>& expected) {
  EXPECT_EQ(tensor.sizes(), at::ArrayRef({(long)expected.size()}));
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(tensor[i].item<T>(), expected[i]) << "pos: " << i;
  }
}

} // namespace

TEST(BatchingTest, SparseCombineTest) {
  const auto jagged0 = createJaggedTensor({{0, 1}, {2}});
  const auto jagged1 = createJaggedTensor({{}, {3}});

  auto request0 = createRequest(1, 2, jagged0);
  auto request1 = createRequest(1, 2, jagged1);

  auto batched =
      combineSparse("id_score_list_features", {request0, request1}, true);

  checkTensor<int32_t>(batched["id_score_list_features.lengths"], {2, 0, 1, 1});
  checkTensor<int32_t>(batched["id_score_list_features.values"], {0, 1, 2, 3});
  checkTensor<float>(
      batched["id_score_list_features.weights"], {1.0f, 1.0f, 1.0f, 1.0f});
}

TEST(BatchingTest, EmbeddingCombineTest) {
  const auto embedding0 = createEmbeddingTensor({{0, 1}, {2, 3}});
  const auto embedding1 = createEmbeddingTensor({{4, 5}});

  auto request0 = createRequest(2, 2, embedding0);
  auto request1 = createRequest(1, 2, embedding1);

  auto batched = combineEmbedding("embedding_features", {request0, request1});
  // num features, num batches, feature dimision
  EXPECT_EQ(batched["embedding_features"].sizes(), at::ArrayRef({2L, 3L, 1L}));
  auto flatten = batched["embedding_features"].flatten();
  checkTensor<float>(flatten, {0, 1, 4, 2, 3, 5});
}

TEST(BatchingTest, DenseCombineTest) {
  auto tensor0 =
      at::tensor({1.1, 2.0, 0.3, 1.2}, at::TensorOptions().dtype(c10::kFloat))
          .reshape({2, 2});
  auto tensor1 = at::tensor({0.9, 2.3}, at::TensorOptions().dtype(c10::kFloat))
                     .reshape({1, 2});

  auto request0 = createRequest(tensor0);
  auto request1 = createRequest(tensor1);

  auto batchedIOBuf = combineFloat("io_buf", {request0, request1});
  auto batchedIValue = combineFloat("ivalue", {request0, request1});

  // num features, num batches, feature dimension
  std::vector<int64_t> expectShape{3L, 2L};
  std::vector<float> expectResult{1.1, 2.0, 0.3, 1.2, 0.9, 2.3};
  EXPECT_EQ(batchedIOBuf["io_buf"].sizes(), expectShape);
  EXPECT_EQ(batchedIValue["ivalue"].sizes(), expectShape);
  auto flatten = batchedIOBuf["io_buf"].flatten();
  checkTensor<float>(flatten, expectResult);
  flatten = batchedIValue["ivalue"].flatten();
  checkTensor<float>(flatten, expectResult);
}

} // namespace torchrec

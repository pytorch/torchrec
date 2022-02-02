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

namespace torchrec {

namespace {

JaggedTensor createJaggedTensor(
    const std::vector<std::vector<int32_t>>& input) {
  std::vector<int32_t> lengths;
  std::vector<int32_t> values;
  std::vector<float> weights;

  for (const auto& vec : input) {
    lengths.push_back(vec.size());
    for (auto v : vec) {
      values.push_back((float)v);
      weights.push_back(1.0);
    }
  }

  return JaggedTensor{
      .lengths =
          at::from_blob(
              lengths.data(), {static_cast<long>(lengths.size())}, at::kInt)
              .clone(),
      .values = at::from_blob(
                    values.data(), {static_cast<long>(values.size())}, at::kInt)
                    .clone(),
      .weights =
          at::from_blob(
              weights.data(), {static_cast<long>(weights.size())}, at::kFloat)
              .clone(),
  };
}

at::Tensor createEmbeddingTensor(
    const std::vector<std::vector<int32_t>>& input) {
  long size = 0;
  for (const auto& vec : input) {
    size += vec.size();
  }

  auto tensor = at::empty({size}, at::kFloat);
  auto accessor = tensor.accessor<float, 1>();
  long curt = 0;
  for (const auto& vec : input) {
    for (const auto v : vec) {
      accessor[curt++] = v;
    }
  }

  return tensor;
}

std::shared_ptr<PredictionRequest> createRequest(
    size_t batchSize,
    size_t numFeatures,
    const JaggedTensor& jagged) {
  auto request = std::make_shared<PredictionRequest>();
  request->batch_size = batchSize;

  {
    SparseFeatures feature;
    feature.num_features = numFeatures;
    feature.lengths = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        jagged.lengths.data_ptr(),
        jagged.lengths.storage().nbytes());
    feature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        jagged.values.data_ptr(),
        jagged.values.storage().nbytes());
    feature.weights = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        jagged.weights.data_ptr(),
        jagged.weights.storage().nbytes());
    request->features["id_score_list_features"] = std::move(feature);
  }

  return request;
}

std::shared_ptr<PredictionRequest>
createRequest(size_t batchSize, size_t numFeatures, at::Tensor embedding) {
  auto request = std::make_shared<PredictionRequest>();
  request->batch_size = batchSize;

  {
    torchrec::FloatFeatures feature;
    feature.num_features = numFeatures;
    feature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        embedding.data_ptr(),
        embedding.storage().nbytes());
    request->features["embedding_features"] = std::move(feature);
  }

  return request;
}

template <typename T>
void checkTensor(at::Tensor& tensor, std::vector<T> expected) {
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

} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/TestUtils.h"

#include <initializer_list>
#include <memory>

#include <folly/io/IOBuf.h>

#include "torchrec/inference/Types.h"

namespace torchrec {

std::shared_ptr<PredictionRequest> createRequest(at::Tensor denseTensor) {
  auto request = std::make_shared<PredictionRequest>();
  request->batch_size = denseTensor.size(0);

  {
    FloatFeatures feature;
    feature.num_features = denseTensor.size(1);
    feature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        denseTensor.data_ptr(),
        denseTensor.storage().nbytes());
    request->features["io_buf"] = std::move(feature);
  }

  {
    c10::IValue feature(denseTensor);
    request->features["ivalue"] = std::move(feature);
  }

  return request;
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

c10::List<at::Tensor> createIValueList(
    const std::vector<std::vector<int32_t>>& input) {
  // Input is batch x num_features
  std::vector<at::Tensor> rows;
  for (const auto& vec : input) {
    rows.push_back(at::tensor(vec, at::TensorOptions().dtype(c10::kFloat)));
  }
  auto combined = at::stack(rows).transpose(0, 1);
  c10::List<at::Tensor> retList;
  for (auto& tensor : combined.split(1)) {
    retList.push_back(tensor.squeeze(0));
  }
  return retList;
}

} // namespace torchrec

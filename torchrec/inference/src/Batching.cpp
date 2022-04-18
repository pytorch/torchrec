/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Batching.h"

#include <c10/core/ScalarType.h>
#include <folly/Range.h>
#include <folly/container/Enumerate.h>
#include <folly/io/Cursor.h>

#include "ATen/Functions.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

C10_DEFINE_REGISTRY(TorchRecBatchingFuncRegistry, BatchingFunc);

std::unordered_map<std::string, at::Tensor> combineFloat(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  auto* maybeIValuePtr =
      std::get_if<c10::IValue>(&requests.front()->features[featureName]);
  at::Tensor combined;
  std::vector<at::Tensor> tensors;

  if (maybeIValuePtr != nullptr) {
    numFeatures = maybeIValuePtr->toTensor().size(1);
    tensors.reserve(requests.size());
  }

  for (const auto& request : requests) {
    if (maybeIValuePtr != nullptr) {
      tensors.push_back(
          std::get<c10::IValue>(request->features[featureName]).toTensor());
      combinedBatchSize += tensors.back().size(0);
    } else {
      const auto& feature =
          std::get<torchrec::FloatFeatures>(request->features[featureName]);
      const auto nf = feature.num_features;
      const auto dataSize = feature.values.computeChainDataLength();
      if (nf * request->batch_size * sizeof(float) != dataSize) {
        throw std::invalid_argument("Invalid float features");
      }
      if (nf > 0) {
        combinedBatchSize += request->batch_size;
        if (numFeatures > 0) {
          if (numFeatures != nf) {
            throw std::invalid_argument("Different number of float features");
          }
        }
        numFeatures = nf;
      }
    }
  }

  if (numFeatures == 0) {
    return {{featureName, at::empty({combinedBatchSize, 0})}};
  }

  if (maybeIValuePtr != nullptr) {
    combined = at::empty(
        {combinedBatchSize, numFeatures},
        maybeIValuePtr->toTensor().options().pinned_memory(true));
    at::cat_out(combined, tensors);
  } else {
    // Create output tensor.
    const auto options =
        at::TensorOptions(at::kCPU).dtype(at::kFloat).pinned_memory(true);
    combined = at::empty({combinedBatchSize, numFeatures}, options);

    // Copy tensor data.
    auto combinedRange = folly::MutableByteRange(
        reinterpret_cast<uint8_t*>(combined.data_ptr()),
        combinedBatchSize * numFeatures * options.dtype().itemsize());
    for (const auto& request : requests) {
      const auto* start =
          &std::get<torchrec::FloatFeatures>(request->features[featureName])
               .values;
      const auto* curr = start;
      do {
        std::memcpy(combinedRange.data(), curr->data(), curr->length());
        combinedRange.advance(curr->length());
        curr = curr->next();
      } while (curr != start);
    }
  }

  return {{featureName, std::move(combined)}};
}

std::unordered_map<std::string, at::Tensor> combineSparse(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests,
    bool isWeighted) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long totalLength = 0;
  // request -> feature -> length of the feature
  std::vector<std::vector<long>> featureLengths;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::SparseFeatures>(request->features[featureName]);

    // validate num_features
    const auto nf = features.num_features;
    if (nf > 0) {
      combinedBatchSize += request->batch_size;
      if (numFeatures > 0) {
        if (numFeatures != nf) {
          throw std::invalid_argument("Different number of float features");
        }
      }
      numFeatures = nf;
    }

    featureLengths.emplace_back().reserve(nf);
    size_t requestLength = 0;
    folly::io::Cursor lengthsCursor(&features.lengths);
    for (int i = 0; i < features.num_features; ++i) {
      size_t featureLength = 0;
      for (int j = 0; j < request->batch_size; ++j) {
        featureLength += lengthsCursor.read<int32_t>();
      }
      featureLengths.back().push_back(featureLength);
      requestLength += featureLength;
    }
    CHECK(lengthsCursor.isAtEnd());
    totalLength += requestLength;
  }

  // Create output tensor.
  const auto options = at::TensorOptions(at::kCPU).pinned_memory(true);
  auto lengths =
      at::empty({numFeatures * combinedBatchSize}, options.dtype(at::kInt));
  auto values = at::empty({totalLength}, options.dtype(at::kInt));
  auto weights =
      at::empty({isWeighted ? totalLength : 0}, options.dtype(at::kFloat));

  std::vector<folly::io::Cursor> lengthsCursors;
  std::vector<folly::io::Cursor> valuesCursor;
  std::vector<folly::io::Cursor> weightsCursor;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::SparseFeatures>(request->features[featureName]);

    lengthsCursors.emplace_back(&features.lengths);
    valuesCursor.emplace_back(&features.values);
    if (isWeighted) {
      weightsCursor.emplace_back(&features.weights);
    }
  }

  auto lengthsRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(lengths.data_ptr()),
      lengths.numel() * lengths.dtype().itemsize());
  auto valuesRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(values.data_ptr()),
      values.numel() * values.dtype().itemsize());
  auto weightsRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(weights.data_ptr()),
      isWeighted ? weights.numel() * weights.dtype().itemsize() : 0);

  for (int i = 0; i < numFeatures; ++i) {
    for (int j = 0; j < requests.size(); ++j) {
      const auto& request = requests[j];

      // TODO: determine values elem size by value_type.
      size_t len = request->batch_size * sizeof(int32_t);
      lengthsCursors[j].pull(lengthsRange.data(), len);
      lengthsRange.advance(len);

      len = featureLengths[j][i] * sizeof(int32_t);
      valuesCursor[j].pull(valuesRange.data(), len);
      valuesRange.advance(len);

      if (isWeighted) {
        len = featureLengths[j][i] * sizeof(float);
        weightsCursor[j].pull(weightsRange.data(), len);
        weightsRange.advance(len);
      }
    }
  }

  std::unordered_map<std::string, at::Tensor> ret = {
      {featureName + ".values", std::move(values)},
      {featureName + ".lengths", std::move(lengths)},
  };
  if (isWeighted) {
    ret[featureName + ".weights"] = std::move(weights);
  }
  return ret;
}

std::unordered_map<std::string, at::Tensor> combineEmbedding(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long dimension = 0;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::FloatFeatures>(request->features[featureName]);
    const auto nf = features.num_features;
    const auto dataSize = features.values.computeChainDataLength();
    if (nf != 0 && dimension == 0) {
      dimension = dataSize / request->batch_size / sizeof(float) / nf;
    }
    if (nf * request->batch_size * dimension * sizeof(float) != dataSize) {
      throw std::invalid_argument("Invalid embedding features");
    }
    if (nf > 0) {
      combinedBatchSize += request->batch_size;
      if (numFeatures > 0) {
        if (numFeatures != nf) {
          throw std::invalid_argument("Different number of embedding features");
        }
      }
      numFeatures = nf;
    }
  }

  if (numFeatures == 0) {
    return {{featureName, at::empty(0)}};
  }

  // Create output tensor.
  const auto options =
      at::TensorOptions(at::kCPU).dtype(at::kFloat).pinned_memory(true);
  auto combined =
      at::empty({numFeatures, combinedBatchSize, dimension}, options);

  // Copy tensor data.
  auto combinedRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(combined.data_ptr()),
      combined.storage().nbytes());
  std::vector<folly::io::Cursor> cursors;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::FloatFeatures>(request->features[featureName]);
    cursors.emplace_back(&features.values);
  }
  for (int i = 0; i < numFeatures; ++i) {
    for (const auto&& it : folly::enumerate(cursors)) {
      auto len = requests[it.index]->batch_size * dimension * sizeof(float);
      it.element.pull(combinedRange.data(), len);
      combinedRange.advance(len);
    }
  }
  return {{featureName, std::move(combined)}};
}

class FloatBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, at::Tensor> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& /* device */,
      LazyTensorRef /* batchItems */) override {
    return combineFloat(featureName, requests);
  }
};

class SparseBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, at::Tensor> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& /* device */,
      LazyTensorRef /* batchItems */) override {
    return combineSparse(featureName, requests, /* isWeighted */ false);
  }
};

class WeightedSparseBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, at::Tensor> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& /* device */,
      LazyTensorRef /* batchItems */) override {
    return combineSparse(featureName, requests, /* isWeighted */ true);
  }
};

class EmbeddingBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, at::Tensor> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& /* device */,
      LazyTensorRef /* batchItems */) override {
    return combineEmbedding(featureName, requests);
  }
};

REGISTER_TORCHREC_BATCHING_FUNC(dense, FloatBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(sparse, SparseBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(weighted_sparse, WeightedSparseBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(embedding, EmbeddingBatchingFunc);

} // namespace torchrec

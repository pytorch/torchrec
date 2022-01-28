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

namespace torchrec {

at::Tensor combineFloat(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  c10::InferenceMode guard;
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  for (const auto& request : requests) {
    const auto nf = request->float_features.num_features;
    const auto dataSize =
        request->float_features.values.computeChainDataLength();
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

  if (numFeatures == 0) {
    return at::empty(0);
  }

  // Create output tensor.
  const auto options =
      at::TensorOptions(at::kCPU).dtype(at::kFloat).pinned_memory(true);
  auto combined = at::empty({combinedBatchSize, numFeatures}, options);

  // Copy tensor data.
  auto combinedRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(combined.data_ptr()),
      combinedBatchSize * numFeatures * options.dtype().itemsize());
  for (const auto& request : requests) {
    const auto* start = &request->float_features.values;
    const auto* curr = start;
    do {
      std::memcpy(combinedRange.data(), curr->data(), curr->length());
      combinedRange.advance(curr->length());
      curr = curr->next();
    } while (curr != start);
  }
  return combined;
}

JaggedTensor combineSparse(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests,
    std::function<const SparseFeatures&(const PredictionRequest&)> accessor,
    bool isWeighted) {
  c10::InferenceMode guard;
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long totalLength = 0;
  // request -> feature -> length of the feature
  std::vector<std::vector<long>> featureLengths;
  for (const auto& request : requests) {
    const auto& features = accessor(*request);

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
      at::empty({isWeighted ? totalLength : 0}, options.dtype(at::kHalf));

  std::vector<folly::io::Cursor> lengthsCursors;
  std::vector<folly::io::Cursor> valuesCursor;
  std::vector<folly::io::Cursor> weightsCursor;
  for (const auto& request : requests) {
    const auto& features = accessor(*request);

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
        len = featureLengths[j][i] * sizeof(at::Half);
        weightsCursor[j].pull(weightsRange.data(), len);
        weightsRange.advance(len);
      }
    }
  }

  return JaggedTensor{
      .values = std::move(values),
      .lengths = std::move(lengths),
      .weights = weights.to(at::kFloat),
  };
}

at::Tensor combineEmbedding(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  c10::InferenceMode guard;
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long dimension = 0;
  for (const auto& request : requests) {
    const auto nf = request->embedding_features.num_features;
    const auto dataSize =
        request->embedding_features.values.computeChainDataLength();
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
    return at::empty(0);
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
    cursors.emplace_back(&request->embedding_features.values);
  }
  for (int i = 0; i < numFeatures; ++i) {
    for (const auto&& it : folly::enumerate(cursors)) {
      auto len = requests[it.index]->batch_size * dimension * sizeof(float);
      it.element.pull(combinedRange.data(), len);
      combinedRange.advance(len);
    }
  }
  return combined;
}

} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include <folly/io/IOBuf.h>

namespace torchrec {

struct SparseFeatures {
  uint32_t num_features;
  // int32: T x B
  folly::IOBuf lengths;
  // T x B x L (jagged)
  folly::IOBuf values;
  // float16
  folly::IOBuf weights;
};

struct FloatFeatures {
  uint32_t num_features;
  // shape: {B}
  folly::IOBuf values;
};

// TODO: Change the input format to torch::IValue.
using Feature = std::variant<SparseFeatures, FloatFeatures>;

struct PredictionRequest {
  uint32_t batch_size;
  std::unordered_map<std::string, Feature> features;
};

struct PredictionResponse {
  // Task name to prediction Tensor
  std::map<std::string, folly::IOBuf> predictions;
};

} // namespace torchrec

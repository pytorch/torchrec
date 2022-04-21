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
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAEvent.h>
#include <folly/ExceptionWrapper.h>
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
// Currently only dense batching function support IValue.
using Feature = std::variant<SparseFeatures, FloatFeatures, c10::IValue>;

struct PredictionRequest {
  uint32_t batch_size;
  std::unordered_map<std::string, Feature> features;
};

struct PredictionResponse {
  c10::IValue predictions;
  // If set, the result is an exception.
  std::optional<folly::exception_wrapper> exception;
};

using PredictionException = std::runtime_error;

using Event = std::
    unique_ptr<at::cuda::CUDAEvent, std::function<void(at::cuda::CUDAEvent*)>>;

} // namespace torchrec

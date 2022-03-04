/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Registry.h>

#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "torchrec/inference/JaggedTensor.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

class BatchingFunc {
 public:
  virtual ~BatchingFunc() = default;

  virtual std::unordered_map<std::string, at::Tensor> batch(
      const std::string& /* featureName */,
      const std::vector<std::shared_ptr<PredictionRequest>>& /* requests */,
      const int64_t& /* totalNumBatch */,
      at::Tensor /* batchOffsets */,
      const c10::Device& /* device */) = 0;
};

/**
 * TorchRecBatchingFuncRegistry is used to register custom batching functions.
 */
C10_DECLARE_REGISTRY(TorchRecBatchingFuncRegistry, BatchingFunc);

#define REGISTER_TORCHREC_BATCHING_FUNC_WITH_PIORITY(name, priority, ...) \
  C10_REGISTER_CLASS_WITH_PRIORITY(                                       \
      TorchRecBatchingFuncRegistry, name, priority, __VA_ARGS__);

#define REGISTER_TORCHREC_BATCHING_FUNC(name, ...) \
  REGISTER_TORCHREC_BATCHING_FUNC_WITH_PIORITY(    \
      name, c10::REGISTRY_DEFAULT, __VA_ARGS__);

std::unordered_map<std::string, at::Tensor> combineFloat(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests);

std::unordered_map<std::string, at::Tensor> combineSparse(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests,
    bool isWeighted);

std::unordered_map<std::string, at::Tensor> combineEmbedding(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests);

} // namespace torchrec

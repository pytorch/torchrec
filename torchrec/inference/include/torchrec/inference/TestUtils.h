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

#include "torchrec/inference/JaggedTensor.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

std::shared_ptr<PredictionRequest> createRequest(at::Tensor denseTensor);

std::shared_ptr<PredictionRequest>
createRequest(size_t batchSize, size_t numFeatures, const JaggedTensor& jagged);

std::shared_ptr<PredictionRequest>
createRequest(size_t batchSize, size_t numFeatures, at::Tensor embedding);

JaggedTensor createJaggedTensor(const std::vector<std::vector<int32_t>>& input);

c10::List<at::Tensor> createIValueList(
    const std::vector<std::vector<int32_t>>& input);

at::Tensor createEmbeddingTensor(
    const std::vector<std::vector<int32_t>>& input);

} // namespace torchrec

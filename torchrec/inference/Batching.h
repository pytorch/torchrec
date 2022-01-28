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

at::Tensor combineFloat(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests);

JaggedTensor combineSparse(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests,
    std::function<const SparseFeatures&(const PredictionRequest&)> accessor,
    bool isWeighted);

at::Tensor combineEmbedding(
    const std::vector<std::shared_ptr<PredictionRequest>>& requests);

} // namespace torchrec

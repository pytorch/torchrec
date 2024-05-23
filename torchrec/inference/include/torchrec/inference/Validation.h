/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "torchrec/inference/Types.h"

namespace torchrec {

// Returns whether sparse features (KeyedJaggedTensor) are valid.
// Currently validates:
//  1. Whether sum(lengths) == size(values)
//  2. Whether there are negative values in lengths
//  3. If weights is present, whether sum(lengths) == size(weights)
bool validateSparseFeatures(
    at::Tensor& values,
    at::Tensor& lengths,
    std::optional<at::Tensor> maybeWeights = c10::nullopt);

// Returns whether dense features are valid.
// Currently validates:
//  1. Whether the size of values is divisable by batch size (request level)
bool validateDenseFeatures(at::Tensor& values, size_t batchSize);

} // namespace torchrec

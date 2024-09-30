/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Validation.h"
#include "ATen/Functions.h"

namespace torchrec {

bool validateSparseFeatures(
    at::Tensor& values,
    at::Tensor& lengths,
    std::optional<at::Tensor> maybeWeights) {
  auto flatLengths = lengths.view(-1);

  // validate sum of lengths equals number of values/weights
  auto lengthsTotal = at::sum(flatLengths).item<int>();
  if (lengthsTotal != values.size(0)) {
    return false;
  }
  if (maybeWeights.has_value() && lengthsTotal != maybeWeights->size(0)) {
    return false;
  }

  // Validate no negative values in lengths.
  // Use faster path if contiguous.
  if (flatLengths.is_contiguous()) {
    int* ptr = (int*)flatLengths.data_ptr();
    for (int i = 0; i < flatLengths.numel(); ++i) {
      if (*ptr < 0) {
        return false;
      }
      ptr++;
    }
  } else {
    // accessor does boundary check (slower)
    auto acc = flatLengths.accessor<int, 1>();
    for (int i = 0; i < acc.size(0); i++) {
      if (acc[i] < 0) {
        return false;
      }
    }
  }

  return true;
}

bool validateDenseFeatures(at::Tensor& values, size_t batchSize) {
  return values.size(0) % batchSize == 0;
}

} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include <ATen/ATen.h>

namespace torchrec {

struct JaggedTensor {
  at::Tensor lengths;
  at::Tensor values;
  at::Tensor weights;
};

struct KeyedJaggedTensor {
  std::vector<std::string> keys;
  at::Tensor lengths;
  at::Tensor values;
  at::Tensor weights;
};

} // namespace torchrec

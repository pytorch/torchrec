/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include <folly/futures/Promise.h>

#include "torchrec/inference/Types.h"

namespace torchrec {

void handleException(
    folly::Promise<std::unique_ptr<PredictionResponse>>& promise,
    const std::string& msg);

} // namespace torchrec

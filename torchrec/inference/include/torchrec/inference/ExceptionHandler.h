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

#include "torchrec/inference/Exception.h"
#include "torchrec/inference/Types.h"

namespace torchrec {
template <typename ExceptionType>
void handleRequestException(
    folly::Promise<std::unique_ptr<PredictionResponse>>& promise,
    const std::string& msg) {
  auto ex = folly::make_exception_wrapper<ExceptionType>(msg);
  auto response = std::make_unique<PredictionResponse>();
  response->exception = std::move(ex);
  promise.setValue(std::move(response));
}

template <typename ExceptionType>
void handleBatchException(
    std::vector<RequestContext>& contexts,
    const std::string& msg) {
  for (auto& context : contexts) {
    handleRequestException<ExceptionType>(context.promise, msg);
  }
}

} // namespace torchrec

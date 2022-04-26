/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Exception.h"

#include <memory>
#include <string>

#include "torchrec/inference/Types.h"

namespace torchrec {

void handleRequestException(
    folly::Promise<std::unique_ptr<PredictionResponse>>& promise,
    const std::string& msg) {
  auto ex = folly::make_exception_wrapper<PredictionException>(msg);
  auto response = std::make_unique<PredictionResponse>();
  response->exception = std::move(ex);
  promise.setValue(std::move(response));
}
void handleBatchException(
    std::vector<RequestContext>& contexts,
    const std::string& msg) {
  for (auto& context : contexts) {
    handleRequestException(context.promise, msg);
  }
}

} // namespace torchrec

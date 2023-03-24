/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <stdexcept>

namespace torchrec {

// We have different error code defined for different kinds of exceptions in
// fblearner/sigrid predictor. (Code pointer:
// fblearner/predictor/if/prediction_service.thrift.) We define different
// exception type here so that in fblearner/sigrid predictor we can detect the
// exception type and return the corresponding error code to reflect the right
// info.
class TorchrecException : public std::runtime_error {
 public:
  explicit TorchrecException(const std::string& error)
      : std::runtime_error(error) {}
};

// GPUOverloadException maps to
// PredictionExceptionCode::GPU_BATCHING_QUEUE_TIMEOUT
class GPUOverloadException : public TorchrecException {
 public:
  explicit GPUOverloadException(const std::string& error)
      : TorchrecException(error) {}
};

// GPUExecutorOverloadException maps to
// PredictionExceptionCode::GPU_EXECUTOR_QUEUE_TIMEOUT
class GPUExecutorOverloadException : public TorchrecException {
 public:
  explicit GPUExecutorOverloadException(const std::string& error)
      : TorchrecException(error) {}
};

// TorchDeployException maps to
// PredictorUserErrorCode::TORCH_DEPLOY_ERROR
class TorchDeployException : public TorchrecException {
 public:
  explicit TorchDeployException(const std::string& error)
      : TorchrecException(error) {}
};
} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/MPMCQueue.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <multipy/runtime/deploy.h>
#include "torchrec/inference/Types.h"

namespace torchrec {

class SingleGPUExecutor {
  constexpr static const size_t kQUEUE_CAPACITY = 10000;

 public:
  struct ExecInfo {
    size_t gpuIdx;
    size_t interpIdx;
    torch::deploy::ReplicatedObj model;
  };
  using ExecInfos = std::vector<ExecInfo>;

  SingleGPUExecutor(
      std::shared_ptr<torch::deploy::InterpreterManager> manager,
      ExecInfos execInfos,
      size_t numGpu,
      c10::Device resultDevice = c10::kCPU);

  // Moveable only
  SingleGPUExecutor(SingleGPUExecutor&& executor) noexcept = default;
  SingleGPUExecutor& operator=(SingleGPUExecutor&& executor) noexcept = default;
  ~SingleGPUExecutor();

  void schedule(std::shared_ptr<PredictionBatch> request);

 private:
  void process();

  std::shared_ptr<torch::deploy::InterpreterManager> manager_;
  const ExecInfos execInfos_;
  const size_t numGpu_;
  const c10::Device resultDevice_;
  folly::MPMCQueue<std::shared_ptr<PredictionBatch>> requests_;

  std::unique_ptr<folly::CPUThreadPoolExecutor> completionExecutor_;
  std::atomic<size_t> roundRobinExecInfoNextIdx_;
  std::thread processThread_;
};
} // namespace torchrec

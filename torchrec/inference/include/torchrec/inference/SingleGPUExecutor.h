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
 public:
  SingleGPUExecutor(
      std::shared_ptr<torch::deploy::InterpreterManager> manager,
      torch::deploy::ReplicatedObj model,
      c10::Device device,
      const std::vector<size_t>& interpreter_idxs,
      c10::Device result_device = c10::kCPU);
  SingleGPUExecutor(SingleGPUExecutor&& executor) noexcept = default;
  SingleGPUExecutor& operator=(SingleGPUExecutor&& executor) noexcept = default;
  ~SingleGPUExecutor();

  void schedule(std::shared_ptr<PredictionBatch> request);

 private:
  void process(const size_t interpreter_idx);

  std::shared_ptr<torch::deploy::InterpreterManager> manager_;
  torch::deploy::ReplicatedObj model_;
  c10::Device device_;
  c10::Device resultDevice_;

  folly::MPMCQueue<std::shared_ptr<PredictionBatch>> requests_;
  std::vector<std::thread> processThreads_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> completionExecutor_;
};
} // namespace torchrec

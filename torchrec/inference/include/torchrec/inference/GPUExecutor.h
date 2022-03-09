/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <stdexcept>

#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/csrc/deploy/deploy.h> // @manual

#include "torchrec/inference/BatchingQueue.h"
#include "torchrec/inference/ResultSplit.h"

namespace torchrec {

class GPUExecutor {
 public:
  GPUExecutor(
      std::shared_ptr<torch::deploy::InterpreterManager> manager,
      torch::deploy::ReplicatedObj model,
      int rank,
      int worldSize,
      std::shared_ptr<torchrec::ResultSplitFunc> func);
  GPUExecutor(GPUExecutor&& executor) noexcept = default;
  GPUExecutor& operator=(GPUExecutor&& executor) noexcept = default;
  ~GPUExecutor();

  void callback(std::shared_ptr<PredictionBatch> batch);

  void process(int idx);

 private:
  // torch deploy
  std::shared_ptr<torch::deploy::InterpreterManager> manager_;
  torch::deploy::ReplicatedObj model_;
  int rank_;
  int worldSize_;

  folly::MPMCQueue<std::shared_ptr<PredictionBatch>> batches_;
  std::vector<std::thread> processThreads_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> completionExecutor_;
  std::shared_ptr<torchrec::ResultSplitFunc> resultSplitFunc_;
};

} // namespace torchrec

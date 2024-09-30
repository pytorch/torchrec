/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

// remove this after we switch over to multipy externally for torchrec
#ifdef FBCODE_CAFFE2
#include <multipy/runtime/deploy.h> // @manual
#else
#include <torch/csrc/deploy/deploy.h> // @manual
#endif

#include "torchrec/inference/BatchingQueue.h"
#include "torchrec/inference/Observer.h"
#include "torchrec/inference/ResultSplit.h"
#include "torchrec/inference/include/torchrec/inference/Observer.h"

namespace torchrec {

class GPUExecutor {
 public:
  // Used to interface with python's garbage collector
  struct GCConfig {
    bool optimizationEnabled = false;
    size_t collectionFreq = 1000;
    size_t statReportingFreq = 10000;
    std::unique_ptr<IDynamicTimeseriesObserver> observer =
        std::make_unique<EmptyDynamicTimeseriesObserver>();
    std::map<int, int> threadIdToNumForwards = std::map<int, int>();
  };

  GPUExecutor(
      std::shared_ptr<torch::deploy::InterpreterManager> manager,
      torch::deploy::ReplicatedObj model,
      size_t rank,
      size_t worldSize,
      std::shared_ptr<torchrec::ResultSplitFunc> func,
      std::chrono::milliseconds queueTimeout,
      std::shared_ptr<IGPUExecutorObserver>
          observer, // shared_ptr because used in completion executor callback
      std::function<void()> warmupFn = {},
      std::optional<size_t> numThreadsPerGPU = c10::nullopt,
      std::unique_ptr<GCConfig> gcConfig = std::make_unique<GCConfig>());
  GPUExecutor(GPUExecutor&& executor) noexcept = default;
  GPUExecutor& operator=(GPUExecutor&& executor) noexcept = default;
  ~GPUExecutor();

  void callback(std::shared_ptr<PredictionBatch> batch);

  void process(int idx);

 private:
  // torch deploy
  std::shared_ptr<torch::deploy::InterpreterManager> manager_;
  torch::deploy::ReplicatedObj model_;
  const size_t rank_;
  const size_t worldSize_;

  folly::MPMCQueue<std::shared_ptr<PredictionBatch>> batches_;
  std::vector<std::thread> processThreads_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> rejectionExecutor_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> completionExecutor_;
  std::shared_ptr<torchrec::ResultSplitFunc> resultSplitFunc_;
  const std::chrono::milliseconds queueTimeout_;
  std::shared_ptr<IGPUExecutorObserver> observer_;
  std::function<void()> warmupFn_;

  std::mutex warmUpMutex_;
  std::mutex warmUpAcquireSessionMutex_;
  std::condition_variable warmUpCV_;
  int warmUpCounter_{0};

  size_t numThreadsPerGPU_;

  std::unique_ptr<GCConfig> gcConfig_;

  void reportGCStats(c10::IValue stats);
};

} // namespace torchrec

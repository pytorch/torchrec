/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h> // @manual
#include <c10/cuda/CUDAStream.h>
#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>
#include <folly/futures/Future.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseThread.h>
#include <folly/synchronization/Baton.h>

#include "torchrec/inference/Batching.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

struct RequestContext {
  uint32_t batchSize;
  folly::Promise<std::unique_ptr<PredictionResponse>> promise;
};

struct PredictionBatch {
  size_t batchSize;

  c10::Dict<std::string, at::Tensor> forwardArgs;

  std::vector<RequestContext> contexts;

  std::chrono::time_point<std::chrono::steady_clock> enqueueTime =
      std::chrono::steady_clock::now();

  std::unique_ptr<at::cuda::CUDAEvent> event =
      std::make_unique<at::cuda::CUDAEvent>(
          cudaEventBlockingSync | cudaEventDisableTiming);

  void cuda();

  size_t size() const;
};

using BatchQueueCb = std::function<void(std::shared_ptr<PredictionBatch>)>;

class BatchingQueue {
 public:
  struct Config {
    std::chrono::milliseconds batchingInterval = std::chrono::milliseconds(10);
    std::chrono::milliseconds queueTimeout = std::chrono::milliseconds(500);
    int numMemPinnerThreads = 4;
    int maxBatchSize = 2000;
    // For feature name to BatchingFunc name.
    const std::unordered_map<std::string, std::string> batchingMetadata;
  };

  BatchingQueue(const BatchingQueue&) = delete;
  BatchingQueue& operator=(const BatchingQueue&) = delete;

  BatchingQueue(
      std::vector<BatchQueueCb> cbs,
      const Config& config,
      int worldSize);
  ~BatchingQueue();

  void add(
      std::shared_ptr<PredictionRequest> request,
      folly::Promise<std::unique_ptr<PredictionResponse>> promise);

  void stop();

 private:
  struct QueryQueueEntry {
    std::shared_ptr<PredictionRequest> request;
    RequestContext context;
    std::chrono::time_point<std::chrono::steady_clock> addedTime;
  };

  struct BatchingQueueEntry {
    std::vector<std::shared_ptr<PredictionRequest>> requests;
    std::vector<RequestContext> contexts;
  };

  void createBatch();

  void pinMemory(int gpuIdx);

  const Config config_;

  // Batching func name to batching func instance.
  std::unordered_map<std::string, std::unique_ptr<torchrec::BatchingFunc>>
      batchingFuncs_;
  std::vector<BatchQueueCb> cbs_;
  std::thread batchingThread_;
  std::vector<std::thread> memPinnerThreads_;
  folly::Synchronized<std::queue<QueryQueueEntry>> requestQueue_;
  std::vector<std::shared_ptr<folly::MPMCQueue<BatchingQueueEntry>>>
      batchingQueues_;
  std::atomic<bool> stopping_;
  int worldSize_;
};

} // namespace torchrec

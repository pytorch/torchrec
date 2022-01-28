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
  bool isTimedOut = false;
};

struct PredictionBatch {
  size_t batch_size;
  // B x F
  at::Tensor float_features;
  // T x B x L (jagged)
  torchrec::JaggedTensor id_list_features;
  // T x B x L (jagged)
  torchrec::JaggedTensor id_score_list_features;
  // B x E x F
  at::Tensor embedding_features;

  std::vector<RequestContext> contexts;

  std::chrono::time_point<std::chrono::steady_clock> enqueueTime =
      std::chrono::steady_clock::now();

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
  };

  BatchingQueue(const BatchingQueue&) = delete;
  BatchingQueue& operator=(const BatchingQueue&) = delete;

  BatchingQueue(
      std::vector<BatchQueueCb> cbs,
      const Config& config,
      int worldSize);

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
    size_t numTimedOutRequests = 0;
  };

  struct BatchQueueEntry {
    std::shared_ptr<PredictionBatch> batch;
    at::cuda::CUDAEvent event;
    std::chrono::time_point<std::chrono::steady_clock> addedTime;
    size_t tensorSize;
  };

  void createBatch();

  void pinMemory(int gpuIdx);

  void processCallback(int gpuIdx);

  const Config config_;

  std::vector<BatchQueueCb> cbs_;
  std::thread batchingThread_;
  std::vector<std::thread> memPinnerThreads_;
  std::vector<std::thread> callbackThreads_;
  folly::Synchronized<std::queue<QueryQueueEntry>> requestQueue_;
  std::vector<folly::Synchronized<std::queue<BatchQueueEntry>>> batchQueues_;
  std::vector<std::shared_ptr<folly::MPMCQueue<BatchingQueueEntry>>>
      batchingQueues_;
  std::atomic<bool> stopping_;
  int worldSize_;
};

} // namespace torchrec

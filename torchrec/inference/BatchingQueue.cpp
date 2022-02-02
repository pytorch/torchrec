/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/BatchingQueue.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include <ATen/Functions.h> // @manual
#include <ATen/core/interned_strings.h>
#include <ATen/record_function.h> // @manual
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fmt/format.h>
#include <folly/ExceptionString.h>
#include <folly/MPMCQueue.h>
#include <folly/Range.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/Cursor.h>
#include <glog/logging.h>

using namespace std::chrono_literals;

namespace torchrec {

void PredictionBatch::cuda() {
  c10::Dict<std::string, at::Tensor> forwardArgsCuda;
  for (auto& iter : forwardArgs) {
    forwardArgsCuda.insert(
        iter.key(), iter.value().to(at::kCUDA, /* non_blocking */ true));
  }
  forwardArgs = std::move(forwardArgsCuda);
}

size_t PredictionBatch::size() const {
  size_t size = 0;
  for (auto& iter : forwardArgs) {
    size += iter.value().storage().nbytes();
  }
  return size;
}

BatchingQueue::BatchingQueue(
    std::vector<BatchQueueCb> cbs,
    const Config& config,
    int worldSize)
    : config_(config),
      cbs_(std::move(cbs)),
      batchQueues_(worldSize),
      stopping_(false),
      worldSize_(worldSize) {
  for (const auto& [_, batchingFuncName] : config_.batchingMetadata) {
    if (batchingFuncs_.count(batchingFuncName) > 0) {
      continue;
    }
    batchingFuncs_[batchingFuncName] =
        TorchRecBatchingFuncRegistry()->Create(batchingFuncName);
  }
  for (int i = 0; i < worldSize_; i++) {
    auto queue = std::make_shared<folly::MPMCQueue<BatchingQueueEntry>>(
        folly::MPMCQueue<BatchingQueueEntry>(1000));
    batchingQueues_.push_back(std::move(queue));
  }
  batchingThread_ = std::thread([&] {
    folly::setThreadName("CreateBatch");
    createBatch();
  });
  for (int i = 0; i < worldSize_; ++i) {
    for (int j = 0; j < config_.numMemPinnerThreads; ++j) {
      memPinnerThreads_.emplace_back([&, i, j] {
        folly::setThreadName(fmt::format("MemoryPinner-{}-GPU-{}", j, i));
        pinMemory(i);
      });
    }
    callbackThreads_.emplace_back([&, i] {
      folly::setThreadName(fmt::format("CallBack-GPU-{}", i));
      processCallback(i);
    });
  }
}

void BatchingQueue::add(
    std::shared_ptr<PredictionRequest> request,
    folly::Promise<std::unique_ptr<PredictionResponse>> promise) {
  CHECK_GT(request->batch_size, 0);
  const auto addedTime = std::chrono::steady_clock::now();
  requestQueue_.withWLock([request = std::move(request),
                           promise = std::move(promise),
                           addedTime = addedTime](auto& queue) mutable {
    const auto batchSize = request->batch_size;
    queue.push(QueryQueueEntry{
        std::move(request),
        RequestContext{batchSize, std::move(promise)},
        addedTime});
  });
}

void BatchingQueue::stop() {
  stopping_ = true;
  // TODO: properly drain the queue before stopping the threads.
  batchingThread_.join();
  for (auto& thread : memPinnerThreads_) {
    thread.join();
  }
  for (auto& thread : callbackThreads_) {
    thread.join();
  }
}

void BatchingQueue::createBatch() {
  auto startTime = std::chrono::steady_clock::now();
  size_t batchSize = 0;
  std::vector<std::shared_ptr<PredictionRequest>> requests;
  std::vector<RequestContext> contexts;
  int roundRobinIdx_ = 0;

  while (!stopping_) {
    const auto now = std::chrono::steady_clock::now();
    size_t numTimedOutRequests = 0;
    bool full = false;

    requestQueue_.withWLock([&](auto& queue) {
      while (!queue.empty()) {
        auto& front = queue.front();
        if (batchSize + front.request->batch_size > config_.maxBatchSize) {
          full = true;
          break;
        }

        auto& context = contexts.emplace_back(std::move(front.context));
        context.isTimedOut =
            std::chrono::steady_clock::now() - front.addedTime >
            config_.queueTimeout;

        if (!context.isTimedOut) {
          requests.push_back(std::move(front.request));
          batchSize += requests.back()->batch_size;
        }

        queue.pop();
      }
    });

    if (full ||
        (std::chrono::steady_clock::now() - startTime >
         config_.batchingInterval)) {
      batchingQueues_[roundRobinIdx_++]->blockingWrite(BatchingQueueEntry{
          .requests = std::move(requests),
          .contexts = std::move(contexts),
          .numTimedOutRequests = numTimedOutRequests});

      startTime = std::chrono::steady_clock::now();
      batchSize = 0;
      roundRobinIdx_ %= worldSize_;
      requests.clear();
      contexts.clear();
    }

    if (!full) {
      /* sleep override */
      std::this_thread::sleep_until(now + config_.batchingInterval);
    }
  }
}

void BatchingQueue::pinMemory(int gpuIdx) {
  at::cuda::CUDAGuard deviceGuard(gpuIdx);
  at::cuda::CUDAStreamGuard streamGuard(
      at::cuda::getStreamFromPool(/* isHighPriority */ false));
  while (!stopping_) {
    BatchingQueueEntry entry;
    if (!batchingQueues_[gpuIdx]->tryReadUntil(
            std::chrono::steady_clock::now() + 10ms, entry)) {
      continue;
    }

    auto& requests = entry.requests;
    auto& contexts = entry.contexts;

    try {
      if (!requests.empty() || !contexts.empty()) {
        RECORD_USER_SCOPE("PinMemory");
        // Combine data.
        size_t combinedBatchSize = 0;
        for (const auto& request : requests) {
          combinedBatchSize += request->batch_size;
        }

        BatchQueueEntry batchedEntry;

        c10::Dict<std::string, at::Tensor> forwardArgs;
        auto combineForwardArgs =
            [&](std::unordered_map<std::string, at::Tensor> map) {
              for (auto& [key, value] : map) {
                CHECK(!forwardArgs.contains(key));
                forwardArgs.insert(key, std::move(value));
              }
            };

        for (auto& [featureName, batchingFuncName] : config_.batchingMetadata) {
          combineForwardArgs(
              batchingFuncs_[batchingFuncName]->batch(featureName, requests));
        }

        batchedEntry.batch = std::make_shared<PredictionBatch>(PredictionBatch{
            combinedBatchSize, std::move(forwardArgs), std::move(contexts)});
        batchedEntry.batch->cuda();
        batchedEntry.event.record();
        batchedEntry.addedTime = std::chrono::steady_clock::now();
        batchQueues_[gpuIdx].withWLock(
            [batchedEntry = std::move(batchedEntry)](auto& queue) mutable {
              queue.push(std::move(batchedEntry));
            });
      }
    } catch (const std::exception& ex) {
      LOG(FATAL) << "Error batching requests, ex: " << folly::exceptionStr(ex);
    }
  }
}

void BatchingQueue::processCallback(int gpuIdx) {
  while (!stopping_) {
    std::shared_ptr<PredictionBatch> batch;
    size_t tensorSize = 0;
    std::chrono::time_point<std::chrono::steady_clock> batchAddedTime;

    batchQueues_[gpuIdx].withWLock([&](auto& queue) mutable {
      if (queue.empty()) {
        return;
      }
      if (auto& front = queue.front(); front.event.query()) {
        batch = std::move(front.batch);
        batchAddedTime = front.addedTime;
        tensorSize = batch->size();
        queue.pop();
      }
    });

    if (!batch) {
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    }

    cbs_[gpuIdx](std::move(batch));
  }
}

} // namespace torchrec

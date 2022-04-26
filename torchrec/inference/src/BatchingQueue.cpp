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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <ATen/Functions.h> // @manual
#include <ATen/core/Dict.h>
#include <ATen/core/interned_strings.h>
#include <ATen/record_function.h> // @manual
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fmt/format.h>
#include <folly/ExceptionString.h>
#include <folly/Lazy.h>
#include <folly/MPMCQueue.h>
#include <folly/Range.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/Cursor.h>
#include <glog/logging.h>

#include "torchrec/inference/Exception.h"
#include "torchrec/inference/Types.h"

using namespace std::chrono_literals;

namespace torchrec {

void PredictionBatch::cuda() {
  for (auto& iter : forwardArgs) {
    if (iter.value().is_cpu()) {
      iter.setValue(iter.value().to(at::kCUDA, /* non_blocking */ true));
    }
  }
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
        c10::InferenceMode guard;
        folly::setThreadName(fmt::format("MemoryPinner-{}-GPU-{}", j, i));
        pinMemory(i);
      });
    }
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
}

void BatchingQueue::createBatch() {
  std::optional<std::chrono::time_point<std::chrono::steady_clock>> startTime;
  size_t batchSize = 0;
  std::vector<std::shared_ptr<PredictionRequest>> requests;
  std::vector<RequestContext> contexts;
  int roundRobinIdx = 0;

  while (!stopping_) {
    bool full = false;

    requestQueue_.withWLock([&](auto& queue) {
      while (!queue.empty()) {
        auto& front = queue.front();

        if (std::chrono::steady_clock::now() - front.addedTime >=
            config_.queueTimeout) {
          handleException(front.context.promise, "Batching queue timeout");
          queue.pop();
          continue;
        }

        if (!startTime) {
          startTime = front.addedTime;
        }

        if (batchSize + front.request->batch_size > config_.maxBatchSize) {
          full = true;
          break;
        }

        auto& context = contexts.emplace_back(std::move(front.context));
        requests.push_back(std::move(front.request));
        batchSize += requests.back()->batch_size;
        queue.pop();
      }
    });

    if (full ||
        (startTime &&
         (std::chrono::steady_clock::now() - *startTime >=
          config_.batchingInterval))) {
      batchingQueues_[roundRobinIdx++]->blockingWrite(BatchingQueueEntry{
          .requests = std::move(requests), .contexts = std::move(contexts)});

      startTime.reset();
      batchSize = 0;
      roundRobinIdx %= worldSize_;
      requests.clear();
      contexts.clear();
    }

    if (!full) {
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void BatchingQueue::pinMemory(int gpuIdx) {
  at::cuda::CUDAGuard deviceGuard(gpuIdx);
  at::cuda::CUDAStreamGuard streamGuard(
      at::cuda::getStreamFromPool(/* isHighPriority */ false));
  if (config_.warmupFn) {
    config_.warmupFn();
  }

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
        for (auto i : c10::irange(requests.size())) {
          combinedBatchSize += requests[i]->batch_size;
        }

        auto batchOffsetsLazy =
            folly::lazy<std::function<at::Tensor()>>([&]() -> at::Tensor {
              size_t batchSize = 0;
              auto batchOffsets = at::empty(
                  {static_cast<long>(requests.size() + 1)},
                  at::TensorOptions().dtype(at::kInt).pinned_memory(true));
              auto batchOffsetsAcc = batchOffsets.accessor<int32_t, 1>();
              batchOffsetsAcc[0] = 0;
              for (auto i : c10::irange(requests.size())) {
                batchSize += requests[i]->batch_size;
                batchOffsetsAcc[i + 1] = batchSize;
              }
              return batchOffsets;
            });

        auto batchItemsLazy =
            folly::lazy<std::function<at::Tensor()>>([&]() -> at::Tensor {
              auto batchItems = at::empty(
                  {static_cast<int64_t>(combinedBatchSize)},
                  at::TensorOptions().dtype(at::kInt).pinned_memory(true));
              auto batchItemsAcc = batchItems.accessor<int32_t, 1>();
              auto batchOffsetsAcc = batchOffsetsLazy().accessor<int32_t, 1>();
              for (auto i = 0; i < requests.size(); ++i) {
                auto start = batchOffsetsAcc[i];
                auto end = batchOffsetsAcc[i + 1];
                for (auto j = start; j < end; ++j) {
                  batchItemsAcc[j] = i;
                }
              }
              return batchItems;
            });

        c10::Dict<std::string, at::Tensor> forwardArgs;
        auto combineForwardArgs =
            [&](std::unordered_map<std::string, at::Tensor> map) {
              for (auto& [key, value] : map) {
                CHECK(!forwardArgs.contains(key));
                forwardArgs.insert(key, std::move(value));
              }
            };

        for (auto& [featureName, batchingFuncName] : config_.batchingMetadata) {
          combineForwardArgs(batchingFuncs_[batchingFuncName]->batch(
              featureName,
              requests,
              combinedBatchSize,
              batchOffsetsLazy,
              c10::Device(c10::kCUDA, gpuIdx),
              batchItemsLazy));
        }

        auto batch = std::make_shared<PredictionBatch>(PredictionBatch{
            combinedBatchSize, std::move(forwardArgs), std::move(contexts)});
        batch->cuda();
        auto createEvent = [&]() {
          return Event(
              std::make_unique<at::cuda::CUDAEvent>(
                  cudaEventBlockingSync | cudaEventDisableTiming)
                  .release(),
              [](at::cuda::CUDAEvent* event) { delete event; });
        };
        batch->event = config_.eventCreationFn ? config_.eventCreationFn(gpuIdx)
                                               : createEvent();
        batch->event->record();
        cbs_[gpuIdx](batch);
      }
    } catch (const std::exception& ex) {
      LOG(FATAL) << "Error batching requests, ex: " << folly::exceptionStr(ex);
    }
  }
}

BatchingQueue::~BatchingQueue() {
  stop();
}

} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <string>

namespace torchrec {

class IBatchingQueueObserver {
 public:
  // Record the amount of time an entry of PredictionRequests
  // in the batching queue waits before they are read and allocated
  // onto a GPU device.
  virtual void recordBatchingQueueLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the amount of time it takes for a batching function
  // to execute.
  virtual void recordBatchingFuncLatency(
      double value,
      std::string batchingFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the amount of time it takes to create a batch of
  // requests.
  virtual void recordBatchCreationLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Increment the number of batching queue timeouts experienced.
  virtual void addBatchingQueueTimeoutCount(double value) = 0;

  // Increment the number of times a GPU could not be chosen
  // for allocation.
  virtual void addGPUBusyCount(double value) = 0;

  // Increment the number of requests entering the batching queue.
  virtual void addRequestsCount(double value) = 0;

  // Increment the number of bytes of tensors moved to cuda.
  virtual void addBytesMovedToGPUCount(double value) = 0;

  // Increment the number of batches processed by the batching
  // queue (moved onto the GPU executor).
  virtual void addBatchesProcessedCount(double value) = 0;

  // Increment the number of requests processed by the batching
  // queue (moved onto the GPU executor).
  virtual void addRequestsProcessedCount(double value) = 0;

  // The obervations that should be made when a batch is completed.
  virtual void observeBatchCompletion(
      size_t batchSizeBytes,
      size_t numRequests) {
    addBytesMovedToGPUCount(batchSizeBytes);
    addBatchesProcessedCount(1);
    addRequestsProcessedCount(numRequests);
  }

  virtual ~IBatchingQueueObserver() {}
};

// Can be used for testing or for opt-ing out of observation.
class EmptyBatchingQueueObserver : public IBatchingQueueObserver {
 public:
  void recordBatchingQueueLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) override {
    (void)value, (void)now;
  }

  void recordBatchingFuncLatency(
      double value,
      std::string batchingFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) override {
    (void)value, (void)batchingFuncName, (void)now;
  }

  void recordBatchCreationLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) override {
    (void)value, (void)now;
  }

  void addBatchingQueueTimeoutCount(double value) override {
    (void)value;
  }

  void addGPUBusyCount(double value) override {
    (void)value;
  }

  void addRequestsCount(double value) override {
    (void)value;
  }

  void addBytesMovedToGPUCount(double value) override {
    (void)value;
  }

  void addBatchesProcessedCount(double value) override {
    (void)value;
  }

  void addRequestsProcessedCount(double value) override {
    (void)value;
  }
};

} // namespace torchrec

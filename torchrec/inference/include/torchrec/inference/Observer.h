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
      double /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordBatchingFuncLatency(
      double /* value */,
      std::string /* batchingFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordBatchCreationLatency(
      double /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void addBatchingQueueTimeoutCount(double /* value */) override {}

  void addGPUBusyCount(double /* value */) override {}

  void addRequestsCount(double /* value */) override {}

  void addBytesMovedToGPUCount(double /* value */) override {}

  void addBatchesProcessedCount(double /* value */) override {}

  void addRequestsProcessedCount(double /* value */) override {}
};

class IGPUExecutorObserver {
 public:
  // Record the amount of time a batch spends in the GPU Executor
  // queue.
  virtual void recordQueueLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of prediction (forward call, H2D).
  virtual void recordPredictionLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of device to host transfer facilitated
  // by result split function.
  virtual void recordDeviceToHostLatency(
      double value,
      std::string resultSplitFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of splitting the result.
  virtual void recordResultSplitLatency(
      double value,
      std::string resultSplitFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency from enqueue to completion.
  virtual void recordTotalLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Increment the number of GPUExecutor queue timeouts.
  virtual void addQueueTimeoutCount(double value) = 0;

  // Increment the number of predict exceptions.
  virtual void addPredictionExceptionCount(double value) = 0;

  // Increment the number of batches successfully processed.
  virtual void addBatchesProcessedCount(double value) = 0;

  virtual ~IGPUExecutorObserver() {}
};

class ISingleGPUExecutorObserver {
 public:
  virtual void addRequestsCount(double value) = 0;
  virtual void addRequestProcessingExceptionCount(double value) = 0;
  virtual void recordQueueLatency(
      double value,
      std::chrono::steady_clock::time_point =
          std::chrono::steady_clock::now()) = 0;

  virtual void recordRequestProcessingLatency(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  virtual ~ISingleGPUExecutorObserver() = default;
};

class EmptySingleGPUExecutorObserver : public ISingleGPUExecutorObserver {
  void addRequestsCount(double) override {}
  void addRequestProcessingExceptionCount(double) override {}
  void recordQueueLatency(
      double,
      std::chrono::steady_clock::time_point =
          std::chrono::steady_clock::now()) override {}

  void recordRequestProcessingLatency(
      double,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) override {}
};

// Can be used for testing or for opt-ing out of observation.
class EmptyGPUExecutorObserver : public IGPUExecutorObserver {
 public:
  void recordQueueLatency(
      double /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordPredictionLatency(
      double /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordDeviceToHostLatency(
      double /* value */,
      std::string /* resultSplitFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordResultSplitLatency(
      double /* value */,
      std::string /* resultSplitFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordTotalLatency(
      double /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void addQueueTimeoutCount(double /* value */) override {}

  void addPredictionExceptionCount(double /* value */) override {}

  void addBatchesProcessedCount(double /* value */) override {}
};

// Helper for determining how much time has elapsed in milliseconds since a
// given time point.
inline std::chrono::milliseconds getTimeElapsedMS(
    std::chrono::steady_clock::time_point startTime) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - startTime);
}

} // namespace torchrec

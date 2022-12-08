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

// Record generic timeseries stat with a key
class IDynamicTimeseriesObserver {
 public:
  virtual void addCount(uint32_t value, std::string key) = 0;

  virtual ~IDynamicTimeseriesObserver() {}
};

// Can be used for testing or for opt-ing out of observation.
class EmptyDynamicTimeseriesObserver : public IDynamicTimeseriesObserver {
 public:
  void addCount(uint32_t /* value */, std::string /* key */) override {}
};

class IBatchingQueueObserver {
 public:
  // Record the amount of time an entry of PredictionRequests
  // in the batching queue waits before they are read and allocated
  // onto a GPU device.
  virtual void recordBatchingQueueLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the amount of time it takes for a batching function
  // to execute.
  virtual void recordBatchingFuncLatency(
      uint32_t value,
      std::string batchingFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the amount of time it takes to create a batch of
  // requests.
  virtual void recordBatchCreationLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Increment the number of batching queue timeouts experienced.
  virtual void addBatchingQueueTimeoutCount(uint32_t value) = 0;

  // Increment the number of times a GPU could not be chosen
  // for allocation.
  virtual void addGPUBusyCount(uint32_t value) = 0;

  // Increment the number of requests entering the batching queue.
  virtual void addRequestsCount(uint32_t value) = 0;

  // Increment the number of bytes of tensors moved to cuda.
  virtual void addBytesMovedToGPUCount(uint32_t value) = 0;

  // Increment the number of batches processed by the batching
  // queue (moved onto the GPU executor).
  virtual void addBatchesProcessedCount(uint32_t value) = 0;

  // Increment the number of requests processed by the batching
  // queue (moved onto the GPU executor).
  virtual void addRequestsProcessedCount(uint32_t value) = 0;

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
      uint32_t /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordBatchingFuncLatency(
      uint32_t /* value */,
      std::string /* batchingFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordBatchCreationLatency(
      uint32_t /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void addBatchingQueueTimeoutCount(uint32_t /* value */) override {}

  void addGPUBusyCount(uint32_t /* value */) override {}

  void addRequestsCount(uint32_t /* value */) override {}

  void addBytesMovedToGPUCount(uint32_t /* value */) override {}

  void addBatchesProcessedCount(uint32_t /* value */) override {}

  void addRequestsProcessedCount(uint32_t /* value */) override {}
};

class IGPUExecutorObserver {
 public:
  // Record the amount of time a batch spends in the GPU Executor
  // queue.
  virtual void recordQueueLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of prediction (forward call, H2D).
  virtual void recordPredictionLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of device to host transfer facilitated
  // by result split function.
  virtual void recordDeviceToHostLatency(
      uint32_t value,
      std::string resultSplitFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency of splitting the result.
  virtual void recordResultSplitLatency(
      uint32_t value,
      std::string resultSplitFuncName,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Record the latency from enqueue to completion.
  virtual void recordTotalLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Increment the number of GPUExecutor queue timeouts.
  virtual void addQueueTimeoutCount(uint32_t value) = 0;

  // Increment the number of predict exceptions.
  virtual void addPredictionExceptionCount(uint32_t value) = 0;

  // Increment the number of batches successfully processed.
  virtual void addBatchesProcessedCount(uint32_t value) = 0;

  virtual ~IGPUExecutorObserver() {}
};

class ISingleGPUExecutorObserver {
 public:
  virtual void addRequestsCount(uint32_t value) = 0;
  virtual void addRequestProcessingExceptionCount(uint32_t value) = 0;
  virtual void recordQueueLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point =
          std::chrono::steady_clock::now()) = 0;

  virtual void recordRequestProcessingLatency(
      uint32_t value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  virtual ~ISingleGPUExecutorObserver() = default;
};

class EmptySingleGPUExecutorObserver : public ISingleGPUExecutorObserver {
  void addRequestsCount(uint32_t) override {}
  void addRequestProcessingExceptionCount(uint32_t) override {}
  void recordQueueLatency(
      uint32_t,
      std::chrono::steady_clock::time_point =
          std::chrono::steady_clock::now()) override {}

  void recordRequestProcessingLatency(
      uint32_t,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) override {}
};

// Can be used for testing or for opt-ing out of observation.
class EmptyGPUExecutorObserver : public IGPUExecutorObserver {
 public:
  void recordQueueLatency(
      uint32_t /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordPredictionLatency(
      uint32_t /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordDeviceToHostLatency(
      uint32_t /* value */,
      std::string /* resultSplitFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordResultSplitLatency(
      uint32_t /* value */,
      std::string /* resultSplitFuncName */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void recordTotalLatency(
      uint32_t /* value */,
      std::chrono::steady_clock::time_point /* now */) override {}

  void addQueueTimeoutCount(uint32_t /* value */) override {}

  void addPredictionExceptionCount(uint32_t /* value */) override {}

  void addBatchesProcessedCount(uint32_t /* value */) override {}
};

class IResourceManagerObserver {
 public:
  // Add the number of requests in flight for a gpu
  virtual void addOutstandingRequestsCount(uint32_t value, int gpuIdx) = 0;

  // Add the most in flight requests on a gpu ever
  virtual void addAllTimeHighOutstandingCount(uint32_t value, int gpuIdx) = 0;

  // Record the latency for finding a device
  virtual void addWaitingForDeviceLatency(
      uint32_t value,
      int gpuIdx,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now()) = 0;

  // Recording all stats related to resource manager at once.
  virtual void recordAllStats(
      uint32_t outstandingRequests,
      uint32_t allTimeHighOutstanding,
      uint32_t waitedForMs,
      int gpuIdx) {
    addOutstandingRequestsCount(outstandingRequests, gpuIdx);
    addAllTimeHighOutstandingCount(allTimeHighOutstanding, gpuIdx);
    addWaitingForDeviceLatency(waitedForMs, gpuIdx);
  }

  virtual ~IResourceManagerObserver() {}
};

// Can be used for testing or for opt-ing out of observation.
class EmptyResourceManagerObserver : public IResourceManagerObserver {
 public:
  void addOutstandingRequestsCount(uint32_t /* value */, int /* gpuIdx */)
      override {}

  void addAllTimeHighOutstandingCount(uint32_t /* value */, int /* gpuIdx */)
      override {}

  void addWaitingForDeviceLatency(
      uint32_t /* value */,
      int /* gpuIdx */,
      std::chrono::steady_clock::time_point /* now */) override {}
};

// Helper for determining how much time has elapsed in milliseconds since a
// given time point.
inline std::chrono::milliseconds getTimeElapsedMS(
    std::chrono::steady_clock::time_point startTime) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - startTime);
}

} // namespace torchrec

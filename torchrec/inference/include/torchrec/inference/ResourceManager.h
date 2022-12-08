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
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <folly/small_vector.h>
#include <glog/logging.h>

#include "torchrec/inference/Observer.h"

namespace torchrec {

/**
 * ResourceManager can be used to limit in-flight batches
 * allocated onto GPUs to prevent OOMing.
 */
class ResourceManager {
 public:
  ResourceManager(
      int worldSize,
      size_t maxOutstandingBatches,
      int logFrequency = 100,
      std::unique_ptr<IResourceManagerObserver> observer =
          std::make_unique<EmptyResourceManagerObserver>());

  // Returns whether batches can be allocated onto a device based on
  // slack provided (ms) and maxOutstandingBatches_).
  bool occupyDevice(int gpuIdx, std::chrono::milliseconds slack);

  void release(int gpuIdx);

 private:
  folly::small_vector<int> gpuToOutstandingBatches_;
  // Helpful for tuning
  folly::small_vector<int> allTimeHigh_;
  const size_t maxOutstandingBatches_;
  const int logFrequency_;
  // Align as 64B to avoid false sharing
  alignas(64) std::mutex mu_;
  std::unique_ptr<IResourceManagerObserver> observer_;
};

class ResourceManagerGuard {
 public:
  ResourceManagerGuard(
      std::weak_ptr<ResourceManager> resourceManager,
      int gpuIdx)
      : resourceManager_(std::move(resourceManager)), gpuIdx_(gpuIdx) {}

  ~ResourceManagerGuard() {
    std::shared_ptr rm = resourceManager_.lock();
    if (rm != nullptr) {
      rm->release(gpuIdx_);
    }
  }

 private:
  std::weak_ptr<ResourceManager> resourceManager_;
  const int gpuIdx_;
};

} // namespace torchrec

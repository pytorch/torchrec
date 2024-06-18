/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include <folly/Random.h>
#include <folly/String.h>
#include <folly/small_vector.h>
#include <glog/logging.h>

#include "torchrec/inference/ResourceManager.h"

namespace torchrec {

ResourceManager::ResourceManager(
    int worldSize,
    size_t maxOutstandingBatches,
    int logFrequency,
    std::unique_ptr<IResourceManagerObserver> observer)
    : gpuToOutstandingBatches_(worldSize),
      allTimeHigh_(worldSize),
      maxOutstandingBatches_(maxOutstandingBatches),
      logFrequency_(logFrequency),
      observer_(std::move(observer)) {
  CHECK(observer_ != nullptr);
}

bool ResourceManager::occupyDevice(
    int gpuIdx,
    std::chrono::milliseconds slack) {
  const auto startTime = std::chrono::steady_clock::now();
  std::chrono::milliseconds waitedFor = std::chrono::milliseconds(0);

  // Exit loop once time expires or device selected.
  while (true) {
    {
      // With lock, try to get device.
      std::lock_guard<std::mutex> lock(mu_);
      if (gpuToOutstandingBatches_[gpuIdx] < maxOutstandingBatches_) {
        // Pick GPU and update stats.
        LOG_EVERY_N(INFO, logFrequency_)
            << "Picked device " << gpuIdx << ", with load "
            << gpuToOutstandingBatches_[gpuIdx]
            << " -- gpuToOutstandingBatches_ list <"
            << folly::join(",", gpuToOutstandingBatches_) << ">. "
            << " -- all time highs: <" << folly::join(",", allTimeHigh_)
            << ">. " << "Waited: " << waitedFor.count()
            << " ms. Slack: " << slack.count() << " ms.";

        gpuToOutstandingBatches_[gpuIdx] += 1;
        observer_->recordAllStats(
            gpuToOutstandingBatches_[gpuIdx],
            allTimeHigh_[gpuIdx],
            waitedFor.count(),
            gpuIdx);

        if (gpuToOutstandingBatches_[gpuIdx] > allTimeHigh_[gpuIdx]) {
          allTimeHigh_[gpuIdx] = gpuToOutstandingBatches_[gpuIdx];
        }

        // Successfully grabbed a GPU slot.
        return true;
      }
    }
    // GPU has too many outstanding batches;
    // Sleep & wait before retrying to get a slot in GPU.
    LOG_EVERY_N(WARNING, logFrequency_)
        << "maxOutstandingBatches_ reached for device " << gpuIdx
        << "! Waiting for " << waitedFor.count() << " ms... "
        << "Current gpuToOutstandingBatches <"
        << folly::join(",", gpuToOutstandingBatches_) << ">.";

    // Sleep to avoid busy waiting
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    waitedFor = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime);
    if (waitedFor >= slack) {
      observer_->recordAllStats(
          gpuToOutstandingBatches_[gpuIdx],
          allTimeHigh_[gpuIdx],
          waitedFor.count(),
          gpuIdx);
      // We have used up all the slack -- requests should time out.
      LOG(WARNING) << "Timing out a batch of requests after slack of "
                   << slack.count() << " ms was exceeded!";
      return false;
    }
  }
}

void ResourceManager::release(int gpuIdx) {
  std::lock_guard<std::mutex> lock(mu_);
  gpuToOutstandingBatches_[gpuIdx] -= 1;
  observer_->addOutstandingRequestsCount(
      gpuToOutstandingBatches_[gpuIdx], gpuIdx);
}

} // namespace torchrec

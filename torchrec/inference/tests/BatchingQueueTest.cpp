/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/BatchingQueue.h"

#include <memory>
#include <thread>

#include <cuda_runtime_api.h> // @manual
#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/io/IOBuf.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

namespace torchrec {

std::shared_ptr<PredictionRequest> createRequest(
    size_t numFeatures,
    size_t batchSize) {
  auto ret = std::make_shared<PredictionRequest>();
  FloatFeatures feature;
  feature.num_features = numFeatures;
  auto values = std::unique_ptr<float[]>(new float[numFeatures * batchSize]);

  for (auto b = 0; b < batchSize; ++b) {
    for (auto f = 0; f < numFeatures; ++f) {
      values.get()[b * numFeatures + f] = static_cast<float>(f);
    }
  }
  ret->batch_size = batchSize;
  feature.values = folly::IOBuf(
      folly::IOBuf::TAKE_OWNERSHIP,
      values.release(),
      numFeatures * batchSize * sizeof(float));
  feature.num_features = numFeatures;
  ret->features["float_features"] = std::move(feature);

  return ret;
}

TEST(BatchingQueueTest, Basic) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }

  folly::Synchronized<std::shared_ptr<PredictionBatch>> res;
  std::vector<BatchQueueCb> batchQueueCbs;
  batchQueueCbs.push_back(
      [&](std::shared_ptr<PredictionBatch> batch) { res = batch; });
  BatchingQueue queue(
      batchQueueCbs,
      BatchingQueue::Config{.batchingMetadata = {{"float_features", "dense"}}},
      /* worldSize */ 1);

  queue.add(
      createRequest(2, 2),
      folly::Promise<std::unique_ptr<PredictionResponse>>());
  queue.add(
      createRequest(2, 4),
      folly::Promise<std::unique_ptr<PredictionResponse>>());

  auto value = std::shared_ptr<PredictionBatch>();
  while (true) {
    value = res.exchange(nullptr);
    if (value) {
      break;
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  ASSERT_EQ(2 * (2 + 4), value->forwardArgs.at("float_features").numel());
}

} // namespace torchrec

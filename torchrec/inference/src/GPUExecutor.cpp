/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/GPUExecutor.h"

#include <chrono>
#include <stdexcept>

#include <c10/cuda/CUDAGuard.h>
#include <fmt/format.h>
#include <folly/MPMCQueue.h>
#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>
#include <folly/stop_watch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/csrc/deploy/deploy.h> // @manual

#include "torchrec/inference/BatchingQueue.h"

DEFINE_int32(copy_timeout, 500, "");

namespace torchrec {

using PredictionException = std::runtime_error;

namespace {
void init_cuda_runtime() {
  LOG(INFO) << "Beginning CUDA runtime warmup, idx: "
            << at::cuda::current_device();
  folly::stop_watch<std::chrono::seconds> timer;

  // Trick to trigger the CUDA runtime initialization so that we don't have
  // slow requests for the first few seconds after service initialization
  auto device = at::Device(at::kCUDA, at::cuda::current_device());

  std::vector<at::Tensor> ts;
  for (auto j = 0; j < 10; ++j) {
    // allocate some matrices that we preserve so they aren't immediately
    // free'd and reused by caching allocator. This should push some persisted
    // blocks into caching allocator, and with variable sizes so we hit a few
    // of the buckets/paths.
    auto size = static_cast<int64_t>(1) << (std::min(j, 24));
    ts.push_back(at::ones({size}, at::TensorOptions().pinned_memory(true)));
    ts.push_back(at::ones({size}, at::TensorOptions().device(device)));

    // allocate some matrices and do a matmul to initialize CuBLAS, etc.
    auto a = at::ones({1024, 1024}, at::TensorOptions().device(device));
    auto b = at::ones({1024, 1024}, at::TensorOptions().device(device));
    ts.push_back(at::matmul(a, b));
  }
  AT_CUDA_CHECK(cudaDeviceSynchronize());
  LOG(INFO) << "CUDA runtime warmup finished in " << timer.elapsed().count()
            << " seconds.";
}
} // namespace

GPUExecutor::GPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    torch::deploy::ReplicatedObj model,
    int rank,
    int worldSize)
    : manager_(manager),
      model_(std::move(model)),
      rank_(rank),
      worldSize_(worldSize),
      batches_(10'000) {
  at::cuda::CUDAGuard guard(rank_);
  init_cuda_runtime();

  int num_threads_per_gpu = manager_->allInstances().size() / worldSize_;
  for (int i = 0; i < num_threads_per_gpu; ++i) {
    LOG(INFO) << "Starting Thread " << i << " for Model Shard Rank " << rank_
              << ", as Global thread: " << rank * num_threads_per_gpu + i;
    processThreads_.emplace_back([this, rank, num_threads_per_gpu, i] {
      process(rank * num_threads_per_gpu + i);
    });
  }
}

GPUExecutor::~GPUExecutor() {
  for (int i = 0; i < processThreads_.size(); ++i) {
    batches_.blockingWrite(nullptr);
  }
  for (auto& thread : processThreads_) {
    thread.join();
  }

  std::shared_ptr<PredictionBatch> batch;
  while (batches_.readIfNotEmpty(batch)) {
    for (auto& context : batch->contexts) {
      PredictionException ex("Server shutdown");
      context.promise.setException(std::move(ex));
    }
  }
}

void GPUExecutor::callback(std::shared_ptr<PredictionBatch> batch) {
  batches_.blockingWrite(std::move(batch));
}

void GPUExecutor::process(int idx) {
  int num_threads_per_gpu = manager_->allInstances().size() / worldSize_;
  folly::setThreadName(
      fmt::format("GPU-{}: Thread-{}", rank_, idx % num_threads_per_gpu));

  // set device guard again to the correct device again because set_device is
  // thread-local
  at::cuda::CUDAGuard deviceGuard(rank_);
  auto stream = at::cuda::getStreamFromPool(true, rank_);
  at::cuda::CUDAStreamGuard streamGuard(stream);

  while (true) {
    std::shared_ptr<PredictionBatch> batch;
    batches_.blockingRead(batch);
    if (batch == nullptr) {
      // shutdown
      break;
    }

    if (std::chrono::steady_clock::now() - batch->enqueueTime >
        std::chrono::milliseconds(FLAGS_copy_timeout)) {
      for (auto& context : batch->contexts) {
        PredictionException ex("GPU Copy timeout");
        context.promise.setException(std::move(ex));
      }
      continue;
    }

    if (batch->batch_size == 0) {
      continue;
    }

    // Free session to avoid accumulating too many PyObjects.
    auto model = model_.acquireSession(&manager_->allInstances().at(idx));
    at::IValue predictions;

    try {
      RECORD_USER_SCOPE("Forward");
      predictions = model.self.attr("__call__")({std::move(batch->forwardArgs)})
                        .toIValue();
    } catch (const std::exception& ex) {
      LOG_EVERY_N(ERROR, 100) << "Exception during predict, msg: " << ex.what();
    }

    auto remainingBatchSize = batch->batch_size;
    for (auto& context : batch->contexts) {
      if (context.isTimedOut) {
        PredictionException ex("Batching queue timeout");
        context.promise.setException(std::move(ex));
      } else if (predictions.isNone()) {
        PredictionException ex("Predict exception");
        context.promise.setException(std::move(ex));
      } else {
        CHECK(predictions.isGenericDict());
        CHECK_GE(remainingBatchSize, context.batchSize);

        auto response = std::make_unique<PredictionResponse>();
        for (const auto& item : predictions.toGenericDict()) {
          auto tensor = item.value().toTensor();
          response->predictions.emplace(
              item.key().toStringRef(),
              folly::IOBuf(
                  folly::IOBuf::COPY_BUFFER,
                  (float*)tensor.data_ptr() +
                      (batch->batch_size - remainingBatchSize),
                  context.batchSize * sizeof(float)));
        }
        context.promise.setValue(std::move(response));
      }
      remainingBatchSize -= context.batchSize;
    }
    CHECK_EQ(remainingBatchSize, 0);
  }
}

} // namespace torchrec

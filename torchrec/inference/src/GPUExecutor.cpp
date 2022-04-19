/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/GPUExecutor.h"

#include <chrono>
#include <memory>
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

#include "ATen/cuda/CUDAEvent.h"
#include "caffe2/torch/csrc/autograd/profiler_legacy.h"
#include "torchrec/inference/BatchingQueue.h"
#include "torchrec/inference/Exception.h"
#include "torchrec/inference/Types.h"

DEFINE_int32(copy_timeout, 500, "");

DEFINE_bool(
    emit_nsys_nvtx,
    false,
    "emit NVTX markers/ranges to be visualized in NSight Systems");

namespace torchrec {

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

// Enable NVTX tracing for the caller thread if the flag is set.
void enable_nvtx_tracing() {
  thread_local static bool emit = false;
  if (FLAGS_emit_nsys_nvtx && emit == false) {
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::NVTX, false, false));
    emit = true;
  }
}
} // namespace

GPUExecutor::GPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    torch::deploy::ReplicatedObj model,
    int rank,
    int worldSize,
    std::shared_ptr<torchrec::ResultSplitFunc> func,
    std::chrono::milliseconds queueTimeout)
    : manager_(manager),
      model_(std::move(model)),
      rank_(rank),
      worldSize_(worldSize),
      batches_(10'000),
      resultSplitFunc_(func),
      queueTimeout_(queueTimeout) {
  at::cuda::CUDAGuard guard(rank_);
  init_cuda_runtime();

  int num_threads_per_gpu = manager_->allInstances().size() / worldSize_;
  for (int i = 0; i < num_threads_per_gpu; ++i) {
    LOG(INFO) << "Starting Thread " << i << " for Model Shard Rank " << rank_
              << ", as Global thread: " << rank * num_threads_per_gpu + i;
    processThreads_.emplace_back([this, rank, num_threads_per_gpu, i] {
      if (FLAGS_emit_nsys_nvtx) {
        enable_nvtx_tracing();
      }
      process(rank * num_threads_per_gpu + i);
    });
  }

  completionExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(2 * num_threads_per_gpu);
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
      handleException(context.promise, "Server shutdown");
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

  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < worldSize_; ++i) {
    streams.push_back(
        at::cuda::getStreamFromPool(/* isHighPriority */ true, i));
  }
  at::cuda::CUDAMultiStreamGuard streamGuard(streams);
  at::cuda::CUDAGuard deviceGuard(rank_);
  auto d2hStream =
      at::cuda::getStreamFromPool(/* isHighPriority */ true, rank_);

  while (true) {
    std::shared_ptr<PredictionBatch> batch;
    batches_.blockingRead(batch);
    if (batch == nullptr) {
      // shutdown
      break;
    }

    if (batch->batchSize == 0) {
      continue;
    }

    if (std::chrono::steady_clock::now() - batch->enqueueTime >=
        queueTimeout_) {
      for (auto& context : batch->contexts) {
        handleException(context.promise, "GPUExecutor queue timeout");
      }
      continue;
    }

    // Free session to avoid accumulating too many PyObjects.

    // AttributeError: 'ShardedTensor' object has no attribute
    // '_sharded_tensor_id' AttributeError: module
    // 'torch.distributed._shard.sharded_tensor.api' has no attribute
    // 'ShardedTensor.ProcessGroupState'
    auto model = model_.acquireSession(&manager_->allInstances().at(idx));
    at::IValue predictions;

    try {
      RECORD_USER_SCOPE("Forward");
      // Block current stream until H2D finishes.
      batch->event->block(streams[rank_]);
      predictions = model.self.attr("__call__")({std::move(batch->forwardArgs)})
                        .toIValue();
    } catch (const std::exception& ex) {
      LOG_EVERY_N(ERROR, 100) << "Exception during predict, msg: " << ex.what();
    }

    completionExecutor_->add(
        // Can not bind the method directly because of the unique_ptr of item.
        [batch = std::move(batch),
         predictions = std::move(predictions),
         resultSplitFunc = resultSplitFunc_,
         rank = rank_,
         d2hStream = d2hStream]() mutable {
          RECORD_USER_SCOPE("CompletionStage");

          at::cuda::CUDAStreamGuard streamGuard(d2hStream);
          at::cuda::CUDAGuard deviceGuard(rank);

          if (!predictions.isNone()) {
            predictions = resultSplitFunc->moveToHost(predictions);
            batch->event->record();
            // Wait for D2H to finish.
            batch->event->synchronize();
          }

          size_t offset = 0;
          for (auto& context : batch->contexts) {
            if (predictions.isNone()) {
              handleException(context.promise, "Predict exception");
            } else {
              CHECK_LT(offset, batch->batchSize);

              auto response = std::make_unique<PredictionResponse>();
              response->predictions = resultSplitFunc->splitResult(
                  predictions, offset, context.batchSize, batch->batchSize);
              context.promise.setValue(std::move(response));
            }
            offset += context.batchSize;
          }
          CHECK_EQ(offset, batch->batchSize);
        });
  }
}

} // namespace torchrec

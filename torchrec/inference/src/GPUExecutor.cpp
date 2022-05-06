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

DEFINE_bool(gpu_executor_use_high_pri_stream_main_device, false, "");

DEFINE_bool(gpu_executor_use_high_pri_stream_peer_device, false, "");

DEFINE_bool(gpu_executor_use_high_pri_stream_d2h, false, "");

namespace torchrec {

namespace {
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
    std::chrono::milliseconds queueTimeout,
    std::function<void()> warmupFn)
    : manager_(manager),
      model_(std::move(model)),
      rank_(rank),
      worldSize_(worldSize),
      batches_(10'000),
      resultSplitFunc_(func),
      queueTimeout_(queueTimeout),
      warmupFn_(std::move(warmupFn)) {
  at::cuda::CUDAGuard guard(rank_);

  int num_threads_per_gpu = manager_->allInstances().size() / worldSize_;
  rejectionExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(2 * num_threads_per_gpu);
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
    rejectionExecutor_->add([batch = std::move(batch)]() {
      handleBatchException(batch->contexts, "Server shutdown");
    });
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
    streams.push_back(at::cuda::getStreamFromPool(
        /* isHighPriority */ i == rank_
            ? FLAGS_gpu_executor_use_high_pri_stream_main_device
            : FLAGS_gpu_executor_use_high_pri_stream_peer_device,
        i));
  }
  at::cuda::CUDAMultiStreamGuard streamGuard(streams);
  at::cuda::CUDAGuard deviceGuard(rank_);
  auto d2hStream = at::cuda::getStreamFromPool(
      /* isHighPriority */ FLAGS_gpu_executor_use_high_pri_stream_d2h, rank_);

  if (warmupFn_) {
    warmupFn_();
  }

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
      rejectionExecutor_->add([batch = std::move(batch)]() {
        handleBatchException(batch->contexts, "GPUExecutor queue timeout");
      });

      continue;
    }

    // Free session to avoid accumulating too many PyObjects.
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

    batch->event->record();

    completionExecutor_->add(
        // Can not bind the method directly because of the unique_ptr of item.
        [this,
         batch = std::move(batch),
         predictions = std::move(predictions),
         resultSplitFunc = resultSplitFunc_,
         rank = rank_,
         d2hStream = d2hStream]() mutable {
          RECORD_USER_SCOPE("CompletionStage");

          batch->event->block(d2hStream);

          at::cuda::CUDAStreamGuard streamGuard(d2hStream);
          at::cuda::CUDAGuard deviceGuard(rank);

          if (!predictions.isNone()) {
            predictions = resultSplitFunc->moveToHost(predictions);
            batch->event->record();
            // Wait for D2H to finish.
            batch->event->synchronize();
          }

          if (predictions.isNone()) {
            rejectionExecutor_->add([batch = std::move(batch)]() {
              handleBatchException(
                  batch->contexts, "GPUExecutor prediction exception");
            });
          } else {
            size_t offset = 0;
            for (auto& context : batch->contexts) {
              CHECK(!predictions.isNone());
              CHECK_LT(offset, batch->batchSize);
              auto response = std::make_unique<PredictionResponse>();
              response->predictions = resultSplitFunc->splitResult(
                  predictions, offset, context.batchSize, batch->batchSize);
              context.promise.setValue(std::move(response));
              offset += context.batchSize;
            }
            CHECK_EQ(offset, batch->batchSize);
          }
        });
  }
}

} // namespace torchrec

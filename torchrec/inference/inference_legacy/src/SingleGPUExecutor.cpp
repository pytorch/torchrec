/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/SingleGPUExecutor.h"
#include <c10/cuda/CUDAGuard.h>
#include <folly/String.h>
#include "torchrec/inference/Assert.h"
#include "torchrec/inference/Observer.h"

namespace torchrec {

SingleGPUExecutor::SingleGPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    ExecInfos execInfos,
    size_t numGpu,
    std::shared_ptr<ISingleGPUExecutorObserver> observer,
    c10::Device resultDevice,
    size_t numProcessThreads,
    bool useHighPriCudaStream)
    : manager_(manager),
      execInfos_(std::move(execInfos)),
      numGpu_(numGpu),
      numProcessThreads_(numProcessThreads),
      useHighPriCudaStream_(useHighPriCudaStream),
      resultDevice_(resultDevice),
      observer_(observer),
      requests_(kQUEUE_CAPACITY),
      processExecutor_(
          std::make_unique<folly::CPUThreadPoolExecutor>(numProcessThreads)),
      completionExecutor_(
          std::make_unique<folly::CPUThreadPoolExecutor>(execInfos_.size())),
      roundRobinExecInfoNextIdx_(0u) {
  for (size_t i = 0; i < numProcessThreads_; ++i) {
    processExecutor_->add([&]() { process(); });
  }
  for (const auto& exec_info : execInfos_) {
    TORCHREC_CHECK(exec_info.interpIdx < manager_->allInstances().size());
  }
  TORCHREC_CHECK(observer_);
}

SingleGPUExecutor::~SingleGPUExecutor() {
  for (size_t i = 0; i < numProcessThreads_; ++i) {
    requests_.blockingWrite(nullptr);
  }
  processExecutor_->join();
  completionExecutor_->join();
}

void SingleGPUExecutor::schedule(std::shared_ptr<PredictionBatch> request) {
  requests_.blockingWrite(std::move(request));
}

namespace {

c10::IValue toDevice(const c10::IValue& iv, c10::Device device) {
  if (iv.isTensor()) {
    return c10::IValue(iv.toTensor().to(device));
  } else if (iv.isList()) {
    const auto list = iv.toList();
    auto copied_list = c10::impl::GenericList(list.elementType());
    for (const auto& entry : list) {
      copied_list.push_back(toDevice(entry, device));
    }
    return c10::IValue(copied_list);
  } else if (iv.isGenericDict()) {
    const auto dict = iv.toGenericDict();
    auto copied_dict = c10::impl::GenericDict(dict.keyType(), dict.valueType());
    for (const auto& entry : dict) {
      copied_dict.insert(entry.key(), toDevice(entry.value(), device));
    }
    return c10::IValue(copied_dict);
  }

  return iv;
}

std::vector<c10::IValue> toDevice(
    std::vector<c10::IValue> args,
    c10::Device device) {
  for (auto& arg : args) {
    arg = toDevice(arg, device);
  }
  return args;
}

const std::pair<std::string, std::string> splitQualname(
    const std::string& qualname) {
  auto idx = qualname.rfind('.');
  auto submodulePath = idx == std::string::npos ? "" : qualname.substr(0, idx);
  auto methodName = idx == std::string::npos
      ? qualname
      : qualname.substr(idx + 1, qualname.size());

  return {submodulePath, methodName};
}

} // namespace

void SingleGPUExecutor::process() {
  c10::InferenceMode inferenceModeGuard;
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < numGpu_; ++i) {
    streams.push_back(
        at::cuda::getStreamFromPool(useHighPriCudaStream_, i /* device */));
  }
  at::cuda::CUDAMultiStreamGuard streamGuard(streams);

  while (true) {
    std::shared_ptr<PredictionBatch> request;
    requests_.blockingRead(request);

    if (!request) {
      break;
    }

    auto timeInQueue = getTimeElapsedMS(request->enqueueTime);
    observer_->recordQueueLatency(timeInQueue.count());

    const size_t execInfoIdx = roundRobinExecInfoNextIdx_;
    roundRobinExecInfoNextIdx_ =
        (roundRobinExecInfoNextIdx_ + 1) % execInfos_.size();

    const auto& execInfo = execInfos_[execInfoIdx];

    const auto device = c10::Device(c10::kCUDA, execInfo.gpuIdx);
    at::cuda::CUDAGuard device_guard(device);

    c10::IValue result;
    try {
      observer_->addRequestsCount(1.0f);
      const auto requestProcessStart = std::chrono::steady_clock::now();

      const auto [submodulePath, methodName] =
          splitQualname(request->methodName);

      auto I = execInfo.model.acquireSession(
          &manager_->allInstances().at(execInfo.interpIdx));

      std::vector<std::string> names;
      folly::split(".", submodulePath, names);
      auto m = I.fromMovable(execInfo.model);
      for (const auto& name : names) {
        if (name == "") {
          break;
        }
        m = m.attr(name.c_str());
      }

      request->event = Event(
          new at::cuda::CUDAEvent(
              cudaEventBlockingSync | cudaEventDisableTiming),
          [](at::cuda::CUDAEvent* event) { delete event; });

      auto out = I.self.attr(methodName.c_str())
                     .callKwargs(toDevice(request->args, device), {})
                     .toIValue();
      result = toDevice(out, resultDevice_);

      observer_->recordRequestProcessingLatency(
          getTimeElapsedMS(requestProcessStart).count());
    } catch (const std::exception& ex) {
      observer_->addRequestProcessingExceptionCount(1.0f);
      LOG_EVERY_N(ERROR, 100)
          << "Exception during request process, msg: " << ex.what();
    }
    request->event->record();

    completionExecutor_->add(
        [result = std::move(result), request = std::move(request)]() {
          request->event->synchronize();
          for (auto& context : request->contexts) {
            auto response = std::make_unique<PredictionResponse>();
            response->batchSize = context.batchSize;
            response->predictions = result;
            context.promise.setValue(std::move(response));
          }
        });
  }
}

} // namespace torchrec

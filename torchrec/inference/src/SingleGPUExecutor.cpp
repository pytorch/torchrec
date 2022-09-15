/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/SingleGPUExecutor.h"
#include <c10/cuda/CUDAGuard.h>
#include "torchrec/inference/Assert.h"

namespace torchrec {

SingleGPUExecutor::SingleGPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    ExecInfos execInfos,
    size_t numGpu,
    c10::Device resultDevice)
    : manager_(manager),
      execInfos_(std::move(execInfos)),
      numGpu_(numGpu),
      resultDevice_(resultDevice),
      requests_(kQUEUE_CAPACITY),
      completionExecutor_(
          std::make_unique<folly::CPUThreadPoolExecutor>(execInfos_.size())),
      roundRobinExecInfoNextIdx_(0u),
      processThread_([&]() { process(); }) {
  for (const auto& exec_info : execInfos_) {
    TORCHREC_CHECK(exec_info.interpIdx < manager_->allInstances().size());
  }
}

SingleGPUExecutor::~SingleGPUExecutor() {
  requests_.blockingWrite(nullptr);
  processThread_.join();
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

} // namespace

void SingleGPUExecutor::process() {
  c10::InferenceMode inferenceModeGuard;
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < numGpu_; ++i) {
    streams.push_back(at::cuda::getStreamFromPool(i));
  }
  at::cuda::CUDAMultiStreamGuard streamGuard(streams);

  while (true) {
    std::shared_ptr<PredictionBatch> request;
    requests_.blockingRead(request);

    if (!request) {
      break;
    }

    const size_t exec_info_idx = roundRobinExecInfoNextIdx_;
    roundRobinExecInfoNextIdx_ =
        (roundRobinExecInfoNextIdx_ + 1) % execInfos_.size();

    const auto& execInfo = execInfos_[exec_info_idx];

    const auto device = c10::Device(c10::kCUDA, execInfo.gpuIdx);
    at::cuda::CUDAGuard device_guard(device);

    auto& model = execInfo.model;
    auto I =
        model.acquireSession(&manager_->allInstances().at(execInfo.interpIdx));

    request->event = Event(
        new at::cuda::CUDAEvent(cudaEventBlockingSync | cudaEventDisableTiming),
        [](at::cuda::CUDAEvent* event) { delete event; });

    // TODO: Support methodName as "model.submodule.method"
    auto out = I.self.attr(request->methodName.c_str())
                   .callKwargs(toDevice(request->args, device), {})
                   .toIValue();
    auto result = toDevice(out, resultDevice_);
    request->event->record();

    completionExecutor_->add(
        [result = std::move(result), request = std::move(request)]() {
          request->event->synchronize();
          for (auto& context : request->contexts) {
            auto response = std::make_unique<PredictionResponse>();
            response->predictions = result;
            context.promise.setValue(std::move(response));
          }
        });
  }
}

} // namespace torchrec

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/SingleGPUExecutor.h"
#include <c10/cuda/CUDAGuard.h>
#include <cassert>

namespace torchrec {

SingleGPUExecutor::SingleGPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    torch::deploy::ReplicatedObj model,
    c10::Device device /* cuda device */,
    const std::vector<size_t>& interpreter_idxs,
    c10::Device result_device /* result device */)
    : manager_(manager),
      model_(std::move(model)),
      device_(device),
      resultDevice_(result_device),
      requests_(10000) {
  for (const auto& interp_idx : interpreter_idxs) {
    assert(interp_idx < manager->allInstances().size());
    processThreads_.emplace_back([this, interp_idx]() { process(interp_idx); });
  }

  completionExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(interpreter_idxs.size());
}

SingleGPUExecutor::~SingleGPUExecutor() {
  for (const auto _ : c10::irange(processThreads_.size())) {
    requests_.blockingWrite(nullptr);
  }
  for (auto& thread : processThreads_) {
    thread.join();
  }
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

void SingleGPUExecutor::process(const size_t interpreter_idx) {
  auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(stream);
  at::cuda::CUDAGuard device_guard(device_);

  while (true) {
    std::shared_ptr<PredictionBatch> request;
    requests_.blockingRead(request);

    if (!request) {
      break;
    }

    auto* onThisInterpreter = &manager_->allInstances().at(interpreter_idx);
    auto I = model_.acquireSession(onThisInterpreter);

    request->event = Event(
        new at::cuda::CUDAEvent(cudaEventBlockingSync | cudaEventDisableTiming),
        [](at::cuda::CUDAEvent* event) { delete event; });

    // TODO: Support methodName as "model.submodle.method"
    auto out = I.self.attr(request->methodName.c_str())
                   .callKwargs(toDevice(request->args, device_), {})
                   .toIValue();
    auto result = toDevice(out, resultDevice_);
    request->event->record(stream);

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

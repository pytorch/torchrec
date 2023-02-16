/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAEvent.h>
#include <boost/noncopyable.hpp>
#include <folly/ExceptionWrapper.h>
#include <folly/container/F14Set.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>

#include "torchrec/inference/ResourceManager.h"

namespace torchrec {

struct SparseFeatures {
  uint32_t num_features;
  // int32: T x B
  folly::IOBuf lengths;
  // T x B x L (jagged)
  folly::IOBuf values;
  // float16
  folly::IOBuf weights;
};

struct FloatFeatures {
  uint32_t num_features;
  // shape: {B}
  folly::IOBuf values;
};

// TODO: Change the input format to torch::IValue.
// Currently only dense batching function support IValue.
using Feature = std::variant<SparseFeatures, FloatFeatures, c10::IValue>;

struct PredictionRequest {
  uint32_t batch_size;
  std::unordered_map<std::string, Feature> features;
};

struct PredictionResponse {
  uint32_t batchSize;
  c10::IValue predictions;
  // If set, the result is an exception.
  std::optional<folly::exception_wrapper> exception;
};

struct RequestContext {
  uint32_t batchSize;
  folly::Promise<std::unique_ptr<PredictionResponse>> promise;
  // folly request context for request tracking in crochet
  std::shared_ptr<folly::RequestContext> follyRequestContext;
};

using PredictionException = std::runtime_error;

using Event = std::
    unique_ptr<at::cuda::CUDAEvent, std::function<void(at::cuda::CUDAEvent*)>>;

struct BatchingMetadata {
  std::string type;
  std::string device;
  folly::F14FastSet<std::string> pinned;
};

// noncopyable because we only want to move PredictionBatch around
// as it holds a reference to ResourceManagerGuard. We wouldn't want
// to inadvertently increase the reference count to ResourceManagerGuard
// with copies of this struct.
struct PredictionBatch : public boost::noncopyable {
  std::string methodName;
  std::vector<c10::IValue> args;

  size_t batchSize;

  c10::impl::GenericDict forwardArgs;

  std::vector<RequestContext> contexts;

  std::unique_ptr<ResourceManagerGuard> resourceManagerGuard = nullptr;

  std::chrono::time_point<std::chrono::steady_clock> enqueueTime =
      std::chrono::steady_clock::now();

  Event event;

  // Need a constructor to use make_shared/unique with
  // noncopyable struct and not trigger copy-constructor.
  PredictionBatch(
      size_t bs,
      c10::impl::GenericDict fa,
      std::vector<RequestContext> ctxs,
      std::unique_ptr<ResourceManagerGuard> rmg = nullptr)
      : batchSize(bs),
        forwardArgs(std::move(fa)),
        contexts(std::move(ctxs)),
        resourceManagerGuard(std::move(rmg)) {}

  PredictionBatch(
      std::string methodNameArg,
      std::vector<c10::IValue> argsArg,
      folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise)
      : methodName(std::move(methodNameArg)),
        args(std::move(argsArg)),
        forwardArgs(
            c10::impl::GenericDict(at::StringType::get(), at::AnyType::get())) {
    contexts.push_back(RequestContext{1u, std::move(promise)});
  }

  size_t sizeOfIValue(const c10::IValue& val) const {
    size_t size = 0;
    if (val.isTensor()) {
      size += val.toTensor().storage().nbytes();
    } else if (val.isList()) {
      for (const auto& v : val.toListRef()) {
        size += sizeOfIValue(v);
      }
    }
    return size;
  }

  inline size_t size() const {
    size_t size = 0;
    for (const auto& iter : forwardArgs) {
      size += sizeOfIValue(iter.value());
    }
    return size;
  }
};

} // namespace torchrec

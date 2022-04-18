/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/ResultSplit.h"

#include <c10/core/ScalarType.h>
#include <folly/Range.h>
#include <folly/container/Enumerate.h>
#include <folly/io/Cursor.h>

#include "ATen/Functions.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

C10_DEFINE_REGISTRY(TorchRecResultSplitFuncRegistry, ResultSplitFunc);

c10::IValue splitDictOfTensor(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(c10::StringType::get(), c10::TensorType::get());
  pred.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTensor());
    const auto& tensor = value.toTensor();
    TORCH_CHECK(tensor.dim() == 1);
    TORCH_CHECK(tensor.size(0) % nTotalLength == 0);
    const auto elemSize = tensor.size(0) / nTotalLength;
    pred.insert(
        key,
        tensor.slice(
            0, nOffset * elemSize, nOffset * elemSize + nLength * elemSize));
  }
  return pred;
}

c10::IValue splitDictOfTensors(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(
      c10::StringType::get(), c10::TupleType::create({c10::TensorType::get()}));
  pred.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTuple());
    const auto tuple = value.toTuple();
    std::vector<c10::IValue> values;
    values.reserve(tuple->size());
    for (int i = 0; i < tuple->size(); ++i) {
      const auto& tensor = tuple->elements()[i].toTensor();
      TORCH_CHECK(tensor.dim() == 1);
      TORCH_CHECK(tensor.size(0) % nTotalLength == 0);
      const auto elemSize = tensor.size(0) / nTotalLength;
      values.push_back(tensor.slice(
          0, nOffset * elemSize, nOffset * elemSize + nLength * elemSize));
    }
    pred.insert(key, c10::ivalue::Tuple::create(std::move(values)));
  }
  return pred;
}

namespace {

class DictOfTensorResultSplitFunc : public ResultSplitFunc {
 public:
  c10::IValue splitResult(
      c10::IValue result,
      size_t nOffset,
      size_t nLength,
      size_t nTotalLength) override {
    return splitDictOfTensor(result, nOffset, nLength, nTotalLength);
  }

  c10::IValue moveToHost(c10::IValue result) {
    const auto& dict = result.toGenericDict();
    c10::impl::GenericDict moved(
        c10::StringType::get(), c10::TensorType::get());
    moved.reserve(dict.size());

    for (auto& entry : dict) {
      const auto& key = entry.key();
      const auto& value = entry.value();
      TORCH_CHECK(value.isTensor());
      const auto& tensor = value.toTensor();
      moved.insert(
          key, tensor.to(at::Device(at::kCPU), /* non_blocking */ true));
    }
    return moved;
  }
};

class DictOfTensorsResultSplitFunc : public ResultSplitFunc {
 public:
  c10::IValue splitResult(
      c10::IValue result,
      size_t nOffset,
      size_t nLength,
      size_t nTotalLength) override {
    return splitDictOfTensors(result, nOffset, nLength, nTotalLength);
  }

  c10::IValue moveToHost(c10::IValue result) {
    const auto& dict = result.toGenericDict();
    c10::impl::GenericDict moved(
        c10::StringType::get(),
        c10::TupleType::create({c10::TensorType::get()}));
    moved.reserve(dict.size());

    for (auto& entry : dict) {
      const auto& key = entry.key();
      const auto& value = entry.value();
      TORCH_CHECK(value.isTuple());
      const auto tuple = value.toTuple();
      std::vector<c10::IValue> values;
      values.reserve(tuple->size());
      for (int i = 0; i < tuple->size(); ++i) {
        auto& elem = tuple->elements()[i];
        TORCH_CHECK(elem.isTensor());
        const auto& tensor = elem.toTensor();
        values.push_back(
            tensor.to(at::Device(at::kCPU), /* non_blocking */ true));
      }
      moved.insert(key, c10::ivalue::Tuple::create(std::move(values)));
    }
    return moved;
  }
};

REGISTER_TORCHREC_RESULTSPLIT_FUNC(dict_of_tensor, DictOfTensorResultSplitFunc);

REGISTER_TORCHREC_RESULTSPLIT_FUNC(
    dict_of_tensors,
    DictOfTensorsResultSplitFunc);

} // namespace
} // namespace torchrec

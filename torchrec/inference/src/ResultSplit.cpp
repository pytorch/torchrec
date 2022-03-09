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
    const size_t& offset,
    const size_t& length) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(c10::StringType::get(), c10::TensorType::get());
  pred.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTensor());
    const auto& tensor = value.toTensor();
    pred.insert(key, tensor.slice(0, offset, offset + length));
  }
  return pred;
}

namespace {

class DictOfTensorResultSplitFunc : public ResultSplitFunc {
 public:
  c10::IValue splitResult(
      c10::IValue result,
      const size_t& offset,
      const size_t& length) override {
    return splitDictOfTensor(result, offset, length);
  }
};

REGISTER_TORCHREC_RESULTSPLIT_FUNC(dict_of_tensor, DictOfTensorResultSplitFunc);

} // namespace
} // namespace torchrec

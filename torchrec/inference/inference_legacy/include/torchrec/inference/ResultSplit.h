/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Registry.h>

namespace torchrec {

class ResultSplitFunc {
 public:
  virtual ~ResultSplitFunc() = default;

  virtual std::string name() = 0;

  virtual c10::IValue splitResult(
      c10::IValue /* result */,
      size_t /* nOffset */,
      size_t /* nLength */,
      size_t /* nTotalLength */) = 0;

  virtual c10::IValue moveToHost(c10::IValue /* result */) = 0;
};

/**
 * TorchRecResultSplitFuncRegistry is used to register custom result split
 * functions.
 */
C10_DECLARE_REGISTRY(TorchRecResultSplitFuncRegistry, ResultSplitFunc);

#define REGISTER_TORCHREC_RESULTSPLIT_FUNC(name, ...) \
  C10_REGISTER_CLASS(TorchRecResultSplitFuncRegistry, name, __VA_ARGS__);

c10::IValue splitDictOfTensor(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength);

c10::IValue splitDictOfTensors(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength);

c10::IValue
splitDictWithMaskTensor(c10::IValue result, size_t nOffset, size_t nLength);

class DictWithMaskTensorResultSplitFunc : public torchrec::ResultSplitFunc {
 public:
  virtual std::string name() override;

  virtual c10::IValue splitResult(
      c10::IValue result,
      size_t offset,
      size_t length,
      size_t /* nTotalLength */) override;

  c10::IValue moveToHost(c10::IValue result) override;
};

} // namespace torchrec

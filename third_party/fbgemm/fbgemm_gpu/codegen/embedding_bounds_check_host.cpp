/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

using namespace at;

void bounds_check_indices_cuda(
    Tensor rows_per_table,
    Tensor indices,
    Tensor offsets,
    int64_t bounds_check_mode,
    Tensor warning);

TORCH_LIBRARY_FRAGMENT(fb, m) {
  // The (a!) tells PyTorch this is an impure operation and so cannot be CSE'd
  // or DCE'd, etc.
  m.def(
      "bounds_check_indices(Tensor rows_per_table, Tensor(a!) indices, Tensor offsets, int bounds_check_mode, Tensor(a!) warning) -> ()");
  m.impl(
      "bounds_check_indices",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(bounds_check_indices_cuda)));
}

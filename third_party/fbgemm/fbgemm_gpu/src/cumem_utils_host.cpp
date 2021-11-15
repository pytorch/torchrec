/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/library.h>

using namespace at;

// Allocate the ATen Tensor with unified managed memory (UVM)
Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes);

// Check if a tensor is allocated with UVM or host-mapped memory
bool is_uvm_tensor(Tensor t);

// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(Tensor t);

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def("is_uvm_tensor(Tensor t) -> bool", TORCH_FN(is_uvm_tensor));
  m.def("uvm_to_cpu(Tensor t) -> Tensor");
  m.impl(
      "uvm_to_cpu",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(uvm_to_cpu)));
  m.def("new_managed_tensor(Tensor self, int[] sizes) -> Tensor");
  m.impl(
      "new_managed_tensor",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(new_managed_tensor)));
}

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/input_combine.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#ifdef FBGEMM_GPU_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace fbgemm {

std::tuple<at::Tensor, at::Tensor, at::Tensor> tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets) {
  TORCH_CHECK(indices_list.size() > 0);
  TORCH_CHECK(offsets_list.size() == indices_list.size());
  TORCH_CHECK(per_sample_weights.size() == indices_list.size());
  TORCH_CHECK(
      static_cast<uint64_t>(include_last_offsets.numel()) ==
      indices_list.size());
  auto include_last_offsets_acc = include_last_offsets.accessor<bool, 1>();
  int64_t total_indices = 0;
  int64_t total_offsets = 1;
  bool need_weights = false;
  bool pin_memory = false;
#ifdef FBGEMM_GPU_WITH_CUDA
  if (at::globalContext().hasCUDA() && at::cuda::is_available()) {
    pin_memory = true;
  }
#endif

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        offsets_list[i].dtype() == c10::kInt ||
        offsets_list[i].dtype() == c10::kLong);
    TORCH_CHECK(indices_list[i].ndimension() == 1);
    TORCH_CHECK(offsets_list[i].ndimension() == 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(offsets_list[i].is_contiguous());
    total_indices += indices_list[i].numel();
    auto num_offset =
        offsets_list[i].numel() - (include_last_offsets_acc[i] ? 1 : 0);
    total_offsets += num_offset == 0 ? 1 : num_offset;

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK(per_sample_weights[i].ndimension() == 1);
      TORCH_CHECK(per_sample_weights[i].numel() == indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
      need_weights = true;
    }
  }

  auto combined_indices = at::empty(
      {total_indices},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(indices_list[0].device())
          .pinned_memory(pin_memory));

  auto combined_indices_acc = combined_indices.accessor<int32_t, 1>();
  size_t idx = 0;

  for (size_t i = 0; i < indices_list.size(); i++) {
    AT_DISPATCH_INDEX_TYPES(
        indices_list[i].scalar_type(),
        "tbe_input_indices_",
        [&]() {
          auto indices_acc = indices_list[i].accessor<index_t, 1>();
          for (auto j = 0; j < indices_list[i].numel(); j++) {
            combined_indices_acc[idx++] = static_cast<int32_t>(indices_acc[j]);
          }
        });
  }


  auto combined_offsets = at::empty(
      {total_offsets},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(offsets_list[0].device())
          .pinned_memory(pin_memory));

  auto combined_offsets_acc = combined_offsets.accessor<int32_t, 1>();
  int32_t offset = 0;
  size_t offsets_acc_idx = 0;
  combined_offsets_acc[offsets_acc_idx++] = 0;

  for (size_t i = 0; i < offsets_list.size(); i++) {
    AT_DISPATCH_INDEX_TYPES(
        offsets_list[i].scalar_type(), "tbe_input_offsets_", [&]() {
          auto offsets_acc = offsets_list[i].accessor<index_t, 1>();
          for (int64_t j = 1,
                       size = offsets_list[i].numel() -
                   (include_last_offsets_acc[i] ? 1 : 0);
               j < size;
               j++) {
            combined_offsets_acc[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_acc[j]);
          }

          offset += static_cast<int32_t>(indices_list[i].numel());
          combined_offsets_acc[offsets_acc_idx++] = offset;
        });
  }

  if (need_weights) {
    auto combined_weights = at::ones(
        {total_indices},
        at::TensorOptions()
            .dtype(c10::kFloat)
            .device(per_sample_weights[0].device())
            .pinned_memory(pin_memory));
    auto* combined_weights_ptr = combined_weights.data_ptr<float>();

    for (size_t i = 0; i < per_sample_weights.size(); i++) {
      auto element_size = per_sample_weights[i].numel();
      if (element_size != 0) {
        memcpy(
            combined_weights_ptr,
            per_sample_weights[i].data_ptr<float>(),
            element_size * sizeof(float));
      }
      combined_weights_ptr += indices_list[i].numel();
    }
    return {combined_indices, combined_offsets, combined_weights};
  }

  return {combined_indices, combined_offsets, at::empty({0})};
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "tbe_input_combine(Tensor[] indices_list, Tensor[] offsets_list, Tensor[] per_sample_weights, Tensor include_last_offsets) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "tbe_input_combine",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(tbe_input_combine_cpu)));
}

} // namespace fbgemm

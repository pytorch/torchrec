// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <ATen/ATen.h>

namespace fbgemm {

std::tuple<at::Tensor, at::Tensor, at::Tensor> tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets);

} // namespace fbgemm

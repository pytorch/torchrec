// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <ATen/ATen.h>

namespace fbgemm {
at::Tensor permute_pooled_embs_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);

at::Tensor permute_pooled_embs_gpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);
} // namespace fbgemm

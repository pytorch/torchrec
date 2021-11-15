/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <THC/THC.h>
#include <c10/util/irange.h>
#include <fbgemm_gpu/permute_pooled_embedding_ops.h>
#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace fbgemm {

at::Tensor permute_pooled_embs_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  TORCH_CHECK(
      offset_dim_list.scalar_type() == at::ScalarType::Long,
      "offset_dim_list needs to have long/int64 type")
  TORCH_CHECK(
      permute_list.scalar_type() == at::ScalarType::Long,
      "permute_list needs to have long/int64 type")
  auto permute = permute_list.data_ptr<int64_t>();
  const auto n = permute_list.numel();
  std::vector<int64_t> dims;
  dims.reserve(n - 1);
  for (const auto i : c10::irange(1, n)) {
    dims.push_back(offset_dim_list[i].item<int64_t>());
  }
  auto ts = pooled_embs.tensor_split(dims, 1);
  std::vector<at::Tensor> permuted_ts;
  permuted_ts.reserve(n);
  for (const auto i : c10::irange(n)) {
    permuted_ts.push_back(ts[permute[i]]);
  }
  return at::cat(permuted_ts, 1);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

template <torch::autograd::Variable (*permute_pooled_embs_op)(
    const at::Tensor&, // [B_local][Sum_T_global(D)]
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&)>
class PermutePooledEmbsFunction
    : public torch::autograd::Function<
          PermutePooledEmbsFunction<permute_pooled_embs_op>> {
 public:
  static Variable forward(
      AutogradContext* ctx,
      const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
      const at::Tensor& offset_dim_list,
      const at::Tensor& permute_list,
      const at::Tensor& inv_offset_dim_list,
      const at::Tensor& inv_permute_list) {
    ctx->saved_data["offset_dim_list"] = offset_dim_list;
    ctx->saved_data["permute_list"] = permute_list;
    ctx->saved_data["inv_offset_dim_list"] = inv_offset_dim_list;
    ctx->saved_data["inv_permute_list"] = inv_permute_list;
    TORCH_CHECK(
        offset_dim_list.scalar_type() == at::ScalarType::Long,
        "offset_dim_list needs to have long/int64 type");
    TORCH_CHECK(
        permute_list.scalar_type() == at::ScalarType::Long,
        "permute_list needs to have long/int64 type");
    return permute_pooled_embs_op(
        pooled_embs,
        offset_dim_list,
        permute_list,
        inv_offset_dim_list,
        inv_permute_list);
  }
  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    const auto& offset_dim_list = ctx->saved_data["offset_dim_list"].toTensor();
    const auto& permute_list = ctx->saved_data["permute_list"].toTensor();
    const auto& inv_offset_dim_list =
        ctx->saved_data["inv_offset_dim_list"].toTensor();
    const auto& inv_permute_list =
        ctx->saved_data["inv_permute_list"].toTensor();
    variable_list grad_inputs(5);
    grad_inputs[0] = permute_pooled_embs_op(
        grad_output[0],
        inv_offset_dim_list,
        inv_permute_list,
        offset_dim_list,
        permute_list);
    return grad_inputs;
  }
};

at::Tensor permute_pooled_embs_auto_grad_gpu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction<permute_pooled_embs_gpu>::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}

at::Tensor permute_pooled_embs_auto_grad_cpu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction<permute_pooled_embs_cpu>::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}
} // namespace fbgemm

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  m.impl(
      "permute_pooled_embs",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(fbgemm::permute_pooled_embs_gpu)));
  m.def(
      "permute_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  m.impl(
      "permute_pooled_embs_auto_grad",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(fbgemm::permute_pooled_embs_auto_grad_cpu)));
  m.impl(
      "permute_pooled_embs_auto_grad",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm::permute_pooled_embs_auto_grad_gpu)));
}

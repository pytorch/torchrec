/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace fbgemm {

class LookupFunctionBatchedUnaryEmbeddingOp
    : public torch::autograd::Function<LookupFunctionBatchedUnaryEmbeddingOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& weight,
      const at::Tensor& table_offsets,
      const at::Tensor& offsets,
      const at::Tensor& indices) {
    ctx->save_for_backward({weight, table_offsets, offsets, indices});
    auto output = fbgemm::batched_unary_embeddings_forward_cuda(
        weight, table_offsets, offsets, indices);
    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto weight = *savedItr++;
    auto table_offsets = *savedItr++;
    auto offsets = *savedItr++;
    auto indices = *savedItr++;
    TORCH_CHECK(grad_outputs.size() == 1);
    auto grad_weight = fbgemm::batched_unary_embeddings_backward_cuda(
        grad_outputs[0], weight, table_offsets, offsets, indices);
    return {grad_weight, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor lookup_batched_unary_embedding_function(
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices) {
  return LookupFunctionBatchedUnaryEmbeddingOp::apply(
      weight, table_offsets, offsets, indices)[0];
}

class Jagged2DToDenseGPUOp
    : public torch::autograd::Function<Jagged2DToDenseGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor embeddings,
      at::Tensor offsets,
      int32_t max_sequence_length) {
    int32_t total_L = embeddings.size(0);
    ctx->save_for_backward({offsets});
    ctx->saved_data["total_L"] = total_L;

    return {jagged_2d_to_dense_forward_cuda(
        embeddings, offsets, max_sequence_length)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto offsets = *savedItr++;
    int32_t total_L = ctx->saved_data["total_L"].toInt();

    using torch::autograd::Variable;
    auto grad_padded_embeddings = grad_outputs[0];
    auto grad_embeddings = jagged_2d_to_dense_backward_cuda(
        grad_padded_embeddings, offsets, total_L);
    return {
        grad_embeddings,
        Variable(), // offsets
        Variable() // max_sequence_length
    };
  }
};

at::Tensor jagged_2d_to_dense_gpu(
    at::Tensor embeddings,
    at::Tensor offsets,
    int64_t max_sequence_length) {
  return Jagged2DToDenseGPUOp::apply(
      embeddings, offsets, static_cast<int32_t>(max_sequence_length))[0];
}

} // namespace fbgemm

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("permute_sparse_data", fbgemm::permute_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "block_bucketize_sparse_features",
      fbgemm::block_bucketize_sparse_features_cuda);
  DISPATCH_TO_CUDA(
      "asynchronous_exclusive_cumsum",
      fbgemm::asynchronous_exclusive_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_complete_cumsum", fbgemm::asynchronous_complete_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_inclusive_cumsum",
      fbgemm::asynchronous_inclusive_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_lengths", fbgemm::reorder_batched_ad_lengths_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_indices", fbgemm::reorder_batched_ad_indices_gpu);
  DISPATCH_TO_CUDA("offsets_range", fbgemm::offsets_range_cuda);
  DISPATCH_TO_CUDA(
      "batched_unary_embeddings",
      fbgemm::lookup_batched_unary_embedding_function);
  DISPATCH_TO_CUDA("jagged_2d_to_dense", fbgemm::jagged_2d_to_dense_gpu);
  DISPATCH_TO_CUDA("jagged_1d_to_dense", fbgemm::jagged_1d_to_dense_gpu);
}

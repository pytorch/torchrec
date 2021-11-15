/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

using namespace at;

Tensor dense_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t BT_block_size);

Tensor dense_embedding_codegen_forward_weighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t BT_block_size);

Tensor dense_embedding_codegen_grad_indice_weights_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    Tensor feature_requires_grad);

Tensor split_embedding_backward_codegen_dense_unweighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    double unused);

Tensor split_embedding_backward_codegen_dense_weighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    double unused);

class SplitLookupFunction_Dense_Op
    : public torch::autograd::Function<SplitLookupFunction_Dense_Op> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      Tensor dev_weights,
      Tensor weights_offsets,
      Tensor D_offsets,
      int64_t total_D,
      int64_t max_D,
      Tensor hash_size_cumsum,
      int64_t total_hash_size_bits,
      Tensor indices,
      Tensor offsets,
      int64_t pooling_mode,
      c10::optional<Tensor> indice_weights,
      c10::optional<Tensor> feature_requires_grad) {
    ctx->save_for_backward({
        dev_weights,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        indice_weights.value_or(Tensor()),
        feature_requires_grad.value_or(Tensor()),
    });

    ctx->saved_data["total_D"] = total_D;
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["pooling_mode"] = pooling_mode;

    constexpr int32_t BT_block_size = 32;
    if (!indice_weights.has_value()) {
      return {dense_embedding_codegen_forward_unweighted_cuda(
          dev_weights,
          weights_offsets,
          D_offsets,
          total_D,
          max_D,
          indices,
          offsets,
          pooling_mode,
          BT_block_size)};
    } else {
      return {dense_embedding_codegen_forward_weighted_cuda(
          dev_weights,
          weights_offsets,
          D_offsets,
          total_D,
          max_D,
          indices,
          offsets,
          pooling_mode,
          indice_weights.value(),
          BT_block_size)};
    }
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto D_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;

    auto total_D = ctx->saved_data["total_D"].toInt();
    auto max_D = ctx->saved_data["max_D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();

    TORCH_CHECK(grad_outputs.size() == 1);

    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
    using torch::autograd::Variable;

    auto grad_output = grad_outputs[0];
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 ||
        grad_output.stride(0) % 4 != 0) {
        grad_output = grad_output.contiguous();
    }

    if (!indice_weights.defined()) {
      auto grad_dev_weights =
          split_embedding_backward_codegen_dense_unweighted_exact_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              hash_size_cumsum,
              total_hash_size_bits,
              indices,
              offsets,
              pooling_mode,
              BT_block_size,
              max_segment_length_per_warp,
              /* unused=*/0.0);
      return {
          grad_dev_weights,
          Variable(), // weights_offsets
          Variable(), // D_offsets
          Variable(), // total_D
          Variable(), // max_D
          Variable(), // hash_size_cumsum
          Variable(), // total_hash_size_bits
          Variable(), // indices
          Variable(), // offsets
          Variable(), // pooling_mode
          Variable(), // indice_weights
          Variable(), // feature_requires_grad
      };
    } else {
      auto grad_indice_weights =
          dense_embedding_codegen_grad_indice_weights_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              indices,
              offsets,
              feature_requires_grad);
      auto grad_dev_weights =
          split_embedding_backward_codegen_dense_weighted_exact_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              hash_size_cumsum,
              total_hash_size_bits,
              indices,
              offsets,
              pooling_mode,
              indice_weights,
              BT_block_size,
              max_segment_length_per_warp,
              /* unused=*/0.0);
      return {
          grad_dev_weights,
          Variable(), // weights_offsets
          Variable(), // D_offsets
          Variable(), // total_D
          Variable(), // max_D
          Variable(), // hash_size_cumsum
          Variable(), // total_hash_size_bits
          Variable(), // indices
          Variable(), // offsets
          Variable(), // pooling_mode
          grad_indice_weights,
          Variable(), // feature_requires_grad
      };
    }
  }
};

at::Tensor split_embedding_codegen_lookup_dense_function(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    c10::optional<Tensor> feature_requires_grad) {
  return SplitLookupFunction_Dense_Op::apply(
      dev_weights,
      weights_offsets,
      D_offsets,
      total_D,
      max_D,
      hash_size_cumsum,
      total_hash_size_bits,
      indices,
      offsets,
      pooling_mode,
      indice_weights,
      feature_requires_grad)[0];
}

TORCH_LIBRARY_FRAGMENT(fb, m) {
    m.def("dense_embedding_codegen_lookup_function(Tensor dev_weights, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, Tensor? feature_requires_grad) -> Tensor");
    m.impl("dense_embedding_codegen_lookup_function", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(split_embedding_codegen_lookup_dense_function)));
}

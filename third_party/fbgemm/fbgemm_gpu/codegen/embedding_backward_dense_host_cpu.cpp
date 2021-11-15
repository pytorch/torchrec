/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "codegen/embedding_forward_split_cpu.h"

using namespace at;

Tensor split_embedding_backward_codegen_dense_cpu(
    Tensor grad_output,
    Tensor host_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    double unused);

namespace {

class SplitLookupFunction_Dense_Op
    : public torch::autograd::Function<SplitLookupFunction_Dense_Op> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      Tensor host_weights,
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
    Tensor indice_weights_value = indice_weights.value_or(Tensor());
    Tensor feature_requires_grad_value =
        feature_requires_grad.value_or(Tensor());
    ctx->save_for_backward({
        host_weights,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        indice_weights_value,
        feature_requires_grad_value,
    });

    ctx->saved_data["total_D"] = total_D;
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["pooling_mode"] = pooling_mode;

    return {split_embedding_codegen_forward_cpu(
        host_weights,
        weights_offsets,
        D_offsets,
        total_D,
        hash_size_cumsum,
        indices,
        offsets,
        pooling_mode,
        indice_weights_value)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto host_weights = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto D_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;

    auto max_D = ctx->saved_data["max_D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();

    TORCH_CHECK(grad_outputs.size() == 1);

    using torch::autograd::Variable;

    auto grad_host_weights = split_embedding_backward_codegen_dense_cpu(
        grad_outputs[0],
        host_weights,
        weights_offsets,
        D_offsets,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
        pooling_mode,
        indice_weights,
        /* unused=*/0.0);
    // NOTE: MEAN pooling will not work with indice_weights!
    auto grad_indice_weights = indice_weights.defined()
        ? split_embedding_codegen_grad_indice_weights_cpu(
              grad_outputs[0],
              host_weights,
              weights_offsets,
              D_offsets,
              indices,
              offsets,
              feature_requires_grad)
        : Variable();
    return {
        grad_host_weights,
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
};

Tensor split_embedding_codegen_lookup_dense_function(
    Tensor host_weights,
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
      host_weights,
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

TORCH_LIBRARY_IMPL(fb, CPU, m) {
  m.impl("dense_embedding_codegen_lookup_function", torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(split_embedding_codegen_lookup_dense_function)));
}

} // namespace

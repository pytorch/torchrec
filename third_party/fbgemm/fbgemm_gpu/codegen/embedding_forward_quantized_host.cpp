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

Tensor int_nbit_split_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t output_dtype,
    int64_t unused);

Tensor int_nbit_split_embedding_codegen_forward_weighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t output_dtype,
    int64_t unused);

Tensor int_nbit_split_embedding_codegen_lookup_function(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype) {
  if (!indice_weights) {
    return int_nbit_split_embedding_codegen_forward_unweighted_cuda(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        D_offsets,
        total_D,
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float16_D,
        max_float32_D,
        indices,
        offsets,
        pooling_mode,
        output_dtype,
        0);
  }
  return int_nbit_split_embedding_codegen_forward_weighted_cuda(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      max_int2_D,
      max_int4_D,
      max_int8_D,
      max_float16_D,
      max_float32_D,
      indices,
      offsets,
      pooling_mode,
      *indice_weights,
      output_dtype,
      0);
}

Tensor pruned_hashmap_lookup_unweighted_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

Tensor pruned_array_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets);

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "int_nbit_split_embedding_codegen_lookup_function(Tensor dev_weights, Tensor uvm_weights, Tensor weights_placements, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, int total_D, int max_int2_D, int max_int4_D, int max_int8_D, int max_float16_D, int max_float32_D, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, int output_dtype=1) -> Tensor");
  m.impl(
      "int_nbit_split_embedding_codegen_lookup_function",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(int_nbit_split_embedding_codegen_lookup_function)));

  m.def(
      "pruned_hashmap_lookup(Tensor indices, Tensor offsets, Tensor hash_table, Tensor hash_table_offsets) -> Tensor");
  m.impl(
      "pruned_hashmap_lookup",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(pruned_hashmap_lookup_unweighted_cuda)));

  m.def(
      "pruned_array_lookup(Tensor indices, Tensor offsets, Tensor index_remappings, Tensor index_remappings_offsets) -> Tensor");
  m.impl(
      "pruned_array_lookup",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(pruned_array_lookup_cuda)));
}

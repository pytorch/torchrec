/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm {

// Return array of size T_in.numel(), representing incomplete exclusive cumsum
at::Tensor asynchronous_exclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_exclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor offsets_range_cuda(const at::Tensor& offsets, int64_t range_size);

at::Tensor offsets_range_cpu(const at::Tensor& offsets, int64_t range_size);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> permute_sparse_data_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cuda(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> permute_sparse_data_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input);
at::Tensor _float_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor& _fused8bitrowwise_to_float_cpu_out(
    at::Tensor& output,
    const at::Tensor& input);
at::Tensor& _float_to_fused8bitrowwise_cpu_out(
    at::Tensor& output,
    const at::Tensor& input);

at::Tensor reorder_batched_ad_lengths_gpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_indices_gpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_lengths_cpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_indices_cpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor recat_embedding_grad_output_cuda(
    at::Tensor grad_output, // [B_local][T_global][D]
    std::vector<int64_t> num_features_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cuda(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_batch_cuda(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const at::Tensor& dim_sum_per_rank,
    const at::Tensor& cumsum_dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cpu(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);

at::Tensor batched_unary_embeddings_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

at::Tensor batched_unary_embeddings_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

at::Tensor jagged_2d_to_dense_forward_cuda(
    at::Tensor embeddings,
    at::Tensor offsets,
    int32_t max_L);

at::Tensor jagged_2d_to_dense_backward_cuda(
    at::Tensor grad_padded_embeddings,
    at::Tensor offsets,
    int32_t total_L);

at::Tensor jagged_1d_to_dense_gpu(
    at::Tensor values,
    at::Tensor offsets,
    int64_t max_L,
    int64_t padding_value);

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda.h>

#include "./fbgemm_cuda_utils.cuh"


// Kernel for recat the embedding gradient output with the mixed dimension
// support
template <typename scalar_t>
__global__ void recat_copy_async_kernel(
    const int64_t* __restrict__ dim_sum_per_rank, // 1D, T
    const int64_t* __restrict__ cum_dim_sum_per_rank, // 1D, T
    const scalar_t* __restrict__ go, // 2D, B x sum(mixed_D)
    scalar_t* __restrict__ sgo, // 1D, B * sum(mixed_D)
    const int64_t T,
    const int64_t B,
    const int64_t dim_sum) {
  auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
  auto b = b_t % B;
  auto t = b_t / B;

  if (b_t >= B * T) {
    return;
  }
  auto dim_current = dim_sum_per_rank[t];
  const auto tgt_base_addr = B * cum_dim_sum_per_rank[t];
  const auto src_base_addr = cum_dim_sum_per_rank[t];

  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &sgo[tgt_base_addr + b * dim_current]) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &go[src_base_addr + b * dim_sum])) {
    int32_t d_base = dim_current / 4 * 4;
    for (int32_t d = threadIdx.x * 4; d < d_base; d += blockDim.x * 4) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(
          &go[src_base_addr + b * dim_sum + d],
          &sgo[tgt_base_addr + b * dim_current + d]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t d_left = threadIdx.x; d_base + d_left < dim_current;
         d_left += blockDim.x) {
      sgo[tgt_base_addr + b * dim_current + d_base + d_left] =
          go[src_base_addr + b * dim_sum + d_base + d_left];
    }
  } else {
    for (int32_t d = threadIdx.x; d < dim_current; d += blockDim.x) {
      sgo[tgt_base_addr + b * dim_current + d] =
          go[src_base_addr + b * dim_sum + d];
    }
  }
}

// Kernerl for permute pooled embedding op.
// This kernel is moving D elements per warp.
template <typename scalar_t>
__global__ void permute_pooled_embs_kernel(
    const scalar_t* __restrict__ go, // 2D, B x sum(mixed_D)
    const int64_t* __restrict__ offset_dim_list, // 1D, T
    const int64_t* __restrict__ permute_list, // 1D, T
    const int64_t* __restrict__ inv_offset_dim_list, // 1D, T+1
    scalar_t* __restrict__ sgo, // 2D, B x sum(mixed_D)
    const int64_t B,
    const int64_t T,
    const int64_t dim_sum) {
  int32_t t = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  int32_t b = blockIdx.y + gridDim.y * blockIdx.z;
  int32_t idx = threadIdx.x % warpSize;
  int32_t blk = warpSize;
  if (b >= B) {
    return;
  }
  if (t >= T) {
    return;
  }
  int64_t permute_idx = permute_list[t];
  int64_t input_dim_start = offset_dim_list[permute_idx];
  int64_t input_dim_end = offset_dim_list[permute_idx + 1];
  int64_t cur_dim = input_dim_end - input_dim_start;
  if (idx >= cur_dim) {
    return;
  }
  // Apply the offsets on B dimension.
  go += b * dim_sum;
  sgo += b * dim_sum;
  int64_t sgo_offset = inv_offset_dim_list[t];
  // Need to check alignment before using vector code path.
  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(&sgo[sgo_offset]) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &go[input_dim_start])) {
    const int32_t vec_size = 4;
    int32_t loop_end = cur_dim / (vec_size) * (vec_size);
    for (int32_t i = idx * vec_size; i < loop_end; i += blk * vec_size) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(
          &go[input_dim_start + i], &sgo[sgo_offset + i]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t i = loop_end + idx; i < cur_dim; i += blk) {
      sgo[sgo_offset + i] = go[input_dim_start + i];
    }
  } else { // Fallback if not aligned.
    for (int32_t i = idx; i < cur_dim; i += blk) {
      sgo[sgo_offset + i] = go[input_dim_start + i];
    }
  }
}

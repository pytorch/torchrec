/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Forward kernel for batched unary embedding op
template <typename scalar_t, typename index_t>
__global__ void batched_unary_embeddings_forward_kernel(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const scalar_t* __restrict__ weight, // N * sum(E) * 1 (embedding dimension
                                         // is 1)
    const index_t* __restrict__ table_offsets,
    const index_t* __restrict__ offsets,
    const index_t* __restrict__ indices,
    scalar_t* __restrict__ output // N * B * T
) {
  index_t sum_E = table_offsets[T];
  int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }
  int32_t t = blockIdx.y;
  int32_t n = blockIdx.z;
  index_t table_offset = table_offsets[t];
  index_t indices_start = offsets[t * B + b];
  index_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;
  // TODO: this should be acc_type<scalar_t, true>
  scalar_t sum = 0.0;
  for (int32_t l = 0; l < L; ++l) {
    auto idx = __ldg(&indices[indices_start + l]);
    sum += weight[n * sum_E + table_offset + idx + 0];
  }
  output[(n * B + b) * T + t] = sum;
}

// Backward kernel for batched unary embedding op
template <typename scalar_t, typename index_t>
__global__ void batched_unary_embeddings_backward_kernel(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const scalar_t* __restrict__ grad_output, // [N * B * T]
    const index_t* __restrict__ table_offsets,
    const index_t* __restrict__ offsets,
    const index_t* __restrict__ indices,
    scalar_t* __restrict__ grad_weight // [N * sum_E * 1] (embedding
                                       // dimension is 1)
) {
  index_t sum_E = table_offsets[T];
  int32_t n_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t n = n_t / T;
  int32_t t = n_t % T;
  if (n >= N) {
    return;
  }
  index_t table_offset = table_offsets[t];

  for (int32_t b = 0; b < B; ++b) {
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const scalar_t go = grad_output[(n * B + b) * T + t];
    for (int32_t l = 0; l < L; ++l) {
      index_t idx = __ldg(&indices[indices_start + l]);
      grad_weight[n * sum_E + table_offset + idx + 0] += go;
    }
  }
}

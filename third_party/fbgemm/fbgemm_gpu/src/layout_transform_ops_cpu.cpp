/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include "ATen/Parallel.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm {

at::Tensor recat_embedding_grad_output_mixed_D_cpu(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank) {
  TORCH_CHECK(grad_output.is_contiguous());
  const auto B_local = grad_output.sizes()[0];

  at::Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());

  int n = dim_sum_per_rank.size();
  std::vector<int64_t> accum_dim_sum(n + 1);
  accum_dim_sum[0] = 0;
  std::partial_sum(
      dim_sum_per_rank.begin(), dim_sum_per_rank.end(), &accum_dim_sum[1]);
  const auto global_dim_sum = accum_dim_sum[n];
  TORCH_CHECK(B_local * global_dim_sum == grad_output.numel());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "recat_embedding_gradients", ([&] {
        const auto go = grad_output.accessor<scalar_t, 2>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        at::parallel_for(
            0, n * B_local, 1, [&](int64_t i_begin, int64_t i_end) {
              const auto dim_begin = i_begin / B_local;
              const auto dim_end = (i_end + B_local - 1) / B_local;
              for (const auto dim : c10::irange(dim_begin, dim_end)) {
                const auto dim_sum = dim_sum_per_rank[dim];
                const auto sgo_offset = B_local * accum_dim_sum[dim];
                scalar_t* dst = &sgo[sgo_offset];
                const scalar_t* src = &go[0][accum_dim_sum[dim]];
                const auto r_begin = (dim == dim_begin) ? i_begin % B_local : 0;
                const auto r_end = (dim == dim_end - 1 && i_end % B_local != 0)
                    ? i_end % B_local
                    : B_local;
                for (const auto r : c10::irange(r_begin, r_end)) {
                  memcpy(
                      dst + r * dim_sum,
                      src + r * global_dim_sum,
                      dim_sum * sizeof(scalar_t));
                }
              }
            });
      }));

  return sharded_grad_output;
}

} // namespace fbgemm


TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "recat_embedding_grad_output_mixed_D_batch(Tensor grad_output, Tensor dim_sum_per_rank, Tensor cumsum_dim_sum_per_rank) -> Tensor");
  m.def(
      "recat_embedding_grad_output_mixed_D(Tensor grad_output, int[] dim_sum_per_rank) -> Tensor");
  m.def(
      "recat_embedding_grad_output(Tensor grad_output, int[] num_features_per_rank) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl(
      "recat_embedding_grad_output_mixed_D",
      fbgemm::recat_embedding_grad_output_mixed_D_cpu);
}

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

namespace fbgemm {

at::Tensor merge_pooled_embeddings_cpu(
    std::vector<at::Tensor> pooled_embeddings,
    int64_t batch_size,
    at::Device target_device) {
  auto cat_host_0 = [&](const std::vector<at::Tensor>& ts) {
    int64_t n = 0;
    for (auto& t : ts) {
      n += t.numel();
    }
    at::Tensor r;
    if (n == 0) {
      r = at::empty({n});
    } else {
      r = at::empty({n}, ts[0].options());
    }
    r.resize_(0);
    return at::cat_out(r, ts, 1); // concat the tensor list in dim = 1
  };
  return cat_host_0(pooled_embeddings);
}

} // namespace fbgemm

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("merge_pooled_embeddings", fbgemm::merge_pooled_embeddings_cpu);
}

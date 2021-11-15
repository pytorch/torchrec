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
#include <torch/library.h>

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "recat_embedding_grad_output_mixed_D_batch",
      fbgemm::recat_embedding_grad_output_mixed_D_batch_cuda);
  DISPATCH_TO_CUDA(
      "recat_embedding_grad_output_mixed_D",
      fbgemm::recat_embedding_grad_output_mixed_D_cuda);
  DISPATCH_TO_CUDA(
      "recat_embedding_grad_output", fbgemm::recat_embedding_grad_output_cuda);
}

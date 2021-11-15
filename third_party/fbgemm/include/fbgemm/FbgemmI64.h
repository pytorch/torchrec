/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

#include "fbgemm/Utils.h"

namespace fbgemm {

FBGEMM_API void cblas_gemm_i64_i64acc(
    matrix_op_t transa,
    matrix_op_t transb,
    int M,
    int N,
    int K,
    const std::int64_t* A,
    int lda,
    const std::int64_t* B,
    int ldb,
    bool accumulate,
    std::int64_t* C,
    int ldc);

} // namespace fbgemm

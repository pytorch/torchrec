/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/UtilsAvx2.h"
#include "fbgemm/spmmUtilsAvx2.h"

namespace fbgemm {

FBGEMM_API void sparseDenseMMRef(
    int M,
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    bool accum = false);

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
FBGEMM_API void sparseDenseInt8MMRef(
    int N,
    const std::unique_ptr<BCSRMatrix<>>& bcsr,
    const uint8_t* B,
    int ldb,
    int32_t* C_i32,
    uint8_t* C_u8,
    int ldc,
    trRequantizationParams_t& rParams,
    bool accum = false,
    int thread_id = 0,
    int num_threads = 1);

template <bool FUSE_RELU, QuantizationGranularity Q_GRAN>
FBGEMM_API void trRequantizeRef(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const trRequantizationParams_t& r);

// Get matrix shapes of interest
FBGEMM_API std::vector<std::vector<int>> getSparseMatrixShapes();

} // namespace fbgemm

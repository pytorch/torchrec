/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint> // for std::int32_t
#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

/**
 * @brief Sum a given vector.
 */
FBGEMM_API std::int32_t reduceAvx2(const std::uint8_t* A, int len);

/**
 * @brief Transpose 8 rows from source matrix.
 */
void transpose_8rows(
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst);

/**
 * @brief avx2 part of the spmdm code.
 */
void spmdmKernelAvx2(
    int N,
    const uint8_t* A_buffer,
    const int32_t* colptr,
    const int8_t* values,
    const int16_t* rowidx,
    int32_t* C_buffer);

} // namespace fbgemm

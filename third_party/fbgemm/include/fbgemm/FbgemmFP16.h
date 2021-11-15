/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cpuinfo.h>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "./Types.h"
#include "./Utils.h"
#include "./FbgemmPackMatrixB.h"

namespace fbgemm {

template<>
struct TypeConverter<float16> {
  float16 operator()(float src) const {
    constexpr float FP16_MAX = 65504.f;
    const float fp16 = std::max(-FP16_MAX, std::min(src, FP16_MAX));
    return cpu_float2half_rn(fp16);
  }
};

using PackedGemmMatrixFP16 = PackedGemmMatrixB<float16>;

template<typename T>
FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<T>& Bp,
    const float beta,
    float* C,
    int thread_id = 0,
    int num_threads = 1);

extern template void cblas_gemm_compute<float16>(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads);

}; // namespace fbgemm

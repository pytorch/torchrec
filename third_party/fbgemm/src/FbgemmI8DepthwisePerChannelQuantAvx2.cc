/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"

#include <stdexcept> // for logic_error
#include <string>

#include "./FbgemmI8Depthwise2DAvx2-inl.h"


namespace fbgemm {

// Old interface
template <typename BIAS_TYPE /*=std::int32_t*/>
void depthwise_2d_per_channel_quantization_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
      N,
      H,
      W,
      IC_OC,
      IC_OC,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      B_zero_point,
      Bp,
      C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void
depthwise_2d_per_channel_quantization_same_pad<int32_t>(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_2d_per_channel_quantization_same_pad<float>(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

} // namespace fbgemm

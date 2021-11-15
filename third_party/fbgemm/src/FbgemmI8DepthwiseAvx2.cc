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

using namespace std;

namespace fbgemm {

// Dispatch input shape and FUSE_RELU
template <QuantizationGranularity Q_GRAN, typename BIAS_TYPE /*=std::int32_t*/>
void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (B.GetKernelProduct() == 3 * 3) {
    if (fuse_relu) {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE, Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE, Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
    return;
  }

  if (B.GetKernelProduct() != 5 * 5) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(5 * 5) + " but has " + to_string(B.GetKernelProduct());
    throw logic_error(msg);
  }
  if (fuse_relu) {
    depthwise_2d_<5, true /* FUSE_RELU */, BIAS_TYPE, Q_GRAN>(
        N,
        H,
        W,
        IC,
        OC,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_2d_<5, false /* FUSE_RELU */, BIAS_TYPE, Q_GRAN>(
        N,
        H,
        W,
        IC,
        OC,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
}

template FBGEMM_API void depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_2d_same_pad<QuantizationGranularity::GROUP>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_2d_same_pad<QuantizationGranularity::GROUP>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

// Old interface
template <typename BIAS_TYPE /*=std::int32_t*/>
void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
      N,
      H,
      W,
      IC_OC,
      IC_OC,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      &B_zero_point,
      B,
      &C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      &act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <cstdint>
#include "fbgemm/ConvUtils.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {

class FBGEMM_API PackedDepthWiseConvMatrix {
 public:
  /**
   * @param IC the number of input channels (same as the number of groups
   *           because depth-wise convolution has one input channel per group)
   * @param OC the number of output channels
   * @param kernel_prod the product of all kernels. For example, kernel_prod =
   *                    9 for 3x3 conv, and 27 for 3x3x3 conv.
   * @param smat the source unpacked weight in GRS layout
   */
  PackedDepthWiseConvMatrix(int OC, int kernel_prod, const std::int8_t* smat);
  virtual ~PackedDepthWiseConvMatrix();

  const std::int8_t* PackedMat() const {
    return pmat_;
  }

  int GetKernelProduct() const {
    return kernel_prod_;
  }

  /**
   * @brief Unpacks pmat_ into unpack_data.
   * Used for recovering the weight matrix into the original format
   */
  void unpack(std::int8_t* unpacked_data);

  /**
   * @brief returns the index into pmat_ given the row and column for smat
   */
  int addr(int r, int c);

 private:
  const int OC_; /**< the number of output channels */
  const int kernel_prod_; /** the product of all kernel dims */
  std::int8_t* pmat_; /** packed weight */
}; // PackedDepthWiseConvMatrix

/**
 * Depth-wise convolution that results in the same output feature size as the
 * input feature. That is PAD_T = PAD_B = (R - 1) / 2 and PAD_L = PAD_R =
 * (S - 1) / 2. This function also does requantization.
 * @param col_offsets nullptr if col_offsets are folded into bias
 * @param act_times_w_scale Only used if BIAS_TYPE is float, i.e., bias is
 *                          unquantized.
 */
template <QuantizationGranularity Q_GRAN, typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

template <typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_2d_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    float act_times_w_scale = 1.0f,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Depth-wise convolution that results in the same output feature size as the
 * input feature. That is PAD_T = PAD_B = (R - 1) / 2 and PAD_L = PAD_R =
 * (S - 1) / 2. This function also does requantization and uses per-channel
 * quantization.
 * @param col_offsets nullptr if col_offsets are folded into bias
 * @param act_times_w_scale Only used if BIAS_TYPE is float, i.e., bias is
 *                          unquantized.
 */
template <typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_2d_per_channel_quantization_same_pad(
    int N,
    int H,
    int W,
    int IC_OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

/**
 * @param col_offsets nullptr if col_offsets are folded into bias
 */
template <QuantizationGranularity Q_GRAN, typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_3d_same_pad(
    const conv_param_t<3>& conv_p,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

} // namespace fbgemm

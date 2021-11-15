/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept> // for logic_error
#include <vector>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <int SPATIAL_DIM, typename ACC_T>
bool takeDepthWiseFastPath(const conv_param_t<SPATIAL_DIM>& conv_p) {
  // Note: Depthwise convolutions (both 2D and 3D) are optimized for the most
  // common case.
  // 3x3 or 5x5 2D
  // (3 or 5)x(3x3 or 5x5) 3D
  bool ret = std::is_same<ACC_T, std::int32_t>::value &&
      conv_p.G == conv_p.IC &&
      (conv_p.G == conv_p.OC || conv_p.G * 2 == conv_p.OC) &&
      conv_p.G % 8 == 0 &&
      std::all_of(
                 conv_p.stride.begin(),
                 conv_p.stride.end(),
                 [](int i) { return i == 1 || i == 2; }) &&
      SPATIAL_DIM >= 2 &&
      conv_p.K[SPATIAL_DIM - 2] == conv_p.K[SPATIAL_DIM - 1] &&
      std::all_of(
                 conv_p.K.begin(),
                 conv_p.K.end(),
                 [](int i) { return i == 3 || i == 5; }) &&
      std::all_of(
                 conv_p.dilation.begin(),
                 conv_p.dilation.end(),
                 [](int i) { return i == 1; }) &&
      !conv_p.transposed;

  // Check pads result in same input and output spatial dim
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    if (conv_p.pad[i] != (conv_p.K[i] - 1) / 2 ||
        conv_p.pad[i] != conv_p.pad[SPATIAL_DIM + i]) {
      ret = false;
    }
  }

  return ret;
}

template <int SPATIAL_DIM>
bool takePointWiseFastPath(const conv_param_t<SPATIAL_DIM>& conv_p) {
  return std::accumulate(conv_p.K.begin(), conv_p.K.end(), 0) == SPATIAL_DIM &&
      std::accumulate(conv_p.stride.begin(), conv_p.stride.end(), 0) ==
      SPATIAL_DIM &&
      std::accumulate(conv_p.dilation.begin(), conv_p.dilation.end(), 0) ==
      SPATIAL_DIM &&
      std::accumulate(conv_p.pad.begin(), conv_p.pad.end(), 0) == 0 &&
      !conv_p.transposed;
}

template <int SPATIAL_DIM>
bool take1DFastPath(const conv_param_t<SPATIAL_DIM>& conv_p) {
  return false && !conv_p.transposed;
}

template <int SPATIAL_DIM, typename ACC_T>
optimized_conv_t ConvFastPath(const conv_param_t<SPATIAL_DIM>& conv_p) {
  if (takeDepthWiseFastPath<SPATIAL_DIM, ACC_T>(conv_p)) {
    return optimized_conv_t::depthwise;
  } else if (fbgemmOptimizedGConv<SPATIAL_DIM>(conv_p)) {
    return optimized_conv_t::groupwise;
  } else if (takePointWiseFastPath<SPATIAL_DIM>(conv_p)) {
    return optimized_conv_t::pointwise;
  } else if (take1DFastPath<SPATIAL_DIM>(conv_p)) {
    return optimized_conv_t::fastpath1d;
  } else {
    return optimized_conv_t::im2col;
  }
}

template <typename processOutputType, int SPATIAL_DIM, typename ACC_T>
int fbgemmConv(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* activations,
    PackWeightsForConv<SPATIAL_DIM, std::int8_t, ACC_T>& packed_weights,
    typename processOutputType::outType* out,
    std::int32_t* outBuffer,
    processOutputType& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params) {
  if (!packed_weights.isPackingCompliant(conv_p)) {
    std::string msg =
        "[FBGEMM_CONV_ERROR] Convolution parameters "
        "mismatch between pre-packed weights and conv invocation! ";
    msg += packed_weights.mismatchingParams(conv_p);
    msg += std::string(
        " Please pack weights using the same parameters "
        "with which convolution operation is invoked!");
    throw std::logic_error(msg);
  }

  switch (ConvFastPath<SPATIAL_DIM, ACC_T>(conv_p)) {
    case optimized_conv_t::depthwise: {
      // 2D and 3D depthwise fast path
      // std::cout << "Depthwise fast path" << std::endl;
      const std::int32_t* B_zero_point = outProcess.getBZeroPoint();
      const float* C_multiplier = outProcess.getCMultiplier();
      const float* act_times_w_scale = outProcess.getActWScale();
      if (SPATIAL_DIM == 3) {
        static_assert(
            std::is_same<typename processOutputType::outType, std::uint8_t>::
                value,
            "For depthwise, only requantized output is supported");

        if (processOutputType::QGRANType == QuantizationGranularity::TENSOR) {
          depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
              *reinterpret_cast<const conv_param_t<3>*>(&conv_p),
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale,
              thread_id,
              num_threads);
        } else if (
            processOutputType::QGRANType == QuantizationGranularity::GROUP) {
          depthwise_3d_same_pad<QuantizationGranularity::GROUP>(
              *reinterpret_cast<const conv_param_t<3>*>(&conv_p),
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale, // act_scale * weight_scale
              thread_id,
              num_threads);
        } else if (
            processOutputType::QGRANType ==
            QuantizationGranularity::OUT_CHANNEL) {
          depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
              *reinterpret_cast<const conv_param_t<3>*>(&conv_p),
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale, // act_scale * weight_scale
              thread_id,
              num_threads);
        } else {
          std::string msg =
              "[FBGEMM_CONV_ERROR] This quantization granularity is "
              "not supported";
          throw std::runtime_error(msg);
        }
      } else if (SPATIAL_DIM == 2) {
        if (processOutputType::QGRANType == QuantizationGranularity::TENSOR) {
          depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
              conv_p.MB, // mini batch
              conv_p.IN_DIM[0], // H
              conv_p.IN_DIM[1], // W
              conv_p.IC, // input channels
              conv_p.OC, // output channels
              conv_p.stride[0], // stride_h
              conv_p.stride[1], // stride_w
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale,
              thread_id,
              num_threads);
        } else if (
            processOutputType::QGRANType == QuantizationGranularity::GROUP) {
          depthwise_2d_same_pad<QuantizationGranularity::GROUP>(
              conv_p.MB, // mini batch
              conv_p.IN_DIM[0], // H
              conv_p.IN_DIM[1], // W
              conv_p.IC, // input channels
              conv_p.OC, // output channels
              conv_p.stride[0], // stride_h
              conv_p.stride[1], // stride_w
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale, // act_scale * weight_scale
              thread_id,
              num_threads);
        } else if (
            processOutputType::QGRANType ==
            QuantizationGranularity::OUT_CHANNEL) {
          // The number of input channels == groups for depthwise convolutions
          depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
              conv_p.MB, // mini batch
              conv_p.IN_DIM[0], // H
              conv_p.IN_DIM[1], // W
              conv_p.IC, // input channels
              conv_p.OC, // output channels
              conv_p.stride[0], // stride_h
              conv_p.stride[1], // stride_w
              outProcess.getAZeroPoint(),
              activations,
              B_zero_point,
              *(packed_weights.getPackedWForDepthwise()),
              C_multiplier,
              outProcess.getCZeroPoint(),
              out,
              outProcess.getColOffsets(),
              outProcess.getBias(),
              outProcess.RELU_FUSED, // fuse_relu
              act_times_w_scale, // act_scale * weight_scale
              thread_id,
              num_threads);
        } else {
          std::string msg =
              "[FBGEMM_CONV_ERROR] This quantization granularity is "
              "not supported";
          throw std::runtime_error(msg);
        }
      } else {
        std::string msg =
            "[FBGEMM_CONV_ERROR] This spatial dim is not supported";
        throw std::runtime_error(msg);
      }
      break;
    }
    case optimized_conv_t::groupwise: {
      // optimized groupwise convolution
      // std::cout << "Groupwise fast path" << std::endl;
      std::vector<int32_t> row_offset_buf(
          rowOffsetBufferSizeGConv<SPATIAL_DIM>(conv_p));
      outProcess.setRowOffsets(row_offset_buf.data());
      fbgemmGroupwiseConv(
          conv_p,
          activations,
          outProcess.getAZeroPoint(),
          row_offset_buf.data(),
          *(packed_weights.getPackedWForGroupwise()),
          out,
          outBuffer,
          outProcess,
          thread_id,
          num_threads);
      break;
    }
    case optimized_conv_t::pointwise: {
      std::vector<int32_t> row_offset_buf(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize(blocking_params));
      int image_dim = std::accumulate(
          conv_p.IN_DIM.begin(),
          conv_p.IN_DIM.end(),
          1,
          std::multiplies<int>());
      PackAWithRowOffset<uint8_t, ACC_T> packA(
          matrix_op_t::NoTranspose,
          conv_p.MB * image_dim,
          conv_p.IC,
          activations,
          conv_p.IC,
          nullptr,
          conv_p.G,
          row_offset_buf.data(),
          blocking_params);

      outProcess.setRowOffsets(row_offset_buf.data());
      fbgemmPacked(
          packA,
          *(packed_weights.getPackedWForPointwise()),
          out,
          outBuffer,
          conv_p.OC,
          outProcess,
          thread_id,
          num_threads,
          blocking_params);
      break;
    }
    case optimized_conv_t::fastpath1d: {
      break;
    }
    case optimized_conv_t::im2col: {
      // All other convolutions go through im2col-based implementation
      // std::cout << "Im2col path" << std::endl;
      std::vector<int32_t> row_offset_buf(
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>::rowOffsetBufferSize(
              blocking_params));

      const std::int32_t* b_zero_point = outProcess.getBZeroPoint();
      bool b_symmetric = false;
      if (processOutputType::QGRANType == QuantizationGranularity::TENSOR) {
        b_symmetric = b_zero_point[0] == 0;
      } else if (
          processOutputType::QGRANType == QuantizationGranularity::GROUP) {
        b_symmetric =
            std::all_of(b_zero_point, b_zero_point + conv_p.G, [](int i) {
              return i == 0;
            });
      } else if (
          processOutputType::QGRANType ==
          QuantizationGranularity::OUT_CHANNEL) {
        b_symmetric =
            std::all_of(b_zero_point, b_zero_point + conv_p.OC, [](int i) {
              return i == 0;
            });
      } else {
        std::string msg =
            "[FBGEMM_CONV_ERROR] This quantization granularity is "
            "not supported";
        throw std::runtime_error(msg);
      }
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM> packA(
          conv_p,
          activations,
          nullptr, /* buffer for packed matrix */
          outProcess.getAZeroPoint(),
          row_offset_buf.data(),
          b_symmetric,
          blocking_params);

      outProcess.setRowOffsets(row_offset_buf.data());
      fbgemmPacked(
          packA,
          *(packed_weights.getPackedWForIm2col()),
          out,
          outBuffer,
          conv_p.OC,
          outProcess,
          thread_id,
          num_threads,
          blocking_params);
      break;
    }
  } // switch

  return 0;
}

#define INSTANTIATE_BASE(ACC_T, Q_GRAN, RELU, SPATIAL_DIM, BIAS_TYPE)      \
  template FBGEMM_API int fbgemmConv(                                      \
      const conv_param_t<SPATIAL_DIM>& conv_p,                             \
      const std::uint8_t* activations,                                     \
      PackWeightsForConv<SPATIAL_DIM, std::int8_t, ACC_T>& packed_weights, \
      std::uint8_t* out,                                                   \
      std::int32_t* outBuffer,                                             \
      ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,               \
      int thread_id,                                                       \
      int num_threads,                                                     \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(ACC_T, Q_GRAN, RELU, SPATIAL_DIM) \
  INSTANTIATE_BASE(ACC_T, Q_GRAN, RELU, SPATIAL_DIM, float)  \
  INSTANTIATE_BASE(ACC_T, Q_GRAN, RELU, SPATIAL_DIM, int32_t)

#define INSTANTIATE_SPATIAL_DIM(ACC_T, Q_GRAN, RELU) \
  INSTANTIATE_BIAS_T(ACC_T, Q_GRAN, RELU, 1)         \
  INSTANTIATE_BIAS_T(ACC_T, Q_GRAN, RELU, 2)         \
  INSTANTIATE_BIAS_T(ACC_T, Q_GRAN, RELU, 3)

#define INSTANTIATE_RELU(ACC_T, Q_GRAN)         \
  INSTANTIATE_SPATIAL_DIM(ACC_T, Q_GRAN, true)  \
  INSTANTIATE_SPATIAL_DIM(ACC_T, Q_GRAN, false)

#define INSTANTIATE_Q_GRANS(ACC_T)                          \
  INSTANTIATE_RELU(ACC_T, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_RELU(ACC_T, QuantizationGranularity::GROUP)   \
  INSTANTIATE_RELU(ACC_T, QuantizationGranularity::OUT_CHANNEL)

INSTANTIATE_Q_GRANS(std::int32_t)

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_RELU
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

template bool takeDepthWiseFastPath<2, std::int32_t>(
    const conv_param_t<2>& conv_p);
template bool takeDepthWiseFastPath<3, std::int32_t>(
    const conv_param_t<3>& conv_p);
template bool takeDepthWiseFastPath<2, std::int16_t>(
    const conv_param_t<2>& conv_p);
template bool takeDepthWiseFastPath<3, std::int16_t>(
    const conv_param_t<3>& conv_p);

template FBGEMM_API optimized_conv_t
ConvFastPath<1, std::int32_t>(const conv_param_t<1>& conv_p);
template FBGEMM_API optimized_conv_t
ConvFastPath<2, std::int32_t>(const conv_param_t<2>& conv_p);
template FBGEMM_API optimized_conv_t
ConvFastPath<3, std::int32_t>(const conv_param_t<3>& conv_p);

template FBGEMM_API optimized_conv_t
ConvFastPath<1, std::int16_t>(const conv_param_t<1>& conv_p);
template FBGEMM_API optimized_conv_t
ConvFastPath<2, std::int16_t>(const conv_param_t<2>& conv_p);
template FBGEMM_API optimized_conv_t
ConvFastPath<3, std::int16_t>(const conv_param_t<3>& conv_p);

} // namespace fbgemm

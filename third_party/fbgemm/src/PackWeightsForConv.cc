/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/Fbgemm.h"

#include <algorithm>
#include <memory>

namespace fbgemm {

template <int SPATIAL_DIM, typename T, typename accT>
PackWeightsForConv<SPATIAL_DIM, T, accT>::PackWeightsForConv(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const T* sdata,
    const BlockingFactors* blocking_params)
    : conv_param_(conv_p) {
  // Note: The following logic should *exactly* match with what we have in
  // FbgemmConv.cc
  switch (ConvFastPath<SPATIAL_DIM, accT>(conv_p)) {
    case optimized_conv_t::depthwise: {
      const int kernel_d = SPATIAL_DIM <= 2 ? 1 : conv_p.K[0];
      const int kernel_h = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
      const int kernel_w = conv_p.K[SPATIAL_DIM - 1];
      W_dw_packed_ = std::make_shared<PackedDepthWiseConvMatrix>(
          conv_p.OC, kernel_d * kernel_h * kernel_w, sdata);
      break;
    }
    case optimized_conv_t::groupwise: {
      W_gconv_packed_ =
          std::make_shared<PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>>(
              matrix_op_t::Transpose, conv_p, sdata, nullptr);
      break;
    }
    case optimized_conv_t::pointwise: {
      const int N = conv_p.OC / conv_p.G;
      const int kernel_d = SPATIAL_DIM <= 2 ? 1 : conv_p.K[0];
      const int kernel_h = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
      const int kernel_w = conv_p.K[SPATIAL_DIM - 1];
      const int K = kernel_d * kernel_h * kernel_w * conv_p.IC;
      W_pointwise_packed_ = std::make_shared<PackBMatrix<T, accT>>(
          matrix_op_t::Transpose,
          K,
          N,
          sdata,
          K / conv_p.G,
          nullptr,
          conv_p.G,
          blocking_params);
      break;
    }
    case optimized_conv_t::fastpath1d: {
      break;
    }
    case optimized_conv_t::im2col: {
      const int N = conv_p.OC / conv_p.G;
      const int kernel_d = SPATIAL_DIM <= 2 ? 1 : conv_p.K[0];
      const int kernel_h = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
      const int kernel_w = conv_p.K[SPATIAL_DIM - 1];
      const int K = kernel_d * kernel_h * kernel_w * conv_p.IC;
      W_im2col_packed_ = std::make_shared<PackBMatrix<T, accT>>(
          matrix_op_t::Transpose,
          K,
          N,
          sdata,
          K / conv_p.G,
          nullptr,
          conv_p.G,
          blocking_params);
      break;
    }
  } // switch
}

template <int SPATIAL_DIM, typename T, typename accT>
void PackWeightsForConv<SPATIAL_DIM, T, accT>::unpack(T* origin_buf) {
  if (W_dw_packed_) {
    W_dw_packed_->unpack(origin_buf);
  } else if (W_gconv_packed_) {
    W_gconv_packed_->unpack(origin_buf);
  } else if (W_im2col_packed_) {
    W_im2col_packed_->unpack(origin_buf);
  } else if (W_pointwise_packed_) {
    W_pointwise_packed_->unpack(origin_buf);
  } else {
    assert(false && "At least one packed weights object should exist");
  }
}

template <int SPATIAL_DIM, typename T, typename accT>
bool PackWeightsForConv<SPATIAL_DIM, T, accT>::isPackingCompliant(
    const conv_param_t<SPATIAL_DIM>& test_conv_p) {
  return conv_param_.IC == test_conv_p.IC && conv_param_.OC == test_conv_p.OC &&
      conv_param_.G == test_conv_p.G &&
      std::equal(
             conv_param_.K.begin(),
             conv_param_.K.end(),
             test_conv_p.K.begin()) &&
      std::equal(
             conv_param_.stride.begin(),
             conv_param_.stride.end(),
             test_conv_p.stride.begin()) &&
      std::equal(
             conv_param_.pad.begin(),
             conv_param_.pad.end(),
             test_conv_p.pad.begin()) &&
      std::equal(
             conv_param_.dilation.begin(),
             conv_param_.dilation.end(),
             test_conv_p.dilation.begin());
}

template <int SPATIAL_DIM, typename T, typename accT>
std::string PackWeightsForConv<SPATIAL_DIM, T, accT>::mismatchingParams(
    const conv_param_t<SPATIAL_DIM>& test_conv_p) {
  std::string msg = "";

  auto combineStr = [](std::string id, std::string str1, std::string str2) {
    std::string out = id + std::string(" ");
    out += str1;
    out += std::string(" vs ") + str2;
    out += std::string(";");
    return out;
  };

  auto combineInt = [&combineStr](std::string id, int int1, int int2) {
    return combineStr(id, std::to_string(int1), std::to_string(int2));
  };

  if (conv_param_.IC != test_conv_p.IC) {
    msg += combineInt("input_channels", conv_param_.IC, test_conv_p.IC);
  }
  if (conv_param_.OC != test_conv_p.OC) {
    msg += combineInt("output_channels", conv_param_.IC, test_conv_p.IC);
  }
  if (conv_param_.G != test_conv_p.G) {
    msg += combineInt("groups", conv_param_.G, test_conv_p.G);
  }

  if (!std::equal(
          conv_param_.K.begin(), conv_param_.K.end(), test_conv_p.K.begin())) {
    msg += combineStr(
        "kernel",
        arrayToString<SPATIAL_DIM>(conv_param_.K),
        arrayToString<SPATIAL_DIM>(test_conv_p.K));
  }

  if (!std::equal(
          conv_param_.stride.begin(),
          conv_param_.stride.end(),
          test_conv_p.stride.begin())) {
    msg += combineStr(
        "stride",
        arrayToString<SPATIAL_DIM>(conv_param_.stride),
        arrayToString<SPATIAL_DIM>(test_conv_p.stride));
  }

  if (!std::equal(
          conv_param_.pad.begin(),
          conv_param_.pad.end(),
          test_conv_p.pad.begin())) {
    msg += combineStr(
        "pad",
        arrayToString<2 * SPATIAL_DIM>(conv_param_.pad),
        arrayToString<2 * SPATIAL_DIM>(test_conv_p.pad));
  }

  if (!std::equal(
          conv_param_.dilation.begin(),
          conv_param_.dilation.end(),
          test_conv_p.dilation.begin())) {
    msg += combineStr(
        "dilation",
        arrayToString<SPATIAL_DIM>(conv_param_.dilation),
        arrayToString<SPATIAL_DIM>(test_conv_p.dilation));
  }

  return msg;
}

template class PackWeightsForConv<1, int8_t, int32_t>;
template class PackWeightsForConv<2, int8_t, int32_t>;
template class PackWeightsForConv<3, int8_t, int32_t>;

} // namespace fbgemm

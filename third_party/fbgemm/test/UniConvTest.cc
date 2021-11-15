/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "./QuantizationHelpers.h"
#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::GROUP,
    QuantizationGranularity::OUT_CHANNEL};

// clang-format off
template <int SPATIAL_DIM = 1>
static typename std::enable_if<SPATIAL_DIM == 1, vector<conv_param_t<1>>>::type
GetShapes_() {
  vector<conv_param_t<1>> shapes = {
    // MB, IC, OC, {IW}, G, {KW}, {stride_w}, {pad_l,pad_r}, {dilation_w}
    // Regular
    conv_param_t<1>(1, 16, 16, {30}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 32, 32, {30}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 32, 16, {30}, 1, {3}, {1}, {0, 0}, {2}),
    // deconv shapes
    // MB, IC, OC, {IW}, G, {KW}, {stride_w}, {pad_l,pad_r}, {dilation_w}, {output_padding}, transposed
    // Regular
    conv_param_t<1>(1, 16, 16, {30}, 1, {3}, {1}, {1, 1}, {1}, {0}, true),
    conv_param_t<1>(1, 32, 32, {30}, 1, {3}, {1}, {1, 1}, {1}, {0}, true),
    conv_param_t<1>(1, 32, 16, {30}, 1, {3}, {1}, {0, 0}, {1}, {0}, true),
    conv_param_t<1>(1, 16, 16, {30}, 1, {3}, {1}, {1, 1}, {2}, {0}, true),
    conv_param_t<1>(1, 32, 32, {30}, 1, {3}, {1}, {1, 1}, {2}, {0}, true),
    conv_param_t<1>(1, 32, 16, {30}, 1, {3}, {1}, {0, 0}, {2}, {0}, true),
    conv_param_t<1>(1, 16, 16, {30}, 1, {3}, {1}, {1, 1}, {1}, {1}, true),
    conv_param_t<1>(1, 32, 32, {30}, 1, {3}, {1}, {1, 1}, {1}, {1}, true),
    conv_param_t<1>(1, 32, 16, {30}, 1, {3}, {1}, {0, 0}, {1}, {1}, true),
    conv_param_t<1>(1, 32, 32, {30}, 1, {3}, {2}, {1, 1}, {2}, {0}, true),

    // some example deconv shapes
    conv_param_t<1>(1, 96, 48, {30}, 1, {16}, {8}, {4, 4}, {1}, {0}, true),
    conv_param_t<1>(1, 48, 24, {30}, 1, {8}, {4}, {2, 2}, {1}, {0}, true),
    conv_param_t<1>(1, 24, 12, {30}, 1, {4}, {2}, {1, 1}, {1}, {0}, true),

    // groupwise
    conv_param_t<1>(1, 32, 16, {30}, 8, {3}, {1}, {0, 0}, {1}, {1}, true),
    conv_param_t<1>(1, 32, 32, {30}, 8, {3}, {2}, {1, 1}, {2}, {0}, true),
  };
  return shapes;
}
// clang-format on

// clang-format off
template <int SPATIAL_DIM = 2>
static typename std::enable_if<SPATIAL_DIM == 2, vector<conv_param_t<2>>>::type
GetShapes_() {
  vector<conv_param_t<>> shapes = {
    // MB, IC, OC, {IH, IW}, G, {KH, KW}, {stride_h, stride_w}, {pad_t, pad_l,
    // pad_b, pad_r}, {dilation_h, dilation_w}
    // Regular
    conv_param_t<>(1, 16, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 16, 32, {30, 10}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}, {2, 2}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}, {3, 3}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}, {2, 2}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}, {2, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 2}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {2, 1, 2, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 2, 1, 2}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 2}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {2, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 16, 16, {10, 30}, 1, {3, 5}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 16, 16, {10, 30}, 1, {5, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 16, 16, {10, 30}, 1, {5, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}),
    // groupwise
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 16, {10, 30}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 16, 32, {10, 30}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {2, 2}, {2, 1, 2, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {1, 2}, {2, 1, 2, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {2, 1}, {2, 1, 2, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 5}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {5, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {5, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}),
    // DW
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 3}, {2, 2}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 3}, {1, 1}, {1, 2, 1, 2}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 3}, {2, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 3}, {1, 2}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {3, 5}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {5, 3}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {5, 5}, {1, 1}, {1, 1, 1, 1}),
    conv_param_t<>(1, 32, 32, {10, 30}, 32, {5, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}),
    conv_param_t<>(1, 128, 256, {32, 100}, 128, {3, 3}, {1, 1}, {1, 1, 1, 1}),
    // Pointwise
    conv_param_t<>(1, 32, 32, {10, 30}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
    conv_param_t<>(1, 16, 32, {10, 30}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {1, 1}, {2, 2}, {0, 0, 0, 0}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {1, 1}, {1, 2}, {0, 0, 0, 0}),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {1, 1}, {2, 1}, {0, 0, 0, 0}),

    // deconv shapes
    // MB, IC, OC, {IH, IW}, G, {KH, KW}, {stride_h, stride_w}, {pad_t, pad_l,
    // pad_b, pad_r}, {dilation_h, dilation_w}, {output_padding_h, output_padding_w}, transposed
    // Regular
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}, {2, 2}, {0, 0}, true),
    conv_param_t<>(1, 32, 16, {10, 30}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}, {0, 0}, true),
    // groupwise
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 16, {10, 30}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {1, 0}, true),
    conv_param_t<>(1, 16, 32, {10, 30}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1}, {0, 1}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {2, 2}, {2, 1, 2, 1}, {1, 1}, {1, 1}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {1, 2}, {2, 1, 2, 1}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 3}, {2, 1}, {2, 1, 2, 1}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {3, 5}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {5, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 32, 32, {10, 30}, 8, {5, 3}, {1, 1}, {1, 1, 1, 1}, {2, 2}, {0, 0}, true),
  };
  return shapes;
}
// clang-format on

namespace {

// tuple represents MB, IC, OC, IT, IH, IW, KH/KW, stride, pad
class uniConvTest
    : public testing::TestWithParam<
          tuple<int, int, int, int, int, int, int, int, int, int>> {};

// tuple represents QuantizationGranularity, A symmetric, B symmetric,
// test_bias, test_float_bias
class UniConvQGranTest
    : public testing::TestWithParam<
          tuple<QuantizationGranularity, bool, bool, bool, bool>> {};

}; // namespace

// Combine only allows at most 10 generators.
INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    uniConvTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2}), // MB
        ::testing::ValuesIn({16, 32}), // IC
        ::testing::ValuesIn({16, 32}), // OC
        ::testing::ValuesIn({17}), // IT
        ::testing::ValuesIn({10, 30}), // IH
        ::testing::ValuesIn({10, 30, 55}), // IW
        ::testing::ValuesIn({1, 4, 16}), // G
        ::testing::ValuesIn({1, 3, 5, 7}), // kernel
        ::testing::ValuesIn({1, 2}), // stride
        ::testing::ValuesIn({0, 1, 2}))); // pad

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    UniConvQGranTest,
    ::testing::Combine(
        ::testing::ValuesIn(qGranularityVals),
        ::testing::Bool(), // A symmetric
        ::testing::Bool(), // B symmetric
        ::testing::Bool(), // test_bias
        ::testing::Bool())); // test_float_bias
/**
 * Test for conv packing
 */
TEST_P(uniConvTest, packingTest) {
  int MB, IC, OC, IT, IH, IW, G, kernel, stride, pad;
  tie(MB, IC, OC, IT, IH, IW, G, kernel, stride, pad) = GetParam();

  conv_param_t<1> conv_p_1d(
      MB, IC, OC, {IW}, G, {kernel}, {stride}, {pad, pad});

  int kernel_dim_1d = kernel;
  aligned_vector<int8_t> Bint8_1d(
      kernel_dim_1d * conv_p_1d.IC * (conv_p_1d.OC / conv_p_1d.G));
  PackWeightsForConv<1> packedB_1D(conv_p_1d, Bint8_1d.data());

  switch (ConvFastPath<1, int32_t>(conv_p_1d)) {
    case optimized_conv_t::depthwise: {
      ASSERT_EQ(packedB_1D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_1D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix is null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_EQ(packedB_1D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_1D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix is null";
      break;
    }
    case optimized_conv_t::pointwise: {
      ASSERT_EQ(packedB_1D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should null";
      ASSERT_EQ(packedB_1D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix should be null";
      ASSERT_NE(packedB_1D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix is null";
      break;
    }
    case optimized_conv_t::fastpath1d: {
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_1D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_1D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_1D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix is null";
      break;
    }
  }

  conv_param_t<2> conv_p_2d(
      MB,
      IC,
      OC,
      {IH, IW},
      G,
      {kernel, kernel},
      {stride, stride},
      {pad, pad, pad, pad});

  int kernel_dim_2d = kernel * kernel;
  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  switch (ConvFastPath<2, int32_t>(conv_p_2d)) {
    case optimized_conv_t::depthwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix is null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix is null";
      break;
    }
    case optimized_conv_t::pointwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix is null";
      break;
    }
    case optimized_conv_t::fastpath1d: {
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_2D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix is null";
      break;
    }
  }

  conv_param_t<3> conv_p_3d(
      MB,
      IC,
      OC,
      {IT, IH, IW},
      G,
      {kernel, kernel, kernel},
      {stride, stride, stride},
      {pad, pad, pad, pad, pad, pad});

  int kernel_dim_3d = kernel * kernel * kernel;
  aligned_vector<int8_t> Bint8_3d(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));
  PackWeightsForConv<3> packedB_3D(conv_p_3d, Bint8_3d.data());

  switch (ConvFastPath<3, int32_t>(conv_p_3d)) {
    case optimized_conv_t::depthwise: {
      ASSERT_EQ(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix is null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_EQ(packedB_3D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix is null";
      break;
    }
    case optimized_conv_t::pointwise: {
      ASSERT_EQ(packedB_3D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix is null";
      break;
    }
    case optimized_conv_t::fastpath1d: {
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_3D.getPackedWForDepthwise(), nullptr)
          << "depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix is null";
      break;
    }
  }
}

/**
 * Test for packing/unpacking
 */
TEST_P(uniConvTest, packUnpackTest) {
  int MB, IC, OC, IT, IH, IW, G, kernel, stride, pad;
  tie(MB, IC, OC, IT, IH, IW, G, kernel, stride, pad) = GetParam();

  conv_param_t<1> conv_p_1d(
      MB, IC, OC, {IW}, G, {kernel}, {stride}, {pad, pad});

  int kernel_dim_1d = kernel;

  aligned_vector<int8_t> Bint8_1d(
      kernel_dim_1d * conv_p_1d.IC * (conv_p_1d.OC / conv_p_1d.G));
  aligned_vector<int8_t> Bint8_1d_unpacked(
      kernel_dim_1d * conv_p_1d.IC * (conv_p_1d.OC / conv_p_1d.G));

  PackWeightsForConv<1> packedB_1D(conv_p_1d, Bint8_1d.data());

  packedB_1D.unpack(Bint8_1d_unpacked.data());

  ASSERT_EQ(Bint8_1d, Bint8_1d_unpacked)
      << "Original and unpacked data elements are not the same [1D]";

  conv_param_t<2> conv_p_2d(
      MB,
      IC,
      OC,
      {IH, IW},
      G,
      {kernel, kernel},
      {stride, stride},
      {pad, pad, pad, pad});

  int kernel_dim_2d = kernel * kernel;

  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  aligned_vector<int8_t> Bint8_2d_unpacked(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));

  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  packedB_2D.unpack(Bint8_2d_unpacked.data());

  ASSERT_EQ(Bint8_2d_unpacked, Bint8_2d)
      << "Original and unpacked data elements are not the same [2D]";

  conv_param_t<3> conv_p_3d(
      MB,
      IC,
      OC,
      {IT, IH, IW},
      G,
      {kernel, kernel, kernel},
      {stride, stride, stride},
      {pad, pad, pad, pad, pad, pad});

  int kernel_dim_3d = kernel * kernel * kernel;

  aligned_vector<int8_t> Bint8_3d(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));

  aligned_vector<int8_t> Bint8_3d_unpacked(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));

  PackWeightsForConv<3> packedB_3D(conv_p_3d, Bint8_3d.data());

  packedB_3D.unpack(Bint8_3d_unpacked.data());

  ASSERT_EQ(Bint8_3d_unpacked, Bint8_3d)
      << "Original and unpacked data elements are not the same [3D]";
}

TEST(uniConvTest, cornerCases) {
  int stride = 1;
  conv_param_t<2> conv_p_2d(
      1, // mini-batch
      16, // input channels
      32, // output channels
      {28, 28}, // input height/width
      4, // groups
      {3, 3}, // kernel height/width
      {stride, stride}, // strides
      {1, 1, 1, 1}); // padding

  int kernel_dim_2d = conv_p_2d.K[0] * conv_p_2d.K[1];

  aligned_vector<uint8_t> Aint8(
      conv_p_2d.MB * conv_p_2d.IN_DIM[0] * conv_p_2d.IN_DIM[1] * conv_p_2d.IC);
  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  aligned_vector<int32_t> Cint32_fb(
      conv_p_2d.MB * conv_p_2d.OUT_DIM[0] * conv_p_2d.OUT_DIM[1] *
      conv_p_2d.OC);
  aligned_vector<uint8_t> Cint8_fb(Cint32_fb.size(), 0);

  // A matrix (input activations)
  randFill<uint8_t>(Aint8, 0, 5);
  int32_t Aint8_zero_point = 4;

  // B matrix (weights)
  randFill<int8_t>(Bint8_2d, -4, 4);
  aligned_vector<int32_t> Bint8_zero_point(1);
  randFill(Bint8_zero_point, -3, -1);

  aligned_vector<float> C_multiplier(Bint8_zero_point.size());
  randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
  int32_t C_zero_point = 5;

  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  vector<int32_t> col_offsets(conv_p_2d.OC);

  DoNothing<> doNothingObj{};
  ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
      doNothingObj,
      C_multiplier.data(),
      C_zero_point,
      Aint8_zero_point,
      Bint8_zero_point.data(),
      nullptr, // row offsets
      col_offsets.data(),
      nullptr, // bias
      conv_p_2d.OC,
      conv_p_2d.G);

  try {
    conv_p_2d.stride[0] = 2;
    fbgemmConv(
        conv_p_2d,
        Aint8.data(),
        packedB_2D,
        Cint8_fb.data(),
        Cint32_fb.data(),
        outputProcObj,
        0,
        1);
  } catch (std::logic_error const& err) {
    std::string s(err.what());
    EXPECT_TRUE(s.rfind("[FBGEMM_CONV_ERROR]", 0) == 0);
  }

  // reset
  conv_p_2d.stride[0] = stride;
  // this should run fine
  fbgemmConv(
      conv_p_2d,
      Aint8.data(),
      packedB_2D,
      Cint8_fb.data(),
      Cint32_fb.data(),
      outputProcObj,
      0,
      1);
}

/**
 * @brief Unit test for uint8 activations, int8 weights, and 32-bit
 * accumulation. Output processing: requantization -> nothing
 */

template <int SPATIAL_DIM = 2>
void runRequantizeTest(
    QuantizationGranularity q_granularity,
    bool a_symmetric,
    bool b_symmetric,
    bool test_bias,
    bool test_float_bias) {
  vector<conv_param_t<SPATIAL_DIM>> shapes(GetShapes_<SPATIAL_DIM>());

  for (auto conv_p : shapes) {
    int R = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
    int S = conv_p.K[SPATIAL_DIM - 1];
    int G = conv_p.G;
    int OC = conv_p.OC;
    int OH = SPATIAL_DIM == 1 ? 1 : conv_p.OUT_DIM[SPATIAL_DIM - 2];
    int OW = conv_p.OUT_DIM[SPATIAL_DIM - 1];
    int IC_per_G = conv_p.IC / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;
    int IH = SPATIAL_DIM == 1 ? 1 : conv_p.IN_DIM[SPATIAL_DIM - 2];
    int IW = conv_p.IN_DIM[SPATIAL_DIM - 1];

    // activations
    aligned_vector<uint8_t> Aint8(conv_p.MB * IH * IW * conv_p.IC, 0);

    // weights
    // The weight matrix is in layout G K/G (R S C/G)
    aligned_vector<int8_t> Bint8(R * S * G * IC_per_G * OC_per_G, 0);
    aligned_vector<int8_t> Bint8_tr(Bint8.size(), 0);

    aligned_vector<int32_t> Cint32_ref(conv_p.MB * OH * OW * OC, 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);

    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = a_symmetric ? 0 : 4;

    randFill<int8_t>(Bint8, -4, 4);

    // computing column offset
    vector<int32_t> col_offsets(OC);

    int ncols_per_quant_group = OC;
    if (q_granularity == QuantizationGranularity::GROUP) {
      ncols_per_quant_group = OC_per_G;
    } else if (q_granularity == QuantizationGranularity::OUT_CHANNEL) {
      ncols_per_quant_group = 1;
    }

    aligned_vector<int32_t> Bint8_zero_point(OC / ncols_per_quant_group);
    if (b_symmetric) {
      randFill(Bint8_zero_point, 0, 0);
    } else {
      randFill(Bint8_zero_point, -3, 3);
    }

    // matrix dimensions after im2col for each GEMM.
    // For each group, there is one GEMM of the following dimensions
    int MDim = conv_p.MB * OH * OW;
    int NDim = OC_per_G;
    int KDim = R * S * IC_per_G;

    vector<uint8_t> Aint8_im2col(MDim * KDim * G);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    vector<int32_t> row_offsets(MDim);

    // activation_scale * weight_scale
    aligned_vector<float> act_times_w_scale(Bint8_zero_point.size());
    randFill(act_times_w_scale, 0.1234f / 2, 0.1234f * 3 / 2);

    float out_scale = 2.0f;
    aligned_vector<float> C_multiplier(Bint8_zero_point.size());
    transform(
        act_times_w_scale.begin(),
        act_times_w_scale.end(),
        C_multiplier.begin(),
        [&out_scale](float i) { return i / out_scale; });

    int32_t C_zero_pt = 5;

    // initialize bias
    aligned_vector<int32_t> bias_int32(OC);
    aligned_vector<float> bias_fp32(OC);
    if (test_bias) {
      randFill(bias_int32, -8, 8);
    }

    // floating point bias
    if (test_float_bias) {
      if (q_granularity == QuantizationGranularity::TENSOR) {
        transform(
            bias_int32.begin(),
            bias_int32.end(),
            bias_fp32.begin(),
            [&act_times_w_scale](float i) { return i * act_times_w_scale[0]; });
      } else if (q_granularity == QuantizationGranularity::GROUP) {
        for (int g = 0; g < G; ++g) {
          for (int c = 0; c < OC_per_G; ++c) {
            bias_fp32[g * OC_per_G + c] = act_times_w_scale[g] *
                static_cast<float>(bias_int32[g * OC_per_G + c]);
          }
        }
      } else { // OUT_CHANNEL
        transform(
            act_times_w_scale.begin(),
            act_times_w_scale.end(),
            bias_int32.begin(),
            bias_fp32.begin(),
            multiplies<float>());
      }
    }
    // reference implementation
    // conv_ref expects weights to be in G (R S C/G) K/G
    transposeConvWeights(conv_p, Bint8.data(), Bint8_tr.data());
    int8_t* rightBData = Bint8_tr.data();
    for (int g = 0; g < G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          R * S * IC_per_G,
          OC_per_G,
          OC_per_G,
          rightBData + g * R * S * IC_per_G * OC_per_G,
          Bint8_zero_point.data() + g * OC_per_G / ncols_per_quant_group,
          col_offsets.data() + g * OC_per_G,
          ncols_per_quant_group);
    }
    conv_ref(
        conv_p, Aint8.data(), Aint8_zero_point, rightBData, Cint32_ref.data());

    for (int g = 0; g < G; ++g) {
      row_offsets_u8acc32_ref(
          MDim,
          KDim,
          KDim * G,
          Aint8_im2col.data() + g * KDim,
          row_offsets.data());

      requantize_u8acc32_ref(
          MDim,
          NDim,
          G * NDim,
          Cint32_ref.data() + g * NDim,
          Cint8_ref.data() + g * NDim,
          C_multiplier.data() + g * NDim / ncols_per_quant_group,
          C_zero_pt,
          Aint8_zero_point,
          Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
          row_offsets.data(),
          col_offsets.data() + g * NDim,
          test_bias ? bias_int32.data() + g * NDim : nullptr,
          ncols_per_quant_group);
    }

    PackWeightsForConv<SPATIAL_DIM> packedWeights(conv_p, Bint8.data());

    // TODO: Uncomment once we support multiple threads in fbgemmGroupwiseConv
    // #ifdef _OPENMP
    // #pragma omp parallel
    // #endif
    {
      vector<int32_t> row_offset_buf(rowOffsetBufferSizeGConv(conv_p));

      DoNothing<> doNothingObj{};

      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();

      if (q_granularity == QuantizationGranularity::TENSOR) {
        if (test_float_bias) {
          ReQuantizeOutput<false, QuantizationGranularity::TENSOR, float>
              reqObj(
                  doNothingObj,
                  C_multiplier.data(),
                  C_zero_pt,
                  Aint8_zero_point,
                  Bint8_zero_point.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  test_bias ? bias_fp32.data() : nullptr,
                  G * NDim,
                  G,
                  act_times_w_scale.data());

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);

        } else {
          ReQuantizeOutput<false, QuantizationGranularity::TENSOR> reqObj(
              doNothingObj,
              C_multiplier.data(),
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point.data(),
              nullptr, /* row offset buffer */
              col_offsets.data(),
              test_bias ? bias_int32.data() : nullptr,
              G * NDim,
              G);

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);
        }

      } else if (q_granularity == QuantizationGranularity::GROUP) {
        if (test_float_bias) {
          ReQuantizeOutput<false, QuantizationGranularity::GROUP, float> reqObj(
              doNothingObj,
              C_multiplier.data(),
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point.data(),
              nullptr, /* row offset buffer */
              col_offsets.data(),
              test_bias ? bias_fp32.data() : nullptr,
              G * NDim,
              G,
              act_times_w_scale.data());

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);

        } else {
          ReQuantizeOutput<false, QuantizationGranularity::GROUP> reqObj(
              doNothingObj,
              C_multiplier.data(),
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point.data(),
              nullptr, /* row offset buffer */
              col_offsets.data(),
              test_bias ? bias_int32.data() : nullptr,
              G * NDim,
              G);

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);
        }

      } else {
        if (test_float_bias) {
          ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL, float>
              reqObj(
                  doNothingObj,
                  C_multiplier.data(),
                  C_zero_pt,
                  Aint8_zero_point,
                  Bint8_zero_point.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  test_bias ? bias_fp32.data() : nullptr,
                  G * NDim,
                  G,
                  act_times_w_scale.data());

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);

        } else {
          ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL> reqObj(
              doNothingObj,
              C_multiplier.data(),
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point.data(),
              nullptr, /* row offset buffer */
              col_offsets.data(),
              test_bias ? bias_int32.data() : nullptr,
              G * NDim,
              G);

          fbgemmConv(
              conv_p,
              Aint8.data(),
              packedWeights,
              Cint8_fb.data(),
              Cint32_fb.data(),
              reqObj,
              tid,
              num_threads);
        }
      }
    } // omp parallel

    compare_validate_buffers(
        Cint8_ref.data(),
        Cint8_fb.data(),
        MDim,
        NDim * G,
        NDim * G,
        static_cast<uint8_t>(0));
  } // for each shape
}

TEST_P(UniConvQGranTest, requantizeTest) {
  QuantizationGranularity q_granularity;
  bool a_symmetric, b_symmetric;
  bool test_bias, test_float_bias;
  tie(q_granularity, a_symmetric, b_symmetric, test_bias, test_float_bias) =
      GetParam();

  runRequantizeTest<1>(
      q_granularity, a_symmetric, b_symmetric, test_bias, test_float_bias);
  runRequantizeTest<2>(
      q_granularity, a_symmetric, b_symmetric, test_bias, test_float_bias);
}

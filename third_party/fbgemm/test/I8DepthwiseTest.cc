/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"
#include "src/RefImplementations.h"

using namespace std;

namespace fbgemm {

// From Xray OCR
// clang-format off
static vector<vector<int>> shapes = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, G, H_in, W_in, stride, kernel
  {   1,  272,  47, 125, 1, 3 },
  {   1,  272,  47, 125, 1, 5 },
//  {   1,  272,  64, 125, 1, 3 },
//  {   1,  272,  66, 125, 1, 3 },
//  {   1,  272,  67, 100, 1, 3 },
//  {   1,  272,  75,  75, 1, 3 },
//   {   1,  272,  75,  76, 1, 3 },
//  {   1,  272,  75, 100, 1, 3 },
//  {   1,  272,  94,  75, 1, 3 },
//  {   1,  272, 109,  75, 1, 3 },
  {   1,  544,  24,  63, 1, 3 },
//  {   1,  544,  33,  63, 1, 3 },
//  {   1,  544,  34,  50, 1, 3 },
//  {   1,  544,  36,  63, 1, 3 },
//  {   1,  544,  38,  38, 1, 3 },
//  {   1,  544,  38,  40, 1, 3 },
  {   1,  544,  47,  38, 1, 3 },
  {   1, 1088,   7,   7, 1, 3 },
  {  2, 1088,   7,   7, 1, 3 },
  {   2, 1088,   7,   7, 1, 5 },
//  { 100, 1088,   7,   7, 1, 3 },

  {   1,  248,  93, 250, 2, 3 },
  {   1,  248,  93, 250, 2, 5 },
//  {   1,  248, 128, 250, 2, 3 },
//  {   1,  248, 133, 200, 2, 3 },
//  {   1,  248, 150, 150, 2, 3 },
  {   1,  248, 150, 151, 2, 3 },
//  {   1,  248, 150, 158, 2, 3 },
//  {   1,  248, 188, 150, 2, 3 },
//  {   1,  248, 225, 150, 2, 3 },
  {   1,  272,  47, 125, 2, 3 },
//  {   1,  272,  64, 125, 2, 3 },
//  {   1,  272,  66, 125, 2, 3 },
//  {   1,  272,  67, 100, 2, 3 },
//  {   1,  272,  75,  75, 2, 3 },
//  {   1,  272,  75,  76, 2, 3 },
  {   1,  272,  94,  75, 2, 3 },
  {   1,  544,  14,  14, 2, 3 },
  // {  51,  544,  14,  14, 2, 3 },
//  { 100,  544,  14,  14, 2, 3 },

  {   1,    8,   4,   4, 1, 3 },
  // Tests for the shapes when OH/OW is less than padding
  {   1,  72,  1, 1, 2, 5 },
  {   1,  72,  7, 1, 2, 5 },
  {   1,  72,  1, 7, 2, 5 },
};

static vector<vector<int>> shapes_3d = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, T_in, H_in, W_in, stride_t, stride_h, stride_w, K_T, K_H, K_W
  {   1,  32,   16,  28, 28, 1, 1, 1, 3, 3, 3, },
  {   1, 128,    8,  14, 14, 2, 2, 2, 3, 3, 3, },
  {   5,  16,   32,  56, 56, 1, 1, 1, 3, 3, 3, },
  {   1,   8,    4,   4,  4, 1, 1, 1, 3, 3, 3, },
  {   1,  32,   16,  28, 28, 1, 1, 1, 3, 5, 5, },
  {   1,  32,   16,  28, 28, 1, 2, 2, 3, 5, 5, },
  {   1,  32,   16,  28, 28, 1, 1, 1, 5, 5, 5, },
};
// clang-format on

namespace {

class FBGemmDepthWiseTest
    : public testing::TestWithParam<tuple<bool, bool, int>> {};

class FBGemmDepthWisePerChannelQuantizationTest
    : public testing::TestWithParam<int> {};

// Two parameters are K (or Groups) and kernel_prod, i.e.,
// (output_channels)(kernel_prod)
// output_channels == Groups.
// For example, kernel_prod for 3x3 convolution is 9
class FBGemmDepthWisePackUnpackTest
    : public testing::TestWithParam<tuple<int, int>> {};

} // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWiseTest,
    ::testing::Combine(
        ::testing::Bool(), // a_symmetric
        ::testing::Bool(), // b_symmetric
        ::testing::Values(1, 2))); // oc_per_g

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWisePerChannelQuantizationTest,
    ::testing::Values(1, 2));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWisePackUnpackTest,
    ::testing::Combine(
        ::testing::Values(8, 16, 24, 32, 40, 64, 72),
        ::testing::Values(1, 2, 3, 4, 5, 9, 10, 11, 27)));

TEST_P(FBGemmDepthWiseTest, Test2D) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  for (auto shape : shapes) {
    int N = shape[0];
    int G = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    int R = shape[5];
    int S = R;
    int PAD_T = (R - 1) / 2, PAD_B = (R - 1) / 2, PAD_L = (S - 1) / 2,
        PAD_R = (S - 1) / 2;
    int OC = G * oc_per_g;

    conv_param_t<2> conv_p(
        N,
        G,
        OC,
        {H, W},
        G,
        {R, S},
        {stride_h, stride_w},
        {PAD_T, PAD_L, PAD_B, PAD_R});
    int H_OUT = conv_p.OUT_DIM[0];
    int W_OUT = conv_p.OUT_DIM[1];

    int MDim = N * H_OUT * W_OUT;
    int KDim = R * S * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = a_symmetric ? 0 : 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = b_symmetric ? 0 : 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<int32_t> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col;
    if (!b_symmetric) {
      A_im2col.resize(MDim * KDim);
      im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());
    }

    // reference implementation conv_ref expects weights to be in G (R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      if (!b_symmetric) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            A_im2col.data() + g * KDimPerGroup,
            row_offsets.data());
      }

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_uint8_ref.data() + g * oc_per_g,
          C_multiplier.data(),
          C_zero_point,
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          OC);
    }

    PackedDepthWiseConvMatrix Bp(OC, R * S, B.data());
    depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
        N,
        H,
        W,
        G,
        OC,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        &B_zero_point,
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        a_symmetric ? nullptr : col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < OC; ++k) {
            int32_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            int32_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * OC + k];
            EXPECT_EQ(actual, expected)
                << "Depthwise " << R << "x" << S << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3

TEST_P(FBGemmDepthWiseTest, Test3D) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  // 3D tests take a long time so for a symmetric quantization, we only
  // test with 2 shapes.
  for (auto shape : shapes_3d) {
    int N = shape[0];
    int G = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = shape[6];
    int stride_w = shape[7];
    int K_T = shape[8];
    int K_H = shape[9];
    int K_W = shape[10];
    int PAD_P = (K_T - 1) / 2, PAD_N = PAD_P, PAD_T = (K_H - 1) / 2,
        PAD_B = PAD_T, PAD_L = (K_W - 1) / 2, PAD_R = PAD_L;
    int OC = G * oc_per_g;

    conv_param_t<3> conv_p(
        N,
        G,
        OC,
        {T, H, W},
        G,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * T * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = a_symmetric ? 0 : 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = b_symmetric ? 0 : 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<int32_t> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col;
    if (!b_symmetric) {
      A_im2col.resize(MDim * KDim);
      im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());
    }

    // reference implementation conv_ref expects weights to be in G (T R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      if (!b_symmetric) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            A_im2col.data() + g * KDimPerGroup,
            row_offsets.data());
      }

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_uint8_ref.data() + g * oc_per_g,
          C_multiplier.data(),
          C_zero_point,
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          OC);
    }

    PackedDepthWiseConvMatrix Bp(OC, K_T * K_H * K_W, B.data());

    depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
        conv_p,
        A_zero_point,
        A.data(),
        &B_zero_point,
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        a_symmetric ? nullptr : col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int k = 0; k < OC; ++k) {
              int32_t expected = C_uint8_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              int32_t actual =
                  C_uint8[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              EXPECT_EQ(actual, expected)
                  << "Depthwise 3D results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3D

TEST_P(
    FBGemmDepthWisePerChannelQuantizationTest,
    Test2DPerChannelQuantization) {
  int oc_per_g = GetParam();

  for (auto shape : shapes) {
    int N = shape[0];
    int G = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    int R = shape[5];
    int S = R;
    int PAD_T = (R - 1) / 2, PAD_B = (R - 1) / 2, PAD_L = (S - 1) / 2,
        PAD_R = (S - 1) / 2;
    int OC = G * oc_per_g;

    conv_param_t<2> conv_p(
        N,
        G,
        OC,
        {H, W},
        G,
        {R, S},
        {stride_h, stride_w},
        {PAD_T, PAD_L, PAD_B, PAD_R});
    int H_OUT = conv_p.OUT_DIM[0];
    int W_OUT = conv_p.OUT_DIM[1];

    int MDim = N * H_OUT * W_OUT;
    int KDim = R * S * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(OC);
    for (auto k = 0; k < OC; ++k) {
      aligned_vector<int8_t> Bk(R * S);
      // limit min, max to int8_t range
      randFill<int8_t>(Bk, -16 + k % 112, 16 + k % 112);
      copy(Bk.begin(), Bk.end(), B.begin() + k * R * S);

      B_zero_point[k] = 5 + k;
    }

    aligned_vector<float> C_multiplier(OC);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<int32_t> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    // im2col to compute row offset later
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

    // reference implementation conv_ref expects weights to be in G (R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          A_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_uint8_ref.data() + g * oc_per_g,
          C_multiplier.data() + g * oc_per_g,
          C_zero_point,
          A_zero_point,
          B_zero_point.data() + g * oc_per_g,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          1);
    }

    PackedDepthWiseConvMatrix Bp(OC, R * S, B.data());
    depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
        N,
        H,
        W,
        G,
        OC,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B_zero_point.data(),
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < OC; ++k) {
            int32_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            int32_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * OC + k];
            EXPECT_EQ(actual, expected)
                << "Depthwise " << R << "x" << S << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3PerChannelQuantization

TEST_P(
    FBGemmDepthWisePerChannelQuantizationTest,
    Test3DPerChannelQuantization) {
  int oc_per_g = GetParam();

  for (auto shape : shapes_3d) {
    int N = shape[0];
    int G = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = shape[6];
    int stride_w = shape[7];
    int K_T = shape[8];
    int K_H = shape[9];
    int K_W = shape[10];
    int PAD_P = (K_T - 1) / 2, PAD_N = PAD_P, PAD_T = (K_H - 1) / 2,
        PAD_B = PAD_T, PAD_L = (K_W - 1) / 2, PAD_R = PAD_L;
    int OC = G * oc_per_g;

    conv_param_t<3> conv_p(
        N,
        G,
        OC,
        {T, H, W},
        G,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * T * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(OC);
    for (auto k = 0; k < OC; ++k) {
      aligned_vector<int8_t> Bk(K_T * K_H * K_W);
      // limit min, max to int8_t range
      randFill<int8_t>(Bk, -16 + k % 112, 16 + k % 112);
      copy(Bk.begin(), Bk.end(), B.begin() + k * K_T * K_H * K_W);

      B_zero_point[k] = 5 + k;
    }

    aligned_vector<float> C_multiplier(OC);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<int32_t> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

    // reference implementation conv_ref expects weights to be in G (T R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          A_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_uint8_ref.data() + g * oc_per_g,
          C_multiplier.data() + g * oc_per_g,
          C_zero_point,
          A_zero_point,
          B_zero_point.data() + g * oc_per_g,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          1);
    }

    PackedDepthWiseConvMatrix Bp(OC, K_T * K_H * K_W, B.data());

    depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
        conv_p,
        A_zero_point,
        A.data(),
        B_zero_point.data(),
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int k = 0; k < OC; ++k) {
              int32_t expected = C_uint8_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              int32_t actual =
                  C_uint8[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              ASSERT_EQ(actual, expected)
                  << "Depthwise 3D results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3DPerChannelQuantization

TEST_P(FBGemmDepthWisePackUnpackTest, TestPackUnpack) {
  int K, kernel_prod;
  tie(K, kernel_prod) = GetParam();

  ASSERT_EQ(K % 8, 0)
      << "output channels (== groups) should be a multiple of 8";
  aligned_vector<int8_t> B(K * kernel_prod);
  randFill<int8_t>(B, -16, 16);

  aligned_vector<int8_t> BUnpacked(K * kernel_prod);

  PackedDepthWiseConvMatrix BPacked(K, kernel_prod, B.data());
  BPacked.unpack(BUnpacked.data());

  ASSERT_EQ(BUnpacked, B)
      << "Original and unpacked data elements are not the same";
} // TestPackUnpack

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "./QuantizationHelpers.h"
#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

vector<matrix_op_t> transposeVals{
    matrix_op_t::NoTranspose,
    matrix_op_t::Transpose};

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::GROUP,
    QuantizationGranularity::OUT_CHANNEL};

namespace {
// class fbgemmGConvAcc32Test
//     : public testing::TestWithParam<tuple<matrix_op_t, matrix_op_t>> {};
class fbgemmGConvAcc32WithQuantGranularityTest
    : public testing::TestWithParam<tuple<
          matrix_op_t,
          matrix_op_t,
          QuantizationGranularity,
          bool,
          bool>> {};
class fbgemmGConvPackTest : public testing::TestWithParam<matrix_op_t> {};
}; // namespace

// INSTANTIATE_TEST_CASE_P(
//     InstantiationName,
//     fbgemmGConvAcc32Test,
//     ::testing::Combine(
//         ::testing::Values(matrix_op_t::NoTranspose),
//         ::testing::ValuesIn(transposeVals)));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmGConvAcc32WithQuantGranularityTest,
    ::testing::Combine(
        ::testing::Values(matrix_op_t::NoTranspose),
        ::testing::ValuesIn(transposeVals),
        ::testing::ValuesIn(qGranularityVals),
        ::testing::Bool(), // A symmetric
        ::testing::Bool())); // B symmetric

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmGConvPackTest,
    ::testing::ValuesIn(transposeVals));
/**
 * @brief 3D Shapes for unit test.
 */
template <int SPATIAL_DIM>
static typename std::enable_if<SPATIAL_DIM == 3, vector<conv_param_t<3>>>::type
GetShapes_() {
  // clang-format off
  vector<conv_param_t<3>> shapes = {
    // MB, IC, OC, {IT, IH, IW}, G, {KT, KH, KW},
    // {stride_t, stride_h, stride_w}, {pad_p, pad_t, pad_l,
    //                                  pad_n, pad_b, pad_r}
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 8, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 4, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 2, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 4, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 2, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 5}, 1, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),

    conv_param_t<3>(1, 16, 16, {4, 4, 4}, 8, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {4, 4, 4}, 4, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {4, 4, 4}, 2, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {4, 4, 4}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {6, 6, 6}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {6, 6, 6}, 4, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {6, 6, 6}, 2, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {6, 6, 6}, 1, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),

    // batch size > 1
    conv_param_t<3>(2, 16, 16, {4, 4, 4}, 8, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {4, 4, 4}, 4, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {4, 4, 4}, 2, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {4, 4, 4}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {6, 6, 6}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {6, 6, 6}, 4, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {6, 6, 6}, 2, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(2, 16, 16, {6, 6, 6}, 1, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),

    conv_param_t<3>(1, 16, 16, {1, 4, 4}, 8, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 4, 4}, 4, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 4, 4}, 2, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 4, 4}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 4, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 2, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 1, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),

    // unequal stride
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 8, {3, 3, 3},
        {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 4, {3, 3, 3},
        {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 2, {3, 3, 3},
        {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {1, 6, 6}, 1, {3, 3, 3},
        {1, 2, 2}, {1, 1, 1, 1, 1, 1}),

    // Small H and W corner cases
    conv_param_t<3>(1, 16, 16, {5, 5, 2}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 2, 5}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 1}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 1, 5}, 1, {3, 3, 3},
        {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 6}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 6, 5}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 5, 3}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 3, 5}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
    conv_param_t<3>(1, 16, 16, {5, 3, 3}, 8, {3, 3, 3},
        {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  };
  return shapes;
  // clang-format off
}

/**
 * @brief 2D Shapes for unit test.
 */
template <int SPATIAL_DIM = 2>
static typename std::enable_if<SPATIAL_DIM == 2, vector<conv_param_t<2>>>::type
GetShapes_() {
    vector<conv_param_t<>> shapes = {
        // MB, IC, OC, {IH, IW}, G, {KH, KW}, {stride_h, stride_w},
        // {pad_t, pad_l, pad_b, pad_r}
        conv_param_t<>(1, 16, 16, {5, 5}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 4, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 2, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 4, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 2, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 5}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 32, 32, {3, 3}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 32, 32, {4, 4}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 32, 32, {3, 5}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 32, 32, {5, 3}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // fix from 8 to 16 to address G_together > G for avx512
        conv_param_t<>(1, 16, 16, {5, 5}, 2, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 128, 128, {56, 48}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 128, 128, {48, 56}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // the line below is from resnext101-32x4d
        conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(2, 128, 128, {56, 56}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // Small H and W corner cases
        conv_param_t<>(1, 16, 16, {5, 2}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {2, 5}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 1}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {1, 5}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 6}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {6, 5}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 4}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {4, 5}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {4, 4}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {5, 2}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {2, 2}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),
        conv_param_t<>(1, 16, 16, {1, 1}, 8, {3, 3}, {2, 2}, {1, 1, 1, 1}),

        // The following lines are commented to reduce test time but still valid
        // when we want more extensive testings.
        // conv_param_t<>(1, 64, 64, {3, 3}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // conv_param_t<>(1, 64, 64, {4, 4}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // conv_param_t<>(1, 64, 64, {3, 5}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // conv_param_t<>(1, 64, 64, {5, 3}, 8, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // conv_param_t<>(1, 16, 16, {5, 5}, 2, {3, 3}, {1, 1}, {1, 1, 1, 1}),
        // conv_param_t<>(1, 256, 256, {56, 48}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 256, 256, {48, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 256, 256, {56, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(2, 256, 256, {56, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),

        // conv_param_t<>(1, 128, 128, {3, 3}, 8, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 128, 128, {4, 4}, 8, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 128, 128, {3, 5}, 8, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 128, 128, {5, 3}, 8, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 32, 32, {5, 5}, 2, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 512, 512, {56, 48}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 512, 512, {48, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(1, 512, 512, {56, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
        // conv_param_t<>(2, 512, 512, {56, 56}, 32, {3, 3}, {1, 1},
        // {1, 1, 1, 1}),
    };
    return shapes;
}

/**
 * @brief Unit test for uint8 activations, int8 weights, and 32-bit
 * accumulation. Output processing: requantization -> nothing
 */
template <int SPATIAL_DIM = 2>
void runRequantizeTest(matrix_op_t /* unused */,
    matrix_op_t btrans,
    QuantizationGranularity q_granularity,
    bool a_symmetric, bool b_symmetric) {
  vector<conv_param_t<SPATIAL_DIM>> shapes(GetShapes_<SPATIAL_DIM>());
  for (auto conv_p : shapes) {
    int T = SPATIAL_DIM <= 2 ? 1 : conv_p.K[SPATIAL_DIM - 3];
    int R = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
    int S = conv_p.K[SPATIAL_DIM - 1];
    int G = conv_p.G;
    int OC = conv_p.OC;
    int IT = SPATIAL_DIM <= 2 ? 1 : conv_p.IN_DIM[SPATIAL_DIM - 3];
    int IH = SPATIAL_DIM == 1 ? 1 : conv_p.IN_DIM[SPATIAL_DIM - 2];
    int IW = conv_p.IN_DIM[SPATIAL_DIM - 1];
    int OT = SPATIAL_DIM <= 2 ? 1 : conv_p.OUT_DIM[SPATIAL_DIM - 3];
    int OH = SPATIAL_DIM == 1 ? 1 : conv_p.OUT_DIM[SPATIAL_DIM - 2];
    int OW = conv_p.OUT_DIM[SPATIAL_DIM - 1];
    int IC_per_G = conv_p.IC / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    // activations
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * IT * IH *IW * conv_p.IC, 0);

    // weights
    // when btrans == Transpose, the weight matrix is
    // in layout G K/G (T R S C/G) instead of G (T R S C/G) K/G
    aligned_vector<int8_t> Bint8(T * R * S * G * IC_per_G * OC_per_G, 0);
    aligned_vector<int8_t> Bint8_tr(Bint8.size(), 0);

    aligned_vector<int32_t> Cint32_ref(conv_p.MB *OT *OH * OW * OC, 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);

    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = a_symmetric ? 0 : 4;

    randFill<int8_t>(Bint8, -4, 4);

    // computing column offset
    vector<int32_t> col_offsets(G * OC_per_G);

    int ncols_per_quant_group = G * OC_per_G;
    if (q_granularity == QuantizationGranularity::GROUP) {
      ncols_per_quant_group = OC_per_G;
    } else if (q_granularity == QuantizationGranularity::OUT_CHANNEL) {
      ncols_per_quant_group = 1;
    }

    aligned_vector<int32_t> Bint8_zero_point(
        G * OC_per_G / ncols_per_quant_group);
    if (b_symmetric) {
      randFill(Bint8_zero_point, 0, 0);
    } else {
      randFill(Bint8_zero_point, -3, -1);
    }

    // matrix dimensions after im2col for each GEMM.
    // For each group, there is one GEMM of the following dimensions
    int MDim = conv_p.MB * OT * OH * OW;
    int NDim = OC_per_G;
    int KDim = T * R * S * IC_per_G;

    vector<uint8_t> Aint8_im2col(MDim * KDim * G);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    vector<int32_t> row_offsets(MDim);

    aligned_vector<float> C_multiplier(Bint8_zero_point.size());
    randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
    int32_t C_zero_pt = 5;

    // reference implementation
    // conv_ref expects weights to be in G (T R S C/G) K/G
    int8_t* rightBData = Bint8.data();
    if (btrans == matrix_op_t::Transpose) {
      transposeConvWeights(conv_p, Bint8.data(), Bint8_tr.data());
      rightBData = Bint8_tr.data();
    }
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
          nullptr,
          ncols_per_quant_group);
    }

    PackWeightMatrixForGConv<int8_t, int32_t, SPATIAL_DIM> packedWeights(
        btrans, conv_p, Bint8.data(), nullptr);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      vector<int32_t> row_offset_buf(rowOffsetBufferSizeGConv(conv_p));

      DoNothing<> doNothingObj{};

      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();

      if (q_granularity == QuantizationGranularity::TENSOR) {
        ReQuantizeOutput<false, QuantizationGranularity::TENSOR> reqObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            Bint8_zero_point[0] ? row_offset_buf.data() : nullptr,
            col_offsets.data(),
            nullptr,
            G * NDim,
            G);

        fbgemmGroupwiseConv(
            conv_p,
            Aint8.data(),
            Aint8_zero_point,
            Bint8_zero_point[0] ? row_offset_buf.data() : nullptr,
            packedWeights,
            Cint8_fb.data(),
            Cint32_fb.data(),
            reqObj,
            tid,
            num_threads);
      } else if (q_granularity == QuantizationGranularity::GROUP) {
        ReQuantizeOutput<false, QuantizationGranularity::GROUP> reqObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            row_offset_buf.data(),
            col_offsets.data(),
            nullptr,
            G * NDim,
            G);

        fbgemmGroupwiseConv(
            conv_p,
            Aint8.data(),
            Aint8_zero_point,
            row_offset_buf.data(),
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
            row_offset_buf.data(),
            col_offsets.data(),
            nullptr,
            G * NDim,
            G);

        fbgemmGroupwiseConv(
            conv_p,
            Aint8.data(),
            Aint8_zero_point,
            row_offset_buf.data(),
            packedWeights,
            Cint8_fb.data(),
            Cint32_fb.data(),
            reqObj,
            tid,
            num_threads);
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

TEST_P(fbgemmGConvAcc32WithQuantGranularityTest, requantizeTest) {
  matrix_op_t atrans, btrans;
  QuantizationGranularity q_granularity;
  bool a_symmetric, b_symmetric;

  tie(atrans, btrans, q_granularity, a_symmetric, b_symmetric) = GetParam();

  runRequantizeTest<2>(atrans, btrans, q_granularity, a_symmetric, b_symmetric);
  runRequantizeTest<3>(atrans, btrans, q_granularity, a_symmetric, b_symmetric);
}

/**
 * @brief Unit test for uint8 activations, int8 weights, and 32-bit
 * accumulation. Output processing: nothing
 */
/*
TEST_P(fbgemmGConvAcc32Test, NoRequantizeTest) {
  vector<conv_param_t<>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  tie(atrans, btrans) = GetParam();

  for (auto conv_p : shapes) {
    int R = conv_p.K[0];
    int S = conv_p.K[1];
    int G = conv_p.G;
    int OC = conv_p.OC;
    int OH = conv_p.OUT_DIM[0];
    int OW = conv_p.OUT_DIM[1];
    int IC_per_G = conv_p.IC / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    // activations
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC, 0);

    // weights
    // when btrans == Transpose, the weight matrix is in layout G K/G (R S C/G)
    // instead of G (R S C/G) K/G
    aligned_vector<int8_t> Bint8(R * S * conv_p.G * IC_per_G * OC_per_G, 0);
    aligned_vector<int8_t> Bint8_tr(R * S * conv_p.G * IC_per_G * OC_per_G, 0);

    aligned_vector<int32_t> Cint32_ref(conv_p.MB * OH * OW * OC, 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);

    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = 4;

    randFill<int8_t>(Bint8, -4, 4);

    // matrix dimensions after im2col for each GEMM.
    // For each group, there is one GEMM of the following dimensions
    int MDim = conv_p.MB * OH * OW;
    int NDim = OC_per_G;
    // int KDim = R * S * IC_per_G;

    // reference implementation
    // conv_ref expects weights to be in G (R S C/G) K/G
    int8_t* rightBData = Bint8.data();
    if (btrans == matrix_op_t::Transpose) {
      transposeConvWeights(conv_p, Bint8.data(), Bint8_tr.data());
      rightBData = Bint8_tr.data();
    }
    conv_ref(
        conv_p, Aint8.data(), Aint8_zero_point, rightBData, Cint32_ref.data());

    PackWeightMatrixForGConv<int8_t> packedWeights(
        btrans, conv_p, Bint8.data(), nullptr);

    // TODO: Uncomment once we support multiple threads in fbgemmGroupwiseConv
    // #ifdef _OPENMP
    // #pragma omp parallel
    // #endif
    {
      vector<int32_t> row_offset_buf(rowOffsetBufferSizeGConv(conv_p));

      DoNothing<int32_t, int32_t> doNothingObj{};

      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();

      fbgemmGroupwiseConv(
          conv_p,
          Aint8.data(),
          Aint8_zero_point,
          row_offset_buf.data(),
          packedWeights,
          Cint32_fb.data(),
          Cint32_fb.data(),
          doNothingObj,
          tid,
          num_threads);
    }

    compare_validate_buffers(
        Cint32_ref.data(),
        Cint32_fb.data(),
        MDim,
        NDim * G,
        NDim * G,
        static_cast<int32_t>(0));
  } // for each shape
}
*/

template <int SPATIAL_DIM = 2>
void runPackUnpackTest(matrix_op_t btrans) {
  vector<conv_param_t<SPATIAL_DIM>> shapes(GetShapes_<SPATIAL_DIM>());

  for (auto conv_p : shapes) {
    int T = SPATIAL_DIM <= 2 ? 1 : conv_p.K[SPATIAL_DIM - 3];
    int R = SPATIAL_DIM == 1 ? 1 : conv_p.K[SPATIAL_DIM - 2];
    int S = conv_p.K[SPATIAL_DIM - 1];
    int IC_per_G = conv_p.IC / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    // Weights -- test the packing/unpacking of only the weights
    // when btrans == Transpose, the weight matrix is in
    // layout G K/G (T R S C/G) instead of G (T R S C/G) K/G
    int weight_len = T * R * S * conv_p.G * IC_per_G * OC_per_G;
    aligned_vector<int8_t> Bint8(weight_len, 0);

    // Random fill the weights
    randFill<int8_t>(Bint8, -4, 4);

    // Instantiate the object
    PackWeightMatrixForGConv<int8_t, int32_t, SPATIAL_DIM> packedWeights(
        btrans, conv_p, Bint8.data(), nullptr);

    // Setup a buffer to get pack -> unpacked results
    aligned_vector<int8_t> unpack_buf(weight_len, 0);

    // START Actual pack-unpack operations
    // Perform packing first. This should populate pdata_ of packedWeights
    packedWeights.pack();

    // Next perform unpacking
    packedWeights.unpack(unpack_buf.data());
    // END actual pack-unpack operations

    // Sanity check
    for (int i = 0; i < weight_len; ++i) {
      EXPECT_EQ(unpack_buf.data()[i], Bint8.data()[i])
        << "Pack/Unpack results differ at index " << i
        << ", Reference: " << static_cast<int>(Bint8.data()[i])
        << ", Pack-Unpacked: " << static_cast<int>(unpack_buf.data()[i]);
    }
  } // for each shape
}

/**
 * @brief Unit test for packing and unpacking the weight tensor
 */
TEST_P(fbgemmGConvPackTest, PackUnpackTest) {
  matrix_op_t btrans = GetParam();
  runPackUnpackTest<2>(btrans);
  runPackUnpackTest<3>(btrans);
}

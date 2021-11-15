/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"

using namespace std;
using namespace fbgemm;

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::OUT_CHANNEL};

namespace {

// tuple represents #rows, #cols, fuse_relu, quantization_granularity, bias_type
class FloatRequantizeTest
    : public testing::TestWithParam<
          tuple<int, int, bool, QuantizationGranularity>> {};

}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FloatRequantizeTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 3, 4}), // number of rows
        ::testing::ValuesIn(
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 32}), // number of
                                                                  // cols
        ::testing::Bool(), // fuse relu
        ::testing::ValuesIn(qGranularityVals))); // requantization granularity

/**
 * Test for float bias
 */
TEST_P(FloatRequantizeTest, floatBiasTest) {
  int rows, cols;
  bool fuse_relu;
  QuantizationGranularity q_gran;
  tie(rows, cols, fuse_relu, q_gran) = GetParam();

  int numElements = rows * cols;

  aligned_vector<float> act_times_w_scale(cols);
  randFill<float>(act_times_w_scale, -8, 8);

  float out_scale = 2.0f;

  aligned_vector<float> C_multiplier(cols);
  transform(
      act_times_w_scale.begin(),
      act_times_w_scale.end(),
      C_multiplier.begin(),
      [&out_scale](float i) { return i / out_scale; });

  aligned_vector<int32_t> Bint8_zero_point(cols);
  randFill<int32_t>(Bint8_zero_point, -8, 8);

  aligned_vector<int32_t> row_offset_buf(rows);
  randFill<int32_t>(row_offset_buf, -8, 8);

  aligned_vector<int32_t> col_offsets(cols);
  randFill<int32_t>(col_offsets, -8, 8);

  // quantized bias
  aligned_vector<int32_t> bias_q(cols);
  randFill<int32_t>(bias_q, -8, 8);

  // floating point bias
  aligned_vector<float> bias_f(cols);
  if (q_gran == QuantizationGranularity::TENSOR) {
    transform(
        bias_q.begin(),
        bias_q.end(),
        bias_f.begin(),
        [&act_times_w_scale](float i) { return i * act_times_w_scale[0]; });
  } else if (q_gran == QuantizationGranularity::OUT_CHANNEL) {
    transform(
        act_times_w_scale.begin(),
        act_times_w_scale.end(),
        bias_q.begin(),
        bias_f.begin(),
        multiplies<float>());

  } else {
    FAIL();
  }

  aligned_vector<int32_t> input(numElements);
  randFill<int32_t>(input, -8, 8);

  aligned_vector<uint8_t> output_q_bias(numElements);
  aligned_vector<uint8_t> output_f_bias(numElements);

  int32_t C_zero_point = 3;
  int32_t Aint8_zero_point = 3;

  block_type_t block{0, rows, 0, cols};

  DoNothing<> doNothingObj{};

#define TESTCODE(FUSE_RELU, Q_GRAN)                           \
  ReQuantizeOutput<FUSE_RELU, Q_GRAN> reqObj_q(               \
      doNothingObj,                                           \
      C_multiplier.data(),                                    \
      C_zero_point,                                           \
      Aint8_zero_point,                                       \
      Bint8_zero_point.data(),                                \
      row_offset_buf.data(),                                  \
      col_offsets.data(),                                     \
      bias_q.data(),                                          \
      cols);                                                  \
  ReQuantizeOutput<FUSE_RELU, Q_GRAN, float> reqObj_f(        \
      doNothingObj,                                           \
      C_multiplier.data(),                                    \
      C_zero_point,                                           \
      Aint8_zero_point,                                       \
      Bint8_zero_point.data(),                                \
      row_offset_buf.data(),                                  \
      col_offsets.data(),                                     \
      bias_f.data(),                                          \
      cols,                                                   \
      1,                                                      \
      act_times_w_scale.data());                              \
  reqObj_q.f<inst_set_t::avx2>(                               \
      output_q_bias.data(), input.data(), block, cols, cols); \
  reqObj_f.f<inst_set_t::avx2>(                               \
      output_f_bias.data(), input.data(), block, cols, cols);

  if (fuse_relu) {
    if (q_gran == QuantizationGranularity::TENSOR) {
      TESTCODE(true, QuantizationGranularity::TENSOR)

    } else if (q_gran == QuantizationGranularity::OUT_CHANNEL) {
      TESTCODE(true, QuantizationGranularity::OUT_CHANNEL)

    } else {
      FAIL();
    }

  } else {
    if (q_gran == QuantizationGranularity::TENSOR) {
      TESTCODE(false, QuantizationGranularity::TENSOR)

    } else if (q_gran == QuantizationGranularity::OUT_CHANNEL) {
      TESTCODE(false, QuantizationGranularity::OUT_CHANNEL)

    } else {
      FAIL();
    }
  }
#undef TESTCODE
  ASSERT_EQ(output_q_bias, output_f_bias)
      << "Requantization with quantized bias and float bias differs";
}

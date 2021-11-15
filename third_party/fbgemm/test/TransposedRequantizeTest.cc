/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/spmmUtils.h"

using namespace std;
using namespace fbgemm;

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::OUT_CHANNEL};

namespace {

// tuple represents #rows, #cols, fuse_relu, quantization_granularity
class RequantizeTest : public testing::TestWithParam<
                           tuple<int, int, bool, bool, QuantizationGranularity>>
                           {};

}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    RequantizeTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 32}), // number of
                                                                  // rows
        ::testing::ValuesIn({1, 2, 3, 4, 16, 31, 32}), // number of columns
        ::testing::Bool(), // fuse relu
        ::testing::Bool(), // use bias
        ::testing::ValuesIn(qGranularityVals))); // requantization granularity

TEST_P(RequantizeTest, reqTest) {
  int rows, cols;
  bool fuse_relu;
  bool use_bias;
  QuantizationGranularity q_gran;
  tie(rows, cols, fuse_relu, use_bias, q_gran) = GetParam();

  int numElements = rows * cols;

  // Each row of weight matrix has it's own scale
  // The following is a multiplication activation scale with
  // weight scales.
  aligned_vector<float> act_times_w_scale(rows);
  randFill<float>(act_times_w_scale, -8.0f, 8.0f);

  random_device rd;
  mt19937 gen(rd());
  auto distribution = uniform_int_distribution<>(0, 5);
  int32_t act_zero_point = distribution(gen);

  // Each row of weight matrix has it's own zero point
  aligned_vector<int32_t> weight_zero_point(rows);
  randFill<int32_t>(weight_zero_point, -8, 8);

  aligned_vector<int32_t> act_col_offsets(cols);
  // We are randomly filling act col offsets for the
  // purpose of this test. In reality, these will be calculated
  // by summing the columns of activations.
  randFill<int32_t>(act_col_offsets, -8, 8);

  aligned_vector<int32_t> weight_row_offsets(rows);
  randFill<int32_t>(weight_row_offsets, -8, 8);

  aligned_vector<float> bias(rows);
  randFill<float>(bias, -8.0f, 8.0f);

  aligned_vector<int32_t> input(numElements);
  randFill<int32_t>(input, -8, 8);

  aligned_vector<uint8_t> output_ref(numElements);
  aligned_vector<uint8_t> output_test(numElements);

  // output scale and zero point
  float scale = 2.0f;
  int32_t zero_point = 2;

  block_type_t block{0, rows, 0, cols};

  auto bias_data_ptr = use_bias ? bias.data() : nullptr;

  bool use_col_offsets = true;

  if (q_gran == QuantizationGranularity::TENSOR) {
    if (weight_zero_point[0] == 0) {
      use_col_offsets = false;
    }
  }
  else {
    auto areEqual = [](int a, int b) { return a == b; };
    if (std::all_of(
        weight_zero_point.begin(),
        weight_zero_point.end(),
        std::bind(areEqual, std::placeholders::_1, 0))) {
      use_col_offsets = false;
    }
  }
  auto col_offsets_ptr = use_col_offsets ? act_col_offsets.data() : nullptr;
  trRequantizationParams_t reqParams = {act_zero_point,
                                        weight_zero_point.data(),
                                        zero_point,
                                        scale,
                                        weight_row_offsets.data(),
                                        col_offsets_ptr,
                                        bias_data_ptr,
                                        act_times_w_scale.data()};

#define TESTCODE(FUSE_RELU, ACT_SYMMETRIC, WEIGHT_SYMMETRIC, HAS_BIAS, Q_GRAN) \
  trRequantizeRef<FUSE_RELU, Q_GRAN>(                                          \
      output_ref.data(), input.data(), block, cols, cols, reqParams);          \
  trRequantizeOpt<FUSE_RELU, ACT_SYMMETRIC, WEIGHT_SYMMETRIC, HAS_BIAS,Q_GRAN>(\
      output_test.data(), input.data(), block, cols, cols, reqParams);

  if (fuse_relu) {
    if (q_gran == QuantizationGranularity::TENSOR) {
      // Assume weight matrix has the same scale and the same
      // zero point for all rows.
      // Only weight_zero_point[0] and act_times_w_scale[0] is used
      // in calculations
      if (weight_zero_point[0] == 0 || !use_col_offsets) {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(true, true, true, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(true, true, true, false, QuantizationGranularity::TENSOR)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(true, false, true, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(true, false, true, false, QuantizationGranularity::TENSOR)
          }
        }
      }
      else {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(true, true, false, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(true, true, false, false, QuantizationGranularity::TENSOR)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(true, false, false, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(true, false, false, false, QuantizationGranularity::TENSOR)
          }
        }
      }
    } else if (q_gran == QuantizationGranularity::OUT_CHANNEL) {
      if (!use_col_offsets) {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(true, true, true, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(true, true, true, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(true, false, true, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(true, false, true, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
      }
      else {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(true, true, false, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(true, true, false, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(true, false, false, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(true, false, false, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
      }
    } else {
      FAIL();
    }

  } else {
    if (q_gran == QuantizationGranularity::TENSOR) {
      if (weight_zero_point[0] == 0 || !use_col_offsets) {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(false, true, true, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(false, true, true, false, QuantizationGranularity::TENSOR)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(false, false, true, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(false, false, true, false, QuantizationGranularity::TENSOR)
          }
        }
      }
      else {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(false, true, false, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(false, true, false, false, QuantizationGranularity::TENSOR)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(false, false, false, true, QuantizationGranularity::TENSOR)
          }
          else {
            TESTCODE(false, false, false, false,
                QuantizationGranularity::TENSOR)
          }
        }
      }
    } else if (q_gran == QuantizationGranularity::OUT_CHANNEL) {
      if (!use_col_offsets) {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(false, true, true, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(false, true, true, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(false, false, true, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(false, false, true, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
      }
      else {
        if (act_zero_point == 0) {
          if (use_bias) {
            TESTCODE(false, true, false, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(false, true, false, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
        else {
          if (use_bias) {
            TESTCODE(false, false, false, true,
                QuantizationGranularity::OUT_CHANNEL)
          }
          else {
            TESTCODE(false, false, false, false,
                QuantizationGranularity::OUT_CHANNEL)
          }
        }
      }
    } else {
      FAIL();
    }
  }
#undef TESTCODE
  ASSERT_EQ(output_ref, output_test) << "reference doesn't match with test";
}

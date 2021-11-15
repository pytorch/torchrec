/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <ostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // num_rows, block_size
      {150, 1},
      {150, 4},
      {10, 8},
      {150, 16},
      {1, 8},
      {1, 16},
      {150, 24},
      {150, 32},
      {150, 40},
      {150, 64},
      {150, 80},
      {150, 128},
      {150, 384},
      {10, 385},
      {10, 769},
  };
  return input_dims;
}

vector<int> prefetch_distances{0, 16, 1000000};

namespace {
class SparseAdagradTest
    : public testing::TestWithParam<tuple<bool, int, bool, bool, bool>> {};
}; // namespace

// Test:
INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    SparseAdagradTest,
    ::testing::Combine(
        ::testing::Bool(), // 64 bit indices
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Bool(), // out of bound indices
        ::testing::Bool(), // use weight decay
        ::testing::Bool())); // adjust weight decay

TEST_P(SparseAdagradTest, basicTest_two_stages) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, out_of_bounds, use_weight_decay, adjust_weight_decay;
  int prefetch;
  tie(isIndex64b,
      prefetch,
      out_of_bounds,
      use_weight_decay,
      adjust_weight_decay) = GetParam();

  for (auto input : inputs) {
    int num_rows = input[0];
    int block_size = input[1];
    int param_size = num_rows * block_size;

    vector<float> g(param_size); // gradients

    vector<float> h(param_size); // input momentums
    vector<float> w(param_size); // input params
    vector<float> h_ref(param_size);
    vector<float> w_ref(param_size);

    default_random_engine generator;

    normal_distribution<float> h_w_distribution;

    uniform_real_distribution<float> values_gen(0, 10);
    for (int i = 0; i < param_size; i++) {
      h_ref[i] = h[i] = values_gen(generator);
    }
    for (int i = 0; i < param_size; i++) {
      w_ref[i] = w[i] = values_gen(generator);
    }
    for (int i = 0; i < param_size; i++) {
      g[i] = values_gen(generator);
    }

    vector<std::int64_t> indices(num_rows);
    vector<std::int32_t> indices_32(num_rows);
    float epsilon = 1e-5;
    float lr = 0.5;
    float weight_decay = use_weight_decay ? 0.1f : 0.0f;

    uniform_int_distribution<std::int64_t> index_distribution(0, num_rows - 1);
    for (int i = 0; i < num_rows; ++i) {
      indices_32[i] = indices[i] = index_distribution(generator);
    }
    if (out_of_bounds) {
      int idx = index_distribution(generator);
      indices_32[idx] = indices[idx] = num_rows;
    }

    vector<double> counters;
    constexpr int64_t counter_halflife = 1e6;
    if (adjust_weight_decay) {
      uniform_real_distribution<> counter_distribution(0, 2 * counter_halflife);
      counters.resize(num_rows);
      for (int i = 0; i < num_rows; ++i) {
        counters[i] = counter_distribution(generator);
      }
    }

    int ret_fbgemm, ret_ref;
    if (isIndex64b) {
      ret_ref = sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per rows
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);

      auto fn_fbgemm = GenerateSparseAdaGrad<std::int64_t>(
          block_size, false, prefetch, use_weight_decay);

      ret_fbgemm = fn_fbgemm(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w.data(), // input parameters
          g.data(), // input gradients
          h.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);
    } else { // 32 bit indices
      ret_ref = sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per rows
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);

      auto fn_fbgemm = GenerateSparseAdaGrad<std::int32_t>(
          block_size, false, prefetch, use_weight_decay);

      ret_fbgemm = fn_fbgemm(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w.data(), // input parameters
          g.data(), // input gradients
          h.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);
    }

    EXPECT_EQ(ret_fbgemm, ret_ref)
        << "return vals differ, reference is: " << ret_ref
        << " ,fbgemm is: " << ret_fbgemm;
    for (int i = 0; i < h.size(); ++i) {
      EXPECT_EQ(h[i], h_ref[i])
          << "results for h differ at (" << i << ") reference: " << h_ref[i]
          << ", FBGEMM: " << h[i] << " emb dim :" << block_size;
    }
    for (int i = 0; i < w.size(); ++i) {
      EXPECT_EQ(w[i], w_ref[i])
          << "results for h differ at (" << i << ") reference: " << w_ref[i]
          << ", FBGEMM: " << w[i] << " emb dim :" << block_size;
    }
  }
}

TEST_P(SparseAdagradTest, rowwiseTest_two_stages) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, out_of_bounds, use_weight_decay, adjust_weight_decay;
  int prefetch;
  tie(isIndex64b,
      prefetch,
      out_of_bounds,
      use_weight_decay,
      adjust_weight_decay) = GetParam();

  for (auto input : inputs) {
    int num_rows = input[0];
    int block_size = input[1];
    int param_size = num_rows * block_size;

    vector<float> g(param_size); // gradients

    vector<float> h(param_size); // input momentums
    vector<float> w(param_size); // input params
    vector<float> h_ref(param_size);
    vector<float> w_ref(param_size);

    default_random_engine generator;
    uniform_real_distribution<float> values_gen(0, 2);
    for (int i = 0; i < param_size; i++) {
      h_ref[i] = h[i] = values_gen(generator);
    }
    for (int i = 0; i < param_size; i++) {
      w_ref[i] = w[i] = values_gen(generator);
    }
    for (int i = 0; i < param_size; i++) {
      g[i] = values_gen(generator);
    }

    vector<std::int64_t> indices(num_rows);
    vector<std::int32_t> indices_32(num_rows);
    float epsilon = 1e-5;
    float lr = 0.5;
    float weight_decay = use_weight_decay ? 0.1f : 0.0f;

    uniform_int_distribution<std::int64_t> index_distribution(0, num_rows - 1);
    for (int i = 0; i < num_rows; ++i) {
      indices_32[i] = indices[i] = index_distribution(generator);
    }
    if (out_of_bounds) {
      int idx = index_distribution(generator);
      indices_32[idx] = indices[idx] = num_rows;
    }

    vector<double> counters;
    constexpr int64_t counter_halflife = 1e6;
    if (adjust_weight_decay) {
      uniform_real_distribution<> counter_distribution(0, 2 * counter_halflife);
      counters.resize(num_rows);
      for (int i = 0; i < num_rows; ++i) {
        counters[i] = counter_distribution(generator);
      }
    }

    int ret_fbgemm, ret_ref;
    if (isIndex64b) {
      ret_ref = rowwise_sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per rows
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);

      auto fn_fbgemm = GenerateSparseAdaGrad<std::int64_t>(
          block_size, true, prefetch, use_weight_decay);

      ret_fbgemm = fn_fbgemm(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w.data(), // input parameters
          g.data(), // input gradients
          h.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);
    } else { // 32 bit indices
      ret_ref = rowwise_sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per rows
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);

      auto fn_fbgemm = GenerateSparseAdaGrad<std::int32_t>(
          block_size, true, prefetch, use_weight_decay);

      ret_fbgemm = fn_fbgemm(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w.data(), // input parameters
          g.data(), // input gradients
          h.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr,
          weight_decay,
          adjust_weight_decay ? counters.data() : nullptr,
          counter_halflife);
    }

    EXPECT_EQ(ret_fbgemm, ret_ref)
        << "return vals differ, reference is: " << ret_ref
        << " ,fbgemm is: " << ret_fbgemm;
    for (int i = 0; i < h.size(); ++i) {
      EXPECT_EQ(h[i], h_ref[i])
          << "results for h differ at (" << i << ") reference: " << h_ref[i]
          << ", FBGEMM: " << h[i] << " emb dim :" << block_size;
    }
    for (int i = 0; i < w.size(); ++i) {
      EXPECT_EQ(w[i], w_ref[i])
          << "results for w differ at (" << i << ") reference: " << w_ref[i]
          << ", FBGEMM: " << w[i] << " emb dim :" << block_size;
    }
  }
}

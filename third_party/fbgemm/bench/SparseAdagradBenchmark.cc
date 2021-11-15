/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // num_rows, emb dim
      {1500000, 24},
      {1500000, 32},
      {1500000, 40},
      {1500000, 64},
      {1500000, 80},
      {1500000, 128},
      {1500000, 144},
      {1500000, 192},
      {1500000, 384},
  };
  return input_dims;
}

void run_benchmark(
    const int num_rows, // number of rows reading
    const int block_size, // number of parameters per row
    const uint64_t param_size, // total number of parameters
    const bool isIndex64b,
    const bool adjust_weight_decay) {
  vector<char> llc(64L * 1024L * 1024L, 1.0);
  vector<float> g(num_rows * block_size); // gradients
  vector<float> h(param_size); // input momentums
  vector<float> w(param_size); // input params
  vector<float> h_ref(param_size);
  vector<float> w_ref(param_size);

  default_random_engine generator;
  // normal_distribution<float> h_w_distribution;

  // TODO: check appropriate vals for g,h,w
  for (int i = 0; i < g.size(); ++i) {
    g[i] = 4 + i; // h_w_distribution(generator);
  }
  for (int i = 0; i < h.size(); ++i) {
    h_ref[i] = h[i] = 2 + i; // h_w_distribution(generator);
  }
  for (int i = 0; i < w.size(); ++i) {
    w_ref[i] = w[i] = 3 + i; // h_w_distribution(generator);
  }

  vector<int64_t> indices(num_rows);
  vector<int32_t> indices_32(num_rows);
  vector<double> counter(num_rows);
  float epsilon = 1e-5;
  float lr = 0.5;
  float weight_decay = adjust_weight_decay ? 1e-7 : 0.0f;
  constexpr int64_t counter_halflife = 1e6;

  uniform_int_distribution<int64_t> length_distribution(0, num_rows - 1);
  uniform_int_distribution<int64_t> counter_distribution(
      0, 2 * counter_halflife);
  for (int i = 0; i < num_rows; ++i) {
    indices_32[i] = indices[i] = length_distribution(generator);
    counter[i] = counter_distribution(generator);
  }

  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 10;
  double data_moved = 5 * sizeof(float) * num_rows * block_size;

  double t = 0.0;
  constexpr bool rowwise = false;
  constexpr int prefetch = 16;
  if (isIndex64b) {
    auto fn_indices_64 = GenerateSparseAdaGrad<int64_t>(
        block_size, rowwise, prefetch, adjust_weight_decay);

    t = measureWithWarmup(
        [&]() {
          fn_indices_64(
              num_rows, // number of rows reading
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices.data(), // indices of each row
              epsilon,
              lr,
              weight_decay, // weight decay (lambda)
              adjust_weight_decay
                  ? counter.data()
                  : nullptr, // feature ID frequency counter data
              counter_halflife); // counter halflife value for adjustments
        },
        NUM_WARMUP,
        NUM_ITER,
        [&]() { llc_flush(llc); });

    for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
      sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per row
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr,
          weight_decay, // weight decay (lambda)
          adjust_weight_decay ? counter.data()
                              : nullptr, // feature ID frequency counter data
          counter_halflife); // counter halflife value for adjustments
    }
  } else {
    auto fn_indices_32 = GenerateSparseAdaGrad<int32_t>(
        block_size, rowwise, prefetch, adjust_weight_decay);

    t = measureWithWarmup(
        [&]() {
          fn_indices_32(
              num_rows, // number of rows reading
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices_32.data(), // indices of each row
              epsilon,
              lr,
              weight_decay, // weight decay (lambda)
              adjust_weight_decay
                  ? counter.data()
                  : nullptr, // feature ID frequency counter data
              counter_halflife); // counter halflife value for adjustments
        },
        NUM_WARMUP,
        NUM_ITER,
        [&]() { llc_flush(llc); });

    for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
      sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per row
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr,
          weight_decay, // weight decay (lambda)
          adjust_weight_decay ? counter.data()
                              : nullptr, // feature ID frequency counter data
          counter_halflife); // counter halflife value for adjustments
    }
  }

  for (int i = 0; i < w.size(); ++i) {
    assert(fabs(w[i] - w_ref[i]) < 1e-5);
    if (fabs(w[i] - w_ref[i]) >= 1e-5) {
      fprintf(stderr, "%d %f %f\n", i, w[i], w_ref[i]);
    }
  }

  for (int i = 0; i < h.size(); ++i) {
    assert(fabs(h[i] - h_ref[i]) < 1e-5);
    if (fabs(h[i] - h_ref[i]) >= 1e-5) {
      fprintf(stderr, "%d %f %f\n", i, h[i], h_ref[i]);
    }
  }

  cout << "indices: " << (isIndex64b ? " 64bits " : " 32bits ") << " | ";
  cout << "weight_decay: " << setw(1) << adjust_weight_decay << " | ";
  cout << "num_rows: " << setw(8) << num_rows << " block_size: " << setw(4)
       << block_size << " | ";
  cout << "time taken by jit code(secs): " << setw(10) << fixed
       << setprecision(6) << t << " | ";
  cout << "bandwidth fbgemm (GB/s) " << setw(10) << fixed << setprecision(6)
       << data_moved / t / 1e9 << endl;
}

int main() {
  int num_rows;
  int block_size;
  uint64_t param_size;
  vector<vector<int>> inputs(GetInputs_());

  for (auto isIndex64b : vector<bool>{true, false}) {
    for (auto adjust_weight_decay : vector<bool>{true, false}) {
      for (auto& input : inputs) {
        assert(input.size() >= 2);
        num_rows = input[0];
        block_size = input[1];
        param_size = num_rows * block_size;
        run_benchmark(
            num_rows, block_size, param_size, isIndex64b, adjust_weight_decay);
      }
    }
  }
  return 0;
}

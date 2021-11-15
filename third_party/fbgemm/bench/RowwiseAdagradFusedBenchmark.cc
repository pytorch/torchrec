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
      // batch size, number of rows of table, emb dim , avg lengthl
      // TODO: Add more inputs
      // Use these -- but they are slow.
      {10, 4000000, 32, 100},
      {10, 4000000, 64, 100},
      {10, 4000000, 128, 100},
      {10, 4000000, 256, 100},
      // Use these for debugging
      // {2, 16, 128, 10},
      // {10, 4000, 128, 100},
      // {10, 4000, 128, 100},
      // {10, 4000, 128, 100},
  };
  return input_dims;
}

void run_benchmark(
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    bool use_32_bit_indices = false,
    bool prefetch = false) {
  vector<char> llc(64L * 1024L * 1024L, 1.0);
  vector<float> g(batch_size * embedding_dim); // gradients
  vector<float> h(num_rows); // input momentums
  vector<float> w(num_rows * embedding_dim); // input params
  vector<float> h_ref(h.size());
  vector<float> w_ref(w.size());

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

  // Generate lengths
  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_rows));
  vector<int> offsets(batch_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < batch_size; ++i) {
    offsets[i + 1] = offsets[i] + length_distribution(generator);
  }

  // Compute the number of indices
  int lengths_sum = offsets[batch_size];
  cout << "lengths_sum " << lengths_sum << endl;

  // Generate indices
  vector<int64_t> indices;
  vector<int32_t> indices_32;

  vector<int> container(num_rows);

  // please note we generate unique indices
  for (int i = 0; i < batch_size; ++i) {
    iota(container.begin(), container.end(), 0);
    random_shuffle(container.begin(), container.end());
    copy(
        container.begin(),
        container.begin() + (offsets[i + 1] - offsets[i]),
        back_inserter(indices));
  }
  copy(begin(indices), end(indices), back_inserter(indices_32));

  float epsilon = 1e-5;
  float lr = 0.5;

  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 10;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes = lengths_sum *
          ((embedding_dim + 1) * sizeof(float) * 2 +
           (use_32_bit_indices ? 4 : 8)) +
      batch_size * (embedding_dim * sizeof(float) + sizeof(int));
  double bytes_padded = lengths_sum *
          (((embedding_dim * sizeof(float) + 63) / 64 + 1) * 64 * 2 +
           (use_32_bit_indices ? 4 : 8)) +
      batch_size * (embedding_dim * sizeof(float) + sizeof(int));

  auto kernel_i32 = GenerateRowWiseSparseAdaGradFused<int32_t>(
      embedding_dim, prefetch ? 16 : 0);
  auto kernel_i64 = GenerateRowWiseSparseAdaGradFused<int64_t>(
      embedding_dim, prefetch ? 16 : 0);

  for (bool flush_cache : {false, true}) {
    double t = measureWithWarmup(
        [&]() {
          if (use_32_bit_indices) {
            kernel_i32(
                batch_size,
                lengths_sum,
                num_rows,
                w.data(),
                g.data(),
                h.data(),
                indices_32.data(),
                offsets.data(),
                epsilon,
                lr);
          } else {
            kernel_i64(
                batch_size,
                lengths_sum,
                num_rows,
                w.data(),
                g.data(),
                h.data(),
                indices.data(),
                offsets.data(),
                epsilon,
                lr);
          }
        },
        NUM_WARMUP,
        NUM_ITER,
        [&]() { llc_flush(llc); });

    if (flush_cache) {
      cout << setw(20) << "cache flushed";
    } else {
      cout << setw(20) << "cache not flushed";
    }
    if (prefetch) {
      cout << setw(16) << "prefetch on";
    } else {
      cout << setw(16) << "prefetch off";
    }

    cout << setw(8) << "b/w" << setw(10) << bytes / 1e9 / t << " GB/s"
         << setw(20) << "effective b/w: " << setw(16) << bytes_padded / 1e9 / t
         << "GB/s" << setw(8) << " time " << setw(16) << t << endl;
  }
}

int main() {
  vector<vector<int>> inputs(GetInputs_());

  for (auto& input : inputs) {
    assert(input.size() > 3);
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    cout << "batch size" << setw(6) << batch_size << setw(10) << "num rows"
         << setw(16) << num_rows << setw(10) << "emb dim" << setw(6)
         << embedding_dim << setw(16) << "avg length" << setw(6) << average_len
         << endl;

    for (bool use_32_bit_indices : {false, true}) {
      for (bool prefetch : {false, true}) {
        // args: batch sz, num rows, emb dim, avg len, use 32b, prefetch
        cout << (use_32_bit_indices ? " 32" : " 64") << " bit indices";
        if (prefetch) {
          cout << " with prefetching";
        }
        cout << ", ";
        run_benchmark(
            batch_size,
            num_rows,
            embedding_dim,
            average_len,
            use_32_bit_indices,
            prefetch);
      } // prefetch
    } // use_32_bit_indices
  } // for each input

  return 0;
}

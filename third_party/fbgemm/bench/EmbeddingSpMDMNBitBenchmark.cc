/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

void print_fused_table(int rows, int embedding_dim, const uint8_t* table) {
  for (int i = 0; i < rows; i++) {
    std::cout << "row: " << i << " : " << std::endl;
    for (int ii = 0; ii < embedding_dim; ii++) {
      std::cout << (int)table[i * (embedding_dim + 2 * sizeof(float)) + ii]
                << ",";
    }
    std::cout << std::endl;
  }
}

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

int run_benchmark(
    int bit_rate,
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false) {
  // Create embedding table
  int num_elem_per_byte = 8 / bit_rate;
  int fused_embedding_dim =
      (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(float16);
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;

  vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
  for (int i = 0; i < num_rows; i++) {
    for (int ii = 0;
         ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
         ii++) {
      fused_embedding_table[i * fused_embedding_dim + ii] = 2;
    }
    float16* scale_bias = reinterpret_cast<float16*>(
        &fused_embedding_table[i * fused_embedding_dim] +
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = 2.0f;
    float bias = 1.0f;
    FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
    FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
  }

  // print_fused_table(num_rows, embedding_dim, fused_embedding_table);

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
  map<int64_t, set<int>> dedup_map; // index -> set(output index)

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

  // Generate weights
  vector<float> weights(lengths_sum);
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  vector<float> output_sls_ref(batch_size * embedding_dim);
  vector<float> output_slws_ref(output_sls_ref.size()),
      output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 10;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes = lengths_sum * fused_embedding_dim;
  constexpr int CACHE_LINE_LEN = 64;
  double bytes_padded = lengths_sum * CACHE_LINE_LEN *
      static_cast<int>((fused_embedding_dim + CACHE_LINE_LEN - 1) /
                       CACHE_LINE_LEN);

  for (bool has_weight : {false, true}) {
    vector<float>& output_ref = has_weight ? output_slws_ref : output_sls_ref;

    bool success = false, success_ref = false;

    if (use_32_bit_indices) {
      success_ref = EmbeddingSpMDMNBit_ref(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices_32.data(),
          offsets.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());
    } else {
      success = EmbeddingSpMDMNBit_ref(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());
    }

    auto kernel_32 = GenerateEmbeddingSpMDMNBit<int32_t>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0);
    auto kernel_64 = GenerateEmbeddingSpMDMNBit<int64_t>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0);

    vector<float>& output = has_weight ? output_slws : output_sls;
    for (bool flush_cache : {false, true}) {
      double t = measureWithWarmup(
          [&]() {
            if (use_32_bit_indices) {
              success = kernel_32(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices_32.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
            } else {
              success = kernel_64(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(fused_embedding_table);
              cache_evict(indices);
              cache_evict(indices_32);
              cache_evict(offsets);
              cache_evict(weights);
              cache_evict(output);
            }
          });

      // printMatrix(
      //     matrix_op_t::NoTranspose,
      //     output.data(),
      //     batch_size,
      //     embedding_dim,
      //     embedding_dim,
      //     "");
      // printMatrix(
      //     matrix_op_t::NoTranspose,
      //     output_ref.data(),
      //     batch_size,
      //     embedding_dim,
      //     embedding_dim,
      //     "");
      // Check correctness
      if (!flush_cache) {
        // vector<float>& output_ref =
        //     has_weight ? output_slws_ref : output_sls_ref;
        if (success != success_ref) {
          assert(
              false && "ERROR: refernce impl and JIT imp did not both succeed");
        } else if (success) {
          for (int i = 0; i < output.size(); ++i) {
            assert(fabs(output[i] - output_ref[i]) < 1e-3);
            if (fabs(output[i] - output_ref[i]) >= 1e-3) {
              cout << i << " " << output[i] << " " << output_ref[i] << endl;
            }
          }
        }
      }

      if (has_weight) {
        cout << setw(16) << "SLW(WEIGHTED) ";
      } else {
        cout << setw(16) << "SLS ";
      }
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
           << setw(20) << "effective b/w: " << setw(16)
           << bytes_padded / 1e9 / t << "GB/s" << setw(8) << " time "
           << setw(16) << t << endl;
    } // flush_cache
  } // has_weight
  return 0;
}

int main() {
  int batch_size;
  int num_rows;
  int embedding_dim;
  int average_len;

  vector<vector<int>> inputs(GetInputs_());

  for (int bit_rate : {2, 4}) {
    for (auto& input : inputs) {
      assert(input.size() > 3);
      batch_size = input[0];
      num_rows = input[1];
      embedding_dim = input[2];
      average_len = input[3];

      cout << "bit_rate" << setw(6) << bit_rate << "batch size" << setw(6)
           << batch_size << setw(10) << "num rows" << setw(16) << num_rows
           << setw(10) << "emb dim" << setw(6) << embedding_dim << setw(16)
           << "avg length" << setw(6) << average_len << endl;
      // args: batch sz, num rows, emb dim, avg len, normalize, use 32b,
      // prefetch
      cout << "64 bit indices, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false); // normalize_by_lengths

      cout << "64 bit indices with prefetching, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false, // normalize_by_lengths
          false, // use_32_bit_indices
          true); // prefetch

      cout << "32 bit indices, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false, // normalize_by_lengths
          true); // use_32_bit_indices

      cout << "32 bit indices with prefetching, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false, // normalize_by_lengths
          true, // use_32_bit_indices
          true); // prefetch

      // running with normalize by lengths
      // run_benchmark(batch_size, num_rows, embedding_dim, average_len,
      // true); run_benchmark(
      //     batch_size, num_rows, embedding_dim, average_len, true,
      //     true);
      // run_benchmark(
      //     batch_size,
      //     num_rows,
      //     embedding_dim,
      //     average_len,
      //     false,
      //     true,
      //     true);
    }
  }
  return 0;
}

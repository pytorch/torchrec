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
    bool normalize_by_lengths,
    bool use_fp16_inputs = false,
    bool use_32_bit_indices = false,
    bool prefetch = false) {
  // Create embedding table
  default_random_engine generator;

  vector<float> embedding_table(num_rows * embedding_dim);
  normal_distribution<float> embedding_distribution;
  for (int i = 0; i < embedding_table.size(); ++i) {
    embedding_table[i] = embedding_distribution(generator);
  }
  vector<float16> embedding_table_fp16;
  if (use_fp16_inputs) {
    embedding_table_fp16.resize(embedding_table.size());
    FloatToFloat16_simd(
        embedding_table.data(),
        embedding_table_fp16.data(),
        embedding_table.size());
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
  int elem_bytes = use_fp16_inputs ? sizeof(float16) : sizeof(float);
  double bytes = lengths_sum *
          (embedding_dim * elem_bytes + (use_32_bit_indices ? 4 : 8)) +
      batch_size * sizeof(int);
  double bytes_padded = lengths_sum *
          ((embedding_dim * elem_bytes + 63) / 64 * 64 +
           (use_32_bit_indices ? 4 : 8)) +
      batch_size * sizeof(int);

  for (bool has_weight : {false, true}) {
    vector<float>& output_ref = has_weight ? output_slws_ref : output_sls_ref;

    bool success = false, success_ref = false;

    if (use_fp16_inputs) {
      if (use_32_bit_indices) {
        success_ref = EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            embedding_table_fp16.data(),
            indices_32.data(),
            offsets.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      } else {
        success_ref = EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            embedding_table_fp16.data(),
            indices.data(),
            offsets.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      }
    } else {
      if (use_32_bit_indices) {
        success_ref = EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            embedding_table.data(),
            indices_32.data(),
            offsets.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      } else {
        success_ref = EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            embedding_table.data(),
            indices.data(),
            offsets.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      }
    }

    auto kernel_fp32_i32 = GenerateEmbeddingSpMDM<float, int32_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
    auto kernel_fp32_i64 = GenerateEmbeddingSpMDM<float, int64_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
    auto kernel_fp16_i32 = GenerateEmbeddingSpMDM<float16, int32_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
    auto kernel_fp16_i64 = GenerateEmbeddingSpMDM<float16, int64_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);

    vector<float>& output = has_weight ? output_slws : output_sls;
    for (bool flush_cache : {false, true}) {
      double t = measureWithWarmup(
          [&]() {
            if (use_fp16_inputs) {
              if (use_32_bit_indices) {
                success = kernel_fp16_i32(
                    batch_size,
                    lengths_sum,
                    num_rows,
                    embedding_table_fp16.data(),
                    indices_32.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    output.data());
              } else {
                success = kernel_fp16_i64(
                    batch_size,
                    lengths_sum,
                    num_rows,
                    embedding_table_fp16.data(),
                    indices.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    output.data());
              }
            } else {
              if (use_32_bit_indices) {
                success = kernel_fp32_i32(
                    batch_size,
                    lengths_sum,
                    num_rows,
                    embedding_table.data(),
                    indices_32.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    output.data());
              } else {
                success = kernel_fp32_i64(
                    batch_size,
                    lengths_sum,
                    num_rows,
                    embedding_table.data(),
                    indices.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    output.data());
              }
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(embedding_table);
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
      // cout << "reference data\n";
      // printMatrix(
      //     matrix_op_t::NoTranspose,
      //     output_ref.data(),
      //     batch_size,
      //     embedding_dim,
      //     embedding_dim,
      //     "");
      // Check correctness
      if (!flush_cache) {
        if (success != success_ref) {
          assert(
              false && "ERROR: refernce impl and JIT imp did not both succeed");
        } else if (success) {
          for (int i = 0; i < output.size(); ++i) {
            assert(output[i] == output_ref[i]);
            if (output[i] != output_ref[i]) {
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

    for (bool normalize_by_lengths : {false, true}) {
      for (bool use_fp16_inputs : {false, true}) {
        for (bool use_32_bit_indices : {false, true}) {
          for (bool prefetch : {false, true}) {
            // args: batch sz, num rows, emb dim, avg len, normalize, use 32b,
            // prefetch
            if (normalize_by_lengths) {
              cout << "Mean";
            }
            if (use_fp16_inputs) {
              cout << "fp16 inputs";
            }
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
                normalize_by_lengths,
                use_fp16_inputs,
                use_32_bit_indices,
                prefetch);
          } // prefetch
        } // use_32_bit_indices
      } // use_fp16_inputs
    } // normalize_by_length
  } // for each input
  return 0;
}

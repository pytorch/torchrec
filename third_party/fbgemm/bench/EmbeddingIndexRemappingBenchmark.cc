/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/EmbeddingSpMDMTestUtils.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, avg lengthl
      {10, 4000000, 100},
      {20, 4000000, 100},
      {40, 4000000, 100},
      {50, 4000000, 100},
      {10, 10000000, 100},
      {20, 10000000, 100},
      {40, 10000000, 100},
      {50, 10000000, 100},
  };
  return input_dims;
}

int run_benchmark(
    int batch_size,
    int num_rows,
    int average_len,
    bool use_32_bit_indices = false) {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 100;
  int offset_numel = batch_size + 1;

  constexpr float sparsity = 0.5;

  vector<int64_t> lengths, offsets, indices;
  vector<int32_t> lengths_32, offsets_32, indices_32;
  vector<float> weights;

  GenerateLengthsIndicesWeights(
      lengths,
      lengths_32,
      offsets,
      offsets_32,
      indices,
      indices_32,
      weights,
      batch_size,
      num_rows,
      64, // embedding_dim (not used)
      average_len, // average number of indices in a batch
      EmbeddingSpMDMCornerCase::NONE);

  // Create mapping table for rowwise sparsity
  vector<int32_t> mapping_table;
  CreateMappingTableForRowWiseSparsity(mapping_table, num_rows, sparsity);

  vector<int32_t> out_indices_32(indices_32.size(), 0);
  vector<int32_t> out_offsets_32(offsets_32.size(), 0);
  vector<float> out_weights(weights.size(), 0);
  vector<int64_t> out_indices(indices.size(), 0);
  vector<int64_t> out_offsets(offsets.size(), 0);

  double duration_ref = measureWithWarmup(
      [&]() {
        if (use_32_bit_indices) {
          compressed_indices_remap_ref<int32_t>(
              offset_numel,
              indices_32.data(),
              mapping_table.data(),
              offsets_32.data(),
              weights.data(),
              out_indices_32.data(),
              out_offsets_32.data(),
              out_weights.data());
        } else {
          compressed_indices_remap_ref<int64_t>(
              offset_numel,
              indices.data(),
              mapping_table.data(),
              offsets.data(),
              weights.data(),
              out_indices.data(),
              out_offsets.data(),
              out_weights.data());
        }
      },
      NWARMUP,
      NITER);

  double duration = measureWithWarmup(
      [&]() {
        if (use_32_bit_indices) {
          compressed_indices_remap<int32_t>(
              offset_numel,
              indices_32.data(),
              mapping_table.data(),
              offsets_32.data(),
              weights.data(),
              out_indices_32.data(),
              out_offsets_32.data(),
              out_weights.data());
        } else {
          compressed_indices_remap<int64_t>(
              offset_numel,
              indices.data(),
              mapping_table.data(),
              offsets.data(),
              weights.data(),
              out_indices.data(),
              out_offsets.data(),
              out_weights.data());
        }
      },
      NWARMUP,
      NITER);
  cout << "reference:" << duration_ref * 1e6 << " (us), ";
  cout << "Opt:" << duration * 1e6 << " (us) " << endl;

  return 0;
}

int main() {
  int batch_size;
  int num_rows;
  int average_len;

  vector<vector<int>> inputs(GetInputs_());

  for (auto& input : inputs) {
    assert(input.size() == 3);
    batch_size = input[0];
    num_rows = input[1];
    average_len = input[2];

    cout << "batch size" << setw(6) << batch_size << setw(10) << "num rows"
         << setw(14) << num_rows << setw(16) << "avg length" << setw(6)
         << average_len << endl;
    cout << "64 bit indices, ";
    run_benchmark(batch_size, num_rows, average_len);

    cout << "32 bit indices, ";
    run_benchmark(batch_size, num_rows, average_len, true);
    cout << endl;
  }
  return 0;
}

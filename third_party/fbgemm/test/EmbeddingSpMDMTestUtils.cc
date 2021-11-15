/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "EmbeddingSpMDMTestUtils.h"

#include <numeric>
#include <random>

namespace fbgemm {

using namespace std;

int GenerateLengthsIndicesWeights(
    vector<int64_t>& lengths,
    vector<int32_t>& lengths_32,
    vector<int64_t>& offsets,
    vector<int32_t>& offsets_32,
    vector<int64_t>& indices,
    vector<int32_t>& indices_32,
    vector<float>& weights,
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    EmbeddingSpMDMCornerCase corner_case) {
  // Generate lengths
  default_random_engine generator;
  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_rows));
  lengths.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    lengths[i] =
        corner_case == EMPTY_INDICES ? 0 : length_distribution(generator);
  }

  // Compute the number of indices
  int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);

  // Generate indices
  indices.resize(lengths_sum);
  indices_32.resize(lengths_sum);

  uniform_int_distribution<int> index_distribution(0, num_rows - 1);
  for (int i = 0; i < lengths_sum; ++i) {
    indices_32[i] = indices[i] = index_distribution(generator);
  }
  if (corner_case != EMPTY_INDICES) {
    // to make sure to exercise out-of-bound cases
    indices_32[0] = indices[0] = num_rows - 1;
  }
  if (corner_case == OUT_OF_BOUND_INDICES) {
    int idx = uniform_int_distribution<int>(0, lengths_sum - 1)(generator);
    indices_32[idx] = indices[idx] = num_rows;
  }
  if (corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
    if (bernoulli_distribution(0.5)(generator)) {
      ++lengths[batch_size - 1];
    } else {
      --lengths[batch_size - 1];
    }
  }

  // Generate offsets
  offsets.resize(lengths.size() + 1);
  lengths_32.resize(lengths.size());
  offsets_32.resize(offsets.size());
  offsets[0] = 0;
  offsets_32[0] = 0;
  for (int i = 0; i < lengths.size(); ++i) {
    offsets_32[i + 1] = offsets[i + 1] = offsets[i] + lengths[i];
    lengths_32[i] = lengths[i];
  }

  // Generate weights
  weights.resize(lengths_sum);
  normal_distribution<float> embedding_distribution;
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  return lengths_sum;
}

int CreateMappingTableForRowWiseSparsity(
    vector<int32_t>& mapping_table,
    int num_rows,
    float sparsity) {
  default_random_engine generator;
  mapping_table.resize(num_rows);
  bernoulli_distribution row_prune_dist(sparsity);
  int num_compressed_rows = 0;
  for (int i = 0; i < num_rows; ++i) {
    if (row_prune_dist(generator)) {
      // pruned
      mapping_table[i] = -1;
    } else {
      mapping_table[i] = num_compressed_rows;
      ++num_compressed_rows;
    }
  }

  return num_compressed_rows;
}

} // namespace fbgemm

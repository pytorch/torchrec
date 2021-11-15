/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "./EmbeddingSpMDMTestUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg lengthl
      {1, 8, 8, 4},
      {2, 8, 16, 4},
      {10, 4000, 32, 100},
      {100, 4000, 32, 100},
      {10, 4000, 64, 100},
      {10, 4000, 128, 100},
      {4, 400, 256, 10},
      {10, 4000, 48, 100},
      {10, 4000, 40, 100},
      {10, 4000, 56, 100},
      {10, 4000, 1, 100},
      {10, 4000, 4, 100},
      // These were  from C2 tests
      {10, 40, 16, 10},
      {10, 40, 85, 10},
      {10, 40, 8, 10},
      {10, 40, 96, 10},
      {10, 40, 163, 10},
  };
  return input_dims;
}

vector<int> prefetch_distances{0, 16, 1000000};

namespace {

class Fused8BitRowwiseEmbeddingLookupTest : public testing::TestWithParam<tuple<
                                                bool,
                                                bool,
                                                int,
                                                EmbeddingSpMDMWeightChoice,
                                                bool,
                                                bool,
                                                EmbeddingSpMDMCornerCase>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    Fused8BitRowwiseEmbeddingLookupTest,
    ::testing::Combine(
        ::testing::Bool(), // isIndex64b
        ::testing::Bool(), // isOffset64b
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Values(
            UNWEIGHTED,
            WEIGHTED,
            POSITIONAL_WEIGHTED), // use_weight
        ::testing::Bool(), // normalize_by_lengths
        ::testing::Bool(), // use_offsets
        ::testing::Values(
            NONE,
            EMPTY_INDICES,
            OUT_OF_BOUND_INDICES,
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM)));

TEST_P(Fused8BitRowwiseEmbeddingLookupTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, isOffset64b, is_wt_positional, use_weight,
      normalize_by_lengths, use_offsets;
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isIndex64b,
      isOffset64b,
      prefetch,
      weight_choice,
      normalize_by_lengths,
      use_offsets,
      corner_case) = GetParam();
  is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  use_weight = weight_choice != UNWEIGHTED;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create embedding table
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    int fused_embedding_dim = embedding_dim + 2 * sizeof(float);
    vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
    for (int i = 0; i < num_rows; i++) {
      for (int ii = 0; ii < embedding_dim; ii++) {
        fused_embedding_table[i * fused_embedding_dim + ii] =
            entries(generator);
      }
      float* scale_bias = reinterpret_cast<float*>(
          fused_embedding_table.data() + i * fused_embedding_dim +
          embedding_dim);
      scale_bias[0] = embedding_distribution(generator);
      scale_bias[1] = embedding_distribution(generator);
    }

    vector<int64_t> lengths, offsets, indices;
    vector<int32_t> lengths_32, offsets_32, indices_32;
    vector<float> weights;
    int lengths_sum = GenerateLengthsIndicesWeights(
        lengths,
        lengths_32,
        offsets,
        offsets_32,
        indices,
        indices_32,
        weights,
        batch_size,
        num_rows,
        embedding_dim,
        average_len,
        corner_case);
    const int64_t* offsets_or_lengths =
        (use_offsets ? offsets : lengths).data();
    const int32_t* offsets_or_lengths_32 =
        (use_offsets ? offsets_32 : lengths_32).data();

    vector<float> output_sls_ref(batch_size * embedding_dim);
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success, success_ref;

    if (isOffset64b) {
      if (isIndex64b) {
        success_ref = EmbeddingSpMDM_ref<uint8_t, int64_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDM<uint8_t, int64_t, int64_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            output.data());
      } else {
        success_ref = EmbeddingSpMDM_ref<uint8_t, int32_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDM<uint8_t, int32_t, int64_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            output.data());
      }
    } else {
      if (isIndex64b) {
        success_ref = EmbeddingSpMDM_ref<uint8_t, int64_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDM<uint8_t, int64_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths_32,
            use_weight ? weights.data() : nullptr,
            output.data());
      } else {
        success_ref = EmbeddingSpMDM_ref<uint8_t, int32_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDM<uint8_t, int32_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths_32,
            use_weight ? weights.data() : nullptr,
            output.data());
      }
    }

    // Check correctness
    EXPECT_EQ(success, success_ref)
        << "Reference and JIT impl did not both succeed";
    if (corner_case == OUT_OF_BOUND_INDICES ||
        corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
      EXPECT_EQ(success, false);
    }
    if (success) {
      for (int i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], output_ref[i])
            << "results differ at (" << i << ") reference: " << output_ref[i]
            << ", FBGEMM: " << output[i] << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}

TEST_P(Fused8BitRowwiseEmbeddingLookupTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, isOffset64b, is_wt_positional, use_weight,
      normalize_by_lengths, use_offsets;
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isIndex64b,
      isOffset64b,
      prefetch,
      weight_choice,
      normalize_by_lengths,
      use_offsets,
      corner_case) = GetParam();
  is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  use_weight = weight_choice != UNWEIGHTED;

  constexpr float sparsity = 0.7;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create mapping table for rowwise sparsity
    vector<int32_t> mapping_table;
    int num_compressed_rows =
        CreateMappingTableForRowWiseSparsity(mapping_table, num_rows, sparsity);

    // Create embedding table
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    int fused_embedding_dim = embedding_dim + 2 * sizeof(float);
    vector<uint8_t> fused_embedding_table(
        num_compressed_rows * fused_embedding_dim);
    for (int i = 0; i < num_compressed_rows; i++) {
      for (int ii = 0; ii < embedding_dim; ii++) {
        fused_embedding_table[i * fused_embedding_dim + ii] =
            entries(generator);
      }
      float* scale_bias = reinterpret_cast<float*>(
          fused_embedding_table.data() + i * fused_embedding_dim +
          embedding_dim);
      scale_bias[0] = embedding_distribution(generator);
      scale_bias[1] = embedding_distribution(generator);
    }

    vector<int64_t> lengths, offsets, indices;
    vector<int32_t> lengths_32, offsets_32, indices_32;
    vector<float> weights;
    int lengths_sum = GenerateLengthsIndicesWeights(
        lengths,
        lengths_32,
        offsets,
        offsets_32,
        indices,
        indices_32,
        weights,
        batch_size,
        num_rows,
        embedding_dim,
        average_len,
        corner_case);
    const int64_t* offsets_or_lengths =
        (use_offsets ? offsets : lengths).data();
    const int32_t* offsets_or_lengths_32 =
        (use_offsets ? offsets_32 : lengths_32).data();

    vector<float> output_sls_ref(batch_size * embedding_dim);
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success, success_ref;

    if (isOffset64b) {
      if (isIndex64b) {
        success_ref = EmbeddingSpMDMRowWiseSparse_ref<uint8_t, int64_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel =
            GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, int64_t, int64_t>(
                embedding_dim,
                use_weight,
                normalize_by_lengths,
                prefetch,
                is_wt_positional,
                use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      } else {
        success_ref = EmbeddingSpMDMRowWiseSparse_ref<uint8_t, int32_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel =
            GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, int32_t, int64_t>(
                embedding_dim,
                use_weight,
                normalize_by_lengths,
                prefetch,
                is_wt_positional,
                use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      }
    } else {
      if (isIndex64b) {
        success_ref = EmbeddingSpMDMRowWiseSparse_ref<uint8_t, int64_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, int64_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths_32,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      } else {
        success_ref = EmbeddingSpMDMRowWiseSparse_ref<uint8_t, int32_t>(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, int32_t>(
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths_32,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      }
    }

    // Check correctness
    EXPECT_EQ(success, success_ref)
        << "Reference and JIT impl did not both succeed";
    if (corner_case == OUT_OF_BOUND_INDICES ||
        corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
      EXPECT_EQ(success, false);
    }
    if (success) {
      for (int i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], output_ref[i])
            << "results differ at (" << i << ") reference: " << output_ref[i]
            << ", FBGEMM: " << output[i] << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}

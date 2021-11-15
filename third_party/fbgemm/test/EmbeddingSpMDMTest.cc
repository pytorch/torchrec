/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <numeric> // for accumulate and iota
#include <ostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "./EmbeddingSpMDMTestUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg length
      {1, 8, 8, 4},
      {10, 4000, 32, 100},
      // {100, 4000, 32, 100},
      {10, 4000, 128, 100},
      {4, 400, 256, 10},
      {10, 4000, 48, 100},
      // {10, 4000, 40, 100},
      {10, 4000, 56, 100},
      {10, 4000, 1, 100},
      // These were  from C2 tests
      {10, 40, 16, 10},
      {10, 40, 85, 10},
      {10, 40, 163, 10},
  };
  return input_dims;
}

namespace {

class EmbeddingSpMDMTest : public testing::TestWithParam<tuple<
                               bool,
                               bool,
                               bool,
                               int,
                               EmbeddingSpMDMWeightChoice,
                               bool,
                               bool,
                               EmbeddingSpMDMCornerCase,
                               bool,
                               bool>> {};

class IndexRemapTest
    : public testing::TestWithParam<tuple<int, int, int, bool, bool>> {};
}; // namespace

vector<int> prefetch_distances = {0, 16, 1000000};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingSpMDMTest,
    ::testing::Combine(
        ::testing::Bool(), // is fp16
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
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM),
        ::testing::Bool(), // output_stride != block_size
        ::testing::Bool())); // input_stride != block_size

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    IndexRemapTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 5, 10}), // batch size
        ::testing::ValuesIn({1, 50, 100, 1000}), // number of rows
        ::testing::ValuesIn({1, 5, 16}), // avg len
        ::testing::Bool(), // is index 64 bit?
        ::testing::Bool())); // per sample weights?

TEST_P(EmbeddingSpMDMTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isFp16, isIndex64b, isOffset64b, is_wt_positional, use_weight,
      normalize_by_lengths, use_offsets, use_output_stride, use_input_stride;
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isFp16,
      isIndex64b,
      isOffset64b,
      prefetch,
      weight_choice,
      normalize_by_lengths,
      use_offsets,
      corner_case,
      use_output_stride,
      use_input_stride) = GetParam();
  is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  use_weight = weight_choice != UNWEIGHTED;

  if (corner_case != NONE) {
    // Check corner case only for subset of tests.
    if (isFp16 || normalize_by_lengths || use_output_stride ||
        use_input_stride) {
      return;
    }
  }
  if (is_wt_positional && !use_weight) {
    // weight positional only makes sense when use_weight is true
    return;
  }

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];
    int output_stride = use_output_stride ? embedding_dim * 2 + 3 : -1;
    int input_stride = use_input_stride ? embedding_dim * 2 + 3 : -1;

    // Create embedding table
    vector<float> embedding_table(
        num_rows * (use_input_stride ? input_stride : embedding_dim));
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < embedding_dim; ++j) {
        embedding_table
            [i * (use_input_stride ? input_stride : embedding_dim) + j] =
                embedding_distribution(generator);
      }
    }
    vector<float16> embedding_table_fp16;
    if (isFp16) {
      embedding_table_fp16.resize(embedding_table.size());
      FloatToFloat16_simd(
          embedding_table.data(),
          embedding_table_fp16.data(),
          embedding_table.size());
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

    vector<float> output_sls_ref(
        batch_size * (use_output_stride ? output_stride : embedding_dim));
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success, success_ref;

    if (isOffset64b) {
      if (isIndex64b) {
        if (isFp16) {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel =
              GenerateEmbeddingSpMDMWithStrides<float16, int64_t, int64_t>(
                  embedding_dim,
                  use_weight,
                  normalize_by_lengths,
                  prefetch,
                  is_wt_positional,
                  use_offsets,
                  output_stride,
                  input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data());
        } else {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel =
              GenerateEmbeddingSpMDMWithStrides<float, int64_t, int64_t>(
                  embedding_dim,
                  use_weight,
                  normalize_by_lengths,
                  prefetch,
                  is_wt_positional,
                  use_offsets,
                  output_stride,
                  input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel =
              GenerateEmbeddingSpMDMWithStrides<float16, int32_t, int64_t>(
                  embedding_dim,
                  use_weight,
                  normalize_by_lengths,
                  prefetch,
                  is_wt_positional,
                  use_offsets,
                  output_stride,
                  input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data());
        } else {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel =
              GenerateEmbeddingSpMDMWithStrides<float, int32_t, int64_t>(
                  embedding_dim,
                  use_weight,
                  normalize_by_lengths,
                  prefetch,
                  is_wt_positional,
                  use_offsets,
                  output_stride,
                  input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data());
        }
      }
    } else {
      if (isIndex64b) {
        if (isFp16) {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel = GenerateEmbeddingSpMDMWithStrides<float16, int64_t>(
              embedding_dim,
              use_weight,
              normalize_by_lengths,
              prefetch,
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data());
        } else {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel = GenerateEmbeddingSpMDMWithStrides<float, int64_t>(
              embedding_dim,
              use_weight,
              normalize_by_lengths,
              prefetch,
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel = GenerateEmbeddingSpMDMWithStrides<float16, int32_t>(
              embedding_dim,
              use_weight,
              normalize_by_lengths,
              prefetch,
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data());
        } else {
          success_ref = EmbeddingSpMDM_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);

          auto kernel = GenerateEmbeddingSpMDMWithStrides<float, int32_t>(
              embedding_dim,
              use_weight,
              normalize_by_lengths,
              prefetch,
              is_wt_positional,
              use_offsets,
              output_stride,
              input_stride);
          success = kernel(
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data());
        }
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
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
          int offset =
              i * (use_output_stride ? output_stride : embedding_dim) + j;
          EXPECT_EQ(output[offset], output_ref[offset])
              << "results differ at (" << i << ") reference: " << output_ref[i]
              << ", FBGEMM: " << output[i] << " emb dim :" << embedding_dim;
        }
      }
    }
  } // end for input
}

TEST_P(EmbeddingSpMDMTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isFp16, isIndex64b, isOffset64b, is_wt_positional, use_weight,
      normalize_by_lengths, use_offsets;
  bool use_output_stride, use_input_stride; // not used
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isFp16,
      isIndex64b,
      isOffset64b,
      prefetch,
      weight_choice,
      normalize_by_lengths,
      use_offsets,
      corner_case,
      use_output_stride,
      use_input_stride) = GetParam();
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
    vector<float> embedding_table(num_compressed_rows * embedding_dim);
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    for (int i = 0; i < embedding_table.size(); ++i) {
      embedding_table[i] = embedding_distribution(generator);
    }
    vector<float16> embedding_table_fp16;
    if (isFp16) {
      embedding_table_fp16.resize(embedding_table.size());
      FloatToFloat16_simd(
          embedding_table.data(),
          embedding_table_fp16.data(),
          embedding_table.size());
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
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float16, int64_t, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float, int64_t, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float16, int32_t, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float, int32_t, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      }
    } else {
      if (isIndex64b) {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float16, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float16, int32_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float, int32_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
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

TEST_P(IndexRemapTest, basicTest) {
  int batch_size, num_rows, avg_len;
  bool isIndex64b, per_sample_weights;
  tie(batch_size, num_rows, avg_len, isIndex64b, per_sample_weights) =
      GetParam();
  constexpr float sparsity = 0.5;

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
      64, // embedding_dim (not used)
      avg_len, // average number of indices in a batch
      EmbeddingSpMDMCornerCase::NONE);

  // Create mapping table for rowwise sparsity
  vector<int32_t> mapping_table;
  int num_compressed_rows =
      CreateMappingTableForRowWiseSparsity(mapping_table, num_rows, sparsity);

  // outputs
  vector<int32_t> out_indices_32(indices_32.size(), 0);
  vector<int32_t> out_offsets_32(offsets_32.size(), 0);
  vector<float> out_weights(weights.size(), 0);

  vector<int64_t> out_indices(indices.size(), 0);
  vector<int64_t> out_offsets(offsets.size(), 0);

  // reference outputs
  vector<int32_t> out_indices_32_ref(indices_32.size(), 0);
  vector<int32_t> out_offsets_32_ref(offsets_32.size(), 0);
  vector<float> out_weights_ref(weights.size(), 0);

  vector<int64_t> out_indices_ref(indices.size(), 0);
  vector<int64_t> out_offsets_ref(offsets.size(), 0);

  // number of elements in the offset array ( it's equal to batch_size + 1)
  int offset_numel = offsets_32.size();

  if (isIndex64b) {
    if (per_sample_weights) {
      compressed_indices_remap<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          weights.data(),
          out_indices.data(),
          out_offsets.data(),
          out_weights.data());

      compressed_indices_remap_ref<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          weights.data(),
          out_indices_ref.data(),
          out_offsets_ref.data(),
          out_weights_ref.data());
    } else {

      compressed_indices_remap<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          nullptr,
          out_indices.data(),
          out_offsets.data(),
          nullptr);

      compressed_indices_remap_ref<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          nullptr,
          out_indices_ref.data(),
          out_offsets_ref.data(),
          nullptr);
    }
  } else {
    if (per_sample_weights) {
      compressed_indices_remap<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          weights.data(),
          out_indices_32.data(),
          out_offsets_32.data(),
          out_weights.data());

      compressed_indices_remap_ref<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          weights.data(),
          out_indices_32_ref.data(),
          out_offsets_32_ref.data(),
          out_weights_ref.data());
    } else {
      compressed_indices_remap<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          nullptr,
          out_indices_32.data(),
          out_offsets_32.data(),
          nullptr);

      compressed_indices_remap_ref<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          nullptr,
          out_indices_32_ref.data(),
          out_offsets_32_ref.data(),
          nullptr);
    }
  }

  if (isIndex64b) {
    EXPECT_EQ(out_offsets, out_offsets_ref) << "offsets don't match";
    for (int i = 0; i < out_offsets[offset_numel - 1]; ++i) {
      EXPECT_EQ(out_indices[i], out_indices_ref[i])
          << "indices don't match at " << i;
    }
  } else {
    EXPECT_EQ(out_offsets_32, out_offsets_32_ref) << "offsets don't match";
    for (int i = 0; i < out_offsets_32[offset_numel - 1]; ++i) {
      EXPECT_EQ(out_indices_32[i], out_indices_32_ref[i])
          << "indices don't match at " << i;
    }
  }

  if (per_sample_weights) {
    size_t len = isIndex64b ? out_offsets[offset_numel - 1]
                            : out_offsets_32[offset_numel - 1];

    for (int i = 0; i < len; ++i) {
      EXPECT_EQ(out_weights[i], out_weights_ref[i])
          << "weights don't match at" << i;
    }
  }
}

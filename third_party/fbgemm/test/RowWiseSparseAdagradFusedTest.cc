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

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include "./EmbeddingSpMDMTestUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg length
      {1, 8, 8, 4},
      {2, 8, 16, 4},
      {10, 4000, 32, 100},
      {100, 4000, 32, 100},
      {10, 4000, 64, 100},
      {10, 4000, 128, 100},
      {4, 400, 256, 10},
      {10, 4000, 48, 100},
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

class RowWiseSparseAdagradFusedTest : public testing::TestWithParam<tuple<
                                          bool,
                                          bool,
                                          bool,
                                          bool,
                                          int,
                                          bool,
                                          EmbeddingSpMDMCornerCase,
                                          bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    RowWiseSparseAdagradFusedTest,
    ::testing::Combine(
        ::testing::Bool(), // isWeightFp16
        ::testing::Bool(), // useStochasticRounding
        ::testing::Bool(), // isIndex64b
        ::testing::Bool(), // isOffset64b
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Bool(), // use_offsets
        ::testing::Values(
            NONE,
            EMPTY_INDICES,
            OUT_OF_BOUND_INDICES,
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM),
        ::testing::Bool())); // grad_stride != block_size

TEST_P(RowWiseSparseAdagradFusedTest, rowwiseTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isWeightFp16, useStochasticRounding, isIndex64b, isOffset64b,
      use_offsets, use_grad_stride;
  int prefetch;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isWeightFp16,
      useStochasticRounding,
      isIndex64b,
      isOffset64b,
      prefetch,
      use_offsets,
      corner_case,
      use_grad_stride) = GetParam();

  if (!isWeightFp16 && useStochasticRounding) {
    // stochastic rounding makes sense only for fp16 weight
    return;
  }

  cpuinfo_initialize();
  int vlen = fbgemmHasAvx512Support()
      ? simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS
      : simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];
    int grad_stride = use_grad_stride ? embedding_dim * 2 + 3 : -1;

    // Create embedding table
    vector<float> w(num_rows * embedding_dim), w_ref(num_rows * embedding_dim),
        h(num_rows), h_ref(num_rows),
        g(batch_size * (use_grad_stride ? grad_stride : embedding_dim));
    vector<float16> w_fp16(w.size()), w_fp16_ref(w.size());
    default_random_engine generator;
    uniform_real_distribution<float> values_gen(0, 2);
    for (int i = 0; i < w.size(); ++i) {
      w_ref[i] = w[i] = values_gen(generator);
      w_fp16_ref[i] = w_fp16[i] = cpu_float2half_rn(w[i]);
    }
    for (int i = 0; i < h.size(); ++i) {
      h_ref[i] = h[i] = values_gen(generator);
    }
    for (int i = 0; i < g.size(); ++i) {
      g[i] = values_gen(generator);
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

    float epsilon = 1e-5;
    float lr = 0.5;

#define REF(Weights, Indices, Offsets)                    \
  do {                                                    \
    success_ref = rowwise_sparse_adagrad_fused_ref(       \
        embedding_dim,                                    \
        batch_size,                                       \
        lengths_sum,                                      \
        num_rows,                                         \
        Weights,                                          \
        g.data(),                                         \
        h_ref.data(),                                     \
        corner_case == EMPTY_INDICES ? nullptr : Indices, \
        Offsets,                                          \
        epsilon,                                          \
        lr,                                               \
        use_offsets,                                      \
        useStochasticRounding,                            \
        vlen,                                             \
        grad_stride);                                     \
  } while (0)

#define JIT(WeightType, IndexType, OffsetType, Weights, Indices, Offsets)     \
  do {                                                                        \
    auto kernel =                                                             \
        GenerateRowWiseSparseAdaGradFused<IndexType, OffsetType, WeightType>( \
            embedding_dim,                                                    \
            prefetch,                                                         \
            use_offsets,                                                      \
            useStochasticRounding,                                            \
            grad_stride);                                                     \
    success = kernel(                                                         \
        batch_size,                                                           \
        lengths_sum,                                                          \
        num_rows,                                                             \
        Weights,                                                              \
        g.data(),                                                             \
        h.data(),                                                             \
        corner_case == EMPTY_INDICES ? nullptr : Indices,                     \
        Offsets,                                                              \
        epsilon,                                                              \
        lr);                                                                  \
  } while (0)

    bool success, success_ref;
    if (isWeightFp16) {
      if (isOffset64b) {
        if (isIndex64b) {
          REF(w_fp16_ref.data(), indices.data(), offsets_or_lengths);
          JIT(float16,
              int64_t,
              int64_t,
              w_fp16.data(),
              indices.data(),
              offsets_or_lengths);
        } else { // 32 bit indices
          REF(w_fp16_ref.data(), indices_32.data(), offsets_or_lengths);
          JIT(float16,
              int32_t,
              int64_t,
              w_fp16.data(),
              indices_32.data(),
              offsets_or_lengths);
        }
      } else { // 32 bit offset
        if (isIndex64b) {
          REF(w_fp16_ref.data(), indices.data(), offsets_or_lengths_32);
          JIT(float16,
              int64_t,
              int32_t,
              w_fp16.data(),
              indices.data(),
              offsets_or_lengths_32);
        } else { // 32 bit indices
          REF(w_fp16_ref.data(), indices_32.data(), offsets_or_lengths_32);
          JIT(float16,
              int32_t,
              int32_t,
              w_fp16.data(),
              indices_32.data(),
              offsets_or_lengths_32);
        }
      }
    } else { // 32 bit of weights
      if (isOffset64b) {
        if (isIndex64b) {
          REF(w_ref.data(), indices.data(), offsets_or_lengths);
          JIT(float,
              int64_t,
              int64_t,
              w.data(),
              indices.data(),
              offsets_or_lengths);
        } else { // 32 bit indices
          REF(w_ref.data(), indices_32.data(), offsets_or_lengths);
          JIT(float,
              int32_t,
              int64_t,
              w.data(),
              indices_32.data(),
              offsets_or_lengths);
        }
      } else { // 32 bit offset
        if (isIndex64b) {
          REF(w_ref.data(), indices.data(), offsets_or_lengths_32);
          JIT(float,
              int64_t,
              int32_t,
              w.data(),
              indices.data(),
              offsets_or_lengths_32);
        } else { // 32 bit indices
          REF(w_ref.data(), indices_32.data(), offsets_or_lengths_32);
          JIT(float,
              int32_t,
              int32_t,
              w.data(),
              indices_32.data(),
              offsets_or_lengths_32);
        }
      }
    }

    EXPECT_EQ(success, success_ref)
        << "return vals differ, reference is: " << success_ref
        << " ,fbgemm is: " << success;
    if (success) {
      for (int i = 0; i < h.size(); ++i) {
        EXPECT_EQ(h[i], h_ref[i])
            << "results for h differ at (" << i << ") reference: " << h_ref[i]
            << ", FBGEMM: " << h[i] << " emb dim :" << embedding_dim;
      }

      for (int i = 0; i < w.size(); ++i) {
        float w_, w_ref_;
        if (isWeightFp16) {
          w_ = cpu_half2float(w_fp16[i]);
          w_ref_ = cpu_half2float(w_fp16_ref[i]);
        } else {
          w_ = w[i];
          w_ref_ = w_ref[i];
        }
        EXPECT_EQ(w_, w_ref_)
            << "results for w differ at (" << i << ") reference: " << w_ref_
            << ", FBGEMM: " << w_ << " emb dim :" << embedding_dim;
      }
    }
  }
}

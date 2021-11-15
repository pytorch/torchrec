/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include <functional>

#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

template <
    typename InType,
    typename IndexType,
    typename OffsetType = std::int32_t>
class EmbeddingSpMDMKernelSignature {
 public:
  /**
   * Behavior is as the follow pseudocode
   * (when use_offsets == true, lengths[i] == offsets[i + 1] - offsets[i])
   * (when is_weight_positional == true, use weights[j - offsets[i]] instead of
   *  weights[j])
   *
   * for i in range(output_size):
   *  out[i * block_size : (i + 1) * block_size] = 0
   *  for j in range(offsets[i], offsets[i + 1]):
   *   for k in range(block_size):
   *    out[i * block_size + k] += input[indices[j] * block_size + k] *
   *                               weights ? weights[j] : 1;
   *  if normalize_weights and lengths[i] > 0:
   *   out[i * block_size : (i + 1) * block_size] /= lengths[i]
   *
   * @param data_size the number of rows in embedding table
   */
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t data_size,
      const InType* input,
      const IndexType* indices,
      const OffsetType* offsets_or_lengths,
      const float* weights, // optional, can be null for non-weighted sum
      float* out)>;
};

/**
 * @tparam InType can be float, float16, or uint8_t
 * @tparam IndexType can be int32_t or int64_t
 * @tparam IndexType can be int32_t or int64_t
 *
 * @param use_offsets If true, the generated code assumes we will pass offsets
 *                    instead of lengths that confirms PyTorch EmbeddingBag
 *                    interface. In this case, the length of offsets array
 *                    should be output_size + 1 and offsets[output_size] should
 *                    be index_size.
 *                    If false, the generate code assumes we will pass lengths
 *                    that confirms Caffe2 SparseLengthsSum interface.
 */
template <
    typename InType,
    typename IndexType,
    typename OffsetType = std::int32_t>
FBGEMM_API
    typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType>::Type
    GenerateEmbeddingSpMDM(
        const std::int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch = 16,
        bool is_weight_positional = false,
        bool use_offsets = true);

/**
 * @param output_stride If -1, output_stride is same as block_size
 * @param input_stride If -1, input_stride is same as block_size
 */
template <
    typename InType,
    typename IndexType,
    typename OffsetType = std::int32_t>
FBGEMM_API
    typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType>::Type
    GenerateEmbeddingSpMDMWithStrides(
        const std::int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch = 16,
        bool is_weight_positional = false,
        bool use_offsets = true,
        std::int64_t output_stride = -1,
        std::int64_t input_stride = -1);

/**
 * @tparam IndexType can be int32_t or int64_t
 * @tparam OffsetType can be int32_t or int64_t
 * @param bit_rate can be 2 or 4
 */
template <typename IndexType, typename OffsetType = std::int32_t>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<
    std::uint8_t,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMNBit(
    int bit_rate,
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false,
    bool use_offsets = true);

template <
    typename InType,
    typename IndexType,
    typename OffsetType = std::int32_t>
class EmbeddingSpMDMRowWiseSparseKernelSignature {
 public:
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t uncompressed_data_size,
      // TODO: add compressed_data_size and check array bound
      const InType* input,
      const IndexType* indices,
      const OffsetType* offsets_or_lengths,
      const float* weights, // optional, can be null for non-weighted sum
      float* out,
      const std::int32_t* compressed_indices_table)>;
};

/**
 * @tparam InType can be float, float16, or uint8_t
 * @tparam IndexType can be int32_t or int64_t
 * @tparam OffsetType can be int32_t or int64_t
 */
template <
    typename InType,
    typename IndexType,
    typename OffsetType = std::int32_t>
FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    InType,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMRowWiseSparse(
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false,
    bool use_offsets = true);

/**
 * @tparam IndexType can be int32_t or int64_t
 * @tparam OffsetType can be int32_t or int64_t
 * @param bit_rate can be 2 or 4
 */
template <typename IndexType, typename OffsetType = std::int32_t>
FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    std::uint8_t,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse(
    int bit_rate,
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false,
    bool use_offsets = true);

/**
 * @return The number of rows processed. If smaller than num_rows, an error
 *         must have happened at the last row processed.
 */
template <typename IndexType>
class SparseAdaGradSignature {
 public:
  using Type = std::function<int(
      int num_rows, // number of rows reading
      std::uint64_t param_size, // total number of parameters
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const IndexType* indices, // indices of each row
      float epsilon,
      float lr,
      float weight_decay,
      const double* counter, // used for weight_decay adjusted for frequency
                             // nullptr when frequency adjustment is not used.
                             // ignored when the kernel is generated with
                             // use_weight_decay = false.
      std::int64_t counter_halflife)>; // frequency adjust happens only after
};

template <typename IndexType>
FBGEMM_API typename SparseAdaGradSignature<IndexType>::Type
GenerateSparseAdaGrad(
    int block_size, // number of parameters per row
    bool rowwise = false,
    int prefetch = 16,
    bool use_weight_decay = false);

// RowWiseSparseAdaGrad fused with SLS gradient
// Weights can be either float or float16
template <
    typename IndexType,
    typename OffsetType = std::int32_t,
    typename DataType = float>
class RowWiseSparseAdaGradFusedSignature {
 public:
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t data_size, // number of rows in w
      DataType* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const IndexType* indices, // indices of each row
      const OffsetType* offsets_or_lengths,
      float epsilon,
      float lr)>;
};

/**
 * @param grad_stride If -1, grad_stride is same as block size
 */
template <
    typename IndexType,
    typename OffsetType = std::int32_t,
    typename DataType = float>
FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<
    IndexType,
    OffsetType,
    DataType>::Type
GenerateRowWiseSparseAdaGradFused(
    int block_size, // number of parameters per row
    int prefetch = 16,
    bool use_offsets = true,
    bool use_stochastic_rounding = true,
    int grad_stride = -1);

namespace internal {
// Specialization for block size 1 internally called by GenerateEmbeddingSpMDM
template <typename InType, typename IndexType, typename OffsetType>
FBGEMM_API bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true);

template <typename IndexType, bool HAS_WEIGHTS>
void compressed_indices_remap_avx512(
    std::int32_t offsets_numel,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights);

} // namespace internal

template <typename IndexType>
FBGEMM_API void compressed_indices_remap(
    std::int32_t offsets_numel,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights);

} // namespace fbgemm

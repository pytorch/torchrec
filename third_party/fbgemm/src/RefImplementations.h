/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <cstdint>

#include "fbgemm/ConvUtils.h"
#include "fbgemm/FbgemmI8Spmdm.h"
#include "fbgemm/Types.h"

namespace fbgemm {

/**
 * @brief Reference implementation of requantization step.
 * int32 multiplier
 * @param bias can be nullptr
 */
FBGEMM_API void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const std::int32_t* inp,
    std::uint8_t* out,
    std::int32_t C_multiplier,
    std::int32_t C_right_shift,
    std::int32_t C_zero_point,
    std::int32_t A_zero_point,
    std::int32_t B_zero_point,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false);

/**
 * @brief Reference implementation of requantization step.
 * float multiplier
 * @param bias can be nullptr
 * @param ncols_per_quant_group the number of columns share the same
 *        quantization parameter.
 *        ncols_per_quant_group == N : per-tensor quantization
 *        ncols_per_quant_group == N / groups : per-group quantization
 *        ncols_per_quant_group == 1 : per-channel quantization
 */
FBGEMM_API void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const std::int32_t* inp,
    std::uint8_t* out,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t A_zero_point,
    const std::int32_t* B_zero_point,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    int ncols_per_quant_group,
    bool fuse_relu = false);

/**
 * @brief Reference implementation of matrix multiply with uint8 for A,
 * int8 for B, and 32-bit accumulation.
 */
FBGEMM_API void matmul_u8i8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const std::uint8_t* Aint8,
    const std::int8_t* Bint8,
    std::int32_t* Cint32);

/**
 * @brief Reference implementation of matrix multiply with uint 8 for A,
 * int8 for B, and 16-bit accumulation.
 */
FBGEMM_API void matmul_u8i8acc16_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int brow,
    const std::uint8_t* Aint8,
    const std::int8_t* Bint8,
    std::int32_t* Cint32);

/**
 * @brief Reference implementation of cblas_sgemm in MKL/BLAS.
 */
FBGEMM_API void cblas_sgemm_ref(
    const matrix_op_t transa,
    const matrix_op_t transb,
    const int m,
    const int n,
    const int k,
    float alpha,
    const float* Afp32,
    int lda,
    const float* Bfp32,
    int ldb,
    float beta,
    float* Cfp32,
    int ldc);

FBGEMM_API void cblas_gemm_i64_i64acc_ref(
    matrix_op_t transa,
    matrix_op_t transb,
    int M,
    int N,
    int K,
    const std::int64_t* A,
    int lda,
    const std::int64_t* B,
    int ldb,
    bool accumulate,
    std::int64_t* C,
    int ldc);

/**
 * @brief Reference implementation to compute row_offsets (sums of rows of A).
 */
FBGEMM_API void row_offsets_u8acc32_ref(
    int M,
    int K,
    int ld,
    const std::uint8_t* Aint8,
    std::int32_t* row_offsets);

/**
 * @brief Reference implementation to compute adjusted col_offsets (sum of
 * columns of B and adjusted with B_zero_point)
 *
 * @param ncols_per_quant_group see ncols_per_quant_group in
 *        requantize_u8acc32_ref
 */
FBGEMM_API void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const std::int8_t* Bint8,
    const std::int32_t* B_zero_point,
    std::int32_t* col_offsets,
    int ncols_per_quant_group);

/**
 * @brief Reference implementation of SPMDM (sparse matrix times dense matrix).
 *
 * @param groups when > 1, for gth group, we multiply
 *               A[:,g*(A.ncols/groups):(g+1)*(A.ncols/groups)] sub-matrix with
 *               B[:,g*(B.ncols/groups):(g+1)*(B.ncols/groups)] sub-matrix .
 */
FBGEMM_API void spmdm_ref(
    int M,
    const std::uint8_t* A,
    int lda,
    CompressedSparseColumn& B,
    bool accumulation,
    std::int32_t* C,
    int ldc,
    int groups = 1);

/*
 * @brief Trim a 32-bit integer to a 16-bit integer.
 */
int32_t clip_16bit(int32_t x);

/*
 * @brief Reference implementation of convolution operation.
 * The activations A are assumed to be in NHiWiC format.
 * The filters B are assumed to be in RSCK format.
 * The output C is assumed to be in NHoWoC format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void conv_ref(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* B,
    std::int32_t* C);

/*
 * @brief Transforms weights from  G K/G (R S C/G) to G (R S C/G) K/G format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void transposeConvWeights(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

/*
 * @brief Reference implementation of im2col operation.
 *
 * For 2D:
 * The input A is assumed to be in NHiWiC format.
 * The output A is assumed to be in NHoWoRSC format.
 *
 * For 3D:
 * The input A is assumed to be in NTiHiWiC format.
 * The output A is assumed to be in NToHoWoK0K1K2C format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void im2col_ref(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    std::uint8_t* Ao);

template <
    typename InType = std::uint8_t,
    typename IndexType = std::int64_t,
    typename OffsetType = std::int32_t>
FBGEMM_API bool EmbeddingSpMDM_ref(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true,
    std::int64_t output_stride = -1,
    std::int64_t input_stride = -1);

template <typename IndexType = std::int64_t, typename OffsetType = std::int32_t>
FBGEMM_API bool EmbeddingSpMDMNBit_ref(
    int bit_rate,
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const std::uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true);

template <
    typename InType = std::uint8_t,
    typename IndexType = std::int64_t,
    typename OffsetType = std::int32_t>
FBGEMM_API bool EmbeddingSpMDMRowWiseSparse_ref(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t uncompressed_data_size,
    // const std::int64_t compressed_data_size,
    const InType* input,
    const IndexType* indices,
    const std::int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true);

template <typename IndexType = std::int64_t, typename OffsetType = std::int32_t>
FBGEMM_API bool EmbeddingSpMDMNBitRowWiseSparse_ref(
    int bit_rate,
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t uncompressed_data_size,
    // const std::int64_t compressed_data_size,
    const std::uint8_t* input,
    const IndexType* indices,
    const std::int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true);

/**
 * @param num_rows number of rows reading
 * @param block_size number of parameters per rows
 * @param param_size total number of parameters
 * @param w input parameters
 * @param g input gradients
 * @param h input momentum
 * @param indices indices of each row
 * @param counter used for weight_decay adjusted for frequency. nullptr when
 *                frequency adjustment is not used. Ignored when weight_decay
 *                == 0
 * @param counter_halflife weight_decay is adjusted only after this number of
 *                         iterations
 */
template <typename IndexType>
FBGEMM_API int sparse_adagrad_ref(
    int num_rows,
    int block_size,
    std::uint64_t param_size,
    float* w,
    const float* g,
    float* h,
    const IndexType* indices,
    float epsilon,
    float lr,
    float weight_decay = 0.f,
    const double* counter = nullptr,
    const int64_t counter_halflife = 0);

/**
 * @param num_rows number of rows reading
 * @param block_size number of parameters per rows
 * @param param_size total number of parameters
 * @param w input parameters
 * @param g input gradients
 * @param h input momentum
 * @param indices indices of each row
 * @param counter used for weight_decay adjusted for frequency. nullptr when
 *                frequency adjustment is not used. Ignored when weight_decay
 *                == 0
 * @param counter_halflife weight_decay is adjusted only after this number of
 *                         iterations
 */
template <typename IndexType>
FBGEMM_API int rowwise_sparse_adagrad_ref(
    int num_rows,
    int block_size,
    std::uint64_t param_size,
    float* w,
    const float* g,
    float* h,
    const IndexType* indices,
    float epsilon,
    float lr,
    float weight_decay = 0.f,
    const double* counter = nullptr,
    const int64_t counter_halflife = 0);

template <typename DataType, typename IndexType, typename OffsetType>
FBGEMM_API int rowwise_sparse_adagrad_fused_ref(
    std::int64_t block_size,
    std::int64_t output_size,
    std::int64_t index_size,
    std::int64_t data_size,
    DataType* w, // input/output parameters
    const float* g, // inupt gradients
    float* h, // input/output momentums
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    float epsilon,
    float lr,
    bool use_offsets = true,
    bool use_stochastic_rounding = true, // For DataType=float16
    int emu_vector_size = 8,
    std::int64_t grad_stride = -1);

template <typename IndexType>
FBGEMM_API void compressed_indices_remap_ref(
    std::int32_t offsets_len,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights);

} // namespace fbgemm

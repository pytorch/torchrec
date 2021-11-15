/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
// This file defines common utilities used in code compiled with avx2/avx512
// flags.

#include <string>

namespace fbgemm {

enum class FBGEMM_ENUM_CLASS_API QuantizationGranularity {
  TENSOR,
  GROUP,
  OUT_CHANNEL,
};

/**
 * @brief A struct to represent a block of a matrix.
 */
struct FBGEMM_API block_type_t {
  int row_start;
  int row_size;
  int col_start;
  int col_size;

  std::string toString() const {
    std::string out = "";
    out += "row start:" + std::to_string(row_start) + ", ";
    out += "row size:" + std::to_string(row_size) + ", ";
    out += "col start:" + std::to_string(col_start) + ", ";
    out += "col size:" + std::to_string(col_size);
    return out;
  }
};

/**
 * @brief A struct to represent all the requantization parameters.
 *
 * Please note that this is different from RequantizationParams in
 * QuantUtilsAvx2.h as it combines all the parameters needed for various
 * quantization granularities
 */
template <typename BIAS_TYPE = std::int32_t>
struct requantizationParams_t {
  using BIAS_T = BIAS_TYPE;
  std::int32_t A_zero_point;
  const std::int32_t* B_zero_point;
  std::int32_t C_zero_point;
  const float* C_multiplier;
  const std::int32_t* row_offsets;
  const std::int32_t* col_offsets;
  const BIAS_T* bias;
  std::uint32_t ncols;
  int groups;
  const float* act_times_w_scale;
};

/**
 * @brief A struct to represent all the parameters for requantizing for floats.
 */
struct requantizationForFloatParams_t {
  std::int32_t A_zero_point;
  const std::int32_t* B_zero_point;
  float A_scale;
  const float* B_scale;
  const std::int32_t* row_offsets;
  const std::int32_t* col_offsets;
  const float* bias;
  std::uint32_t ncols;
  int groups;
};

/**
 * @brief Allocate size bytes of uninitialized storage whose alignment is
 * specified by align.
 */
FBGEMM_API void*
fbgemmAlignedAlloc(size_t align, size_t size, bool raiseException = false);

/**
 * @brief Free memory allocated by fbgemmAlignedAlloc
 */
FBGEMM_API void fbgemmAlignedFree(void* p);

} // namespace fbgemm

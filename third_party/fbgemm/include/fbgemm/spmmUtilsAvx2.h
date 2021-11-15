/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "./FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {
struct FBGEMM_API trRequantizationParams_t {
  std::int32_t act_zero_point; // activation zero point
  const std::int32_t* weight_zero_points; // weight zero point(s)
  std::int32_t C_zero_point;
  const float C_scale;
  const std::int32_t* weight_row_offsets;
  const std::int32_t* act_col_offsets;
  const float* bias;
  const float* act_times_w_scale;
};

template <
    bool FUSE_RELU,
    bool ACT_SYMMETRIC, // whether activation matrix is symmetric
    bool WEIGHT_SYMMETRIC, // whether weight matrix is symmetric
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN>
FBGEMM_API void trRequantizeOpt(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const trRequantizationParams_t& rParams);
} // namespace fbgemm

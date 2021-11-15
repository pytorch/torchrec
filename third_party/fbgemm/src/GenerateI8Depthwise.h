/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <cstdint>

namespace fbgemm {

class GenI8Depthwise {
 public:
  using jit_kernel_signature = void (*)(
      const std::uint8_t* a,
      const std::int8_t* b,
      std::int32_t* c,
      std::int32_t* a_sum, // row_wise sum of A
      int h,
      int w,
      int ic, // the number of input channels == the number of groups
      const int* mask,
      int A_zero_point);

  jit_kernel_signature getOrCreate(
      int D, // dimension
      std::array<int, 3> F, // filter size (K_T, K_H, K_W)
      int oc_per_g, // the number of output channels per group
      bool compute_a_sum,
      int remainder, // the number of channels in the remainder loop
      int prev_skip,
      int next_skip,
      int top_skip,
      int bottom_skip,
      int left_skip,
      int right_skip);
};

} // namespace fbgemm

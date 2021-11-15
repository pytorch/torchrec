/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>

namespace fbgemm {

namespace internal {

// A constant array to initialize an AVX2 register to be used as a 32-bit
// granularity mask.
// clang-format off
alignas(64) static const int avx2_ps_or_epi32_masks[9][8] = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  {  0,  0,  0,  0,  0,  0,  0,  0,  },
  { -1,  0,  0,  0,  0,  0,  0,  0,  },
  { -1, -1,  0,  0,  0,  0,  0,  0,  },
  { -1, -1, -1,  0,  0,  0,  0,  0,  },
  { -1, -1, -1, -1,  0,  0,  0,  0,  },
  { -1, -1, -1, -1, -1,  0,  0,  0,  },
  { -1, -1, -1, -1, -1, -1,  0,  0,  },
  { -1, -1, -1, -1, -1, -1, -1,  0,  },
  { -1, -1, -1, -1, -1, -1, -1, -1,  },
};

// mask can be accessed by avx2_ps_or_epi32_combined_mask[(8 - remainder) % 8]
static const int avx2_ps_or_epi32_combined_mask[16] = {
  -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
};

// A constant array to initialize an SSE register to be used as a 8-bit
// granularity mask.
alignas(64) static const std::int8_t sse_epi8_masks[17][16] = {
  {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, },
  { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, },
};
// clang-format on

} // namespace internal

} // namespace fbgemm

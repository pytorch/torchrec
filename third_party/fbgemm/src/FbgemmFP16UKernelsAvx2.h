/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include "fbgemm/Types.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmFPCommon.h"

namespace fbgemm {

using GemmParamsFP16 = GemmParams<float16>;

void NOINLINE gemmkernel_1x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_2x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_3x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_4x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_5x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_6x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp);

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "./GenerateKernel.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * Get or Create the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate<
    inst_set_t::avx512_vnni>(bool accum, int32_t mc, int32_t nc, int32_t kc) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  return codeObj.getOrCreate<inst_set_t::avx512_vnni>(accum, mc, nc, kc);
}

/**
 * Get or Create the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate<
    inst_set_t::avx512_vnni_ymm>(bool accum, int32_t mc, int32_t nc, int32_t kc) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  return codeObj.getOrCreate<inst_set_t::avx512_vnni_ymm>(accum, mc, nc, kc);
}

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * @brief Create instruction sequence to generate 16-bit 1s
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:15] will have 0x0001, dest[16:31]
 *             will have 0x0001 and so on
 */
template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::avx2, int>::type = 0>
void gen16BitVectorOne(x86::Emitter* a, T dest) {
  a->vpcmpeqw(dest, dest, dest);
  a->vpsrlw(dest, dest, 15);
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<
        instSet == inst_set_t::avx512 || instSet == inst_set_t::avx512_ymm ||
            instSet == inst_set_t::avx512_vnni ||
            instSet == inst_set_t::avx512_vnni_ymm,
        int>::type = 0>
void gen16BitVectorOne(x86::Emitter* a, T dest) {
  a->vpternlogd(dest, dest, dest, 0xff);
  a->vpsrlw(dest, dest, 15);
}

/**
 * @brief Emit instruction do load 32-bit integer. AVX512 has
 *        different instrunction to load registers with index >= 16
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param dest Destination vector register
 */
template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::avx2, int>::type = 0>
void emitLoadDWord(x86::Emitter* a, T dest, const x86::Mem& ptr) {
  a->vmovdqa(dest, ptr);
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<
        instSet == inst_set_t::avx512 || instSet == inst_set_t::avx512_ymm ||
            instSet == inst_set_t::avx512_vnni ||
            instSet == inst_set_t::avx512_vnni_ymm,
        int>::type = 0>
void emitLoadDWord(x86::Emitter* a, T dest, const x86::Mem& ptr) {
  a->vmovdqa32(dest, ptr);
}

/**
 * @brief Emit partial extract from Wide regiter to Half Register, eg.
 *        Zmm -> Ymm or Ymm -> Xmm
 * @tparam instSet instruction set to be used
 *
 * @param half Destination (half) vector register
 * @param vec Source (full) vector register
 * @param idx Index of of the half vector 0 or 1
 */
template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<
        instSet == inst_set_t::avx512 || instSet == inst_set_t::avx512_ymm ||
            instSet == inst_set_t::avx512_vnni ||
            instSet == inst_set_t::avx512_vnni_ymm,
        int>::type = 0>
void emitExtractHalfVector(
    x86::Emitter* a,
    x86::Ymm half,
    const x86::Zmm vec,
    int idx) {
  a->vextracti32x8(half, vec, idx);
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<
        instSet == inst_set_t::avx512 || instSet == inst_set_t::avx512_ymm ||
            instSet == inst_set_t::avx512_vnni ||
            instSet == inst_set_t::avx512_vnni_ymm,
        int>::type = 0>
void emitExtractHalfVector(
    x86::Emitter* a,
    x86::Xmm half,
    x86::Ymm vec,
    int idx) {
  a->vextracti32x4(half, vec, idx);
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::avx2, int>::type = 0>
void emitExtractHalfVector(
    x86::Emitter* a,
    x86::Xmm half,
    x86::Ymm vec,
    int idx) {
  a->vextracti128(half, vec, idx);
}

/**
 * @brief Create instruction sequence to generate 8-bit 1s
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:7] will have 0x01, dest[8:15]
 *             will have 0x01 and so on
 */
template <
    typename T,
    typename std::enable_if<std::is_same<T, x86::Ymm>::value, int>::type = 0>
void gen8BitVectorOne(x86::Emitter* a, T dest) {
  a->vpcmpeqw(dest, dest, dest);
  a->vpabsb(dest, dest);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, x86::Zmm>::value, int>::type = 0>
void gen8BitVectorOne(x86::Emitter* a, T dest) {
  a->vpternlogd(dest, dest, dest, 0xff);
  a->vpabsb(dest, dest);
}

/**
 * @brief Generates instruction sequence to compute s32 += U8 * I8
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param cReg contains result
 *
 */

template <
    inst_set_t INST_SET,
    typename std::enable_if<
        INST_SET == inst_set_t::avx2 || INST_SET == inst_set_t::avx512,
        int>::type = 0>
void genU8I8S32FMA(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t aReg,
    typename simd_info<INST_SET>::vec_reg_t bReg,
    typename simd_info<INST_SET>::vec_reg_t cReg,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  a->vpmaddubsw(tmpReg, aReg, bReg);
  a->vpmaddwd(tmpReg, oneReg16Bit, tmpReg);
  a->vpaddd(cReg, tmpReg, cReg);
}

template <
    inst_set_t INST_SET,
    typename std::enable_if<INST_SET == inst_set_t::avx512_vnni, int>::type = 0>
void genU8I8S32FMA(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t aReg,
    typename simd_info<INST_SET>::vec_reg_t bReg,
    typename simd_info<INST_SET>::vec_reg_t cReg,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  a->vpdpbusd(cReg, aReg, bReg);
}

/**
 * @brief Add 4 consecutive numbers of type uint8
 *        and emit their sum as 32-bit numbers.
 *        i.e., dest[0:31] contains
 *        src[0:7] + src[8:15] + src[16:23] + src[24:31]
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param dest contains result
 *
 */
template <
    inst_set_t INST_SET,
    typename std::enable_if<
        INST_SET == inst_set_t::avx2 || INST_SET == inst_set_t::avx512,
        int>::type = 0>
void genU8Sum4(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t src,
    typename simd_info<INST_SET>::vec_reg_t dest,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  gen8BitVectorOne(a, tmpReg);
  a->vpmaddubsw(tmpReg, src, tmpReg);
  a->vpmaddwd(tmpReg, tmpReg, oneReg16Bit);
  a->vpaddd(dest, tmpReg, dest);
  /*a->vxorps(tmpReg, tmpReg, tmpReg);*/
  /*a->vmpsadbw(tmpReg, src, tmpReg, static_cast<asmjit::Imm>(0));*/
  /*a->vpermilps(tmpReg, tmpReg, static_cast<asmjit::Imm>(4));*/
  /*a->vpmovzxwd(tmpReg, tmpReg.half());*/
  /*a->vpaddd(dest, tmpReg, dest);*/
}

template <
    inst_set_t INST_SET,
    typename std::enable_if<INST_SET == inst_set_t::avx512_vnni, int>::type = 0>
void genU8Sum4(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t src,
    typename simd_info<INST_SET>::vec_reg_t dest,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  gen8BitVectorOne(a, tmpReg);
  a->vpdpbusd(dest, src, tmpReg);
}

/**
 * @brief Add 8 consecutive numbers of type uint8
 *        and emit their sum as 16-bit numbers.
 *        i.e., dest[0:15] contains
 *        src[0:7] + src[8:15] + src[16:23] + src[24:31]
 *        src[32:39] + src[40:47] + src[48:55] + src[56:63]
 *
 *        and
 *
 *        dest[64:79] contains
 *        src[64:71] + src[71:79] + src[80:87] + src[88:95]
 *        src[96:103] + src[104:111] + src[112:119] + src[120:127]
 *
 *        so on
 *
 * @tparam T Register type of destination, e.g., x86::Ymm or x86::Zmm
 *
 * @param dest contains result
 *
 */
template <typename T>
void genU8Sum8(x86::Emitter* a, T src, T dest, T tmpReg) {
  a->vxorps(tmpReg, tmpReg, tmpReg);
  a->vpsadbw(tmpReg, src, tmpReg);
  a->vpaddd(dest, tmpReg, dest);
}

/**
 * @brief Broadcast lower 8-bits of src to destination vector
 *        register.
 */
template <typename T>
void broadcast8Bit(x86::Emitter* a, x86::Gp src, T dest) {
  // move src to dest
  auto xmm = dest.xmm();
  a->movq(xmm, src);
  a->vpbroadcastb(dest, xmm);
}

} // namespace fbgemm

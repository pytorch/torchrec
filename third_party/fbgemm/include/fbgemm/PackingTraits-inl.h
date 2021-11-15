/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

/*
 * This file configures the important cache blocking parameters and registers
 * blocking parameters for the matrix multiplication loops inside FBGEMM.
 *
 * ROW_INTERLEAVE: the number of interleaved rows to use vpmaddubsw instructions
 * for packing B matrix. For 32-bit accumulation, ROW_INTERLEAVE = 4; For 16-bit
 * accumulation, ROW_INTERLEAVE = 2.
 *
 * VLEN: the vector length of one SIMD register. For avx2, VLEN = 256; For
 * avx512, VLEN = 512.
 *
 * NR: the register blocking parameters for N dimension. The total registers
 * used in N dimension for C accumulations are NR * ROW_INTERLEAVE * 8 (int8) /
 * VLEN.
 *
 * MR: the register blocking parameters for M dimension. The total number of
 * registers used in M dimension for C accumulations is MR.  This indicates the
 * number of vpbroadcastw instructions for A.
 *
 * (MR) * (NR * ROW_INTERLEAVE * 8 (int8) / VLEN): the number of registers used
 * for C accumulations. This number should be less than the maximum registers we
 * can use for C accumulations (A max of 12 out of 16 ymm registers for avx2; a
 * max of 28 out of 32 zmm registers for avx512 ). The remaining are used for A
 * matrix loading, B matrix loading and as temp registers. C accumulation
 * registers should be as large as possible to increase the register
 * utilization.
 *
 * MCB: the cache blocking parameters for M dimension. MCB needs to be a
 * multiple of MR.
 *
 * NCB: the cache blocking parameters for N dimension. NCB needs to be a
 * multiple of NR.
 *
 * KCB: the cache blocking parameters for K dimension. KCB needs to be a
 * multiple of ROW_INTERLEAVE.
 */

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{12}; ///< Register block for M dimension.
  static constexpr int NR_MIN{8}; ///< Minimum register block for N dimension.
                                  ///< 8 because 8*ROW_INTERLEAVE int8 elements
                                  ///< completely fill a 256-bit wide vector.
  static constexpr int NR{8}; ///< Register block for N dimension.
                              ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 4 = 8.
                              ///< Total registers used for N dimension: NCB/NR.
                              ///< Here we use 12 x 1 ymm register blocking for
                              ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      120}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      8}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{3}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 256-bit wide vector.

  static constexpr int NR{
      16}; ///< Register block for N dimension;
           ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 2 = 16.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 3 x 4 ymm register blocking for the
           ///< registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for float input and float
 * accumulation.
 *
 * This is picked when template paramtere T is of float type and instruction
 * set is avx2.
 */
template <>
struct PackingTraits<float, float, inst_set_t::avx2> {
  static constexpr int MR{3}; ///< Register block for M dimension
  static constexpr int NR{32}; ///< Register block for N dimension

  static constexpr int ROW_INTERLEAVE{1}; ///< No Row interleave.

  static constexpr int MCB{
      24}; ///< Cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< Cache block for K dimension

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for fp16 input and float
 * accumulation.
 *
 * This is picked when template parameter T is of float16 type and instruction
 * set is avx2
 */
template <>
struct PackingTraits<float16, float, inst_set_t::avx2> {
  static constexpr int BCOL{8};
  static constexpr int ROW_INTERLEAVE{1};
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{14}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      32}; ///< Register block for N dimension.
           ///< Must be a multiple of 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector. Total registers used for
           ///< N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
           ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
           ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      56}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      32}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512_ymm.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx512_ymm,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{7}; ///< Register block for M dimension.
  static constexpr int NR_MIN{16}; ///< Minimum register block for N dimension.
                                  ///< 8 because 8*ROW_INTERLEAVE int8 elements
                                  ///< completely fill a 256-bit wide vector.
  static constexpr int NR{32}; ///< Register block for N dimension.
                              ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 4 = 8.
                              ///< Total registers used for N dimension: NCB/NR.
                              ///< Here we use 12 x 1 ymm register blocking for
                              ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      56}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      32}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{6}; ///< Register block for M dimension
  static constexpr int NR_MIN{
      32}; ///< Minimum register block for N dimension;
           ///< 32 because 32*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      128}; ///< Register block for N dimension;
            ///< Must be a multiple of 32 because 32*ROW_INTERLEAVE int8
            ///< elements completely fill a 512-bit wide vector. Total registers
            ///< used for N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
            ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
            ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      128}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512_ymm.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx512_ymm,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{6}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 256-bit wide vector.

  static constexpr int NR{
      16}; ///< Register block for N dimension;
           ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 2 = 16.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 3 x 4 ymm register blocking for the
           ///< registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Helper struct to type specialize for int16_t and int32_t together.
 */
template <typename T>
struct is_16or32bit {
  static constexpr bool value =
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value;
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit/16-bit
 * integers.
 *
 * Since there is no int16_t accumulation for AVX512 VNNI, we redirect int16_t
 * to int32_t accumulation and use the same blocking parameters as int32_t.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512_vnni.
 */
template <typename T, typename accT>
struct PackingTraits<
    T,
    accT,
    inst_set_t::avx512_vnni,
    typename std::enable_if<
        is_8bit<T>::value && is_16or32bit<accT>::value>::type> {
  static constexpr int MR{8}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      48}; ///< Register block for N dimension.
           ///< Must be a multiple of 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector. Total registers used for
           ///< N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
           ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
           ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      384}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      48}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit/16-bit
 * integers.
 *
 * Since there is no int16_t accumulation for AVX512 VNNI, we redirect int16_t
 * to int32_t accumulation and use the same blocking parameters as int32_t.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512_vnni_ymm.
 */
template <typename T, typename accT>
struct PackingTraits<
    T,
    accT,
    inst_set_t::avx512_vnni_ymm,
    typename std::enable_if<
        is_8bit<T>::value && is_16or32bit<accT>::value>::type> {
  static constexpr int MR{4}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      48}; ///< Register block for N dimension.
           ///< Must be a multiple of 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector. Total registers used for
           ///< N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
           ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
           ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      384}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      48}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.

  static std::tuple<int, int, int> getCacheBlockParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(MR));
  }
  static std::tuple<int, int, int, int> getKernelParams() {
      return std::tuple<int, int, int,int>(int(MCB), int(NCB), int(NR_MIN), int(NR));
  }
  static std::tuple<int, int, int> getMatrixPackAParams() {
      return std::tuple<int, int, int>(int(MCB), int(KCB), int(ROW_INTERLEAVE));
  }
  static std::tuple<int, int, int> getMatrixPackBParams() {
      return std::tuple<int, int, int>(int(KCB), int(NCB), int(ROW_INTERLEAVE));
  }
};

/**
 * @brief Packing parameter specialization for I64 GEMM
 * integers.
 *
 * This is picked when T is of int64 type and instruction
 * set is avx512.
 */
template <>
struct PackingTraits<int64_t, int64_t, inst_set_t::avx512> {
  static constexpr int MR{2}; ///< Register block for M dimension.
  static constexpr int NR_MIN{8}; ///< Minimum register block for N dimension.
                                  ///< 8 because 8 int64 elements
                                  ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      32}; ///< Register block for N dimension.
           ///< Must be a multiple of 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector. Total registers used for
           ///< N dimension: NR*8/VLEN. We use MR x
           ///< NR*8/VLEN zmm registers
           ///< for C accumulations.

  static constexpr int MCB{
      16}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{8}; ///< Cache block for K dimension.
};

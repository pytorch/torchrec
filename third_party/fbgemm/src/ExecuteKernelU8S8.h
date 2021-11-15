/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "./ExecuteKernel.h"

namespace fbgemm {

/**
 * @brief Execute Engine of uint 8 and int8 matrix
 * multiplication for the macro-kernel and output processing. ExecuteKernel is a
 * derived class of CodeGenBase.
 */
template <typename packingAMatrix, typename cT, typename processOutputType>
class ExecuteKernel<
    packingAMatrix,
    PackBMatrix<int8_t, typename packingAMatrix::accType>,
    cT,
    processOutputType>
    : public CodeGenBase<
          uint8_t,
          int8_t,
          int32_t,
          typename packingAMatrix::accType> {
 public:
  using BaseType =
      CodeGenBase<uint8_t, int8_t, int32_t, typename packingAMatrix::accType>;
  /**
   * @brief Constructor for initializing the parameters for macro-kernel and
   * output processing type.
   */
  ExecuteKernel(
      PackMatrix<packingAMatrix, uint8_t, typename packingAMatrix::accType>&
          packA,
      PackMatrix<
          PackBMatrix<int8_t, typename packingAMatrix::accType>,
          int8_t,
          typename packingAMatrix::accType>& packB,
      cT* matC,
      int32_t* C_buffer,
      int32_t ldc,
      const processOutputType& outputProcess,
      thread_type_t th_info,
      const BlockingFactors* params = nullptr);
  void execute(int kBlock);


 private:
  PackMatrix<packingAMatrix, uint8_t, typename packingAMatrix::accType>&
      packedA_; ///< Packed uint8 block of matrix A.
  PackMatrix<
      PackBMatrix<int8_t, typename packingAMatrix::accType>,
      int8_t,
      typename packingAMatrix::accType>& packedB_; ///< Packed int8 matrix B.
  cT* matC_; ///< Output for matrix C.
  int32_t* C_buffer_; ///< the accumulation buffer for matrix C.
  int32_t ldc_; ///< the leading dimension of matrix C.
  const processOutputType& outputProcess_; ///< output processing function for
                                           ///< matrix C in the macro-kernel.
  thread_type_t
      th_info_; ///<< the thread partition information (thread id and the number
                ///< of threads across the group, m, n dimensions.
  int mbSize_; ///< block size in the m dimension.
  int nbSize_; ///< block size in the n dimension.
  int nrMinSize_; ///< minimum register size in the n dimension.
  int nrSize_; ///< register size in the n dimension.
};

} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include "./GenerateKernel.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

/**
 * @brief Execute Engine for the macro-kernel and output processing.
 * ExecuteKernel is a derived class of CodeGenBase.
 */
template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename cT,
    typename processOutputType>
class ExecuteKernel : public CodeGenBase<
                          typename packingAMatrix::inpType,
                          typename packingBMatrix::inpType,
                          cT,
                          typename packingBMatrix::accType> {
 public:
  ExecuteKernel(
      PackMatrix<
          packingAMatrix,
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType>& packA,
      PackMatrix<
          packingBMatrix,
          typename packingBMatrix::inpType,
          typename packingBMatrix::accType>& packB,
      cT* matC,
      typename packingBMatrix::accType* C_buffer,
      int32_t ldc,
      const processOutputType& outputProcess,
      thread_type_t th_info,
      const BlockingFactors* params = nullptr);
  void execute(int kBlock);

 private:
  PackMatrix<
      packingAMatrix,
      typename packingAMatrix::inpType,
      typename packingAMatrix::accType>&
      packedA_; ///< Packed block of matrix A.
  PackMatrix<
      packingBMatrix,
      typename packingBMatrix::inpType,
      typename packingBMatrix::accType>& packedB_; ///< Packed matrix B.
  cT* matC_; ///< Output for matrix C.
  typename packingAMatrix::accType*
      C_buffer_; ///< the accumulation buffer for matrix C.
  int32_t ldc_; ///< the leading dimension of matrix C.
  const processOutputType& outputProcess_; ///< output processing function for
                                           ///< the C tile in the macro-kernel.
};

} // namespace fbgemm

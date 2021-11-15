/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "./GenerateKernel.h"
#include "./CodeGenHelpers.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * Generate AVX512 instructions for computing block in the rank-k update of
 * 16-bit Accmulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::genComputeBlock(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp /* unused (reserved for prefetching)*/,
    int rowRegs,
    int colRegs,
    int lda) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  // used for matrix A
  VecRegT AReg(29);

  VecRegT tmpReg(30);

  // We start allocating BRegs from zmm28 and then allocate zmm27 and so on.
  for (int j = 0; j < colRegs; ++j) {
    a->vmovups(
        VecRegT(28 - j), x86::dword_ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
  }

  for (int i = 0; i < rowRegs; ++i) {
    // broadcast A
    a->vpbroadcastw(
        AReg, x86::dword_ptr(buffer_A, (i * lda) * sizeof(uint8_t)));
    for (int j = 0; j < colRegs; ++j) {
      a->vpmaddubsw(tmpReg, AReg, VecRegT(28 - j));
      a->vpaddsw(VecRegT(i * colRegs + j), tmpReg, VecRegT(i * colRegs + j));
      // Prefetching is hurting performance in some cases
      // because prefetch instructions itself consumes a slot
      // in pipeline issue thus slowing down the kernel.
      // if((i == rowRegs - 1) && j % 2 == 0){
      // a->prefetcht0(x86::dword_ptr(B_pf, j*VecLen*sizeof(int8_t)));
      //}
    }
  }
}

/**
 * Get or Create the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <inst_set_t instSet>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc) {
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int, int, int, int> kernelSig;
  int kBlock;
  int nBlock;
  int mRegBlockSize;
  int nRegBlockSize;
  int nRegBlockSizeMin;
  int row_interleave;

  if (blocking_params) {
    kBlock = blocking_params->KCB;
    nBlock = blocking_params->NCB;
    mRegBlockSize = blocking_params->MR;
    nRegBlockSize = blocking_params->NR;
    nRegBlockSizeMin = blocking_params->NR_MIN;
    row_interleave = blocking_params->ROW_INTERLEAVE;
  } else {
    kBlock = PackingTraits<uint8_t, int16_t, instSet>::KCB;
    nBlock = PackingTraits<uint8_t, int16_t, instSet>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int16_t, instSet>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int16_t, instSet>::NR;
    nRegBlockSizeMin = PackingTraits<uint8_t, int16_t, instSet>::NR_MIN;
    row_interleave = PackingTraits<uint8_t, int16_t, instSet>::ROW_INTERLEAVE;
  }

  kernelSig = std::make_tuple(
      accum, mc, nc, nBlock, kBlock, mRegBlockSize, nRegBlockSize);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    x86::Assembler assembler(&code);
    x86::Emitter* a = assembler.as<x86::Emitter>();

#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum,
            mc,
            nc,
            nBlock,
            kBlock,
            mRegBlockSize,
            nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    assert(
        kc % row_interleave == 0 && "kc must be a multiple of row_interleave");
    assert(nc % nRegBlockSizeMin == 0 && "nc must be a multiple of NR_MIN");
    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        (maxMRegs + 1) * maxNRegs <= 29 &&
        "number of zmm registers for C + one row for loading B: \
      MR*(NR*ROW_INTERLEAVE*8/512) + (NR*ROW_INTERLEAVE*8/512)  \
      must be <= 29(available registers constraint)");
    int mRegBlocks = mc / mRegBlockSize;
    int mRegBlocksRem = mc % mRegBlockSize;

    // arguments to the function created
    x86::Gp buffer_A = a->zdi();
    x86::Gp buffer_B = a->zsi();
    x86::Gp B_pf = a->zdx();
    x86::Gp CBase = a->zcx();
    x86::Gp kSize = a->gpz(8);
    x86::Gp ldcReg = a->gpz(9);

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            uint8_t*,
            int8_t*,
            int8_t*,
            int32_t*,
            int,
            int>(asmjit::CallConv::kIdHost),
        a->environment());

    asmjit::FuncFrame frame;
    frame.init(func);

    frame.setDirtyRegs(
        x86::Reg::kGroupVec,
        asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
            asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
            asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
    frame.setDirtyRegs(
        x86::Reg::kGroupGp,
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();
    asmjit::Label LoopNBlocks = a->newLabel();
    asmjit::Label Loopk = a->newLabel();

    x86::Gp buffer_B_saved = a->gpz(10);
    x86::Gp C_Offset = a->gpz(11);
    // x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    x86::Gp jIdx = a->gpz(14);
    x86::Gp kIdx = a->gpz(15);

    // save B_buffer address
    a->mov(buffer_B_saved, buffer_B);
    // a->mov(B_pf_saved, B_pf);

    int currColRegs = nc * row_interleave / vectorLen;
    int colRegs = std::min(currColRegs, maxNRegs);
    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx.r32(), iIdx.r32());

      a->bind(LoopMBlocks);
      a->inc(iIdx);
      a->xor_(jIdx.r32(), jIdx.r32());

      a->bind(LoopNBlocks);
      a->inc(jIdx);

      int rowRegs = mRegBlockSize;

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(Loopk);
      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      genComputeBlock<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->add(
          buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(
          buffer_B,
          static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));
      // a->add(B_pf, static_cast<asmjit::Imm>(nBlock * row_interleave *
      // sizeof(int8_t)));

      a->cmp(kIdx, kSize);
      a->jl(Loopk);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);

      // reset A
      a->sub(buffer_A, kSize);

      // B for next block
      a->mov(buffer_B, buffer_B_saved);
      // using C_Offset as temp reg
      a->imul(
          C_Offset,
          jIdx,
          static_cast<asmjit::Imm>(
              nRegBlockSize * row_interleave * sizeof(int8_t)));
      a->add(buffer_B, C_Offset);

      // increment C for next block
      a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

      int jLoopTrips = currColRegs / maxNRegs;
      // jLoopTrips should be at least 1
      jLoopTrips = jLoopTrips ? jLoopTrips : 1;
      a->cmp(jIdx, jLoopTrips);
      a->jl(LoopNBlocks);

      // increment A for next block
      a->add(
          buffer_A,
          static_cast<asmjit::Imm>((rowRegs)*kBlock * sizeof(uint8_t)));

      // increment C for next A block
      a->sub(
          CBase,
          static_cast<asmjit::Imm>(
              jLoopTrips * nRegBlockSize * sizeof(int32_t)));
      a->imul(
          C_Offset,
          ldcReg,
          static_cast<asmjit::Imm>(rowRegs * sizeof(int32_t)));
      a->add(CBase, C_Offset);

      // reset B
      a->mov(buffer_B, buffer_B_saved);
      // a->mov(B_pf, B_pf_saved);

      a->cmp(iIdx, mRegBlocks);
      a->jl(LoopMBlocks);
    }
    // generate code for remainder
    if (mRegBlocksRem > 0) {
      asmjit::Label LoopNRem = a->newLabel();
      asmjit::Label LoopkRem = a->newLabel();
      int rowRegs = mRegBlocksRem;

      a->xor_(jIdx.r32(), jIdx.r32());
      a->bind(LoopNRem);
      a->inc(jIdx);

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopkRem);

      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      genComputeBlock<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->add(
          buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(
          buffer_B,
          static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));
      // a->add(B_pf, static_cast<asmjit::Imm>(nBlock * row_interleave *
      // sizeof(int8_t)));

      a->cmp(kIdx, kSize);
      a->jl(LoopkRem);

      // reset A
      a->sub(buffer_A, kSize);

      // B for next block
      a->mov(buffer_B, buffer_B_saved);
      // using C_Offset as temp reg
      a->imul(
          C_Offset,
          jIdx,
          static_cast<asmjit::Imm>(
              nRegBlockSize * row_interleave * sizeof(int8_t)));
      a->add(buffer_B, C_Offset);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);

      // increment C for next block
      a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

      int jLoopTrips = currColRegs / maxNRegs;
      // jLoopTrips should be at least 1
      jLoopTrips = jLoopTrips ? jLoopTrips : 1;
      a->cmp(jIdx, jLoopTrips);
      a->jl(LoopNRem);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp fn;
    asmjit::Error err;
    {
      std::unique_lock<std::mutex> lock(rtMutex_);
      err = runtime().add(&fn, &code);
    }
    if (err) {
      std::cout << "Error: in fn add" << std::endl;
      return nullptr;
    }

#if defined(FBGEMM_LOG_CODE)
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

/**
 * Instatiate the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
getOrCreate<inst_set_t::avx512>(bool accum, int32_t mc, int32_t nc, int32_t kc);

/**
 * Instatiate the AVX512_256 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
getOrCreate<inst_set_t::avx512_ymm>(bool accum, int32_t mc, int32_t nc, int32_t kc);

} // namespace fbgemm

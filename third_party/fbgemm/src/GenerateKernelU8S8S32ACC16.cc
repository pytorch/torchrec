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
 * Generate AVX2 instructions for computing block in the rank-k update of 16-bit
 * Accmulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
genComputeBlock<inst_set_t::avx2>(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp /* unused (reserved for prefetching)*/,
    int rowRegs,
    int colRegs,
    int lda) {
  using CRegs = x86::Ymm;
  static constexpr int vectorLen = simd_info<inst_set_t::avx2>::WIDTH_BYTES;

  // used for matrix A
  x86::Ymm AReg = x86::ymm13;
  x86::Ymm tmpReg = x86::ymm14;
  for (int i = 0; i < rowRegs; ++i) {
    // broadcast A
    a->vpbroadcastw(
        AReg, x86::dword_ptr(buffer_A, (i * lda) * sizeof(uint8_t)));
    for (int j = 0; j < colRegs; ++j) {
      a->vpmaddubsw(
          tmpReg, AReg, x86::dword_ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
      a->vpaddsw(CRegs(i * colRegs + j), tmpReg, CRegs(i * colRegs + j));
      // Prefetching is hurting performance in some cases
      // because prefetch instructions itself consumes a slot
      // in pipeline issue thus slowing down the kernel.
      // if((i == rowRegs - 1) && j % 2 == 0){
      // a->prefetcht0(x86::dword_ptr(B_pf, j*VLEN_*sizeof(int8_t)));
      //}
    }
  }
}

/**
 * Generate instructions for storing the C registers back to the memory
 * in 16-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::storeCRegs(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  VecT extractDestFull(simd_info<instSet>::NUM_VEC_REGS - 1);
  auto extractDestHalf = extractDestFull.half();

  for (int i = 0; i < rowRegs; ++i) {
    a->imul(C_Offset, ldcReg, static_cast<asmjit::Imm>(i * sizeof(int32_t)));
    for (int j = 0; j < colRegs; ++j) {
      for (int idx = 0; idx < 2; ++idx) {
        emitExtractHalfVector<instSet, VecT>(
            a, extractDestHalf, VecT(i * colRegs + j), idx);
        a->vpmovsxwd(extractDestFull, extractDestHalf);
        x86::Mem destAddr = x86::dword_ptr(
            a->zcx(), C_Offset, 0, (j * 2 + idx) * vectorLen);
        if (accum) {
          a->vpaddd(extractDestFull, extractDestFull, destAddr);
        }
        a->vmovups(destAddr, extractDestFull);
      }
    }
  }
}

/**
 * Get or Create the AVX2 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
getOrCreate<inst_set_t::avx2>(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc) {
  constexpr int vectorLen = simd_info<inst_set_t::avx2>::WIDTH_BYTES;

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
    kBlock = PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::KCB;
    nBlock = PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::NR;
    nRegBlockSizeMin =
        PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::NR_MIN;
    row_interleave =
        PackingTraits<uint8_t, int16_t, inst_set_t::avx2>::ROW_INTERLEAVE;
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
        getCodeLoggingFile<inst_set_t::avx2>(
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
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    (void)maxMRegs; // Suppress unused variable warning
    (void)maxNRegs; // Suppress unused variable warning
    assert(
        maxMRegs * maxNRegs <= 13 &&
        "MR*(NR*ROW_INTERLEAVE*8/256"
        "must be <= 13(available registers constraint)");

    int mRegBlocks = mc / mRegBlockSize;
    int mRegBlocksRem = mc % mRegBlockSize;

    // assert((nc == nRegBlockSize) &&
    //"nc must be equal to the number of register blocks");

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
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
    frame.setDirtyRegs(
        x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label Loopk = a->newLabel();
    asmjit::Label LoopMBlocks = a->newLabel();

    x86::Gp buffer_B_saved = a->gpz(10);
    x86::Gp C_Offset = a->gpz(11);
    // x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    x86::Gp kIdx = a->gpz(14);

    int colRegs = nc * row_interleave / vectorLen;
    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx.r32(), iIdx.r32());

      // save B_buffer address
      a->mov(buffer_B_saved, buffer_B);
      // a->mov(B_pf_saved, B_pf);

      a->bind(LoopMBlocks);
      a->inc(iIdx);

      int rowRegs = mRegBlockSize;

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(Loopk);
      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      genComputeBlock<inst_set_t::avx2>(
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
      storeCRegs<inst_set_t::avx2>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);

      // increment A for next block
      a->sub(buffer_A, kSize);
      a->add(
          buffer_A,
          static_cast<asmjit::Imm>((rowRegs)*kBlock * sizeof(uint8_t)));
      // increment C for next block
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
      asmjit::Label LoopkRem = a->newLabel();
      int rowRegs = mRegBlocksRem;

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopkRem);

      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      genComputeBlock<inst_set_t::avx2>(
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

      // store C matrix
      storeCRegs<inst_set_t::avx2>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);
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
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
storeCRegs<inst_set_t::avx2>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum);

/**
 * Instantiate the inst_set_t::avx512 instructions for store kernel.
 *
 */
template
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
storeCRegs<inst_set_t::avx512>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum);

/**
 * Instantiate the inst_set_t::avx512_ymm instructions for store kernel.
 *
 */
template
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::
storeCRegs<inst_set_t::avx512_ymm>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum);

} // namespace fbgemm

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
 * 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::genComputeBlock(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp B_pf,
    int rowRegs,
    int colRegs,
    int lda) {
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;

  // used for matrix A
  VecRegT AReg(numRegs - 1);

  // used for matrix B
  VecRegT BReg(numRegs - 2);

  // Contains 16-bit 1s
  VecRegT oneReg(numRegs - 3);

  // temporary register
  VecRegT res1(numRegs - 4);

  for (int j = 0; j < colRegs; ++j) {
    // load B
    emitLoadDWord<instSet, VecRegT>(
        a, BReg, x86::dword_ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {
      a->vpbroadcastd(
          AReg, x86::dword_ptr(buffer_A, (i * lda) * sizeof(uint8_t)));
      a->vpmaddubsw(res1, AReg, BReg);
      a->vpmaddwd(res1, oneReg, res1);
      a->vpaddd(VecRegT(i * colRegs + j), res1, VecRegT(i * colRegs + j));
    }
    a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int8_t)));
  }
}

/**
 * Generate AVX512 instructions for storing the C registers back to the memory
 * in 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add(C_Offset, ldcReg);
    } else {
      a->xor_(C_Offset.r32(), C_Offset.r32());
    }
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddd(
            VecT(i * colRegs + j),
            VecT(i * colRegs + j),
            x86::dword_ptr(a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int8_t)));
      }
      a->vmovups(
          x86::dword_ptr(a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int8_t)),
          VecT(i * colRegs + j));
    }
  }
}

/**
 * Get or Create the AVX512 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template <>
template <inst_set_t instSet>
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreate(
    bool accum, int32_t mc, int32_t nc, int32_t kc) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
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
    kBlock = PackingTraits<uint8_t, int32_t, instSet>::KCB;
    nBlock = PackingTraits<uint8_t, int32_t, instSet>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int32_t, instSet>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int32_t, instSet>::NR;
    nRegBlockSizeMin = PackingTraits<uint8_t, int32_t, instSet>::NR_MIN;
    row_interleave = PackingTraits<uint8_t, int32_t, instSet>::ROW_INTERLEAVE;
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
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

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

    auto dirtyVecRegs =
        asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15);
    if (numRegs >= 16) {
      dirtyVecRegs |=
          asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
          asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31);
    }

    frame.setDirtyRegs(x86::Reg::kGroupVec, dirtyVecRegs);
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

    x86::Gp buffer_B_saved = a->gpz(10);
    x86::Gp C_Offset = a->gpz(11);
    x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    x86::Gp jIdx = a->gpz(14);
    x86::Gp kIdx = a->gpz(15);
    // x86::Gp B_pf = a->gpz(8);

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int32_t)));
    // a->xor_(C_Offset.r32(), C_Offset.r32());

    // save B_buffer address
    a->mov(buffer_B_saved, buffer_B);
    a->mov(B_pf_saved, B_pf);

    int currColRegs = nc * row_interleave / vectorLen;
    int colRegs = std::min(currColRegs, maxNRegs);

    auto issueLoopOverK = [&](int rowRegs) {
      asmjit::Label LoopKLabel = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopKLabel);

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
      a->add(
          B_pf,
          static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));

      a->cmp(kIdx, kSize);
      a->jl(LoopKLabel);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);
    };

    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx.r32(), iIdx.r32());

      a->bind(LoopMBlocks);
      a->inc(iIdx);

      a->xor_(jIdx.r32(), jIdx.r32());

      a->bind(LoopNBlocks);
      a->inc(jIdx);

      issueLoopOverK(mRegBlockSize);

      int rowRegs = mRegBlockSize;

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
      a->mov(B_pf, B_pf_saved);
      a->add(B_pf, C_Offset);

      // increment C for next B block
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
      a->imul(C_Offset, ldcReg, static_cast<asmjit::Imm>(rowRegs));
      a->add(CBase, C_Offset);

      // reset B
      a->mov(buffer_B, buffer_B_saved);
      a->mov(B_pf, B_pf_saved);
      a->cmp(iIdx, mRegBlocks);
      a->jl(LoopMBlocks);
    }
    // generate code for remainder
    if (mRegBlocksRem > 0) {
      asmjit::Label LoopNRem = a->newLabel();

      a->xor_(jIdx.r32(), jIdx.r32());
      a->bind(LoopNRem);
      a->inc(jIdx);

      issueLoopOverK(mRegBlocksRem);

      // increment C for next B block
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
 * Instantiate the AVX512 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
getOrCreate<inst_set_t::avx512>(bool accum, int32_t mc, int32_t nc, int32_t kc);

/**
 * Instatiate the AVX512_256 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
getOrCreate<inst_set_t::avx512_ymm>(bool accum, int32_t mc, int32_t nc, int32_t kc);

/**
 * Instantiate the AVX2 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
getOrCreate<inst_set_t::avx2>(bool accum, int32_t mc, int32_t nc, int32_t kc);

/**
 * Instantiate the inst_set_t::avx512 instructions for store kernel.
 *
 */
template
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
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
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
storeCRegs<inst_set_t::avx512_ymm>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum);

/**
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
storeCRegs<inst_set_t::avx2>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum);

} // namespace fbgemm

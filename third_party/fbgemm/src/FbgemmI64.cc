/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI64.h"

#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "./GenerateKernel.h"
#include "./RefImplementations.h"
#include "fbgemm/PackingTraits-inl.h"

using namespace std;

namespace fbgemm {

/**
 * Generate AVX2 instructions for computing block in the rank-k update of 32-bit
 * Accmulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<int64_t, int64_t, int64_t, int64_t>::genComputeBlock(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp B_pf,
    int rowRegs,
    int colRegs,
    int lda) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int vectorLen = simd_info<instSet>::WIDTH_BITS / 64;

  // used for matrix B
  VecRegT BReg(31);

  // temporary register
  VecRegT res1(30);

  for (int j = 0; j < colRegs; ++j) {
    // load B
    a->vmovaps(
      BReg,
      x86::Mem(
          buffer_B,
          j * vectorLen * sizeof(int64_t),
          simd_info<instSet>::WIDTH_BYTES));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {
      a->vpmullq(
          res1,
          BReg,
          x86::qword_ptr(buffer_A, (i * lda) * sizeof(int64_t))._1to8());
      a->vpaddq(VecRegT(i * colRegs + j), res1, VecRegT(i * colRegs + j));
    }
    // TODO: need to tune
    a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int64_t)));
  }
}

/**
 * Generate AVX2 instructions for storing the C registers back to the memory in
 * 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<int64_t, int64_t, int64_t, int64_t>::storeCRegs(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BITS / 64;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add(C_Offset, ldcReg);
    } else {
      a->xor_(C_Offset.r32(), C_Offset.r32());
    }
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddq(
            VecT(i * colRegs + j),
            VecT(i * colRegs + j),
            x86::dword_ptr(
                a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int64_t)));
      }
      a->vmovups(
          x86::dword_ptr(
              a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int64_t)),
          VecT(i * colRegs + j));
    }
  }
}

/**
 * Get or Create the avx512 instructions for int64_t GEMM macro-kernel.
 */
template <>
template <inst_set_t instSet>
CodeGenBase<int64_t, int64_t, int64_t, int64_t>::jit_micro_kernel_fp
CodeGenBase<int64_t, int64_t, int64_t, int64_t>::getOrCreate(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t /* unused */) {
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BITS / 64;

  tuple<bool, int, int, int, int, int, int> kernelSig;
  int kBlock;
  int nBlock;
  int mRegBlockSize;
  int nRegBlockSize;

  if (blocking_params) {
    kBlock = blocking_params->KCB;
    nBlock = blocking_params->NCB;
    mRegBlockSize = blocking_params->MR;
    nRegBlockSize = blocking_params->NR;
  } else {
    kBlock = PackingTraits<int64_t, int64_t, instSet>::KCB;
    nBlock = PackingTraits<int64_t, int64_t, instSet>::NCB;
    mRegBlockSize = PackingTraits<int64_t, int64_t, instSet>::MR;
    nRegBlockSize = PackingTraits<int64_t, int64_t, instSet>::NR;
  }

  kernelSig =
      make_tuple(accum, mc, nc, nBlock, kBlock, mRegBlockSize, nRegBlockSize);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    x86::Assembler assembler(&code);
    x86::Emitter* a = assembler.as<x86::Emitter>();
#ifdef FBGEMM_LOG_CODE
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum, mc, nc, nBlock, kBlock, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize / vectorLen;
    assert(
        maxMRegs * maxNRegs <= 30 &&
        "MR*(NR*64/512) \
        must be <= 29 (available registers constraint)");

    const int mRegBlocks = mc / mRegBlockSize;
    const int mRegBlocksRem = mc % mRegBlockSize;

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
            int64_t*,
            int64_t*,
            int64_t*,
            int64_t*,
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
    x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    x86::Gp jIdx = a->gpz(14);
    x86::Gp kIdx = a->gpz(15);

    a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int64_t)));
    a->imul(kSize, kSize, static_cast<asmjit::Imm>(sizeof(int64_t)));

    // save B_buffer address
    a->mov(buffer_B_saved, buffer_B);
    a->mov(B_pf_saved, B_pf);

    int currColRegs = nc / vectorLen;
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

      // k is incremented by 1
      a->add(kIdx, static_cast<asmjit::Imm>(sizeof(int64_t)));

      genComputeBlock<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->add(buffer_A, static_cast<asmjit::Imm>(sizeof(int64_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(nBlock * sizeof(int64_t)));
      a->add(B_pf, static_cast<asmjit::Imm>(nBlock * sizeof(int64_t)));

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
          static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int64_t)));
      a->add(buffer_B, C_Offset);
      a->mov(B_pf, B_pf_saved);
      a->add(B_pf, C_Offset);

      // increment C for next B block
      a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int64_t)));

      int jLoopTrips = currColRegs / maxNRegs;
      // jLoopTrips should be at least 1
      jLoopTrips = jLoopTrips ? jLoopTrips : 1;
      a->cmp(jIdx, jLoopTrips);
      a->jl(LoopNBlocks);

      // increment A for next block
      a->add(
          buffer_A,
          static_cast<asmjit::Imm>(rowRegs * kBlock * sizeof(int64_t)));

      // increment C for next A block
      a->sub(
          CBase,
          static_cast<asmjit::Imm>(
              jLoopTrips * nRegBlockSize * sizeof(int64_t)));
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
      assert(false);
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

      // k is incremented by 1
      a->add(kIdx, static_cast<asmjit::Imm>(sizeof(int64_t)));

      genComputeBlock<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->add(buffer_A, static_cast<asmjit::Imm>(sizeof(int64_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(nBlock * sizeof(int64_t)));
      a->add(B_pf, static_cast<asmjit::Imm>(nBlock * sizeof(int64_t)));

      a->cmp(kIdx, kSize);
      a->jl(LoopkRem);

      // reset A
      a->sub(buffer_A, kSize);

      // B for next block
      // using C_Offset as temp reg
      a->imul(
          C_Offset,
          jIdx,
          static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int64_t)));
      a->mov(buffer_B, buffer_B_saved);
      a->add(buffer_B, C_Offset);
      a->mov(B_pf, B_pf_saved);
      a->add(B_pf, C_Offset);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);

      // increment C for next B block
      a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int64_t)));

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
      unique_lock<mutex> lock(rtMutex_);
      err = runtime().add(&fn, &code);
    }
    if (err) {
      cout << "Error: in fn add" << endl;
      return nullptr;
    }

#ifdef FBGEMM_LOG_CODE
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

/**
 * Instatiate the AVX512 instructions for int64_t GEMM macro-kernel.
 */
template
CodeGenBase<int64_t, int64_t, int64_t, int64_t>::jit_micro_kernel_fp
CodeGenBase<int64_t, int64_t, int64_t, int64_t>::
getOrCreate<inst_set_t::avx512>(bool accum, int32_t mc, int32_t nc, int32_t kc);

// Expected to have overflows
NO_SANITIZE("undefined")
void cblas_gemm_i64_i64acc(
    matrix_op_t transa,
    matrix_op_t transb,
    int M,
    int N,
    int K,
    const int64_t* A,
    int lda,
    const int64_t* B,
    int ldb,
    bool accumulate,
    int64_t* C,
    int ldc) {
  cpuinfo_initialize();
  if (!fbgemmHasAvx512Support()) {
    cblas_gemm_i64_i64acc_ref(
        transa, transb, M, N, K, A, lda, B, ldb, accumulate, C, ldc);
    return;
  }
  constexpr int MCB = PackingTraits<int64_t, int64_t, inst_set_t::avx512>::MCB;
  constexpr int NCB = PackingTraits<int64_t, int64_t, inst_set_t::avx512>::NCB;
  constexpr int KCB = PackingTraits<int64_t, int64_t, inst_set_t::avx512>::KCB;
  constexpr int MR = PackingTraits<int64_t, int64_t, inst_set_t::avx512>::MR;
  constexpr int NR = PackingTraits<int64_t, int64_t, inst_set_t::avx512>::NR;
  static_assert(MCB % MR == 0, "MR must divide MCB");
  static_assert(NCB % NR == 0, "NR must divide NCB");
  constexpr int VLEN =
      simd_info<inst_set_t::avx512>::WIDTH_BYTES / sizeof(int64_t);
  static_assert(NR % VLEN == 0, "VLEN must divide NR");

  using CodeGenType = CodeGenBase<int64_t, int64_t, int64_t, int64_t>;
  CodeGenType codeObj;
  CodeGenType::jit_micro_kernel_fp fn =
      codeObj.getOrCreate<inst_set_t::avx512>(
          true /* accum */, MCB, NCB, KCB);
  CodeGenType::jit_micro_kernel_fp fn_noacc;
  if (!accumulate) {
    fn_noacc = codeObj.getOrCreate<inst_set_t::avx512>(
        false /* accum */, MCB, NCB, KCB);
  }

  vector<int64_t> At, Bt;
  // TODO: handle transpose during packing
  if (transa == matrix_op_t::Transpose) {
    At.resize(M * K);
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        At.at(i * K + k) = A[i + k * lda];
      }
    }
    A = At.data();
    lda = K;
  }
  if (transb == matrix_op_t::Transpose) {
    Bt.resize(K * N);
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        Bt.at(k * N + j) = B[k + j * ldb];
      }
    }
    B = Bt.data();
    ldb = N;
  }

  alignas(64) array<int64_t, MCB * KCB> packA;
  alignas(64) array<int64_t, KCB * NCB> packB;
  alignas(64) array<int64_t, MCB * NCB> packC;

  for (int ic = 0; ic < M; ic += MCB) {
    for (int kc = 0; kc < K; kc += KCB) {
      // pack A
      for (int i = 0; i < std::min(MCB, M - ic); ++i) {
        memcpy(
            &packA[i * KCB],
            A + (ic + i) * lda + kc,
            std::min(K - kc, KCB) * sizeof(int64_t));
      }

      for (int jc = 0; jc < N; jc += NCB) {
        // pack B
        for (int i = 0; i < std::min(KCB, K - kc); ++i) {
          memcpy(
              &packB[i * NCB],
              B + (kc + i) * ldb + jc,
              std::min(NCB, N - jc) * sizeof(int64_t));
        }

        if (M - ic >= MCB && N - jc >= NCB) {
          if (kc == 0 && !accumulate) {
            fn_noacc(
                packA.data(),
                packB.data(),
                packB.data(),
                C + ic * ldc + jc,
                std::min(KCB, K - kc),
                ldc);
          } else {
            fn(packA.data(),
               packB.data(),
               packB.data(),
               C + ic * ldc + jc,
               std::min(KCB, K - kc),
               ldc);
          }
        } else {
          // remainder
          if (kc == 0 && !accumulate) {
            fn_noacc(
                packA.data(),
                packB.data(),
                packB.data(),
                packC.data(),
                std::min(KCB, K - kc),
                NCB);
          } else {
            for (int i = 0; i < std::min(MCB, M - ic); ++i) {
              memcpy(
                  &packC[i * NCB],
                  C + (ic + i) * ldc + jc,
                  std::min(NCB, N - jc) * sizeof(int64_t));
            }
            fn(packA.data(),
               packB.data(),
               packB.data(),
               packC.data(),
               std::min(KCB, K - kc),
               NCB);
          }
          for (int i = 0; i < std::min(MCB, M - ic); ++i) {
            memcpy(
                C + (ic + i) * ldc + jc,
                &packC[i * NCB],
                std::min(NCB, N - jc) * sizeof(int64_t));
          }
        }
      } // jc
    } // kc
  } // ic
}

} // namespace fbgemm

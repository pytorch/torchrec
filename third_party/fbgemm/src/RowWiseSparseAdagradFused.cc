/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmEmbedding.h"

#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <iostream>
#include <mutex>
#include "./CodeCache.h"
#include "./MaskAvx2.h"
#include "./RefImplementations.h"
#include "fbgemm/Utils.h"

using namespace std;

namespace fbgemm {
namespace {
namespace x86 = asmjit::x86;

template <typename indxType, typename offsetType, typename dataType>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size, // number of rows in w
      dataType* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const indxType* indices, // indices of each row
      const offsetType* offsets_or_lengths,
      float epsilon,
      float lr,
      uint32_t* rand_buffer);
};

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet = inst_set_t::avx2>
class GenRowWiseSparseAdagradFused {
 public:
  GenRowWiseSparseAdagradFused() {}

  typename ReturnFunctionSignature<indxType, offsetType, dataType>::
      jit_sparse_adagrad_kernel
      getOrCreate(
          const int* mask_avx2,
          int block_size,
          int prefetch,
          bool use_offsets,
          bool use_stochastic_rounding,
          int grad_stride);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on:
  // avx2 mask array, embedding dimension (block size), prefetch distance,
  // use_offsets and use_stochastic_rouding switch
  static CodeCache<
      tuple<const int*, int, int, bool, bool, int>,
      typename ReturnFunctionSignature<indxType, offsetType, dataType>::
          jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // class GenRowWiseSparseAdagradFused

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
mutex GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
    rtMutex_;

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
CodeCache<
    tuple<const int*, int, int, bool, bool, int>,
    typename ReturnFunctionSignature<indxType, offsetType, dataType>::
        jit_sparse_adagrad_kernel>
    GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
        codeCache_;

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
typename ReturnFunctionSignature<indxType, offsetType, dataType>::
    jit_sparse_adagrad_kernel
    GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
        getOrCreate(
            const int* mask_avx2, // runtime constant
            int block_size,
            int prefetch,
            bool use_offsets,
            bool use_stochastic_rounding,
            int grad_stride) {
  tuple<const int*, int, int, bool, bool, int> kernelSig = make_tuple(
      mask_avx2,
      block_size,
      prefetch,
      use_offsets,
      use_stochastic_rounding,
      grad_stride);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() -> typename ReturnFunctionSignature<
                indxType,
                offsetType,
                dataType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().environment());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
        bool areIndices64b = is_same<indxType, int64_t>::value;
        bool areWeightsFp16 = is_same<dataType, float16>::value;
#if defined(FBGEMM_LOG_CODE)
        string filename = "RowWiseSparseAdagradFused";
        filename += "_emd_dim_" + to_string(block_size);
        filename += "_wei_float";
        filename += areWeightsFp16 ? "16" : "32";
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::avx512 ? "_avx512" : "_avx2";
        if (prefetch) {
          filename += "_prefetch";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        x86::Gp rand_buffer = a->zax();
        x86::Gp output_size = a->zdi();
        x86::Gp index_size = a->zsi();
        x86::Gp data_size = a->zdx();
        x86::Gp w = a->zcx();
        x86::Gp g = a->gpz(8);
        x86::Gp h = a->gpz(9);
        x86::Gp indices = a->gpz(10);
        x86::Gp lengths = a->gpz(11);
        x86::Xmm epsilon(0);
        x86::Xmm lr(1);
        x86::Gpd lengths_R = a->gpz(12).r32();
        x86::Gp scratchReg1 = a->gpz(13);
        x86::Gp scratchReg2 = a->gpz(14); // for prefetching

        asmjit::FuncDetail func;
        func.init(
            asmjit::FuncSignatureT<
                bool, // return type
                int64_t, // output_size
                int64_t, // index_size
                int64_t, // data_size
                dataType*, // w
                const float*, // g
                float*, // h
                const indxType*, // indices
                const int*, // lengths
                float, // epsilon
                float, // lr then rand_buffer
                uint32_t*>(asmjit::CallConv::kIdHost),
            a->environment());

        asmjit::FuncFrame frame;
        frame.init(func);

        if (instSet == inst_set_t::avx2) {
          frame.setDirtyRegs(
              x86::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
        } else {
          frame.setDirtyRegs(
              x86::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                  asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
                  asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
        }

        frame.setDirtyRegs(
            x86::Reg::kGroupGp,
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(
            output_size,
            index_size,
            data_size,
            w,
            g,
            h,
            indices,
            lengths,
            epsilon,
            lr,
            rand_buffer);

        args.updateFuncFrame(frame);
        frame.finalize();
        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t src_vreg; // for holding embedding value temporarily
        x86::Ymm mask_vreg;

        // Reserve registers with small ids first because some of them need to
        // be used with an instruction not supported in avx512 for which a big
        // register id won't work.
        int first_available_vec_reg_id = 0;
        x86::Ymm partial_sum_vreg = x86::Ymm(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t float_step_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t epsilon_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t lr_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;

        a->vpbroadcastd(epsilon_vreg, epsilon);
        a->vpbroadcastd(lr_vreg, lr);

        // Reserve vector registers for random buffer generating
        // S0...S3: global random buffer state
        // R: generated random number in uint32_t
        // r0: extracted random byte (uint8_t) shifted to bits[5...13]
        // r1: temp
        vec_reg_t R_vreg, S0_vreg, S1_vreg, S2_vreg, S3_vreg, r0_vreg, r1_vreg;
        if (areWeightsFp16 && use_stochastic_rounding) {
          R_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S0_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S1_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S2_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S3_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          r0_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          r1_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;

          // Load random buffer for FP16 stochastic rounding
          if (instSet == inst_set_t::avx2) {
            a->vmovdqa(S0_vreg.ymm(), x86::dword_ptr(rand_buffer));
            a->vmovdqa(
                S1_vreg.ymm(),
                x86::dword_ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)));
            a->vmovdqa(
                S2_vreg.ymm(),
                x86::dword_ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)));
            a->vmovdqa(
                S3_vreg.ymm(),
                x86::dword_ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)));
          } else { // AVX512
            a->vmovdqa32(S0_vreg, x86::dword_ptr(rand_buffer));
            a->vmovdqa32(
                S1_vreg,
                x86::dword_ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)));
            a->vmovdqa32(
                S2_vreg,
                x86::dword_ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)));
            a->vmovdqa32(
                S3_vreg,
                x86::dword_ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)));
          }
        }

        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            src_vreg = vec_reg_t(first_available_vec_reg_id);
            ++first_available_vec_reg_id;

            mask_vreg = x86::Ymm(first_available_vec_reg_id);
            ++first_available_vec_reg_id;
            // Use scratchReg1 as temp
            a->mov(scratchReg1, asmjit::imm(mask_avx2));
            a->vmovups(
                mask_vreg,
                x86::ymmword_ptr(
                    scratchReg1, (vlen - remainder) % vlen * sizeof(int32_t)));
          } else {
            a->mov(scratchReg1, (1 << remainder) - 1);
            a->kmovw(x86::k(1), scratchReg1);
          }
        }
        // Need an extra mask for computing sum of gradients
        int remainder_avx2 =
            block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        x86::KReg reduce_mask_avx512;
        if (remainder_avx2 && instSet == inst_set_t::avx512) {
          reduce_mask_avx512 = x86::k(2);
          a->mov(scratchReg1, (1 << remainder_avx2) - 1);
          a->kmovw(reduce_mask_avx512, scratchReg1);
        }

        int unroll_factor = NUM_VEC_REG - first_available_vec_reg_id;

        // Compute the end address of indices
        a->imul(
            scratchReg1,
            index_size,
            static_cast<asmjit::Imm>(sizeof(indxType)));
        a->add(scratchReg1, indices);
        a->mov(index_size, scratchReg1);

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begin (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->dec(output_size);
        a->jl(LoopRangeIndexEnd);

        // Compute sq avg of gradients
        // Even with avx512, we only need to use avx2 registers when computing
        // partial_sum because some instructions we're using like vhaddps
        // are only in avx2.
        constexpr int vlen_avx2 =
            simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        int num_vec_regs_per_block_avx2 =
            (block_size + vlen_avx2 - 1) / vlen_avx2;

        a->vxorps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

        // TODO: need to do a tree-reduction to fully take advantage of
        // unrolling
        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block_avx2;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block_avx2 - vec_idx);
          for (int v = 0; v < cur_unroll_factor; ++v) {
            x86::Ymm out_vreg = x86::Ymm(v + first_available_vec_reg_id);

            auto g_ptr =
                x86::dword_ptr(g, (vec_idx + v) * vlen_avx2 * sizeof(float));
            if (block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS &&
                vec_idx + v == num_vec_regs_per_block_avx2 - 1) {
              if (instSet == inst_set_t::avx2) {
                a->vmaskmovps(out_vreg, mask_vreg, g_ptr);
              } else {
                a->k(reduce_mask_avx512).z().vmovups(out_vreg, g_ptr);
              }
            } else {
              a->vmovups(out_vreg, g_ptr);
            }
            a->vmulps(out_vreg, out_vreg, out_vreg);
            a->vaddps(partial_sum_vreg, partial_sum_vreg, out_vreg);
          }
        }
        // Reduce sum to 1 value
        // __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
        // __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
        // Use YMM/XMMs with smaller ids for AVX2 specific instructions like
        // vhaddps
        x86::Xmm partial_sum_xmm(partial_sum_vreg.id());
        x86::Xmm float_step_xmm(float_step_vreg.id());
        // a->vmovups(partial_sum_temp0_ymm, partial_sum_vreg);
        a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);
        a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

        //_mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3))
        a->movss(float_step_xmm, partial_sum_xmm);
        //_mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1))
        a->vextractf128(partial_sum_xmm, partial_sum_vreg, 1);

        // final_sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
        //    _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
        a->addss(partial_sum_xmm, float_step_xmm);

        // This fragment moves block size (N) to stack and bcasts it to xmm reg
        a->lea(
            x86::rsp,
            x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
        a->mov(x86::dword_ptr(x86::rsp), block_size);
        a->vbroadcastss(
            float_step_xmm,
            x86::dword_ptr(x86::rsp)); // N is partial_sum_xmm1
        a->vcvtdq2ps(float_step_xmm, float_step_xmm);
        a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

        // final_sum /= N
        a->divss(partial_sum_xmm, float_step_xmm);

        if (use_offsets) {
          a->mov(lengths_R, x86::dword_ptr(lengths, sizeof(offsetType)));
          a->sub(lengths_R, x86::dword_ptr(lengths));
        } else {
          a->mov(lengths_R, x86::dword_ptr(lengths));
        }

        // Array out of bound check
        a->imul(
            scratchReg1, lengths_R, static_cast<asmjit::Imm>(sizeof(indxType)));

        a->add(scratchReg1, indices);
        a->cmp(scratchReg1, index_size);
        a->jg(error);

        asmjit::Label LoopDataIndexBegin = a->newLabel();
        asmjit::Label LoopDataIndexEnd = a->newLabel();

        // dataIndex loop begins (iterate lengths_R_ times)
        a->bind(LoopDataIndexBegin);
        a->dec(lengths_R);
        a->jl(LoopDataIndexEnd);

        // Array out of bound check
        if (areIndices64b) {
          a->mov(scratchReg1, x86::qword_ptr(indices));
        } else {
          a->mov(scratchReg1.r32(), x86::dword_ptr(indices));
        }
        // A trick to check x >= data_size or x < 0 in one shot by treating
        // scratchReg1_ as if it has unsigned value
        // (https://stackoverflow.com/a/34072155).
        a->cmp(scratchReg1, data_size);
        a->jae(error);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();
          // out of bound handling for prefetch
          a->mov(scratchReg2, indices);
          a->add(
              scratchReg2,
              static_cast<asmjit::Imm>(prefetch * sizeof(indxType)));
          a->cmp(scratchReg2, index_size);
          a->jge(pref_dist_reset_start);

          if (areIndices64b) {
            a->mov(
                scratchReg2,
                x86::qword_ptr(indices, prefetch * sizeof(indxType)));
          } else {
            a->mov(
                scratchReg2.r32(),
                x86::dword_ptr(indices, prefetch * sizeof(indxType)));
          }

          a->jmp(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);
          // things are not okay just get the current row
          // this can be improved to getting the max dist row.
          if (areIndices64b) {
            a->mov(scratchReg2, x86::qword_ptr(indices));
          } else {
            a->mov(scratchReg2.r32(), x86::dword_ptr(indices));
          }

          a->bind(pref_dist_reset_end);
        }

        a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));

        if (prefetch) {
          a->prefetchw(x86::dword_ptr(h, scratchReg2, 2));
        }
        // load h
        a->movss(float_step_xmm, x86::dword_ptr(h, scratchReg1, 2));
        // *h + final_sum
        a->addss(float_step_xmm, partial_sum_xmm);
        // store h
        a->movss(x86::dword_ptr(h, scratchReg1, 2), float_step_xmm);
        // sqrt(hi)
        a->sqrtss(float_step_xmm, float_step_xmm);
        // bcast partial to all of ymm/zmm reg
        a->vpbroadcastd(float_step_vreg, float_step_xmm);
        // lr / sqrt(hi) + epsilon
        a->vaddps(float_step_vreg, float_step_vreg, epsilon_vreg);
        a->vdivps(float_step_vreg, lr_vreg, float_step_vreg);

        a->imul(scratchReg1, static_cast<asmjit::Imm>(block_size));
        if (prefetch) {
          a->imul(scratchReg2, static_cast<asmjit::Imm>(block_size));
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // The main computation
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v + first_available_vec_reg_id);

            auto g_ptr =
                x86::dword_ptr(g, (vec_idx + v) * vlen * sizeof(float));
            if (!areWeightsFp16) { // float weights
              auto w_ptr = x86::dword_ptr(
                  w, scratchReg1, 2, (vec_idx + v) * vlen * sizeof(dataType));
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::avx2) {
                  a->vmaskmovps(src_vreg.ymm(), mask_vreg, g_ptr);
                  a->vmulps(src_vreg, float_step_vreg, src_vreg);

                  a->vmaskmovps(out_vreg.ymm(), mask_vreg, w_ptr);
                  a->vaddps(out_vreg, src_vreg, out_vreg);

                  a->vmaskmovps(w_ptr, mask_vreg, out_vreg.ymm());
                } else {
                  a->k(x86::k(1)).vmulps(out_vreg, float_step_vreg, g_ptr);
                  a->k(x86::k(1)).vaddps(out_vreg, out_vreg, w_ptr);
                  a->k(x86::k(1)).vmovups(w_ptr, out_vreg);
                }
              } else {
                a->vmulps(out_vreg, float_step_vreg, g_ptr);
                a->vaddps(out_vreg, out_vreg, w_ptr);
                a->vmovups(w_ptr, out_vreg);
              }
            } else { // float16 weights
              auto w_ptr = x86::word_ptr(
                  w, scratchReg1, 1, (vec_idx + v) * vlen * sizeof(dataType));

              if (use_stochastic_rounding) {
                // Index [0..3] for extracted bytes
                // Each int32 has 4 8-bit rand byte
                int sr_idx = (vec_idx + v) % 4;

                if (sr_idx == 0) {
                  // Generate R buffer every 4 steps of num_vec_regs_per_block
                  // loop. Each 8-bit in R (uint32_t) will be used once. It is
                  // shifted to the bits [5-13] then added to FP32 weights
                  // before FP16 conversion.
                  //
                  // The shifted 8 bit region
                  // +-------+--------+--------+--------+
                  // |       |        |   xxxxx|xxx     |
                  //  31      23       15       7      0
                  //
                  // Half float has 10 bits of mantissa, and float has 23, we
                  // are shifting the bits to cover the region where half
                  // floats can't represent data. This is bits[13..23] of the
                  // mantissa of FP32. This will be effectively adding a random
                  // variable of [0,1]

                  // Random generator using xoshiro128++
                  // Ref: http://prng.di.unimi.it/xoshiro128plusplus.c
                  a->vpaddd(r0_vreg, S0_vreg, S3_vreg);
                  a->vpslld(r1_vreg, r0_vreg, 7);
                  a->vpsrld(r0_vreg, r0_vreg, 25);
                  if (instSet == inst_set_t::avx2) {
                    a->vpor(R_vreg.ymm(), r0_vreg.ymm(), r1_vreg.ymm());
                  } else {
                    a->vpord(R_vreg, r0_vreg, r1_vreg);
                  }
                  a->vpaddd(R_vreg, R_vreg, S0_vreg);

                  a->vpslld(r0_vreg, S1_vreg, 9);

                  if (instSet == inst_set_t::avx2) {
                    a->vpxor(S2_vreg.ymm(), S2_vreg.ymm(), S0_vreg.ymm());
                    a->vpxor(S3_vreg.ymm(), S3_vreg.ymm(), S1_vreg.ymm());
                    a->vpxor(S1_vreg.ymm(), S1_vreg.ymm(), S2_vreg.ymm());
                    a->vpxor(S0_vreg.ymm(), S0_vreg.ymm(), S3_vreg.ymm());

                    a->vpxor(S2_vreg.ymm(), S2_vreg.ymm(), r0_vreg.ymm());
                  } else {
                    a->vpxord(S2_vreg, S2_vreg, S0_vreg);
                    a->vpxord(S3_vreg, S3_vreg, S1_vreg);
                    a->vpxord(S1_vreg, S1_vreg, S2_vreg);
                    a->vpxord(S0_vreg, S0_vreg, S3_vreg);

                    a->vpxord(S2_vreg, S2_vreg, r0_vreg);
                  }
                  a->vpslld(r0_vreg, S3_vreg, 11);
                  a->vpsrld(r1_vreg, S3_vreg, 21);
                  if (instSet == inst_set_t::avx2) {
                    a->vpor(S3_vreg.ymm(), r0_vreg.ymm(), r1_vreg.ymm());
                  } else {
                    a->vpord(S3_vreg, r0_vreg, r1_vreg);
                  }

                  // Extract byte 0 and shift to bits[5..13]
                  a->vpslld(r0_vreg, R_vreg, 24);
                  a->vpsrld(r0_vreg, r0_vreg, 19);
                } else if (sr_idx == 1) {
                  // Extract byte 1 and shift to bits[[5..13]
                  a->vpsrld(r0_vreg, R_vreg, 8);
                  a->vpslld(r0_vreg, r0_vreg, 24);
                  a->vpsrld(r0_vreg, r0_vreg, 19);
                } else if (sr_idx == 2) {
                  // Extract byte 2 and shift to bits[5..13]
                  a->vpslld(r0_vreg, R_vreg, 8);
                  a->vpsrld(r0_vreg, r0_vreg, 24);
                  a->vpslld(r0_vreg, r0_vreg, 5);
                } else { // sr_idx == 3
                  // Extract byte 3 and shift to bits[5..13]
                  a->vpsrld(r0_vreg, R_vreg, 24);
                  a->vpslld(r0_vreg, r0_vreg, 5);
                }
              }

              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::avx2) {
                  a->vmaskmovps(src_vreg.ymm(), mask_vreg, g_ptr);
                  // No AVX2 mask load/store for 16bit
                  // Copy input to stack using loop instead and reuse GPR for h
                  a->lea(x86::rsp, x86::ptr(x86::rsp, -8));
                  a->mov(x86::ptr(x86::rsp), h);
                  a->lea(
                      x86::rsp,
                      x86::ptr(
                          x86::rsp, static_cast<int>(-vlen * sizeof(float16))));
                  for (int r = 0; r < remainder; ++r) {
                    a->mov(
                        h.r16(),
                        x86::word_ptr(
                            w,
                            scratchReg1,
                            1,
                            ((vec_idx + v) * vlen + r) * sizeof(dataType)));
                    a->mov(x86::ptr(x86::rsp, sizeof(dataType) * r), h.r16());
                  }
                  a->vcvtph2ps(out_vreg, x86::word_ptr(x86::rsp));
                  a->vfmadd231ps(out_vreg, float_step_vreg, src_vreg);
                  if (use_stochastic_rounding) {
                    a->vpaddd(out_vreg, r0_vreg, out_vreg);
                  }
                  // Truncate rounding to 'counterwork' the random added part
                  a->vcvtps2ph(x86::word_ptr(x86::rsp), out_vreg, 11);
                  // Copy results back
                  for (int r = 0; r < remainder; ++r) {
                    a->mov(h.r16(), x86::ptr(x86::rsp, sizeof(dataType) * r));
                    a->mov(
                        x86::word_ptr(
                            w,
                            scratchReg1,
                            1,
                            ((vec_idx + v) * vlen + r) * sizeof(dataType)),
                        h.r16());
                  }
                  a->lea(
                      x86::rsp,
                      x86::ptr(
                          x86::rsp, static_cast<int>(vlen * sizeof(float16))));
                  a->mov(h, x86::ptr(x86::rsp));
                  a->lea(x86::rsp, x86::ptr(x86::rsp, 8));
                } else {
                  a->k(x86::k(1)).vcvtph2ps(out_vreg, w_ptr);
                  a->k(x86::k(1)).vfmadd231ps(out_vreg, float_step_vreg, g_ptr);
                  if (use_stochastic_rounding) {
                    a->vpaddd(out_vreg, r0_vreg, out_vreg);
                  }
                  // Truncate rounding
                  a->k(x86::k(1)).vcvtps2ph(w_ptr, out_vreg, 11);
                }
              } else {
                a->vcvtph2ps(out_vreg, w_ptr);
                a->vfmadd231ps(out_vreg, float_step_vreg, g_ptr);
                if (use_stochastic_rounding) {
                  a->vpaddd(out_vreg, r0_vreg, out_vreg);
                }
                // Truncate rounding
                a->vcvtps2ph(w_ptr, out_vreg, 11);
              }
            }

            constexpr int CACHE_LINE_LEN = 64;
            constexpr int BYTES_PER_VLOAD = vlen * sizeof(dataType);
            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            if (prefetch && (vec_idx + v) % VLOAD_PER_CACHE_LINE == 0) {
              a->prefetchw(x86::dword_ptr(
                  w,
                  scratchReg2,
                  areWeightsFp16 ? 1 : 2,
                  (vec_idx + v) * BYTES_PER_VLOAD));
            }
          }
        }

        a->jmp(LoopDataIndexBegin);
        a->bind(LoopDataIndexEnd);

        a->add(lengths, static_cast<asmjit::Imm>(sizeof(offsetType)));
        a->add(g, static_cast<asmjit::Imm>(grad_stride * sizeof(float)));

        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->cmp(indices, index_size);
        a->jne(error);
        a->mov(scratchReg1.r32(), 1);
        a->jmp(exit);
        a->bind(error);
        a->mov(scratchReg1.r32(), 0);
        a->bind(exit);

        if (areWeightsFp16 && use_stochastic_rounding) {
          if (instSet == inst_set_t::avx2) {
            a->vmovdqa(x86::dword_ptr(rand_buffer), S0_vreg.ymm());
            a->vmovdqa(
                x86::dword_ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)),
                S1_vreg.ymm());
            a->vmovdqa(
                x86::dword_ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)),
                S2_vreg.ymm());
            a->vmovdqa(
                x86::dword_ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)),
                S3_vreg.ymm());
          } else {
            a->vmovdqa32(x86::dword_ptr(rand_buffer), S0_vreg);
            a->vmovdqa32(
                x86::dword_ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)),
                S1_vreg);
            a->vmovdqa32(
                x86::dword_ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)),
                S2_vreg);
            a->vmovdqa32(
                x86::dword_ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)),
                S3_vreg);
          }
        }

        a->mov(x86::eax, scratchReg1.r32());
        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<indxType, offsetType, dataType>::
            jit_sparse_adagrad_kernel fn;
        asmjit::Error err;
        {
          unique_lock<mutex> lock(rtMutex_);
          err = runtime().add(&fn, &code);
        }
        if (err) {
          cout << "Error: in fn add" << endl;
          return nullptr;
        }

#if defined(FBGEMM_LOG_CODE)
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
} // getOrCreate

// Per-thread global buffer for random number generating, with max vector size
constexpr size_t VLEN_MAX = simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS;
alignas(64) static thread_local uint32_t g_rnd128v_buffer[4 * VLEN_MAX];
static thread_local bool g_rnd128v_initialized = false;

void rand_initialize() {
  // Splitmix64: http://prng.di.unimi.it/splitmix64.c
  auto rnd128_init_next = [](uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  };

  if (!g_rnd128v_initialized) {
    uint64_t h0 = std::hash<std::thread::id>{}(std::this_thread::get_id());
    for (auto i = 0; i < 4; ++i) {
      g_rnd128v_buffer[i * VLEN_MAX] = rnd128_init_next(h0);
      uint64_t h1 = g_rnd128v_buffer[i * VLEN_MAX];
      for (size_t v = 1; v < VLEN_MAX; ++v) {
        g_rnd128v_buffer[i * VLEN_MAX + v] = rnd128_init_next(h1);
      }
    }
    g_rnd128v_initialized = true;
  }
}

} // namespace

template <typename IndexType, typename OffsetType, typename DataType>
FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<
    IndexType,
    OffsetType,
    DataType>::Type
GenerateRowWiseSparseAdaGradFused(
    int block_size, // number of parameters per row
    int prefetch,
    bool use_offsets,
    bool use_stochastic_rounding,
    int grad_stride) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (grad_stride == -1) {
    grad_stride = block_size;
  }

  // Use avx512 only for fp16 + stochastic rounding
  if (fbgemmHasAvx512Support() && std::is_same<DataType, float16>::value &&
      use_stochastic_rounding) {
    static GenRowWiseSparseAdagradFused<
        IndexType,
        OffsetType,
        DataType,
        inst_set_t::avx512>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        nullptr,
        block_size,
        prefetch,
        use_offsets,
        use_stochastic_rounding,
        grad_stride);
    const auto lambda_func = [=](int64_t output_size,
                                 int64_t index_size,
                                 int64_t data_size,
                                 DataType* w,
                                 const float* g,
                                 float* h,
                                 const IndexType* indices,
                                 const OffsetType* offsets_or_lengths,
                                 float epsilon,
                                 float lr) {
      // Initialize random buffer in the first execution
      // TODO: JIT
      if (std::is_same<DataType, float16>::value && use_stochastic_rounding) {
        rand_initialize();
      }

      return original_func(
          output_size,
          index_size,
          data_size,
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          offsets_or_lengths,
          epsilon,
          lr,
          g_rnd128v_buffer);
    };
    return lambda_func;
  } else if (fbgemmHasAvx2Support()) {
    static GenRowWiseSparseAdagradFused<
        IndexType,
        OffsetType,
        DataType,
        inst_set_t::avx2>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        internal::avx2_ps_or_epi32_combined_mask,
        block_size,
        prefetch,
        use_offsets,
        use_stochastic_rounding,
        grad_stride);
    const auto lambda_func = [=](int64_t output_size,
                                 int64_t index_size,
                                 int64_t data_size,
                                 DataType* w,
                                 const float* g,
                                 float* h,
                                 const IndexType* indices,
                                 const OffsetType* offsets_or_lengths,
                                 float epsilon,
                                 float lr) {
      // Initialize random buffer in the first execution
      // TODO: JIT
      if (std::is_same<DataType, float16>::value && use_stochastic_rounding) {
        rand_initialize();
      }

      return original_func(
          output_size,
          index_size,
          data_size,
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          offsets_or_lengths,
          epsilon,
          lr,
          g_rnd128v_buffer);
    };
    return lambda_func;
  } else {
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               DataType* w,
               const float* g,
               float* h,
               const IndexType* indices,
               const OffsetType* offsets_or_lengths,
               float epsilon,
               float lr) {
      return rowwise_sparse_adagrad_fused_ref(
          block_size,
          output_size,
          index_size,
          data_size,
          w,
          g,
          h,
          indices,
          offsets_or_lengths,
          epsilon,
          lr,
          use_offsets,
          use_stochastic_rounding,
          /*emu_vector_size=*/8,
          grad_stride);
    };
  }
}

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int32_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int32_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int64_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int64_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int32_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int32_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int64_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int64_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int32_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int32_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int64_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int64_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int32_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int32_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int64_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int64_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

} // namespace fbgemm

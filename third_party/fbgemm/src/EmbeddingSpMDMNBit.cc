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
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./MaskAvx2.h"
#include "./RefImplementations.h"
#include "fbgemm/Types.h"

using namespace std;

namespace fbgemm {

namespace {

template <typename T>
T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

namespace x86 = asmjit::x86;

template <typename indxType, typename offsetType, bool ROWWISE_SPARSE>
class ReturnFunctionSignature {};

template <typename indxType, typename offsetType>
class ReturnFunctionSignature<indxType, offsetType, false> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size,
      const uint8_t* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      float* out,
      const int* mask);
};

template <typename indxType, typename offsetType>
class ReturnFunctionSignature<indxType, offsetType, true> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t uncompressed_data_size,
      // int64_t compressed_data_size,
      const uint8_t* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      float* out,
      const int32_t* compressed_indices_table,
      const int* mask);
};

template <
    typename indxType,
    typename offsetType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE = false>
class GenEmbeddingSpMDMNBitLookup {
 public:
  GenEmbeddingSpMDMNBitLookup() {}
  typename ReturnFunctionSignature<indxType, offsetType, ROWWISE_SPARSE>::
      jit_embedding_kernel
      getOrCreate(
          int bit_rate,
          int block_size,
          bool has_weight,
          bool is_weight_positional,
          bool normalize_by_lengths,
          int prefetch,
          bool use_offsets);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on bit_rate, embedding dimension (block size), weighted
  // sls, positional weights, normalize by lenths, prefetch distance, and
  // use_offsets.
  static CodeCache<
      tuple<int, int, bool, bool, bool, int, bool>,
      typename ReturnFunctionSignature<indxType, offsetType, ROWWISE_SPARSE>::
          jit_embedding_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // GenEmbeddingSpmDMLookup

template <
    typename indxType,
    typename offsetType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
mutex
    GenEmbeddingSpMDMNBitLookup<indxType, offsetType, instSet, ROWWISE_SPARSE>::
        rtMutex_;

template <
    typename indxType,
    typename offsetType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
CodeCache<
    tuple<int, int, bool, bool, bool, int, bool>,
    typename ReturnFunctionSignature<indxType, offsetType, ROWWISE_SPARSE>::
        jit_embedding_kernel>
    GenEmbeddingSpMDMNBitLookup<indxType, offsetType, instSet, ROWWISE_SPARSE>::
        codeCache_;

template <
    typename indxType,
    typename offsetType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
typename ReturnFunctionSignature<indxType, offsetType, ROWWISE_SPARSE>::
    jit_embedding_kernel
    GenEmbeddingSpMDMNBitLookup<indxType, offsetType, instSet, ROWWISE_SPARSE>::
        getOrCreate(
            int bit_rate,
            int block_size,
            bool has_weight,
            bool is_weight_positional,
            bool normalize_by_lengths,
            int prefetch,
            bool use_offsets) {
  tuple<int, int, bool, bool, bool, int, bool> kernelSig = make_tuple(
      bit_rate,
      block_size,
      has_weight,
      is_weight_positional,
      normalize_by_lengths,
      prefetch,
      use_offsets);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() -> typename ReturnFunctionSignature<
                indxType,
                offsetType,
                ROWWISE_SPARSE>::jit_embedding_kernel {
        // TODO: Make this tunable
        int pref_dist = prefetch;
        bool areIndices64b = is_same<indxType, int64_t>::value;

        asmjit::CodeHolder code;
        code.init(runtime().environment());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
        string filename = "embeddinglookup_" + to_string(bit_rate) + "bit";
        filename += "_emd_dim_" + to_string(block_size);
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::avx512 ? "_avx512" : "_avx2";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight) {
          filename += "_hasweight";
        }
        if (normalize_by_lengths) {
          filename += "_normalize_by_lengths";
        }
        if (!use_offsets) {
          filename += "_use_lengths";
        }
        if (ROWWISE_SPARSE) {
          filename += "_rowwise_sparse";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif
        // arguments to the function created
        x86::Gp output_size = a->zdi();
        // index_size will be overwritten to hold the end address of indices
        x86::Gp index_size = a->zsi();
        x86::Gp data_size = a->zdx();
        x86::Gp input = a->zcx();
        int reg_id = 8;
        x86::Gp indices = a->gpz(reg_id); // 8
        ++reg_id;
        x86::Gp lengths = a->gpz(reg_id); // 9
        ++reg_id;
        x86::Gp weights = a->gpz(reg_id); // 10
        ++reg_id;
        x86::Gp out = a->gpz(reg_id); // 11

        x86::Gp compressed_indices_table;
        if (ROWWISE_SPARSE) {
          ++reg_id;
          compressed_indices_table = a->gpz(reg_id); // 12
        }

        ++reg_id;
        x86::Gp scratchReg1_ = a->gpz(reg_id); // 12 or 13
        x86::Gp scratchReg1D_ = a->gpz(reg_id).r32();

        ++reg_id;
        x86::Gpd lengths_R_ = a->gpz(reg_id).r32(); // 13 or 14

        ++reg_id;
        x86::Gp scratchReg2_ = a->gpz(reg_id); // 14 or 15
        x86::Gp scratchReg3_;
        if (instSet == inst_set_t::avx2) {
          scratchReg3_ = a->zax();
        }

        asmjit::FuncDetail func;

        if (ROWWISE_SPARSE) {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // uncompressed_data_size
                  const uint8_t*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  float*, // out
                  const int32_t* /* compressed_indices_table */,
                  const int* /* mask */>(asmjit::CallConv::kIdHost),
              a->environment());
        } else {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // data_size
                  const uint8_t*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  float*, // out
                  const int* /* mask */>(asmjit::CallConv::kIdHost),
              a->environment());
        }

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
            reg_id == 15
                ? asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15)
                : asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

        asmjit::FuncArgsAssignment args(&func);
        if (ROWWISE_SPARSE) {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              compressed_indices_table,
              scratchReg1_);
        } else {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              scratchReg1_);
        }

        args.updateFuncFrame(frame);
        frame.finalize();

        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = ceil_div(block_size, vlen);
        int remainder = block_size % vlen;

        // Compute a remainder for vector load
        // Since every row is followed by 2 fp16 (scale and bias), luckily
        // we don't need mask at bit-rate granularity but just at 32-bit
        // granularity.
        int num_elem_per_32bit = 32 / bit_rate;
        // multiply by 4 because we're handling 4 vlen per iteration
        int num_of_32bit_per_vload = vlen * 4 / num_elem_per_32bit;
        int remainder_32bit_granularity =
            ceil_div(block_size, num_elem_per_32bit) % num_of_32bit_per_vload;

        vec_reg_t scale_vreg; // holds scale
        vec_reg_t bias_vreg; // holds bias
        vec_reg_t w_vreg; // for weighted sls -- weights
        vec_reg_t
            vlen_inv_vreg; // used for normalize by lengths -- 1/ lengths[i]
        vec_reg_t src_vreg; // for holding embedding value temporarily
        x86::Ymm mask_vreg; // mask for avx2
        x86::Xmm mask2_vreg;

        // We need 2 vec registers for 1. scale 2. bias
        --unroll_factor;
        scale_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        bias_vreg = vec_reg_t(unroll_factor);

        --unroll_factor;
        src_vreg = vec_reg_t(unroll_factor);
        // temporary register for bit manipulation instructions
        --unroll_factor;
        vec_reg_t temp_vreg = vec_reg_t(unroll_factor);
        vec_reg_t temp2_vreg;

        --unroll_factor;
        temp2_vreg = vec_reg_t(unroll_factor);

        // Create a mask that extracts lower bit_rate bits from each 8-bit block
        --unroll_factor;
        vec_reg_t extract_mask_vreg = vec_reg_t(unroll_factor);
        a->lea(
            x86::rsp,
            x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
        if (bit_rate == 4) {
          a->mov(x86::word_ptr(x86::rsp), 0x0f0f);
          a->vpbroadcastw(extract_mask_vreg, x86::word_ptr(x86::rsp));
        } else {
          a->mov(x86::dword_ptr(x86::rsp), 0x03030303);
          a->vpbroadcastd(extract_mask_vreg, x86::dword_ptr(x86::rsp));
        }
        a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

        if (has_weight) {
          --unroll_factor;
          w_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder && instSet == inst_set_t::avx2) {
          // AVX512 doesn't need to use vector register for masking
          --unroll_factor;
          mask_vreg = x86::ymm(unroll_factor);
        }

        // Creating a mask for vector load
        if (remainder_32bit_granularity && instSet == inst_set_t::avx2) {
          // AVX512 doesn't need to use vector register for masking
          --unroll_factor;
          mask2_vreg = x86::xmm(unroll_factor);
        }

        if (normalize_by_lengths) {
          --unroll_factor;
          vlen_inv_vreg = vec_reg_t(unroll_factor);
        }

        // Make unroll_factor a multiple of 4
        unroll_factor = unroll_factor / 4 * 4;

        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            a->vmovups(
                mask_vreg,
                x86::ymmword_ptr(
                    scratchReg1_, (vlen - remainder) % vlen * sizeof(int32_t)));
          } else {
            a->mov(scratchReg1_, (1 << remainder) - 1);
            a->kmovw(x86::k(1), scratchReg1_);
          }
        }

        if (remainder_32bit_granularity) {
          if (instSet == inst_set_t::avx2) {
            a->lea(
                x86::rsp,
                x86::dword_ptr(
                    x86::rsp, (int32_t)(-(vlen / 2) * sizeof(int32_t))));
            for (int i = 0; i < remainder_32bit_granularity; i++) {
              a->mov(x86::dword_ptr(x86::rsp, i * sizeof(int32_t)), -1);
            }
            for (int i = remainder_32bit_granularity; i < vlen / 2; i++) {
              a->mov(x86::dword_ptr(x86::rsp, i * sizeof(int32_t)), 0);
            }
            a->vmovups(mask2_vreg, x86::dword_ptr(x86::rsp));
            a->lea(
                x86::rsp,
                x86::dword_ptr(
                    x86::rsp, (int32_t)((vlen / 2) * sizeof(int32_t))));
          } else {
            a->mov(scratchReg1_, (1 << remainder_32bit_granularity) - 1);
            a->kmovw(x86::k(2), scratchReg1_);
          }
        }

        // Compute the end address of indices
        a->lea(
            index_size, x86::ptr(indices, index_size, areIndices64b ? 3 : 2));

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begins (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->dec(output_size);
        a->jl(LoopRangeIndexEnd);

        if (normalize_by_lengths) {
          asmjit::Label IfLengthsBegin = a->newLabel();
          asmjit::Label IfLengthsEnd = a->newLabel();
          a->bind(IfLengthsBegin);
          if (use_offsets) {
            a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
            a->sub(lengths_R_, x86::dword_ptr(lengths));
          } else {
            a->mov(lengths_R_, x86::dword_ptr(lengths));
          }
          a->cmp(lengths_R_, 1);
          // Initialize vlen_inv as 0 in case lengths is 0
          a->vxorps(vlen_inv_vreg, vlen_inv_vreg, vlen_inv_vreg);
          a->jl(IfLengthsEnd);

          vec_reg_t temp_vreg0(0);
          if (instSet == inst_set_t::avx2) {
            a->mov(scratchReg1_, 1);
            a->cvtsi2ss(vlen_inv_vreg.xmm(), scratchReg1_);
            a->cvtsi2ss(temp_vreg0.xmm(), lengths_R_);
            a->divss(vlen_inv_vreg.xmm(), temp_vreg0.xmm());
            a->vpbroadcastd(vlen_inv_vreg, vlen_inv_vreg.xmm());
          } else {
            a->mov(scratchReg1_, 1);
            a->cvtsi2ss(temp_vreg0.xmm(), scratchReg1_);
            a->vpbroadcastd(vlen_inv_vreg, temp_vreg0.xmm());
            a->vpbroadcastd(temp_vreg0, lengths_R_);
            a->vcvtdq2ps(temp_vreg0, temp_vreg0);
            a->vdivps(vlen_inv_vreg, vlen_inv_vreg, temp_vreg0);
          }
          a->bind(IfLengthsEnd);
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // Initialize output regs
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v);
            a->vxorps(out_vreg, out_vreg, out_vreg);
          }

          if (use_offsets) {
            a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
            a->sub(lengths_R_, x86::dword_ptr(lengths));
          } else {
            a->mov(lengths_R_, x86::dword_ptr(lengths));
          }

          // Array out of bound check
          a->lea(
              scratchReg1_,
              x86::ptr(indices, lengths_R_, areIndices64b ? 3 : 2));
          a->cmp(scratchReg1_, index_size);
          a->jg(error);

          asmjit::Label LoopDataIndexBegin = a->newLabel();
          asmjit::Label LoopDataIndexEnd = a->newLabel();

          // dataIndex loop begins (iterate lengths_R_ times)
          a->bind(LoopDataIndexBegin);
          a->dec(lengths_R_);
          a->jl(LoopDataIndexEnd);

          // Array out of bound check
          if (areIndices64b) {
            a->mov(scratchReg1_, x86::qword_ptr(indices));
          } else {
            a->mov(scratchReg1_.r32(), x86::dword_ptr(indices));
          }
          // A trick to check x >= data_size or x < 0 in one shot by treating
          // scratchReg1_ as if it has unsigned value
          // (https://stackoverflow.com/a/34072155).
          a->cmp(scratchReg1_, data_size);
          a->jae(error);

          if (ROWWISE_SPARSE) {
            a->mov(
                scratchReg1_.r32(),
                x86::dword_ptr(
                    compressed_indices_table,
                    scratchReg1_,
                    2)); // use of 2 is to multiply by 4
          }

          int num_elem_per_byte = 8 / bit_rate;
          int fused_block_size =
              ceil_div(block_size, num_elem_per_byte) + 2 * sizeof(float16);
          if (pref_dist) {
            asmjit::Label pref_dist_reset_start = a->newLabel();
            asmjit::Label pref_dist_reset_end = a->newLabel();
            // out of bound handling for prefetch
            a->lea(
                scratchReg2_, x86::ptr(indices, pref_dist * sizeof(indxType)));
            a->cmp(scratchReg2_, index_size);
            a->jge(pref_dist_reset_start);

            if (areIndices64b) {
              a->mov(
                  scratchReg2_,
                  x86::qword_ptr(indices, pref_dist * sizeof(indxType)));
            } else {
              a->mov(
                  scratchReg2_.r32(),
                  x86::dword_ptr(indices, pref_dist * sizeof(indxType)));
            }

            a->jmp(pref_dist_reset_end);

            a->bind(pref_dist_reset_start);
            // things are not okay just get the current row
            // this can be improved to getting the max dist row.
            if (areIndices64b) {
              a->mov(scratchReg2_, x86::qword_ptr(indices));
            } else {
              a->mov(scratchReg2_.r32(), x86::dword_ptr(indices));
            }

            a->bind(pref_dist_reset_end);
            if (ROWWISE_SPARSE) {
              asmjit::Label rowwise_sparse_pref_corner_case_begin =
                  a->newLabel();
              asmjit::Label rowwise_sparse_pref_corner_case_end = a->newLabel();
              a->cmp(scratchReg2_, data_size);
              a->jae(rowwise_sparse_pref_corner_case_begin);

              a->mov(
                  scratchReg2_.r32(),
                  x86::dword_ptr(
                      compressed_indices_table,
                      scratchReg2_,
                      2)); // use of 2 is to multiply by 4
              a->test(scratchReg2_.r32(), scratchReg2_.r32());
              // Check negative
              a->jns(rowwise_sparse_pref_corner_case_end);

              a->bind(rowwise_sparse_pref_corner_case_begin);
              // For corner case, just set prefetch row id to 0.
              a->xor_(scratchReg2_.r32(), scratchReg2_.r32());
              a->bind(rowwise_sparse_pref_corner_case_end);
            }
            // This has to be fused_block_size
            a->imul(scratchReg2_, static_cast<asmjit::Imm>(fused_block_size));
          }

          a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));

          if (has_weight) {
            a->vbroadcastss(w_vreg, x86::dword_ptr(weights));
            a->add(weights, static_cast<asmjit::Imm>(sizeof(float)));
          }

          if (ROWWISE_SPARSE) {
            a->cmp(scratchReg1_.r32(), static_cast<asmjit::Imm>(-1));
            a->je(LoopDataIndexBegin);
          }

          a->imul(scratchReg1_, static_cast<asmjit::Imm>(fused_block_size));

          // broadcast the scale
          x86::Mem scale_src, bias_src;
          scale_src = x86::word_ptr(
              input, scratchReg1_, 0, ceil_div(block_size, num_elem_per_byte));
          bias_src = x86::word_ptr(
              input,
              scratchReg1_,
              0,
              ceil_div(block_size, num_elem_per_byte) + sizeof(float16));
          a->vpbroadcastw(scale_vreg.half(), scale_src);
          a->vpbroadcastw(bias_vreg.half(), bias_src);
          a->vcvtph2ps(scale_vreg, scale_vreg.half());
          a->vcvtph2ps(bias_vreg, bias_vreg.half());
          constexpr int CACHE_LINE_LEN = 64;
          if (pref_dist && fused_block_size % CACHE_LINE_LEN > 0 &&
              fused_block_size % CACHE_LINE_LEN <= 2 * sizeof(float16)) {
            a->prefetcht0(x86::dword_ptr(
                input,
                scratchReg2_,
                0,
                fused_block_size / CACHE_LINE_LEN * CACHE_LINE_LEN));
          }

          if (has_weight) {
            a->vmulps(scale_vreg, scale_vreg, w_vreg);
            a->vmulps(bias_vreg, bias_vreg, w_vreg);
          }

          // The main computation
          // Handling 4 vector registers per iteration because
          // 1) when bit_rate == 4, we get zmm from ymm load via vpmovzxbw
          // (epu8->epi16), and then get 4 zmms from each 128-bit portion of
          // zmm via vpmovsxbd (epi8->epi32).
          // 2) when bit_rate == 2, we get zmm from xmm load via vpmovzxbd
          // (epu8->epi32), and then get 4 zmms from each 128-bit portion of
          // zmm via vpmovsxbd (epi8->epi32).
          for (int v = 0; v < cur_unroll_factor; v += 4) {
            int bytes_per_vload = (vlen / num_elem_per_byte) * sizeof(uint8_t);
            auto src_addr = x86::dword_ptr(
                input, scratchReg1_, 0, (vec_idx + v) * bytes_per_vload);

            if (bit_rate == 4) {
              if (num_vec_regs_per_block - (vec_idx + v) < 4 &&
                  remainder_32bit_granularity) {
                if (instSet == inst_set_t::avx512) {
                  a->k(x86::k(2)).vmovups(src_vreg.ymm(), src_addr);
                } else {
                  a->vpmaskmovd(src_vreg.xmm(), mask2_vreg.xmm(), src_addr);
                }
                a->vpmovzxbw(src_vreg, src_vreg.half());
              } else {
                a->vpmovzxbw(src_vreg, src_addr);
              }
              a->vpslld(temp_vreg, src_vreg, asmjit::Imm(4));
              if (instSet == inst_set_t::avx512) {
                a->vpord(src_vreg, src_vreg, temp_vreg);
                a->vpandd(src_vreg, src_vreg, extract_mask_vreg);
              } else {
                a->vpor(src_vreg.ymm(), src_vreg.ymm(), temp_vreg.ymm());
                a->vpand(
                    src_vreg.ymm(), src_vreg.ymm(), extract_mask_vreg.ymm());
              }
            } else {
              if (num_vec_regs_per_block - (vec_idx + v) < 4 &&
                  remainder_32bit_granularity) {
                if (instSet == inst_set_t::avx512) {
                  a->k(x86::k(2)).vmovups(src_vreg.xmm(), src_addr);
                  a->vpmovzxbd(src_vreg, src_vreg.xmm());
                } else {
                  a->vpmaskmovd(src_vreg.xmm(), mask2_vreg.xmm(), src_addr);
                  a->vpmovzxbd(src_vreg, src_vreg.xmm());
                }
              } else {
                a->vpmovzxbd(src_vreg, src_addr);
              }
              a->vpslld(temp_vreg, src_vreg, 2 * 8 + 2);
              a->vpslld(temp2_vreg, src_vreg, 8 + 4);
              if (instSet == inst_set_t::avx512) {
                a->vpord(temp_vreg, temp_vreg, temp2_vreg);
              } else {
                a->vpor(temp_vreg.ymm(), temp_vreg.ymm(), temp2_vreg.ymm());
              }
              a->vpslld(temp2_vreg, src_vreg, 6);
              if (instSet == inst_set_t::avx512) {
                a->vpord(temp_vreg, temp_vreg, temp2_vreg);
                a->vpord(src_vreg, temp_vreg, src_vreg);
                a->vpandd(src_vreg, src_vreg, extract_mask_vreg);
              } else {
                a->vpor(temp_vreg.ymm(), temp_vreg.ymm(), temp2_vreg.ymm());
                a->vpor(src_vreg.ymm(), temp_vreg.ymm(), src_vreg.ymm());
                a->vpand(
                    src_vreg.ymm(), src_vreg.ymm(), extract_mask_vreg.ymm());
              }
            }

            // AVX2: For the following loop, operations on src_vreg impact the
            // next iteration. For i = 0, we make a copy. i = 1 just right
            // shifts and uses it. i = 2 we extract upper 128 bits from the copy
            // to src_vreg and use it. i = 3 just right shifts it and uses it.
            for (int i = 0;
                 i < std::min(4, num_vec_regs_per_block - (vec_idx + v));
                 ++i) {
              vec_reg_t out_vreg = vec_reg_t(v + i);
              if (i == 0) {
                a->vpmovsxbd(temp_vreg, src_vreg.xmm());
                // this is only needed for avx2
                if (instSet == inst_set_t::avx2) {
                  a->vmovups(temp2_vreg, src_vreg);
                }
              } else {
                if (instSet == inst_set_t::avx512) {
                  // We could've used avx512_ymm for clock frequency advantage,
                  // if there's an instruction to extract a 64-bit portion from
                  // a YMM as an XMM register.
                  a->vextracti32x4(temp_vreg.xmm(), src_vreg, asmjit::Imm(i));
                  a->vpmovsxbd(temp_vreg, temp_vreg.xmm());
                } else {
                  if (i == 1) {
                    a->vpsrldq(src_vreg, src_vreg, asmjit::Imm(8));
                  } else if (i == 2) {
                    a->vextractf128(
                        src_vreg.xmm(), temp2_vreg.ymm(), asmjit::Imm(i >> 1));
                  } else {
                    a->vpsrldq(src_vreg, src_vreg, asmjit::Imm(8));
                  }
                  a->vpmovsxbd(temp_vreg, src_vreg.xmm());
                } // avx2
              } // i > 0
              a->vcvtdq2ps(temp_vreg, temp_vreg);
              a->vaddps(out_vreg, out_vreg, bias_vreg);
              a->vfmadd231ps(out_vreg, temp_vreg, scale_vreg);
            } // for each i

            int vload_per_cache_line = CACHE_LINE_LEN / bytes_per_vload;
            int v_aligned = ceil_div(vec_idx + v, 4) * 4;
            if (pref_dist && v_aligned % vload_per_cache_line == 0) {
              a->prefetcht0(x86::dword_ptr(
                  input, scratchReg2_, 0, v_aligned * bytes_per_vload));
            }
          }

          a->jmp(LoopDataIndexBegin);
          a->bind(LoopDataIndexEnd);

          // This loop is for writing back out_vreg (results)
          // back to memory
          for (int v = 0; v < cur_unroll_factor; ++v) {
            auto dst_addr =
                x86::dword_ptr(out, (vec_idx + v) * vlen * sizeof(float));
            vec_reg_t out_vreg = vec_reg_t(v);

            if (normalize_by_lengths) {
              a->vmulps(out_vreg, out_vreg, vlen_inv_vreg);
            }

            if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
              if (instSet == inst_set_t::avx512) {
                a->k(x86::k(1)).vmovups(dst_addr, out_vreg);
              } else {
                a->vmaskmovps(dst_addr, mask_vreg, out_vreg.ymm());
              }
            } else {
              a->vmovups(dst_addr, out_vreg);
            }
          }

          if (vec_idx + unroll_factor < num_vec_regs_per_block ||
              (has_weight && is_weight_positional)) {
            // Reset lengths_R_, indices, weights to run the dataIndex loop
            // again
            if (use_offsets) {
              a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
              a->sub(lengths_R_, x86::dword_ptr(lengths));
            } else {
              a->mov(lengths_R_, x86::dword_ptr(lengths));
            }

            if (has_weight) {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(float)));
              a->sub(weights, scratchReg1_);

              if (vec_idx + unroll_factor < num_vec_regs_per_block) {
                a->imul(
                    scratchReg1_,
                    static_cast<asmjit::Imm>(sizeof(indxType) / sizeof(float)));
                a->sub(indices, scratchReg1_);
              }
            } else {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(indxType)));
              a->sub(indices, scratchReg1_);
            }
          }
        }

        a->add(lengths, static_cast<asmjit::Imm>(sizeof(offsetType)));
        a->add(out, static_cast<asmjit::Imm>(block_size * sizeof(float)));

        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->cmp(indices, index_size);
        a->jne(error);
        a->mov(x86::eax, true);
        a->jmp(exit);
        a->bind(error);
        a->mov(x86::eax, false);
        a->bind(exit);

        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<indxType, offsetType, ROWWISE_SPARSE>::
            jit_embedding_kernel fn;
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
}

} // namespace

template <typename indxType, typename offsetType>
typename EmbeddingSpMDMKernelSignature<uint8_t, indxType, offsetType>::Type
GenerateEmbeddingSpMDMNBit(
    int bit_rate,
    const int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");

  if (!cpuinfo_initialize()) {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  if (fbgemmHasAvx512Support()) {
    static GenEmbeddingSpMDMNBitLookup<indxType, offsetType, inst_set_t::avx512>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out) {
      return original_func(
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          nullptr /* mask not used in avx512 */);
    };
  } else if (fbgemmHasAvx2Support()) {
    static GenEmbeddingSpMDMNBitLookup<indxType, offsetType, inst_set_t::avx2>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out) {
      return original_func(
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          internal::avx2_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out) {
      return EmbeddingSpMDMNBit_ref(
          bit_rate,
          block_size,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets);
    };
  }
}

template <typename indxType, typename offsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    uint8_t,
    indxType,
    offsetType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse(
    int bit_rate,
    const int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");

  if (!cpuinfo_initialize()) {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  if (fbgemmHasAvx512Support()) {
    static GenEmbeddingSpMDMNBitLookup<
        indxType,
        offsetType,
        inst_set_t::avx512,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return original_func(
          output_size,
          index_size,
          uncompressed_data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          compressed_indices_table,
          nullptr /* mask not used in avx512 */);
    };
  } else if (fbgemmHasAvx2Support()) {
    static GenEmbeddingSpMDMNBitLookup<
        indxType,
        offsetType,
        inst_set_t::avx2,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return original_func(
          output_size,
          index_size,
          uncompressed_data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          compressed_indices_table,
          internal::avx2_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return EmbeddingSpMDMNBitRowWiseSparse_ref(
          bit_rate,
          block_size,
          output_size,
          index_size,
          uncompressed_data_size,
          // compressed_data_size,
          input,
          indices,
          compressed_indices_table,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets);
    };
  }
}

#define INSTANTIATE_SPMDM_BASE(RET_TYPE, NAME, INDEX_TYPE, OFFSET_TYPE) \
  template FBGEMM_API                                                   \
      typename RET_TYPE<uint8_t, INDEX_TYPE, OFFSET_TYPE>::Type         \
      NAME<INDEX_TYPE, OFFSET_TYPE>(                                    \
          int bit_rate,                                                 \
          const int64_t block_size,                                     \
          bool has_weight,                                              \
          bool normalize_by_lengths,                                    \
          int prefetch,                                                 \
          bool is_weight_positional,                                    \
          bool use_offsets);

#define INSTANTIATE_SPMDM_NAME(INDEX_TYPE, OFFSET_TYPE) \
  INSTANTIATE_SPMDM_BASE(                               \
      EmbeddingSpMDMKernelSignature,                    \
      GenerateEmbeddingSpMDMNBit,                       \
      INDEX_TYPE,                                       \
      OFFSET_TYPE)                                      \
  INSTANTIATE_SPMDM_BASE(                               \
      EmbeddingSpMDMRowWiseSparseKernelSignature,       \
      GenerateEmbeddingSpMDMNBitRowWiseSparse,          \
      INDEX_TYPE,                                       \
      OFFSET_TYPE)

#define INSTANTIATE_SPMDM_OFFSET_T(INDEX_TYPE) \
  INSTANTIATE_SPMDM_NAME(INDEX_TYPE, int32_t)  \
  INSTANTIATE_SPMDM_NAME(INDEX_TYPE, int64_t)

INSTANTIATE_SPMDM_OFFSET_T(int32_t)
INSTANTIATE_SPMDM_OFFSET_T(int64_t)

#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_NAME
#undef INSTANTIATE_SPMDM_BASE

} // namespace fbgemm

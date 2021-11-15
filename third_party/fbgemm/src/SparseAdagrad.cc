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
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./MaskAvx2.h"
#include "./RefImplementations.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {
namespace x86 = asmjit::x86;

template <typename indxType = std::int64_t>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = int (*)(
      int num_rows, // number of rows reading
      std::uint64_t param_size, // total number of parameters
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const indxType* indices, // indices of each row
      float epsilon,
      float lr,
      const int* mask_avx2,
      float weight_decay,
      const double* counter,
      std::int64_t counter_halflife);
};

template <
    typename indxType = std::int64_t,
    inst_set_t instSet = inst_set_t::avx2>
class GenSparseAdagrad {
 public:
  GenSparseAdagrad() {}
  void genSparseAdagrad(
      x86::Emitter* a,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      x86::Ymm mask_vreg,
      typename simd_info<instSet>::vec_reg_t temp_vreg,
      typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
      bool has_weight_decay);

  void genRowwiseSparseAdagrad(
      x86::Emitter* a,
      int block_size,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      x86::Ymm mask_vreg,
      typename simd_info<instSet>::vec_reg_t temp_vreg,
      typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
      bool has_weight_decay);

  typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
  getOrCreate(
      int block_size,
      int prefetch,
      bool rowwise,
      bool has_weight_decay);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static std::mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on embedding dimension (block size), prefetch distance,
  // rowwise, and has_weight_decay
  static CodeCache<
      std::tuple<int, int, bool, bool>,
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.

  // These are register we share accross SparseAdagrad and RowwiseSparseAdagrad
  x86::Gp w;
  x86::Gp g;
  x86::Gp h;
  x86::Gp indices;
  x86::Gp base_offset;
  x86::Gp temp1_; // loop counter
  x86::Gp temp2_; // prefetch offset
  x86::Gp temp3_; // prefetch offset of moment in rowwise adagrad

  x86::KReg reduce_mask_avx512_;
}; // GenEmbeddingLookup

template <typename indxType, inst_set_t instSet>
std::mutex GenSparseAdagrad<indxType, instSet>::rtMutex_;

template <typename indxType, inst_set_t instSet>
CodeCache<
    std::tuple<int, int, bool, bool>,
    typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
    GenSparseAdagrad<indxType, instSet>::codeCache_;

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genSparseAdagrad(
    x86::Emitter* a,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    x86::Ymm mask_vreg,
    typename simd_info<instSet>::vec_reg_t temp_vreg,
    typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
    bool has_weight_decay) {
  // NOTE: temp_vreg is defined only when remainder is true and instSet == avx2
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;
  constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      vec_reg_t out_vreg = vec_reg_t(v);
      vec_reg_t g_vreg = vec_reg_t(v + cur_unroll_factor);

      if (prefetch && ((vec_idx + v) % (64 / (vlen * sizeof(float))) == 0)) {
        // Intel SDE (wrongly) thinks prefetchwt1 is not available in BDW
        a->prefetchw(
            x86::dword_ptr(h, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->prefetchw(
            x86::dword_ptr(w, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));
      }

      auto g_ptr = x86::dword_ptr(g, (vec_idx + v) * vlen * sizeof(float));
      auto h_ptr = x86::dword_ptr(
          h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float));
      auto w_ptr = x86::dword_ptr(
          w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float));
      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(g_vreg.ymm(), mask_vreg, g_ptr);
          if (has_weight_decay) {
            // TODO(@taiqing) use a vreg for weights to avoid duplicate indexing
            a->vmaskmovps(temp_vreg.ymm(), mask_vreg, w_ptr);
            a->vfmadd231ps(g_vreg, temp_vreg, weight_decay_vreg);
          }
          a->vmulps(out_vreg, g_vreg, g_vreg);
          a->vmaskmovps(temp_vreg.ymm(), mask_vreg, h_ptr);
          a->vaddps(out_vreg, out_vreg, temp_vreg);

          a->vmaskmovps(h_ptr, mask_vreg, out_vreg.ymm());

          a->vsqrtps(out_vreg, out_vreg);
          a->vaddps(out_vreg, out_vreg, epsilon_vreg);

          a->vmulps(g_vreg, lr_vreg, g_vreg);
          a->vdivps(out_vreg, g_vreg, out_vreg);

          a->vmaskmovps(temp_vreg.ymm(), mask_vreg, w_ptr);
          a->vaddps(out_vreg, out_vreg, temp_vreg);

          a->vmaskmovps(w_ptr, mask_vreg, out_vreg.ymm());
        } else if (instSet == inst_set_t::avx512) {
          a->k(x86::k(1)).vmovups(g_vreg, g_ptr);
          if (has_weight_decay) {
            a->k(x86::k(1)).vfmadd231ps(g_vreg, weight_decay_vreg, w_ptr);
          }
          a->k(x86::k(1)).vmulps(out_vreg, g_vreg, g_vreg);
          a->k(x86::k(1)).vaddps(out_vreg, out_vreg, h_ptr);

          a->k(x86::k(1)).vmovups(h_ptr, out_vreg);

          a->k(x86::k(1)).vsqrtps(out_vreg, out_vreg);
          a->k(x86::k(1)).vaddps(out_vreg, out_vreg, epsilon_vreg);

          a->k(x86::k(1)).vmulps(g_vreg, lr_vreg, g_vreg);
          a->k(x86::k(1)).vdivps(out_vreg, g_vreg, out_vreg);

          a->k(x86::k(1)).vaddps(out_vreg, out_vreg, w_ptr);

          a->k(x86::k(1)).vmovups(w_ptr, out_vreg);
        }
      } else {
        a->vmovups(g_vreg, g_ptr);
        if (has_weight_decay) {
          a->vfmadd231ps(g_vreg, weight_decay_vreg, w_ptr);
        }
        a->vmulps(out_vreg, g_vreg, g_vreg);
        a->vaddps(out_vreg, out_vreg, h_ptr);

        a->vmovups(h_ptr, out_vreg);

        a->vsqrtps(out_vreg, out_vreg);
        a->vaddps(out_vreg, out_vreg, epsilon_vreg);

        a->vmulps(g_vreg, lr_vreg, g_vreg);
        a->vdivps(out_vreg, g_vreg, out_vreg);

        a->vaddps(out_vreg, out_vreg, w_ptr);

        a->vmovups(w_ptr, out_vreg);
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genRowwiseSparseAdagrad(
    x86::Emitter* a,
    int block_size,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    x86::Ymm mask_vreg,
    typename simd_info<instSet>::vec_reg_t temp_vreg,
    typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
    bool has_weight_decay) {
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;
  constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;

  // Reduce the unroll factor by 1 for partial sum
  --unroll_factor;
  vec_reg_t partial_sum_vreg = vec_reg_t(unroll_factor);

  if (prefetch) {
    a->prefetchw(x86::dword_ptr(h, temp3_));
  }

  bool areIndices64b = std::is_same<indxType, std::int64_t>::value;
  auto indices_ptr = areIndices64b
      ? x86::qword_ptr(
            indices, temp1_, 3) // use of 3 is to muliply by 8 (int64_t)
      : x86::dword_ptr(
            indices, temp1_, 2); // use of 2 is to muliply by 4 (int32_t)
  if (has_weight_decay) {
    // set base_offset for fetching w in the calculation of gradient square sum
    a->imul(
        areIndices64b ? base_offset : base_offset.r32(),
        indices_ptr,
        static_cast<asmjit::Imm>(block_size * sizeof(float)));
  }

  // Even with avx512, we only need to use avx2 registers when computing
  // partial_sum because some instructions we're using like vhaddps
  // are only in avx2.
  constexpr int vlen_avx2 = simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
  int num_vec_regs_per_block_avx2 = (block_size + vlen_avx2 - 1) / vlen_avx2;

  // Use YMM/XMMs with smaller ids for AVX2 specific instructions like vhaddps
  x86::Ymm partial_sum_vreg_avx2(0);
  x86::Xmm partial_sum_xmm0(partial_sum_vreg_avx2.id());

  a->vxorps(
      partial_sum_vreg_avx2, partial_sum_vreg_avx2, partial_sum_vreg_avx2);

  // TODO: need to do a tree-reduction to fully take advantage of unrolling
  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block_avx2;
       vec_idx += unroll_factor - 1) {
    int cur_unroll_factor =
        std::min(unroll_factor - 1, num_vec_regs_per_block_avx2 - vec_idx);
    for (int v = 0; v < cur_unroll_factor; ++v) {
      x86::Ymm out_vreg = x86::Ymm(v + 1);
      if (has_weight_decay && prefetch &&
          ((vec_idx + v) % (64 / (vlen_avx2 * sizeof(float))) == 0)) {
        a->prefetchw(x86::dword_ptr(
            w, temp2_, 0, (vec_idx + v) * vlen_avx2 * sizeof(float)));
      }

      auto g_ptr = x86::dword_ptr(g, (vec_idx + v) * vlen_avx2 * sizeof(float));
      auto w_ptr = x86::dword_ptr(
          w, base_offset, 0, (vec_idx + v) * vlen_avx2 * sizeof(float));
      if (block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS &&
          vec_idx + v == num_vec_regs_per_block_avx2 - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(out_vreg, mask_vreg, g_ptr);
          if (has_weight_decay) {
            a->vmaskmovps(temp_vreg.ymm(), mask_vreg, w_ptr);
            a->vfmadd231ps(out_vreg, temp_vreg, weight_decay_vreg);
          }
        } else {
          a->k(reduce_mask_avx512_).z().vmovups(out_vreg, g_ptr);
          if (has_weight_decay) {
            a->k(reduce_mask_avx512_)
                .z()
                .vfmadd231ps(out_vreg, weight_decay_vreg, w_ptr);
          }
        }
      } else {
        a->vmovups(out_vreg, g_ptr);
        if (has_weight_decay) {
          a->vfmadd231ps(out_vreg, weight_decay_vreg, w_ptr);
        }
      }
      a->vmulps(out_vreg, out_vreg, out_vreg);
      a->vaddps(partial_sum_vreg_avx2, partial_sum_vreg_avx2, out_vreg);
    }
  }
  // Reduce sum to 1 value
  // __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
  // __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
  a->vhaddps(
      partial_sum_vreg_avx2, partial_sum_vreg_avx2, partial_sum_vreg_avx2);
  a->vhaddps(
      partial_sum_vreg_avx2, partial_sum_vreg_avx2, partial_sum_vreg_avx2);

  x86::Xmm partial_sum_xmm1(1);

  //_mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3))
  a->movss(partial_sum_xmm1, partial_sum_xmm0);
  //_mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1))
  a->vextractf128(partial_sum_xmm0, partial_sum_vreg_avx2, 1);

  // final_sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
  //    _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
  a->addss(partial_sum_xmm0, partial_sum_xmm1);

  // This fragment moves block size (N) to stack and bcasts it to xmm reg
  a->lea(
      x86::rsp,
      x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
  a->mov(x86::dword_ptr(x86::rsp), block_size);
  a->vbroadcastss(
      partial_sum_xmm1, x86::dword_ptr(x86::rsp)); // N is partial_sum_xmm1
  a->vcvtdq2ps(partial_sum_xmm1, partial_sum_xmm1);
  a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

  if (has_weight_decay) {
    // set base_offset for fetching h
    a->imul(
        areIndices64b ? base_offset : base_offset.r32(),
        indices_ptr,
        static_cast<asmjit::Imm>(sizeof(float)));
  }

  // final_sum /= N
  a->divss(partial_sum_xmm0, partial_sum_xmm1);
  // load h
  a->movss(partial_sum_xmm1, x86::dword_ptr(h, base_offset));
  // *h + final_sum
  a->addss(partial_sum_xmm0, partial_sum_xmm1);
  // store h
  a->movss(x86::dword_ptr(h, base_offset), partial_sum_xmm0);
  // sqrt(hi)
  a->sqrtss(partial_sum_xmm0, partial_sum_xmm0);
  // bcast partial to all of ymm/zmm reg
  a->vpbroadcastd(partial_sum_vreg, partial_sum_xmm0);
  // lr / sqrt(hi) + epsilon
  a->vaddps(partial_sum_vreg, partial_sum_vreg, epsilon_vreg);
  a->vdivps(partial_sum_vreg, lr_vreg, partial_sum_vreg);
  // partial_sum_vreg now has float_step

  // set base_offset for fetching w in updating weights
  a->imul(
      areIndices64b ? base_offset : base_offset.r32(),
      indices_ptr,
      static_cast<asmjit::Imm>(block_size * sizeof(float)));

  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      vec_reg_t out_vreg = vec_reg_t(v);

      if (!has_weight_decay && prefetch &&
          ((vec_idx + v) % (64 / (vlen * sizeof(float))) == 0)) {
        a->prefetchw(
            x86::dword_ptr(w, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));
      }

      auto g_ptr = x86::dword_ptr(g, (vec_idx + v) * vlen * sizeof(float));
      auto w_ptr = x86::dword_ptr(
          w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float));
      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(temp_vreg.ymm(), mask_vreg, g_ptr);
          if (has_weight_decay) {
            a->vmaskmovps(out_vreg.ymm(), mask_vreg, w_ptr);
            // TODO(@taiqing): have vreg for weights
            a->vfmadd231ps(temp_vreg, weight_decay_vreg, out_vreg);
          }
          a->vmulps(temp_vreg, partial_sum_vreg, temp_vreg);

          a->vmaskmovps(out_vreg.ymm(), mask_vreg, w_ptr);
          a->vaddps(out_vreg, temp_vreg, out_vreg);

          a->vmaskmovps(w_ptr, mask_vreg, out_vreg.ymm());
        } else {
          if (has_weight_decay) {
            a->k(x86::k(1)).vmovups(out_vreg, g_ptr);
            a->k(x86::k(1)).vfmadd231ps(out_vreg, weight_decay_vreg, w_ptr);
            a->k(x86::k(1)).vmulps(out_vreg, partial_sum_vreg, out_vreg);
          } else {
            a->k(x86::k(1)).vmulps(out_vreg, partial_sum_vreg, g_ptr);
          }
          a->k(x86::k(1)).vaddps(out_vreg, out_vreg, w_ptr);
          a->k(x86::k(1)).vmovups(w_ptr, out_vreg);
        }
      } else {
        if (has_weight_decay) {
          a->vmovups(out_vreg, g_ptr);
          a->vfmadd231ps(out_vreg, weight_decay_vreg, w_ptr);
          a->vmulps(out_vreg, partial_sum_vreg, out_vreg);
        } else {
          a->vmulps(out_vreg, partial_sum_vreg, g_ptr);
        }
        a->vaddps(out_vreg, out_vreg, w_ptr);
        a->vmovups(w_ptr, out_vreg);
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
GenSparseAdagrad<indxType, instSet>::getOrCreate(
    int block_size,
    int prefetch,
    bool rowwise,
    bool has_weight_decay) {
  std::tuple<int, int, bool, bool> kernelSig =
      std::make_tuple(block_size, prefetch, rowwise, has_weight_decay);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() ->
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().environment());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
        bool areIndices64b = std::is_same<indxType, std::int64_t>::value;
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "SparseAdagrad";
        filename += "_emd_dim_" + std::to_string(block_size);
        if (rowwise) {
          filename += "_rowwise";
        }
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::avx512 ? "_avx512" : "_avx2";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight_decay) {
          filename += "weight_decay";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        x86::Gpd num_rows = a->zdi().r32();
        x86::Gp param_size = a->zsi();
        w = a->zdx();
        g = a->zcx();
        h = a->gpz(8);
        indices = a->gpz(9);
        x86::Xmm epsilon(0);
        x86::Xmm lr(1);
        x86::Gp mask_avx2 = a->gpz(10);
        x86::Xmm weight_decay(2);
        x86::Gp counter = a->gpz(11);
        x86::Gp counter_halflife = a->gpz(12);

        // reuse mask_avx2 because mask_avx2 is used only at the beginning
        base_offset = a->gpz(10);
        temp1_ = a->gpz(13);
        temp2_ = a->gpz(14);
        temp3_ = a->gpz(15);

        asmjit::FuncDetail func;
        func.init(
            asmjit::FuncSignatureT<
                int, // return type
                int, // num rows
                std::uint64_t, // param_size
                float*, // w
                const float*, // g
                float*, // h
                const indxType*, // indices
                float, // epsilon
                float, // lr
                const int*, // mask_avx2
                float, // weight_decay
                const double*, // counter then counter_halflife
                std::int64_t>(asmjit::CallConv::kIdHost),
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
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(
            num_rows,
            param_size,
            w,
            g,
            h,
            indices,
            epsilon,
            lr,
            mask_avx2,
            weight_decay,
            counter,
            counter_halflife);

        args.updateFuncFrame(frame);
        frame.finalize();
        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t epsilon_vreg;
        vec_reg_t lr_vreg;
        vec_reg_t weight_decay_vreg;
        vec_reg_t adjusted_weight_decay_vreg;
        x86::Ymm mask_vreg; // mask for avx2
        vec_reg_t
            temp_vreg; // temp vreg for avx2 to handle remainder computation

        --unroll_factor;
        epsilon_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        lr_vreg = vec_reg_t(unroll_factor);
        if (has_weight_decay) {
          --unroll_factor;
          weight_decay_vreg = vec_reg_t(unroll_factor);
          --unroll_factor;
          adjusted_weight_decay_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            --unroll_factor;
            temp_vreg = vec_reg_t(unroll_factor);
          }

          // Creating masks for non multiples of vlen iterations
          if (instSet == inst_set_t::avx2) {
            --unroll_factor;
            mask_vreg = x86::Ymm(unroll_factor);
            a->vmovups(mask_vreg, x86::dword_ptr(mask_avx2));
          } else {
            a->mov(temp1_, (1 << remainder) - 1);
            a->kmovw(x86::k(1), temp1_);
          }
        }
        // Need an extra mask for computing sum of gradients
        int remainder_avx2 =
            block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        if (remainder_avx2 && instSet == inst_set_t::avx512 && rowwise) {
          reduce_mask_avx512_ = x86::k(2);
          a->mov(temp1_, (1 << remainder_avx2) - 1);
          a->kmovw(reduce_mask_avx512_, temp1_);
        }

        if (!rowwise) {
          unroll_factor = unroll_factor / 2; // accont for g_vreg
        }

        asmjit::Label exit = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        a->vpbroadcastd(epsilon_vreg, epsilon);
        a->vpbroadcastd(lr_vreg, lr);
        if (has_weight_decay) {
          a->vpbroadcastd(weight_decay_vreg, weight_decay);
        }

        a->xor_(temp1_, temp1_);

        a->bind(LoopRangeIndexBegin);
        a->cmp(temp1_.r32(), num_rows); // temp1_ is the loop trip counter
        a->jge(LoopRangeIndexEnd);

        auto indices_ptr = areIndices64b
            ? x86::qword_ptr(
                  indices, temp1_, 3) // use of 3 is to muliply by 8 (int64_t)
            : x86::dword_ptr(
                  indices, temp1_, 2); // use of 2 is to muliply by 4 (int32_t)
        a->imul(
            areIndices64b ? base_offset : base_offset.r32(),
            indices_ptr,
            static_cast<asmjit::Imm>(
                (rowwise ? 1 : block_size) * sizeof(float)));

        // Perform this check
        // if (block_size + offsetIdx > param_size) {
        //   return i;
        // }
        if (areIndices64b) {
          a->mov(temp2_, indices_ptr);
        } else {
          a->mov(temp2_.r32(), indices_ptr);
        }

        if (has_weight_decay) {
          // Check counter != nullptr && counter[idx] > 0
          a->vmovaps(adjusted_weight_decay_vreg, weight_decay_vreg);

          asmjit::Label skip_adjust_freq = a->newLabel();

          a->cmp(counter, 0);
          a->je(skip_adjust_freq);

          // temp3_ : counter[idx]
          a->mov(temp3_, x86::qword_ptr(counter, temp2_, 3));
          a->cmp(temp3_, 0);
          a->jle(skip_adjust_freq);

          // OK to use Xmm registers with small ids that are reserved for temp
          // values in the inner most loop.
          vec_reg_t counter_halflife_vreg(0);
          x86::Xmm counter_vreg(1);
          a->cvtsi2sd(counter_halflife_vreg.xmm(), counter_halflife);
          a->movq(counter_vreg, temp3_);
          a->divpd(counter_halflife_vreg.xmm(), counter_vreg);
          a->vcvtpd2ps(
              counter_halflife_vreg.xmm(), counter_halflife_vreg.ymm());
          a->vbroadcastss(counter_halflife_vreg, counter_halflife_vreg.xmm());
          a->vmulps(
              adjusted_weight_decay_vreg,
              adjusted_weight_decay_vreg,
              counter_halflife_vreg);

          a->bind(skip_adjust_freq);
        }

        a->inc(temp2_);
        a->imul(
            temp2_,
            static_cast<asmjit::Imm>(block_size)); //(offsetIdx+1)*blocksize
        a->cmp(temp2_, param_size);
        a->jg(exit);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();

          a->mov(temp2_, temp1_);
          a->add(temp2_, prefetch);
          a->cmp(temp2_.r32(), num_rows);
          a->jge(pref_dist_reset_start);

          auto pref_indices_ptr = areIndices64b
              ? x86::qword_ptr(indices, temp2_, 3)
              : x86::dword_ptr(indices, temp2_, 2);
          if (rowwise) {
            a->imul(
                areIndices64b ? temp3_ : temp3_.r32(),
                pref_indices_ptr,
                static_cast<asmjit::Imm>(sizeof(float)));
          }
          a->imul(
              areIndices64b ? temp2_ : temp2_.r32(),
              pref_indices_ptr,
              static_cast<asmjit::Imm>(block_size * sizeof(float)));

          a->jmp(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);
          a->imul(
              areIndices64b ? temp2_ : temp2_.r32(),
              indices_ptr,
              static_cast<asmjit::Imm>(block_size * sizeof(float)));
          if (rowwise) {
            a->imul(
                areIndices64b ? temp3_ : temp3_.r32(),
                indices_ptr,
                static_cast<asmjit::Imm>(sizeof(float)));
          }

          a->bind(pref_dist_reset_end);
        } // prefetch

        if (rowwise) {
          genRowwiseSparseAdagrad(
              a,
              block_size,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg,
              temp_vreg,
              adjusted_weight_decay_vreg,
              has_weight_decay);
        } else {
          genSparseAdagrad(
              a,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg,
              temp_vreg,
              adjusted_weight_decay_vreg,
              has_weight_decay);
        }

        a->add(g, static_cast<asmjit::Imm>(block_size * sizeof(float)));
        a->inc(temp1_);
        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->bind(exit);
        a->mov(x86::eax, temp1_.r32());
        a->emitEpilog(frame);

        typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
            fn;
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
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
} // getOrCreate

// Specialization for block size 1 internally called by GenerateSparseAdaGrad
template <typename IndexType>
int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input/output parameters
    const float* g, // input gradients
    float* h, // input/output momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife) {
  if (weight_decay != 0.0f) {
    for (int i = 0; i < num_rows; ++i) {
      IndexType idx = indices[i];
      if (idx >= static_cast<int64_t>(param_size)) {
        return i;
      }

      float freq = (counter && counter[idx] > 0)
          ? counter_halflife / counter[idx]
          : 1.0f;
      float gi = std::fma(freq * weight_decay, w[idx], g[i]);
      float hi = h[idx] = h[idx] + gi * gi;
      if (rowwise) {
        w[idx] += lr / (std::sqrt(hi) + epsilon) * gi;
      } else {
        w[idx] += lr * gi / (std::sqrt(hi) + epsilon);
      }
    }
  } else {
    for (int i = 0; i < num_rows; ++i) {
      IndexType idx = indices[i];
      if (idx >= static_cast<int64_t>(param_size)) {
        return i;
      }
      float gi = g[i];
      float hi = h[idx] = h[idx] + gi * gi;
      if (rowwise) {
        w[idx] += lr / (std::sqrt(hi) + epsilon) * gi;
      } else {
        w[idx] += lr * gi / (std::sqrt(hi) + epsilon);
      }
    }
  }
  return num_rows;
}

template int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int64_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife);

template int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int32_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife);

} // namespace

template <typename IndexType>
typename SparseAdaGradSignature<IndexType>::Type GenerateSparseAdaGrad(
    int block_size,
    bool rowwise,
    int prefetch,
    bool use_weight_decay) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  if (fbgemmHasAvx512Support() || fbgemmHasAvx2Support()) {
    if (block_size == 1) {
      return [=](int num_rows, // number of rows reading
                 std::uint64_t param_size, // total number of parameters
                 float* w, // input/output parameters
                 const float* g, // input gradients
                 float* h, // input/output momentums
                 const IndexType* indices, // indices of each row
                 float epsilon,
                 float lr,
                 float weight_decay,
                 const double* counter,
                 std::int64_t counter_halflife) {
        return SparseAdaGradBlockSize1_(
            num_rows,
            param_size,
            w,
            g,
            h,
            indices,
            epsilon,
            lr,
            rowwise,
            weight_decay,
            counter,
            counter_halflife);
      };
    }
    static GenSparseAdagrad<IndexType, inst_set_t::avx2> kernel_generator;
    constexpr int VLEN = simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
    const int* mask_avx2 = &internal::avx2_ps_or_epi32_combined_mask
                               [(VLEN - (block_size % VLEN)) % VLEN];
    const auto original_func = kernel_generator.getOrCreate(
        block_size, prefetch, rowwise, use_weight_decay);
    return [=](int num_rows, // number of rows reading
               std::uint64_t param_size, // total number of parameters
               float* w, // input/output parameters
               const float* g, // input gradients
               float* h, // input/output momentums
               const IndexType* indices, // indices of each row
               float epsilon,
               float lr,
               float weight_decay,
               const double* counter,
               std::int64_t counter_halflife) {
      return original_func(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          epsilon,
          lr,
          mask_avx2,
          weight_decay,
          counter,
          counter_halflife);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return [=](int num_rows, // number of rows reading
               std::uint64_t param_size, // total number of parameters
               float* w, // input/output parameters
               const float* g, // input gradients
               float* h, // input/output momentums
               const IndexType* indices, // indices of each row
               float epsilon,
               float lr,
               float weight_decay,
               const double* counter,
               std::int64_t counter_halflife) {
      if (rowwise) {
        return rowwise_sparse_adagrad_ref(
            num_rows, // number of rows reading
            block_size, // number of parameters per rows
            param_size, // total number of parameters
            w, // input/output parameters
            g, // input gradients
            h, // input/output momentums
            indices,
            epsilon,
            lr,
            weight_decay,
            counter,
            counter_halflife);
      } else {
        return sparse_adagrad_ref(
            num_rows, // number of rows reading
            block_size, // number of parameters per rows
            param_size, // total number of parameters
            w, // input/output parameters
            g, // input gradients
            h, // input/output momentums
            indices,
            epsilon,
            lr,
            weight_decay,
            counter,
            counter_halflife);
      }
    };
  }
}

template FBGEMM_API typename SparseAdaGradSignature<std::int64_t>::Type
GenerateSparseAdaGrad<std::int64_t>(
    int block_size, // number of parameters per rows
    bool rowwise,
    int prefetch,
    bool use_weight_decay);

template FBGEMM_API typename SparseAdaGradSignature<std::int32_t>::Type
GenerateSparseAdaGrad<std::int32_t>(
    int block_size, // number of parameters per rows
    bool rowwise,
    int prefetch,
    bool use_weight_decay);

} // namespace fbgemm

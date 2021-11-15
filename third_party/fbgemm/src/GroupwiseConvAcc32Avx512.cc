/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <asmjit/asmjit.h>
#include "./CodeGenHelpers.h"
#include "./GroupwiseConv.h"

namespace fbgemm {

using namespace std;

namespace x86 = asmjit::x86;

GCONV_INST_DEF_AVX512_AND_VNNI_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genConstForPermutations(x86::Emitter* a) {
  x86::Gp permute_const_reg_upper_half = a->gpz(12);
  x86::Gp permute_const_reg_lower_half = a->gpz(13);
  x86::Xmm const_reg_xmm = x86::xmm11;
  if (this->C_per_G_ == 4) {
    // 4 group together
    // We have 1st group in position 0 and 4, 2nd group 1 and 5 and so on.
    // Permute to put 1st group to lower 128-bit and 2nd group to next
    // 128-bit and so on.
    // load f, b, 7, 3,
    //      e, a, 6, 2,
    //      d, 9, 5, 1,
    //      c, 8, 4, 0 in a 128-bit Xmm
    a->mov(
        permute_const_reg_lower_half,
        static_cast<asmjit::Imm>(0x0d0905010c080400));
    a->mov(
        permute_const_reg_upper_half,
        static_cast<asmjit::Imm>(0x0f0b07030e0a0602));
  } else {
    // this->C_per_G_ == 2
    // 8 group together
    // We have 1st group in position 0 and 8, 2nd group 1 and 9 and so on.
    // Permute to put 1st group to lower 128-bit and 2nd group to next
    // 128-bit and so on.
    // load
    //      f, 7
    //      e, 6
    //      d, 5
    //      c, 4
    //      b, 3
    //      a, 2
    //      9, 1
    //      8, 0 in a 128-bit Xmm
    a->mov(
        permute_const_reg_lower_half,
        static_cast<asmjit::Imm>(0x0b030a0209010800));
    a->mov(
        permute_const_reg_upper_half,
        static_cast<asmjit::Imm>(0x0f070e060d050c04));
  }

  a->movq(const_reg_xmm, permute_const_reg_lower_half);
  a->pinsrq(const_reg_xmm, permute_const_reg_upper_half, 1);
  // Zero extend 16 packed 8-bit integers in the low 8 bytes of const_reg_xmm
  // to 16 packed 32-bit integers in stPermReg_V_
  a->vpmovzxbd(stPermReg_V_, const_reg_xmm);
}

GCONV_INST_DEF_AVX512_AND_VNNI_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genForLoadingWeights(x86::Emitter* a) {
  using WRegs = x86::Zmm;
  int paddedICPerG = (this->C_per_G_ + 3) / 4 * 4;
  // load weights
  for (int r = 0; r < this->R_; ++r) {
    for (int s = 0; s < this->S_; ++s) {
      // For other cases, weights are too big to be kept in registers
      // and are loaded as they are used.
      if (this->C_per_G_ != 16) {
        // still use aligned move since the weigh buffer is 64bytes aligned.
        a->vmovaps(
            WRegs(r * this->S_ + s),
            // load 512 bits for weights, different grouping for different
            // workload
            x86::zmmword_ptr(
                wghts_R_,
                (r * this->S_ + s) * this->K_per_G_ * GTogether_ *
                    paddedICPerG * sizeof(int8_t)));
      }
    }
  }
}

GCONV_INST_DEF_AVX512_AND_VNNI_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::storeResult(x86::Emitter* a) {
  if (GTogether_ > 1) {
    // store with permutation
    a->vpermd(x86::Zmm(9), stPermReg_V_, x86::Zmm(9));
    if (this->accum_) {
      a->vpaddd(x86::Zmm(9), x86::Zmm(9), x86::zmmword_ptr(out_acts_R_));
    }
    a->vmovups(x86::zmmword_ptr(out_acts_R_), x86::Zmm(9));
  } else {
    // horizontal add and store
    if (this->C_per_G_ == 8) {
      a->vextracti32x8(tmpReg1_V_.ymm(), x86::Zmm(9), 1);
      a->vphaddd(x86::Ymm(9), x86::Ymm(9), tmpReg1_V_.ymm());
      a->vpermq(x86::Ymm(9), x86::Ymm(9), static_cast<asmjit::Imm>(0xd8));
      if (this->accum_) {
        a->vpaddd(x86::Ymm(9), x86::Ymm(9), x86::ymmword_ptr(out_acts_R_));
      }
      a->vmovups(x86::ymmword_ptr(out_acts_R_), x86::Ymm(9));
    } else if (this->K_per_G_ == 16) {
      // we have results in 4 Zmm registers, need to reduce them to 2 Ymm
      // register 2 * 8 * 32 where 16 is K_per_g
      // first reduce 4 * 16 * 32bits to 4 * 8 * 32bits
      for (int k = 0; k < kLoopIters_; ++k) {
        auto source_reg = x86::Zmm(9 - k);
        auto result_reg = x86::Ymm(9 - k);
        a->vextracti32x8(x86::Ymm(0), source_reg, 1);
        a->vphaddd(result_reg, result_reg, x86::Ymm(0));
        a->vpermq(result_reg, result_reg, static_cast<asmjit::Imm>(0xd8));
      }
      // secondly reduce 4 * 8 * 32  to 2 * 8 * 32 bits;
      for (int k = 0, i = 0; k < kLoopIters_; k += 2, i++) {
        auto result_reg = x86::Ymm(9 - k);
        auto adjacent_result_reg = x86::Ymm(9 - k - 1);
        a->vphaddd(result_reg, result_reg, adjacent_result_reg);
        a->vpermq(result_reg, result_reg, static_cast<asmjit::Imm>(0xd8));
        if (this->accum_) {
          a->vpaddd(
              result_reg, result_reg, x86::ymmword_ptr(out_acts_R_, 32 * i));
        }
        a->vmovups(x86::ymmword_ptr(out_acts_R_, 32 * i), result_reg);
      }
    }
  }
}

GCONV_INST_DEF_AVX512_AND_VNNI_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::storeOffset(x86::Emitter* a) {
  auto rowOffsetReg_V_Ymm = rowOffsetReg_V_.half();
  auto rowOffsetReg_V_Xmm = rowOffsetReg_V_Ymm.half();
  auto tmpReg1_V_Xmm = tmpReg1_V_.half().half();
  switch (this->C_per_G_) {
    case 2:
      // store 256-bits containing rowoffset for eight groups
      if (this->accum_) {
        a->vpaddd(
            rowOffsetReg_V_Ymm,
            rowOffsetReg_V_Ymm,
            x86::ymmword_ptr(row_offset_R_));
      }
      a->vmovdqu(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_Ymm);
      break;
    case 4:
      // store 128-bits containing rowoffset for four groups
      if (this->accum_) {
        a->vmovdqu(tmpReg1_V_Xmm, x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_Xmm, tmpReg1_V_Xmm);
      }
      a->vmovups(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_Xmm);
      break;
    case 8:
      // store 32-bits of one group
      if (this->accum_) {
        a->vmovd(tmpReg1_V_Xmm, x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_Xmm, tmpReg1_V_Xmm);
      }
      a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_Xmm);
      break;
    case 16:
      // rowOffsetReg_V_[0:63] has sum for first 8 and
      // rowOffsetReg_V_[64:127] has sum for second 8
      // execute vphaddd twice to sum the two
      a->vphaddd(rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm);
      a->vphaddd(rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm);
      if (this->accum_) {
        a->vmovd(tmpReg1_V_Xmm, x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_Xmm, tmpReg1_V_Xmm);
      }
      a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_Xmm);
      break;
    default:
      assert(0 && "not supported case");
  }
}

GCONV_INST_DEF_AVX512_AND_VNNI_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genForSingleFilterPoint(
    x86::Emitter* a,
    int r,
    int s,
    int act_s,
    bool use_zero_reg) {
  using WRegs = x86::Zmm;

  if (use_zero_reg) {
    a->vmovapd(actReg_V_, zeroPTReg_V_); // 64 * 8 bit zero points
  } else {
    if (this->C_per_G_ != 8) {
      // 2(C_Per_g) * 8 (g_together) or
      // 4(C_Per_g) * 4 (g_together) or
      // 16(C_Per_g) broadcasted into 4 slots of ZMM
      a->vbroadcasti32x4(
          actReg_V_,
          x86::oword_ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
    } else {
      // 8(C_Per_g) broadcasted into 8 slots of ZMM
      a->vbroadcasti32x2(
          actReg_V_,
          x86::qword_ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
    }
  }

  // zero extend if C_per_g smaller than 4(the accumulation width our FMA
  // instruction)
  if (this->C_per_G_ == 2) {
    // only use the lower half and extend them to 32bits(4 uint8's)
    a->vpmovzxwd(actReg_V_, actReg_V_.half());
  }

  // row offset
  if (this->needRowOffset_) {
    if (this->C_per_G_ == 2 || this->C_per_G_ == 4) {
      genU8Sum4<INST_SET>(
          a, actReg_V_, rowOffsetReg_V_, oneReg16Bit_V_, tmpReg1_V_);
    } else {
      // still use Ymm for Sum8
      genU8Sum8(a, actReg_V_.half(), rowOffsetReg_V_.half(), tmpReg1_V_.half());
    }
  }
  // FMA
  if (this->C_per_G_ != 16) {
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(r * this->S_ + s),
        WRegs(9),
        oneReg16Bit_V_,
        tmpReg1_V_);
  } else {
    // simd_info<inst_set_t::avx512>::WIDTH_BYTES
    int kLoopMultiplier = 64 / this->C_per_G_;
    for (int k = 0; k < kLoopIters_; ++k) {
      a->vmovaps(
          WRegs(0),
          // copy 512 bits of weights into ZMM, 16(C_Per_g) * 4(1/4 of K_Per_g)
          x86::zmmword_ptr(
              wghts_R_,
              (((r * this->S_ + s) * this->K_per_G_) + k * kLoopMultiplier) *
                  this->C_per_G_ * sizeof(int8_t)));
      // FMA result is not final reduction on C_per_G, producing 4 * 16 outputs
      // in which consectutive 4 elements if summed forms one final output over
      // K_Per_G dimension, we need 16 final 32bits outputs.
      genU8I8S32FMA<INST_SET>(
          a, actReg_V_, WRegs(0), WRegs(9 - k), oneReg16Bit_V_, tmpReg1_V_);
    }
  }
}
#define GENCONVKERNEL_FUNCS(S, IN)                                           \
  template void GenConvKernel<S, IN>::genForLoadingWeights<IN>(x86::Emitter* a); \
  template void GenConvKernel<S, IN>::genConstForPermutations<IN>(               \
      x86::Emitter* a);                                                      \
  template void GenConvKernel<S, IN>::genForSingleFilterPoint<IN>(               \
      x86::Emitter* a, int r, int s, int act_s, bool use_zero_reg);          \
  template void GenConvKernel<S, IN>::storeResult<IN>(x86::Emitter* a);          \
  template void GenConvKernel<S, IN>::storeOffset<IN>(x86::Emitter* a);
GENCONVKERNEL_FUNCS(1, inst_set_t::avx512)
GENCONVKERNEL_FUNCS(1, inst_set_t::avx512_vnni)
GENCONVKERNEL_FUNCS(2, inst_set_t::avx512)
GENCONVKERNEL_FUNCS(2, inst_set_t::avx512_vnni)
GENCONVKERNEL_FUNCS(3, inst_set_t::avx512)
GENCONVKERNEL_FUNCS(3, inst_set_t::avx512_vnni)
#undef GENCONVKERNEL_FUNCS

template class GenConvKernel<1, inst_set_t::avx512>;
template class GenConvKernel<1, inst_set_t::avx512_vnni>;
template class GenConvKernel<2, inst_set_t::avx512>;
template class GenConvKernel<2, inst_set_t::avx512_vnni>;
template class GenConvKernel<3, inst_set_t::avx512>;
template class GenConvKernel<3, inst_set_t::avx512_vnni>;

} // namespace fbgemm

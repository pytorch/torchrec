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
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

namespace x86 = asmjit::x86;

GCONV_INST_DEF_AVX2_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::genConstForPermutations(
    x86::Emitter* a) {
  if (this->C_per_G_ == 4) {
    x86::Gp permute_const_reg = a->gpz(12);
    x86::Xmm const_reg_xmm = x86::xmm11;
    // We have 1st group in even lanes and 2nd group in odd lanes.
    // Permute to put 1st group to lower 128-bit and 2nd group in upper
    // 128-bit.
    // load 7, 5, 3, 1, 6, 4, 2, 0 in a 64-bit reg
    a->mov(permute_const_reg, static_cast<asmjit::Imm>(0x0705030106040200));
    a->movq(const_reg_xmm, permute_const_reg);
    // Zero extend 8 packed 8-bit integers in the low 8 bytes of const_reg_xmm
    // to 8 packed 32-bit integers in stPermReg_V_
    a->vpmovzxbd(stPermReg_V_, const_reg_xmm);
  } else {
    // this->C_per_G_ == 2
    x86::Gp permute_const_reg = a->gpz(12);
    x86::Xmm const_reg_xmm = x86::xmm11;
    // We have 1st group in position 0 and 4, 2nd group 1 and 5 and so on.
    // Permute to put 1st group to lower 64-bit and 2nd group to next
    // 64-bit and so on.
    // load 7, 3, 6, 2, 5, 1, 4, 0 in a 64-bit reg
    a->mov(permute_const_reg, static_cast<asmjit::Imm>(0x0703060205010400));
    a->movq(const_reg_xmm, permute_const_reg);
    a->vpmovzxbd(stPermReg_V_, const_reg_xmm);
  }
}

GCONV_INST_DEF_AVX2_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::genForLoadingWeights(
    x86::Emitter* a) {
  using WRegs = x86::Ymm;
  int paddedICPerG = (this->C_per_G_ + 3) / 4 * 4;
  // load weights
  for (int r = 0; r < this->R_; ++r) {
    for (int s = 0; s < this->S_; ++s) {
      // For other cases, weights are too big to be kept in registers
      // and are loaded as they are used.
      if (this->C_per_G_ == 4 || this->C_per_G_ == 2) {
        a->vmovaps(
            WRegs(r * this->S_ + s),
            x86::dword_ptr(
                wghts_R_,
                (r * this->S_ + s) * this->K_per_G_ * GTogether_ *
                    paddedICPerG * sizeof(int8_t)));
      }
    }
  }
}

GCONV_INST_DEF_AVX2_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::storeResult(x86::Emitter* a) {
  if (GTogether_ > 1) {
    // store with permutation
    a->vpermd(x86::Ymm(9), stPermReg_V_, x86::Ymm(9));
    if (this->accum_) {
      a->vpaddd(x86::Ymm(9), x86::Ymm(9), x86::dword_ptr(out_acts_R_));
    }
    a->vmovups(x86::dword_ptr(out_acts_R_), x86::Ymm(9));
  } else {
    // horizontal add and store
    if (this->C_per_G_ == 8) {
      a->vphaddd(x86::Ymm(9), x86::Ymm(9), x86::Ymm(8));
      a->vpermq(x86::Ymm(9), x86::Ymm(9), static_cast<asmjit::Imm>(0xd8));
      if (this->accum_) {
        a->vpaddd(x86::Ymm(9), x86::Ymm(9), x86::dword_ptr(out_acts_R_));
      }
      a->vmovups(x86::dword_ptr(out_acts_R_), x86::Ymm(9));
    } else if (this->K_per_G_ == 16) {
      a->vphaddd(x86::Ymm(9), x86::Ymm(9), x86::Ymm(8));
      a->vpermq(x86::Ymm(9), x86::Ymm(9), static_cast<asmjit::Imm>(0xd8));

      a->vphaddd(x86::Ymm(7), x86::Ymm(7), x86::Ymm(6));
      a->vpermq(x86::Ymm(7), x86::Ymm(7), static_cast<asmjit::Imm>(0xd8));

      a->vphaddd(x86::Ymm(5), x86::Ymm(5), x86::Ymm(4));
      a->vpermq(x86::Ymm(5), x86::Ymm(5), static_cast<asmjit::Imm>(0xd8));

      a->vphaddd(x86::Ymm(3), x86::Ymm(3), x86::Ymm(2));
      a->vpermq(x86::Ymm(3), x86::Ymm(3), static_cast<asmjit::Imm>(0xd8));

      a->vphaddd(x86::Ymm(9), x86::Ymm(9), x86::Ymm(7));
      a->vpermq(x86::Ymm(9), x86::Ymm(9), static_cast<asmjit::Imm>(0xd8));

      a->vphaddd(x86::Ymm(5), x86::Ymm(5), x86::Ymm(3));
      a->vpermq(x86::Ymm(5), x86::Ymm(5), static_cast<asmjit::Imm>(0xd8));

      if (this->accum_) {
        a->vpaddd(x86::Ymm(9), x86::Ymm(9), x86::dword_ptr(out_acts_R_));
        a->vpaddd(x86::Ymm(5), x86::Ymm(5), x86::dword_ptr(out_acts_R_, 32));
      }
      a->vmovups(x86::dword_ptr(out_acts_R_), x86::Ymm(9));
      a->vmovups(x86::dword_ptr(out_acts_R_, 32), x86::Ymm(5));
    }
  }
}

GCONV_INST_DEF_AVX2_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::storeOffset(x86::Emitter* a) {
  switch (this->C_per_G_) {
    case 2:
      // store 128-bits containing rowoffset for four groups
      if (this->accum_) {
        a->paddd(rowOffsetReg_V_.half(), x86::dword_ptr(row_offset_R_));
      }
      a->vmovdqu(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half());
      break;
    case 4:
      // store 64-bits containing rowoffset for two groups
      if (this->accum_) {
        a->vmovq(tmpReg1_V_.half(), x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->vmovq(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half());
      break;
    case 8:
      if (this->accum_) {
        a->vmovd(tmpReg1_V_.half(), x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half());
      break;
    case 16:
      // rowOffsetReg_V_[0:63] has sum for first 8 and
      // rowOffsetReg_V_[64:127] has sum for second 8
      // execute vphaddd twice to sum the two
      a->vphaddd(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);
      a->vphaddd(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);
      if (this->accum_) {
        a->vmovd(tmpReg1_V_.half(), x86::dword_ptr(row_offset_R_));
        a->paddd(rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half());
      break;
    default:
      assert(0 && "not supported case");
  }
}

GCONV_INST_DEF_AVX2_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::genForSingleFilterPoint(
    x86::Emitter* a,
    int r,
    int s,
    int act_s,
    bool use_zero_reg) {
  using WRegs = x86::Ymm;

  if (GTogether_ > 1) {
    if (this->C_per_G_ == 2) { // group together = 4
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_); // 32 * 8 bit zero points
      } else {
        a->vbroadcastsd( // 64 bits broadcast, 2(C_per_g) * 4 (g_together) and
                         // broadcast them to align with weights layout
            actReg_V_,
            x86::word_ptr(in_acts_R_, (act_s * this->C_) * sizeof(uint8_t)));
      }
      // 8 * 16 bit activation to 8 * 32 bit activation( C_per_G = 2)
      // zero extend because vpmaddubsw and vpmaddwd together sum 4 consecutive
      // elements
      a->vpmovzxwd(actReg_V_, actReg_V_.half());
    } else if (this->C_per_G_ == 4) { // group together = 2
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_); // 32 * 8 bit zero points
      } else {
        a->vbroadcastsd(
            actReg_V_,
            x86::dword_ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
      }
    }
    // row offset
    if (this->needRowOffset_) {
      genU8Sum4<INST_SET>(
          a, actReg_V_, rowOffsetReg_V_, oneReg16Bit_V_, tmpReg1_V_);
    }
    // 32 * int8 weight product 32 * uint8 activation -> 8
    // output(K_per_g * group_together)
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(r * this->S_ + s),
        x86::Ymm(9),
        oneReg16Bit_V_,
        tmpReg1_V_);
  } else {
    if (this->C_per_G_ == 8) {
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_);
      } else {
        a->vbroadcastsd(
            actReg_V_,
            x86::qword_ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
      }
    } else {
      // this->C_per_G_ == 16
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_);
      } else {
        a->vbroadcasti128(
            actReg_V_,
            x86::oword_ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
      }
    }
    // row offset
    if (this->needRowOffset_) {
      genU8Sum8(a, actReg_V_, rowOffsetReg_V_, tmpReg1_V_);
    }
    int kLoopMultiplier = 32 / this->C_per_G_;
    for (int k = 0; k < kLoopIters_; ++k) {
      a->vmovaps(
          WRegs(0),
          x86::dword_ptr(
              wghts_R_,
              (((r * this->S_ + s) * this->K_per_G_) + k * kLoopMultiplier) *
                  this->C_per_G_ * sizeof(int8_t)));
      // FMA result is not final reduction on C_per_G, producing 8 output in
      // which consectutive 2 elements if summedforms one final output over
      // K_Per_G dimension
      genU8I8S32FMA<INST_SET>(
          a, actReg_V_, WRegs(0), x86::Ymm(9 - k), oneReg16Bit_V_, tmpReg1_V_);
    }
  }
}

#define GENCONVKERNEL_FUNCS(S, IN)                                       \
  template void GenConvKernel<S, IN>::genForLoadingWeights<IN>(          \
      x86::Emitter * a);                                                 \
  template void GenConvKernel<S, IN>::genConstForPermutations<IN>(       \
      x86::Emitter * a);                                                 \
  template void GenConvKernel<S, IN>::genForSingleFilterPoint<IN>(       \
      x86::Emitter * a, int r, int s, int act_s, bool use_zero_reg);     \
  template void GenConvKernel<S, IN>::storeResult<IN>(x86::Emitter * a); \
  template void GenConvKernel<S, IN>::storeOffset<IN>(x86::Emitter * a);
GENCONVKERNEL_FUNCS(1, inst_set_t::avx2)
GENCONVKERNEL_FUNCS(2, inst_set_t::avx2)
GENCONVKERNEL_FUNCS(3, inst_set_t::avx2)
#undef GENCONVKERNEL_FUNCS

template class GenConvKernel<1, inst_set_t::avx2>;
template class GenConvKernel<2, inst_set_t::avx2>;
template class GenConvKernel<3, inst_set_t::avx2>;

} // namespace fbgemm

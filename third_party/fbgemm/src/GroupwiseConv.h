/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include "./CodeCache.h"
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
/*#define FBGEMM_LOG_CODE 1*/

#define GCONV_INST_AVX2_HEADER          \
  template <inst_set_t ISET = INST_SET> \
  typename std::enable_if<ISET == inst_set_t::avx2, void>::type

#define GCONV_INST_AVX512_AND_VNNI_HEADER                            \
  template <inst_set_t ISET = INST_SET>                              \
  typename std::enable_if<                                           \
      ISET == inst_set_t::avx512 || ISET == inst_set_t::avx512_vnni, \
      void>::type

#define GCONV_INST_DEF_AVX2_HEADER                \
  template <int SPATIAL_DIM, inst_set_t INST_SET> \
  template <inst_set_t ISET>                      \
  typename std::enable_if<ISET == inst_set_t::avx2, void>::type

#define GCONV_INST_DEF_AVX512_AND_VNNI_HEADER                        \
  template <int SPATIAL_DIM, inst_set_t INST_SET>                    \
  template <inst_set_t ISET>                                         \
  typename std::enable_if<                                           \
      ISET == inst_set_t::avx512 || ISET == inst_set_t::avx512_vnni, \
      void>::type

namespace fbgemm {

namespace x86 = asmjit::x86;

template <typename>
struct is_requantization : std::false_type {};

template <
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    typename BIAS_TYPE,
    typename outT,
    typename inT,
    typename nextOPType>
struct is_requantization<
    ReQuantizeOutput<FUSE_RELU, Q_GRAN, BIAS_TYPE, outT, inT, nextOPType>>
    : std::true_type {};

using jit_conv_kernel_fp = void (*)(
    const uint8_t* in_acts,
    int8_t* wghts,
    int32_t* out_acts,
    int32_t a_zero_pt,
    int32_t oh_start,
    int32_t oh_end,
    int32_t ow,
    int32_t* row_offset);

using kernel_sig_t = std::tuple<
    bool, /* is A zero point 0 */
    bool, /* should row offset be calculated */
    bool, /* is top edge included */
    bool, /* is bottom edge included */
    bool, /* is top bottom edge same? */
    bool, /* use paddings on bottom side? */
    bool, /* use paddings on right side? */
    bool, /* accumulate rowoffsets and output instead of overwrite? */
    int, /* groups */
    int, /* stride */
    int, /* number of input channels per group */
    int>; /* number of output channels per group */

// Common code in a base class
template <int SPATIAL_DIM, inst_set_t INST_SET>
class GenConvKernelBase {
 public:
  GenConvKernelBase(
      const conv_param_t<SPATIAL_DIM>& conv_param,
      std::int32_t a_zero_point,
      bool needRowOffset,
      bool isTopEdgeIncluded,
      bool isBottomEdgeIncluded,
      bool isTopBottomEdgeSame,
      bool accum) {
    assert(fbgemmOptimizedGConv(conv_param));

    isAZeroPointZero_ = a_zero_point == 0;
    needRowOffset_ = needRowOffset;
    isTopEdgeIncluded_ = isTopEdgeIncluded;
    isBottomEdgeIncluded_ = isBottomEdgeIncluded;
    isTopBottomEdgeSame_ = isTopBottomEdgeSame;
    accum_ = accum;

    G_ = conv_param.G;
    K_per_G_ = conv_param.OC / conv_param.G;
    K_ = conv_param.OC;
    C_per_G_ = conv_param.IC / conv_param.G;
    C_ = conv_param.IC;

    // Strides are assumed to be the same in all directions
    STRIDE_ = conv_param.stride[0];
    R_ = conv_param.K[0];
    S_ = conv_param.K[1];
    OH_ = conv_param.OUT_DIM[0];
    OW_ = conv_param.OUT_DIM[1];
    H_PAD_ = conv_param.pad[0];
    W_PAD_ = conv_param.pad[1];

    use_bottom_padding_ =
        !(STRIDE_ > 1 && conv_param.IN_DIM[SPATIAL_DIM - 2] % 2 == 0);
    use_right_padding_ =
        !(STRIDE_ > 1 && conv_param.IN_DIM[SPATIAL_DIM - 1] % 2 == 0);
  }

  ~GenConvKernelBase() {}

  static std::string getCodeLoggingFile(kernel_sig_t kernel_sig) {
    std::ostringstream oss;
    oss << "conv";
    oss << "_G-" << std::get<8>(kernel_sig);
    oss << "_stride-" << std::get<9>(kernel_sig);
    oss << "_IC_per_G-" << std::get<10>(kernel_sig);
    oss << "_OC_per_G-" << std::get<11>(kernel_sig);
    oss << "_isZeroPointZero-" << std::get<0>(kernel_sig);
    oss << "_rowoffset-" << std::get<1>(kernel_sig);
    oss << "_topEdge-" << std::get<2>(kernel_sig);
    oss << "_bottomEdge-" << std::get<3>(kernel_sig);
    oss << "_isTopBottomSame-" << std::get<4>(kernel_sig);
    oss << "_useBottomPadding-" << std::get<5>(kernel_sig);
    oss << "_useRightPadding-" << std::get<6>(kernel_sig);
    oss << "_accum-" << std::get<7>(kernel_sig);

    if (INST_SET == inst_set_t::avx512) {
      oss << "_avx512";
    } else if (INST_SET == inst_set_t::avx2) {
      oss << "_avx2";
    } else {
      oss << "_unknown";
    }

    oss << ".txt";
    return oss.str();
  }

  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static std::mutex rtMutex_; ///< Control access to runtime;

  static CodeCache<
      kernel_sig_t,
      jit_conv_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.

 protected:
  // current conv parameters
  int G_; ///< Number of groups
  int K_; ///< Number of output channels
  int K_per_G_; ///< Number of output channels per group
  int C_; ///< Number of input channels
  int STRIDE_; ///< Stride in either direction
  int C_per_G_; ///< Number of input channels per group
  int R_; ///< Filter/Kernel height
  int S_; ///< Filter/Kernel width
  int OH_; ///< output height
  int OW_; ///< output width
  int H_PAD_; ///< Padding for height (top and bottom)
  int W_PAD_; ///< Padding for width (left and right)

  // Other parameters
  bool isAZeroPointZero_;
  bool needRowOffset_;
  bool isTopEdgeIncluded_;
  bool isBottomEdgeIncluded_;
  bool isTopBottomEdgeSame_;
  bool accum_;
  // For 3x3 kernels with pad == 1: If stride is 2 and image height/width are
  // even, the right or bottom paddings are not used. This variables is set to
  // false if paddings on the left and bottom are not used and kernel generation
  // takes care to not generate code with paddings on the right and bottom side.
  bool use_bottom_padding_;
  bool use_right_padding_;
};

// Generic class
template <int SPATIAL_DIM, inst_set_t INST_SET>
class FBGEMM_API GenConvKernel
    : public GenConvKernelBase<SPATIAL_DIM, INST_SET> {
  typedef typename simd_info<INST_SET>::vec_reg_t vec_reg_t;

 public:
  GenConvKernel(
      const conv_param_t<SPATIAL_DIM>& conv_param,
      std::int32_t a_zero_point,
      bool needRowoffset,
      bool isTopEdgeIncluded,
      bool isBottomEdgeIncluded,
      bool isTopBottomEdgeSame,
      bool accum)
      : GenConvKernelBase<SPATIAL_DIM, INST_SET>(
            conv_param,
            a_zero_point,
            needRowoffset,
            isTopEdgeIncluded,
            isBottomEdgeIncluded,
            isTopBottomEdgeSame,
            accum) {
    constexpr int SIMD_WIDTH = simd_info<INST_SET>::WIDTH_BYTES;
    GTogether_ = PackWeightMatrixForGConv<int8_t, int32_t, SPATIAL_DIM>::
        numOfGroupsTogether(conv_param);
    kLoopIters_ = this->K_per_G_ * this->C_per_G_ / SIMD_WIDTH;
    // y/zmm0-8 are used for holding weights
    zeroPTReg_V_ = vec_reg_t(10);
    tmpReg1_V_ = vec_reg_t(11);
    stPermReg_V_ = vec_reg_t(12);
    actReg_V_ = vec_reg_t(13);
    oneReg16Bit_V_ = vec_reg_t(15);
    rowOffsetReg_V_ = vec_reg_t(14);
  }

  jit_conv_kernel_fp getOrCreate();

  GCONV_INST_AVX2_HEADER genForLoadingWeights(x86::Emitter* a);

  GCONV_INST_AVX512_AND_VNNI_HEADER genForLoadingWeights(x86::Emitter* a);

  GCONV_INST_AVX2_HEADER genConstForPermutations(x86::Emitter* a);

  GCONV_INST_AVX512_AND_VNNI_HEADER genConstForPermutations(x86::Emitter* a);

  GCONV_INST_AVX2_HEADER genForSingleFilterPoint(
      x86::Emitter* a,
      int r,
      int s,
      int act_s,
      bool use_zero_reg);

  GCONV_INST_AVX512_AND_VNNI_HEADER genForSingleFilterPoint(
      x86::Emitter* a,
      int r,
      int s,
      int act_s,
      bool use_zero_reg);

  GCONV_INST_AVX2_HEADER storeResult(x86::Emitter* a);

  GCONV_INST_AVX512_AND_VNNI_HEADER storeResult(x86::Emitter* a);

  GCONV_INST_AVX2_HEADER storeOffset(x86::Emitter* a);

  GCONV_INST_AVX512_AND_VNNI_HEADER storeOffset(x86::Emitter* a);

  void genForTopOrBottomEdge(x86::Emitter* a, bool isTop, bool isBottom);

  void initResultRegs(x86::Emitter* a);

  void genCoreInsts(x86::Emitter* a);

  void genForSingleOutput(
      x86::Emitter* a,
      bool isLeft,
      bool isRight,
      bool isTop,
      bool isBottom);

 private:
  int GTogether_;
  // The number of iterations needed for K dim.
  // e.g., C_per_G_ = K_per_G_ = 8, we have to iterate
  // twice on K dim because 4 (from K dim) * 8 ( from C dim)
  // fill the full avx2 vector width.
  int kLoopIters_;
  asmjit::FuncDetail func_;
  asmjit::FuncFrame frame_;
  vec_reg_t zeroPTReg_V_;
  vec_reg_t tmpReg1_V_;
  vec_reg_t stPermReg_V_;
  vec_reg_t actReg_V_;
  vec_reg_t resultReg_V_;
  vec_reg_t oneReg8Bit_V_;
  vec_reg_t oneReg16Bit_V_;
  vec_reg_t rowOffsetReg_V_;

  // arguments to the function created
  x86::Gp in_acts_R_;
  x86::Gp wghts_R_;
  x86::Gp out_acts_R_;
  x86::Gp a_zero_pt_R_;
  x86::Gp H_R_;
  x86::Gp H_start_R_;
  x86::Gp H_end_R_;
  x86::Gp W_R_;
  x86::Gp row_offset_R_;

  // Used registers
  x86::Gp loopR1_;
  x86::Gp loopR2_;
  x86::Gp scratchReg1_;
  x86::Gp scratchReg2_;
};

template <int SPATIAL_DIM, inst_set_t INST_SET>
std::mutex GenConvKernelBase<SPATIAL_DIM, INST_SET>::rtMutex_;

template <int SPATIAL_DIM, inst_set_t INST_SET>
CodeCache<kernel_sig_t, jit_conv_kernel_fp>
    GenConvKernelBase<SPATIAL_DIM, INST_SET>::codeCache_;

} // namespace fbgemm

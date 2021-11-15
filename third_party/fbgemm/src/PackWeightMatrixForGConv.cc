/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <cassert>
#include <iomanip>
#include <numeric>
#include "./RefImplementations.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT, int SPATIAL_DIM>
PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::PackWeightMatrixForGConv(
    matrix_op_t trans,
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const T* sdata,
    T* pdata)
    : trans_(trans), conv_param_(conv_param), sdata_(sdata) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  GTogether_ = numOfGroupsTogether(conv_param_);
  assert(
      GTogether_ <= conv_param_.G &&
      "Number of groups together smaller than total number of groups");
  if (!pdata) {
    bufAllocatedHere_ = true;
    int kernel_prod = std::accumulate(
        conv_param.K.begin(), conv_param.K.end(), 1, std::multiplies<int>());
    // we make it a multiple of 4
    int paddedICPerG = ((conv_param_.IC / conv_param_.G) + 3) / 4 * 4;
    pdata_ = static_cast<T*>(fbgemmAlignedAlloc(
        64,
        (conv_param_.G + GTogether_ - 1) / GTogether_ * GTogether_ *
            kernel_prod * (conv_param_.OC / conv_param_.G) * paddedICPerG *
            sizeof(T)));
  } else {
    bufAllocatedHere_ = false;
    pdata_ = pdata;
  }

  pack();
}

template <typename T, typename accT, int SPATIAL_DIM>
int PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::numOfGroupsTogether(
    const conv_param_t<SPATIAL_DIM>& conv_param) {
  int OC_per_G = conv_param.OC / conv_param.G;
  int IC_per_G = conv_param.IC / conv_param.G;
  if (fbgemmHasAvx512Support() || fbgemmHasAvx512VnniSupport()) {
    // TODO: change to avx512 when avx512 support is available
    return std::max(
        simd_info<inst_set_t::avx512>::WIDTH_BYTES / OC_per_G /
            std::max(IC_per_G, 4),
        1);
  } else {
    // avx2
    // e.g., IC_per_G == 4, we need to work on 2 groups at a time
    return std::max(
        simd_info<inst_set_t::avx2>::WIDTH_BYTES / OC_per_G /
            std::max(IC_per_G, 4),
        1);
  }
  return 1;
}

/**
 * @brief Get the index of the unpacked data
 *        for a given <t, r, s, k, g, c, tr>
 *
 * Non-transposed: G (T R S C/G) K/G
 * Transposed: G K/G (T R S C/G)
 * Using inline as this will be called frequently
 */
template <typename T, typename accT, int SPATIAL_DIM>
inline int PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::unpacked_index_(
    int t,
    int r,
    int s,
    int k,
    int g,
    int c,
    bool tr) {
  // Get the full dimensions
  // Can't use T as varname because T is a template parameter.
  int F = SPATIAL_DIM <= 2 ? 1 : conv_param_.K[SPATIAL_DIM - 3];
  int R = SPATIAL_DIM == 1 ? 1 : conv_param_.K[SPATIAL_DIM - 2];
  int S = conv_param_.K[SPATIAL_DIM - 1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;

  int idx;
  if (tr) {
    idx = ((((g * OC_per_G + k) * F + t) * R + r) * S + s) * IC_per_G + c;
  } else {
    idx = ((((g * F + t) * R + r) * S + s) * IC_per_G + c) * OC_per_G + k;
  }
  return idx;
}

/**
 * @brief Get the index of the packed data for a given <t, r, s, k, g, c>
 *
 * The index may differ depending on IC_per_G.
 * Using inline as this will be called frequently
 */
template <typename T, typename accT, int SPATIAL_DIM>
inline int PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::packed_index_(
    int t,
    int r,
    int s,
    int k,
    int g,
    int c) {
  // Get the full dimensions
  // Can't use T as varname because T is a template parameter.
  int F = SPATIAL_DIM <= 2 ? 1 : conv_param_.K[SPATIAL_DIM - 3];
  int R = SPATIAL_DIM == 1 ? 1 : conv_param_.K[SPATIAL_DIM - 2];
  int S = conv_param_.K[SPATIAL_DIM - 1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;
  int paddedICPerG = (IC_per_G + 3) / 4 * 4;

  int idx = ((((((g / GTogether_) * F + t) * R + r) * S + s) * OC_per_G + k) *
                 GTogether_ +
             (g % GTogether_)) *
          paddedICPerG +
      c;
  return idx;
}

/**
 * @brief Pack or unpack matrix
 *
 * Let IC_per_G be number of input channels per group and OC_per_G be number of
 * output channels per group.
 *
 * For IC_per_G == 4 && OC_per_G == 4 optimized
 * kernel works on 2 groups at a time hence input channels for g and g+1 group
 * are laid out sequentially for each output channel, i.e., the layout is (G/2)
 * R S K (2C) and K (2C) is in each 32B vector.
 * We work on two groups at a time to fully utilize the avx2 SIMD width of
 * 256-bits.
 *
 * For IC_per_G == 8, 16, 32 && OC_per_G == 8, 16, 32 there is no need to work
 * on 2 groups at a time and full SIMD width can be efficiently utilized even
 * while working on 1 group at a time.
 * In this case, the layout is G R S K_per_G paddedICPerG
 */

template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack_unpack_(
    const T* src,
    T* dst,
    bool ispack) {
  // Can't use T as varname because T is a template parameter.
  int F = SPATIAL_DIM <= 2 ? 1 : conv_param_.K[SPATIAL_DIM - 3];
  int R = SPATIAL_DIM == 1 ? 1 : conv_param_.K[SPATIAL_DIM - 2];
  int S = conv_param_.K[SPATIAL_DIM - 1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;
  int paddedICPerG = (IC_per_G + 3) / 4 * 4;

  // If transpose option is set, the weight matrix is in layout G K/G (T R S
  // C/G) instead of G (T R S C/G) K/G
  bool tr = (trans_ == matrix_op_t::Transpose);
  if (fbgemmOptimizedGConv(conv_param_)) {
    // currently only this case is supported
    for (int t = 0; t < F; ++t) {
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
          for (int k = 0; k < OC_per_G; ++k) {
            for (int g = 0; g < G; ++g) {
              for (int c = 0; c < IC_per_G; ++c) {
                int p_idx = packed_index_(t, r, s, k, g, c);
                int up_idx = unpacked_index_(t, r, s, k, g, c, tr);
                // Pack: src (unpacked) -> dst (packed)
                if (ispack) {
                  dst[p_idx] = src[up_idx];
                } else {
                  dst[up_idx] = src[p_idx];
                }
              }
              if (ispack) {
                for (int c = IC_per_G; c < paddedICPerG; ++c) {
                  int p_idx = packed_index_(t, r, s, k, g, c);
                  dst[p_idx] = 0;
                }
              }
            }
          }
        }
      }
    }
  } else {
    // For pack & transposed, call transposeConvWeights()
    // G K/G (T R S C/G) => G (T R S C/G) K/G
    if (tr) {
      if (ispack) {
        transposeConvWeights(conv_param_, src, dst);
      } else {
        // TODO: Wrap this as a inverseTransposeConvWeights()?
        // For unpack & transposed, call transposeConvWeights()
        // G (T R S C/G) K/G => G K/G (T R S C/G)
        for (int t = 0; t < F; ++t) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              for (int k = 0; k < OC_per_G; ++k) {
                for (int g = 0; g < G; ++g) {
                  for (int c = 0; c < IC_per_G; ++c) {
                    dst[((((g * OC_per_G + k) * F + t) * R + r) * S + s) *
                            IC_per_G +
                        c] =
                        src[((((g * F + t) * R + r) * S + s) * IC_per_G + c) *
                                OC_per_G +
                            k];
                  }
                }
              }
            }
          }
        }
      } // end if(ispack)
    } else {
      // just copy the data for not supported cases
      int kernel_prod = std::accumulate(
          conv_param_.K.begin(),
          conv_param_.K.end(),
          1,
          std::multiplies<int>());
      memcpy(dst, src, G * kernel_prod * OC_per_G * IC_per_G * sizeof(inpType));
    } // end if(tr)
  } // end if(fbgemmOptimizedGConv(conv_param_)
}

/**
 * @brief Pack weight tensor in a suitable format required for the optimized
 * kernel.
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack() {
  pack_unpack_(sdata_, pdata_, true);
}

/**
 * @brief Unpack the packed weight tensor (for the optimized kernel)
 * to the original form.
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::unpack(T* origin_buf) {
  pack_unpack_(const_cast<const T*>(pdata_), origin_buf, false);
}

template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int32_t, 1>;
template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int16_t, 1>;
template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int32_t, 2>;
template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int16_t, 2>;
template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int32_t, 3>;
template class FBGEMM_API PackWeightMatrixForGConv<int8_t, int16_t, 3>;
} // namespace fbgemm

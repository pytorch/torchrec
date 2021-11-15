/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "./OptimizedKernelsAvx2.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT, int SPATIAL_DIM>
PackAWithIm2Col<T, accT, SPATIAL_DIM>::PackAWithIm2Col(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const T* sdata,
    inpType* pmat,
    int32_t a_zero_pt,
    int32_t* row_offset,
    bool b_symmetric,
    const BlockingFactors* params)
    : PackMatrix<PackAWithIm2Col<T, accT, SPATIAL_DIM>, T, accT>(
          conv_p.MB *
              std::accumulate(
                  conv_p.OUT_DIM.begin(),
                  conv_p.OUT_DIM.end(),
                  1,
                  std::multiplies<int>()),
          std::accumulate(
              conv_p.K.begin(),
              conv_p.K.end(),
              1,
              std::multiplies<int>()) *
              conv_p.IC,
          pmat,
          conv_p.G,
          params),
      conv_p_(conv_p),
      sdata_(sdata),
      a_zero_pt_(a_zero_pt) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if ((!fbgemmHasAvx512VnniSupport() && !fbgemmHasAvx512Support() &&
       !fbgemmHasAvx2Support())) {
    assert(0 && "unknown architecure");
  }

  if (params) {
    BaseType::brow_ = params->MCB;
    BaseType::bcol_ = params->KCB;
    row_interleave_B_ = params->ROW_INTERLEAVE;
  } else {
    const inst_set_t isa = fbgemmInstructionSet();
    switch (isa) {
      case inst_set_t::avx512_vnni:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_B_) =
            PackingTraits<T, accT, inst_set_t::avx512_vnni>::
                getMatrixPackAParams();
        break;

      case inst_set_t::avx512_vnni_ymm:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_B_) =
            PackingTraits<T, accT, inst_set_t::avx512_vnni_ymm>::
                getMatrixPackAParams();
        break;

      case inst_set_t::avx512:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_B_) =
            PackingTraits<T, accT, inst_set_t::avx512>::getMatrixPackAParams();
        break;

      case inst_set_t::avx512_ymm:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_B_) =
            PackingTraits<T, accT, inst_set_t::avx512_ymm>::
                getMatrixPackAParams();
        break;

      case inst_set_t::avx2:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_B_) =
            PackingTraits<T, accT, inst_set_t::avx2>::getMatrixPackAParams();
        break;

      default:
        assert(0 && "unknown architecure");
        throw std::runtime_error("unknown architecure");
    }
  }

  if (BaseType::numCols() % conv_p.G != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(conv_p.G) +
        " does not divide numCols = " + std::to_string(BaseType::numCols()));
  }
  if (pmat) {
    BaseType::buf_ = pmat;
  } else {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = static_cast<T*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
    // aligned_alloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
  }
  if (!b_symmetric) {
    if (row_offset) {
      rowOffsetAllocatedHere = false;
      row_offset_ = row_offset;
    } else {
      rowOffsetAllocatedHere = true;
      row_offset_ = static_cast<int32_t*>(
          fbgemmAlignedAlloc(64, BaseType::brow_ * sizeof(int32_t)));
    }
  }
}

template <int SPATIAL_DIM, int BCOL>
void pack_a_with_im2col_opt(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const block_type_t& block,
    const uint8_t* sdata,
    uint8_t* out,
    int32_t a_zero_pt,
    int32_t* row_offset_buf,
    int COL_SIZE,
    int COL_P_SIZE,
    bool row_offset_acc) {
  constexpr int IC = 3;
  int IN_DIM_H = conv_p.IN_DIM[0];
  int IN_DIM_W = conv_p.IN_DIM[1];
  int K_H = conv_p.K[0];
  int K_W = conv_p.K[1];
  constexpr int STRIDE_H = 2;
  constexpr int STRIDE_W = 2;
  int PAD_H = conv_p.pad[0];
  int PAD_W = conv_p.pad[1];
  int OUT_DIM_H = conv_p.OUT_DIM[0];
  int OUT_DIM_W = conv_p.OUT_DIM[1];
  int OUT_DIM_HW = OUT_DIM_H * OUT_DIM_W;

  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int n = i / OUT_DIM_HW;
    int hw = i % OUT_DIM_HW;
    int w = hw % OUT_DIM_W;
    int h = hw / OUT_DIM_W;

    // j refers to column index within block
    int j = 0;
    // r and s iterate over K_H and K_W, respectively
    for (int r = 0; r < K_H; ++r) {
      int h_in = -PAD_H + h * STRIDE_H + r;
      if (h_in < 0 || h_in >= IN_DIM_H) {
        // Short-circuit if h_in is in padding.
        std::memset(
            out + (i - block.row_start) * BCOL + j,
            a_zero_pt,
            sizeof(uint8_t) * K_W * IC);
        j += K_W * IC;
        continue;
      }

      int s = 0;
      // left_pad_len : the number of spatial pixels we need to pad at the
      // beginning
      int left_pad_len = PAD_W - w * STRIDE_W;
      if (left_pad_len > 0) {
        std::memset(
            out + (i - block.row_start) * BCOL + j,
            a_zero_pt,
            sizeof(uint8_t) * left_pad_len * IC);
        s += left_pad_len;
      }

      // mid_len : the number of spatial pixels that we handle normally
      // (no padding)
      int mid_len = std::min(IN_DIM_W + PAD_W - w * STRIDE_W, K_W) - s;
      std::memcpy(
          out + (i - block.row_start) * BCOL + j + s * IC,
          sdata +
              ((n * IN_DIM_H + h_in) * IN_DIM_W + -PAD_W + w * STRIDE_W + s) *
                  IC,
          sizeof(uint8_t) * mid_len * IC);
      s += mid_len;

      // right_pad_len : the number of spatial pixels we need to pad at the end
      int right_pad_len = K_W - s;
      if (right_pad_len > 0) {
        std::memset(
            out + (i - block.row_start) * BCOL + j + s * IC,
            a_zero_pt,
            sizeof(uint8_t) * right_pad_len * IC);
      }
      j += K_W * IC;
    } // r loop

    // zero fill
    // Please see the comment in PackAMatrix.cc for zero vs zero_pt fill.
    if (COL_P_SIZE - COL_SIZE > 0) {
      std::memset(
          &out[(i - block.row_start) * BCOL + COL_SIZE],
          0,
          sizeof(uint8_t) * COL_P_SIZE - COL_SIZE);
    }

    if (row_offset_buf) {
      int32_t row_sum =
          row_offset_acc ? row_offset_buf[i - block.row_start] : 0;
      row_sum += reduceAvx2(out + (i - block.row_start) * BCOL, COL_SIZE);
      row_offset_buf[i - block.row_start] = row_sum;
    }
  }
}

template <typename T, typename accT, int SPATIAL_DIM>
void PackAWithIm2Col<T, accT, SPATIAL_DIM>::pack(const block_type_t& block) {
  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};
  BaseType::packedBlock(block_p);
  T* out = BaseType::getBuf();
  // accumulate into row offset?
  bool row_offset_acc =
      (block.col_start % (this->numCols() / this->numGroups())) != 0;
  int32_t* row_offset_buf = getRowOffsetBuffer();

  bool point_wise = true;
  for (int d = 0; d < SPATIAL_DIM; ++d) {
    if (conv_p_.K[d] != 1 || conv_p_.pad[d] != 0 || conv_p_.stride[d] != 1 ||
        conv_p_.dilation[d] != 1) {
      point_wise = false;
      break;
    }
  }
  for (int d = SPATIAL_DIM; d < SPATIAL_DIM * 2; ++d) {
    if (conv_p_.pad[d] != 0) {
      point_wise = false;
      break;
    }
  }

  // reduceAvx2 only written for T == uint8_t
  static_assert(
      std::is_same<T, uint8_t>::value,
      "PackAWithIm2Col<T, accT>::pack only works for T == uint8_t");
  if (point_wise) {
    int32_t ld = this->numCols();
    if (row_offset_buf) {
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        int buf_idx = i - block.row_start;
        memcpy(
            out + buf_idx * BaseType::blockColSize(),
            sdata_ + i * ld + block.col_start,
            block.col_size * sizeof(T));
        // zero fill
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
        int32_t row_sum =
            row_offset_acc ? row_offset_buf[i - block.row_start] : 0;
        row_sum +=
            reduceAvx2(sdata_ + i * ld + block.col_start, block.col_size);
        row_offset_buf[i - block.row_start] = row_sum;
      }
    } else {
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        int buf_idx = i - block.row_start;
        memcpy(
            out + buf_idx * BaseType::blockColSize(),
            sdata_ + i * ld + block.col_start,
            block.col_size * sizeof(T));
        // zero fill
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
      }
    }

    return;
  }

  int ic_per_group = conv_p_.IC / conv_p_.G;

  if (!conv_p_.transposed && SPATIAL_DIM == 2 && conv_p_.IC == 3 &&
      conv_p_.G == 1 && conv_p_.stride[0] == 2 && conv_p_.stride[1] == 2 &&
      block.col_start == 0 && conv_p_.pad[0] == ((conv_p_.K[0] - 1) / 2) &&
      conv_p_.pad[1] == ((conv_p_.K[1] - 1) / 2) &&
      block_p.col_size <= BaseType::blockColSize() &&
      conv_p_.dilation[0] == 1 && conv_p_.dilation[1] == 1 &&
      std::is_same<T, uint8_t>::value) {
    if (BaseType::blockColSize() == 256) {
      pack_a_with_im2col_opt<SPATIAL_DIM, 256>(
          conv_p_,
          block,
          reinterpret_cast<const uint8_t*>(sdata_),
          reinterpret_cast<uint8_t*>(out),
          a_zero_pt_,
          row_offset_buf,
          block.col_size,
          block_p.col_size,
          row_offset_acc);
      return;
    } else if (BaseType::blockColSize() == 512) {
      pack_a_with_im2col_opt<SPATIAL_DIM, 512>(
          conv_p_,
          block,
          reinterpret_cast<const uint8_t*>(sdata_),
          reinterpret_cast<uint8_t*>(out),
          a_zero_pt_,
          row_offset_buf,
          block.col_size,
          block_p.col_size,
          row_offset_acc);
      return;
    }
  }
  if (conv_p_.transposed) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      if (SPATIAL_DIM == 1) { // static if
        int n = i / (conv_p_.OUT_DIM[0]);
        int ow = i % (conv_p_.OUT_DIM[0]);
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int grs = j / ic_per_group;
          int s = grs % conv_p_.K[0];
          int g = grs / conv_p_.K[0];

          int w = ow + conv_p_.pad[0] - s * conv_p_.dilation[0];
          int w_in = w / conv_p_.stride[0];
          if (w_in * conv_p_.stride[0] == w && w_in >=0 && w_in < conv_p_.IN_DIM[0]) {
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ + (n * conv_p_.IN_DIM[0] + w_in) * conv_p_.IC +
                    g * ic_per_group + (j_blk_start % ic_per_group),
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    (j_blk_start - block.col_start),
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }

      } else if (SPATIAL_DIM == 2) { // static if
        int n = i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
        int hw = i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
        int ow = hw % conv_p_.OUT_DIM[1];
        int oh = hw / conv_p_.OUT_DIM[1];
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int grs = j / ic_per_group;
          int s = grs % conv_p_.K[1];
          int r = grs / conv_p_.K[1] % conv_p_.K[0];
          int g = grs / conv_p_.K[1] / conv_p_.K[0];

          int h = oh + conv_p_.pad[0] - r * conv_p_.dilation[0];
          int w = ow + conv_p_.pad[1] - s * conv_p_.dilation[1];

          int h_in = h / conv_p_.stride[0];
          int w_in = w / conv_p_.stride[1];

          if (h_in * conv_p_.stride[0] == h && h_in >=0 && h_in < conv_p_.IN_DIM[0] &&
              w_in * conv_p_.stride[1] == w && w_in >=0 && w_in < conv_p_.IN_DIM[1]) {
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ +
                    ((n * conv_p_.IN_DIM[0] + h_in) * conv_p_.IN_DIM[1] +
                     w_in) *
                        conv_p_.IC +
                    g * ic_per_group + (j_blk_start % ic_per_group),
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    (j_blk_start - block.col_start),
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }
      } else if (SPATIAL_DIM == 3) { // static if
        int n =
            i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
        int thw =
            i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
        int ow = thw % conv_p_.OUT_DIM[2];
        int oh = thw / conv_p_.OUT_DIM[2] % conv_p_.OUT_DIM[1];
        int ot = thw / conv_p_.OUT_DIM[2] / conv_p_.OUT_DIM[1];
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int gqrs = j / ic_per_group;
          int s = gqrs % conv_p_.K[2];
          int r = gqrs / conv_p_.K[2] % conv_p_.K[1];
          int q = gqrs / conv_p_.K[2] / conv_p_.K[1] % conv_p_.K[0];
          int g = gqrs / conv_p_.K[2] / conv_p_.K[1] / conv_p_.K[0];

          int t = ot + conv_p_.pad[0] - q * conv_p_.dilation[0];
          int h = oh + conv_p_.pad[1] - r * conv_p_.dilation[1];
          int w = ow + conv_p_.pad[2] - s * conv_p_.dilation[2];
          int t_in = t / conv_p_.stride[0];
          int h_in = h / conv_p_.stride[1];
          int w_in = w / conv_p_.stride[2];

          if (t_in * conv_p_.stride[0] == t && t_in >= 0 &&
              t_in < conv_p_.IN_DIM[0] && h_in * conv_p_.stride[1] == h &&
              h_in >= 0 && h_in < conv_p_.IN_DIM[1] &&
              w_in * conv_p_.stride[2] == w && w_in >= 0 &&
              w_in < conv_p_.IN_DIM[2]) {
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ +
                    (((n * conv_p_.IN_DIM[0] + t_in) * conv_p_.IN_DIM[1] +
                      h_in) *
                         conv_p_.IN_DIM[2] +
                     w_in) *
                        conv_p_.IC +
                    g * ic_per_group + (j_blk_start % ic_per_group),
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                &out
                    [(i - block.row_start) * BaseType::blockColSize() +
                     (j_blk_start - block.col_start)],
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }
      }

      // zero fill
      // Please see the comment in PackAMatrix.cc for zero vs zero_pt fill.
      if ((block_p.col_start + block_p.col_size) -
              (block.col_start + block.col_size) >
          0) {
        std::memset(
            &out
                [(i - block.row_start) * BaseType::blockColSize() +
                 (block.col_size)],
            0,
            sizeof(T) *
                ((block_p.col_start + block_p.col_size) -
                 (block.col_start + block.col_size)));
      }

      if (row_offset_buf) {
        int32_t row_sum =
            row_offset_acc ? row_offset_buf[i - block.row_start] : 0;
        row_sum += reduceAvx2(
            out + (i - block.row_start) * this->blockColSize(), block.col_size);
        row_offset_buf[i - block.row_start] = row_sum;
      }
    } // for each i
  } else {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      if (SPATIAL_DIM == 1) { // static if
        int n = i / (conv_p_.OUT_DIM[0]);
        int w = i % (conv_p_.OUT_DIM[0]);
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int grs = j / ic_per_group;
          int s = grs % conv_p_.K[0];
          int g = grs / conv_p_.K[0];

          int w_in =
              -conv_p_.pad[0] + w * conv_p_.stride[0] + s * conv_p_.dilation[0];
          if (w_in < 0 || w_in >= conv_p_.IN_DIM[0]) {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    (j_blk_start - block.col_start),
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ + (n * conv_p_.IN_DIM[0] + w_in) * conv_p_.IC +
                    g * ic_per_group + (j_blk_start % ic_per_group),
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }

      } else if (SPATIAL_DIM == 2) { // static if
        int n = i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
        int hw = i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
        int w = hw % conv_p_.OUT_DIM[1];
        int h = hw / conv_p_.OUT_DIM[1];
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int grs = j / ic_per_group;
          int s = grs % conv_p_.K[1];
          int r = grs / conv_p_.K[1] % conv_p_.K[0];
          int g = grs / conv_p_.K[1] / conv_p_.K[0];

          int h_in =
              -conv_p_.pad[0] + h * conv_p_.stride[0] + r * conv_p_.dilation[0];
          int w_in =
              -conv_p_.pad[1] + w * conv_p_.stride[1] + s * conv_p_.dilation[1];

          if (h_in < 0 || h_in >= conv_p_.IN_DIM[0] || w_in < 0 ||
              w_in >= conv_p_.IN_DIM[1]) {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    (j_blk_start - block.col_start),
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            int chn_start_idx = j_blk_start % ic_per_group;
            int src_offset =
                ((n * conv_p_.IN_DIM[0] + h_in) * conv_p_.IN_DIM[1] + w_in) *
                conv_p_.IC + g * ic_per_group + chn_start_idx;
            // fast path
            // Copy across pixels of input width if we can. We can only do this
            // if the following conditions are met. 1) If the number of groups
            // is 1. For number of groups > 1, im2col
            //    doesn't copy data across groups.
            // 2) If dilation is 1. For dilation > 1, copying from input
            //    across channels is not sequential.
            // 3) For copy from the last channel (end of filter or
            //    end of image width) for the current filter,
            //    only copy if we have enough in the current channel.
            //
            if (conv_p_.G == 1 && conv_p_.dilation[1] == 1 &&
                ((s < (conv_p_.K[1] - 1) && w_in < (conv_p_.IN_DIM[1] - 1)) ||
                 ((chn_start_idx + block.col_size) <= ic_per_group))) {
              // left edge adjustment with s
              j_blk_end = std::min(
                  (j_blk_id + conv_p_.K[1] - s) * ic_per_group,
                  block.col_start + block.col_size);
              // right edge adjustment with w_in
              j_blk_end = std::min(
                  (j_blk_id + conv_p_.IN_DIM[1] - w_in) * ic_per_group,
                  j_blk_end);
              j += j_blk_end - j_blk_start - ic_per_group;
            }
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ + src_offset,
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }
      } else if (SPATIAL_DIM == 3) { // static if
        int n =
            i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
        int thw =
            i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
        int w = thw % conv_p_.OUT_DIM[2];
        int h = thw / conv_p_.OUT_DIM[2] % conv_p_.OUT_DIM[1];
        int t = thw / conv_p_.OUT_DIM[2] / conv_p_.OUT_DIM[1];
        for (int j = block.col_start;
             j < block.col_start + block.col_size + ic_per_group - 1;
             j += ic_per_group) {
          int j_blk_id = j / ic_per_group;
          // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
          int j_blk_start = std::max(j_blk_id * ic_per_group, block.col_start);
          int j_blk_end = std::min(
              (j_blk_id + 1) * ic_per_group, block.col_start + block.col_size);
          if (j_blk_start >= j_blk_end) {
            break;
          }

          int gqrs = j / ic_per_group;
          int s = gqrs % conv_p_.K[2];
          int r = gqrs / conv_p_.K[2] % conv_p_.K[1];
          int q = gqrs / conv_p_.K[2] / conv_p_.K[1] % conv_p_.K[0];
          int g = gqrs / conv_p_.K[2] / conv_p_.K[1] / conv_p_.K[0];

          int t_in =
              -conv_p_.pad[0] + t * conv_p_.stride[0] + q * conv_p_.dilation[0];
          int h_in =
              -conv_p_.pad[1] + h * conv_p_.stride[1] + r * conv_p_.dilation[1];
          int w_in =
              -conv_p_.pad[2] + w * conv_p_.stride[2] + s * conv_p_.dilation[2];

          if (t_in < 0 || t_in >= conv_p_.IN_DIM[0] || h_in < 0 ||
              h_in >= conv_p_.IN_DIM[1] || w_in < 0 ||
              w_in >= conv_p_.IN_DIM[2]) {
            // Please note that padding for convolution should be filled with
            // zero_pt
            std::memset(
                &out
                    [(i - block.row_start) * BaseType::blockColSize() +
                     (j_blk_start - block.col_start)],
                a_zero_pt_,
                sizeof(T) * (j_blk_end - j_blk_start));
          } else {
            std::memcpy(
                out + (i - block.row_start) * BaseType::blockColSize() +
                    j_blk_start - block.col_start,
                sdata_ +
                    (((n * conv_p_.IN_DIM[0] + t_in) * conv_p_.IN_DIM[1] +
                      h_in) *
                         conv_p_.IN_DIM[2] +
                     w_in) *
                        conv_p_.IC +
                    g * ic_per_group + (j_blk_start % ic_per_group),
                sizeof(T) * (j_blk_end - j_blk_start));
          }
        }
      }

      // zero fill
      // Please see the comment in PackAMatrix.cc for zero vs zero_pt fill.
      if ((block_p.col_start + block_p.col_size) -
              (block.col_start + block.col_size) >
          0) {
        std::memset(
            &out
                [(i - block.row_start) * BaseType::blockColSize() +
                 (block.col_size)],
            0,
            sizeof(T) *
                ((block_p.col_start + block_p.col_size) -
                 (block.col_start + block.col_size)));
      }

      if (row_offset_buf) {
        int32_t row_sum =
            row_offset_acc ? row_offset_buf[i - block.row_start] : 0;
        row_sum += reduceAvx2(
            out + (i - block.row_start) * this->blockColSize(), block.col_size);
        row_offset_buf[i - block.row_start] = row_sum;
      }
    } // for each i
  }
}

template <typename T, typename accT, int SPATIAL_DIM>
void PackAWithIm2Col<T, accT, SPATIAL_DIM>::printPackedMatrix(
    std::string name) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;

  T* out = BaseType::getBuf();
  for (auto r = 0; r < BaseType::numPackedRows(); ++r) {
    for (auto c = 0; c < BaseType::numPackedCols(); ++c) {
      T val = out[r * BaseType::blockColSize() + c];
      if (std::is_integral<T>::value) {
        // cast to int64 because cout doesn't print int8_t type directly
        std::cout << std::setw(5) << static_cast<int64_t>(val) << " ";
      } else {
        std::cout << std::setw(5) << val << " ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T, typename accT, int SPATIAL_DIM>
int PackAWithIm2Col<T, accT, SPATIAL_DIM>::rowOffsetBufferSize(
    const BlockingFactors* params) {
  if (cpuinfo_initialize()) {
    if (params) {
      return params->MCB;
    } else {
      if (fbgemmHasAvx512VnniSupport()) {
        return PackingTraits<T, accT, inst_set_t::avx512_vnni>::MCB;
      } else if (fbgemmHasAvx512Support()) {
        return PackingTraits<T, accT, inst_set_t::avx512>::MCB;
      } else if (fbgemmHasAvx2Support()) {
        return PackingTraits<T, accT, inst_set_t::avx2>::MCB;
      } else {
        // TODO: Have default slower path
        assert(0 && "unsupported architecture");
        return -1;
      }
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template class PackAWithIm2Col<uint8_t, int32_t, 1>;
template class PackAWithIm2Col<uint8_t, int16_t, 1>;
template class PackAWithIm2Col<uint8_t, int32_t, 2>;
template class PackAWithIm2Col<uint8_t, int16_t, 2>;
template class PackAWithIm2Col<uint8_t, int32_t, 3>;
template class PackAWithIm2Col<uint8_t, int16_t, 3>;

} // namespace fbgemm

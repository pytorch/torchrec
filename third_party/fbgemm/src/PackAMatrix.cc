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
#include <iostream>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT>
PackAMatrix<T, accT>::PackAMatrix(
    matrix_op_t trans,
    int32_t nRow,
    int32_t nCol,
    const T* smat,
    int32_t ld,
    inpType* pmat,
    int groups,
    const BlockingFactors* params)
    : PackMatrix<PackAMatrix<T, accT>, T, accT>(
          nRow,
          nCol,
          pmat,
          groups,
          params),
      trans_(trans),
      smat_(smat),
      ld_(ld) {
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

  if (BaseType::numCols() % groups != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(groups) +
        " does not divide numCols = " + std::to_string(BaseType::numCols()));
  }
  if (pmat) {
    BaseType::buf_ = pmat;
  } else {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = static_cast<T*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
  }
}

template <typename T, typename accT>
void PackAMatrix<T, accT>::pack(const block_type_t& block) {
  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};

  BaseType::packedBlock(block_p);
  bool tr = (trans_ == matrix_op_t::Transpose);
  T* out = BaseType::getBuf();
  if (tr) {
    // TODO: should print warning because this path is not optimized yet
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      int buf_idx = i - block.row_start;
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        T val = smat_[i + j * ld_];
        out[buf_idx * BaseType::blockColSize() + (j - block.col_start)] = val;
      }
      // zero fill
      // Please note that we zero fill, not zero_pt fill, because for
      // requantization original, i.e., not padded, dimensions are used. If we
      // were to use padded dimensions for requantization, we would zero_pt
      // fill.
      // For example, consider the following dot product:
      // A = .3(5-15), .3(20-15) //.3 is scale and 15 is zero_pt
      // B = .4(1+10), .4(4+10) // .4 is scale and -10 is zero_pt
      //
      // numElements(A) = 2 and numElements(B) = 2
      //
      // Dot product is (real): -3*4.4+1.5*5.6 = -4.8
      // Dot product is (quantized): 5*1+20*4 = 85
      //
      // requantization: .3*.4(85 - (5+20)*(-10) - (1+4)*(15) +
      //                       numElements(A)*(15)(-10)) = -4.8
      //
      // In the above adding one more element zero in the quantized domain,
      // i.e., the quantized vectors become:
      // A_q = 5, 20, 0
      // B_q = 1, 4, 0
      //
      // and requantization with numElements(A) = 2 will produce the same
      // answer (-4.8).
      //
      // Also in the above adding one more element zero_pt in the quantized
      // domain, i.e., the quantized vectors become:
      // A_q = 5, 20, 15
      // B_q = 1, 4, -10
      //
      // and requantization with numElements(A) = 3 will produce the same
      // answer (-4.8).
      for (int j = block.col_size; j < block_p.col_size; ++j) {
        out[buf_idx * BaseType::blockColSize() + j] = 0;
      }
    }
  } else {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      int buf_idx = i - block.row_start;
      memcpy(
          out + buf_idx * BaseType::blockColSize(),
          smat_ + i * ld_ + block.col_start,
          block.col_size * sizeof(T));
      // zero fill
      for (int j = block.col_size; j < block_p.col_size; ++j) {
        out[buf_idx * BaseType::blockColSize() + j] = 0;
      }
    }
  }
}

template <typename T, typename accT>
int32_t PackAMatrix<T, accT>::addr(int32_t r, int32_t c) const {
  int32_t block_row_id = r / BaseType::blockRowSize();
  int32_t brow_offset = (block_row_id * BaseType::blockCols()) *
      (BaseType::blockRowSize() * BaseType::blockColSize());

  int32_t block_col_id = c / BaseType::blockColSize();
  int32_t bcol_offset =
      block_col_id * BaseType::blockRowSize() * BaseType::blockColSize();
  int32_t block_offset = brow_offset + bcol_offset;
  int32_t inblock_offset =
      (r % BaseType::blockRowSize()) * BaseType::blockColSize() +
      (c % BaseType::blockColSize());

  int32_t index = block_offset + inblock_offset;

  return index;
}

template <typename T, typename accT>
void PackAMatrix<T, accT>::printPackedMatrix(std::string name) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;

  T* out = BaseType::getBuf();
  for (auto r = 0; r < BaseType::numPackedRows(); ++r) {
    for (auto c = 0; c < BaseType::numPackedCols(); ++c) {
      T val = out[addr(r, c)];
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

template class PackAMatrix<uint8_t, int32_t>;
template class PackAMatrix<uint8_t, int16_t>;
} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "./OptimizedKernelsAvx2.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtilsAvx2.h"

namespace fbgemm {

template <typename T, typename accT>
PackAWithQuantRowOffset<T, accT>::PackAWithQuantRowOffset(
    matrix_op_t trans,
    int32_t nRow,
    int32_t nCol,
    const float* smat,
    int32_t ld,
    inpType* pmat,
    float scale,
    int32_t zero_pt,
    int groups,
    int32_t* row_offset,
    const BlockingFactors* params)
    : PackMatrix<PackAWithQuantRowOffset<T, accT>, T, accT>(
          nRow,
          nCol,
          pmat,
          groups,
          params),
      trans_(trans),
      smat_(smat),
      ld_(ld),
      scale_(scale),
      zero_pt_(zero_pt),
      row_offset_(row_offset) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (scale_ == 0.0f) {
    throw std::runtime_error("scale cannot be zero");
  }
  if (std::isinf(1.0f / scale_)) {
    throw std::runtime_error("scale's reciprocal cannot be infinity");
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

  rowOffsetAllocatedHere = false;

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
  if (!row_offset_) {
    rowOffsetAllocatedHere = true;
    row_offset_ = static_cast<int32_t*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * sizeof(accT)));
  }
}

template <typename T, typename accT>
void PackAWithQuantRowOffset<T, accT>::pack(const block_type_t& block) {
  // assert(block.row_start % BaseType::blockRowSize() == 0);
  assert(block.row_size <= BaseType::blockRowSize());
  assert(block.col_size <= BaseType::blockColSize());

  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};
  assert(block_p.col_size <= BaseType::blockColSize());
  BaseType::packedBlock(block_p);

  T* out = BaseType::getBuf();
  bool tr = (trans_ == matrix_op_t::Transpose);
  // accumulate into row offset?
  bool row_offset_acc =
      (block.col_start % (this->numCols() / this->numGroups())) != 0;
  int32_t* row_offset_buf = getRowOffsetBuffer();

  float* smat_transposed = nullptr;
  if (tr) {
    smat_transposed = static_cast<float*>(
        fbgemmAlignedAlloc(64, block.row_size * block.col_size * sizeof(float)));
    transpose_simd(
        block.col_size,
        block.row_size,
        smat_ + block.col_start * ld_ + block.row_start,
        ld_,
        smat_transposed,
        block.col_size);
  }
  const float* smat_temp =
      tr ? smat_transposed : smat_ + block.row_start * ld_ + block.col_start;
  int32_t ld_temp = tr ? block.col_size : ld_;

  static_assert(
      std::is_same<T, uint8_t>::value,
      "PackAWithQuantRowOffset<T, accT>::pack only works for T == uint8_t");

  // Only scale and zero points are used in QuantizeAvx2
  TensorQuantizationParams qparams;
  qparams.scale = scale_;
  qparams.zero_point = zero_pt_;

  for (int i = 0; i < block.row_size; ++i) {
    QuantizeAvx2(
        smat_temp + i * ld_temp,
        out + i * BaseType::blockColSize(),
        block.col_size,
        qparams);
    int32_t row_sum = row_offset_acc ? row_offset_buf[i] : 0;
    row_sum += reduceAvx2(out + i * BaseType::blockColSize(), block.col_size);
    row_offset_buf[i] = row_sum;

    // zero fill
    // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
    for (int j = block.col_start + block.col_size; j < block_p.col_size; ++j) {
      out[i * BaseType::blockColSize() + j] = 0;
    }
  }
  if (smat_transposed) {
    fbgemmAlignedFree(smat_transposed);
  }
}

template <typename T, typename accT>
int32_t PackAWithQuantRowOffset<T, accT>::addr(int32_t r, int32_t c) const {
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
void PackAWithQuantRowOffset<T, accT>::printPackedMatrix(std::string name) {
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

template <typename T, typename accT>
int PackAWithQuantRowOffset<T, accT>::rowOffsetBufferSize(
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
        assert(0 && "unsupported architecture");
        return -1;
      }
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template class PackAWithQuantRowOffset<uint8_t, int32_t>;

} // namespace fbgemm

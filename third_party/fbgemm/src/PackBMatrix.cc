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

/*
 * We pass in weights for Fully-connected and Convolution layers as B matrix.
 * Since weights are constant during inference, B matrix is constant
 * during inference so it's packed once and used multiple times. The code in
 * this file takes care of fully packing B matrix. Fully packing means dividing
 * the whole B matrix into blocks and storing all the blocks in the packed
 * buffer instead of just 1 or some blocks.
 *
 * Packing refers to the rearranging of B elements to make it suitable to the
 * way we access B in the inner compute kernel.
 *
 * Packing of B is dependent on three parameters: KCB, NCB and ROW_INTERLEAVE.
 *
 * Note 1: B is assumed to be in row-major format with K
 * rows and N columns, i.e., the following B matrix with 3 rows and 5 columns
 *
 *    B Matrix:
 *    b00 b01 b02 b03 b04
 *    b10 b11 b12 b13 b14
 *    b20 b21 b22 b23 b24
 *
 * is layed out in the memory as follows:
 *
 *    B layout in memory (row major):
 *    b00 b01 b02 b03 b04 b10 b11 b12 b13 b14 b20 b21 b22 b23 b24
 *
 * Note 2: KCB is always restricted/expected to be a multiple of ROW_INTERLEAVE
 * and thus it's minimum value is equal to ROW_INTERLEAVE.
 *
 * Note 3: ROW_INTERLEAVE is 2 for when we accumulate into 16-bits and 4 for
 * when we accumulate into 32-bits.
 *
 * Note 4: Minimum value of NCB is such that the number of bits in
 * NCB*ROW_INTERLEAVE elements at the very minimum is equal to the vector length
 * (i.e., 256 for avx2 and 512 for avx512).
 *
 * Minimum NCB value for int8 data type:
 *            avx2     avx512
 *    acc16   16       32
 *    acc32   8        16
 *
 * Packing examples:
 * Let us assume KCB=4, NCB=6 and ROW_INTERLEAVE=4 for the following examples.
 * To keep things manageable in the examples, NCB is 6 which is less than the
 * minimum value allowed for NCB as per the table above.
 *
 * * * * * * * * * * * * * * * * * * * *
 *
 * Example 1:
 *    Original B is an 8x4 matrix as follows:
 *    b00 b01 b02 b03
 *    b10 b11 b12 b13
 *    b20 b21 b22 b23
 *    b30 b31 b32 b33
 *    b40 b41 b42 b43
 *    b50 b51 b52 b53
 *    b60 b61 b62 b63
 *    b70 b71 b72 b73
 *
 * Packed matrix has 2 tiles along rows and 1 tile along columns. So
 * allocated/needed memory for B buffer is (2*4)*(1*6) elements.
 *
 * Packed B matrix looks like as follows:
 *
 * b00 b10 b20 b30 b01 b11 b21 b31 b02 b12 b22 b32 b03 b13 b23 b33 x x x x x \
 * x x x | b40 b50 b60 b70 b41 b51 b61 b71 b42 b52 b62 b72 b43 b53 b63 b73 x x \
 * x x x x x x
 *
 *    ROW_INTERLEAVE rows are mixed with columns and layed out sequentially.
 *
 *    ("x" indicates uninitialized locations)
 *    ("|" indicates start of the next block; A block here refers to KCB*NCB
 *     elements.)
 *    ("\" indicates that the elements continue on the next line)
 *    (block 1 of size KCB*NCB directly follows block 0 of the same size)
 *
 * * * * * * * * * * * * * * * * * * * *
 *
 * Example 2:
 *    Original B is a 3x4 matrix as follows:
 *    b00 b01 b02 b03
 *    b10 b11 b12 b13
 *    b20 b21 b22 b23
 *
 * Packed matrix has 1 tile along rows and 1 tile along columns. So
 * allocated/needed memory for B buffer is (1*4)*(1*6) elements.
 *
 * Packed B matrix looks like as follows:
 *
 * b00 b10 b20 0 b01 b11 b21 0 b02 b12 b22 0 b03 b13 b23 0 x x x x x x x x
 *
 * If a tile along rows has less than ROW_INTERLEAVE rows, interleaved elements
 * are zero initialized.
 *
 * * * * * * * * * * * * * * * * * * * *
 *
 * Example 3:
 *    Original B is a 5x4 matrix as follows:
 *    b00 b01 b02 b03
 *    b10 b11 b12 b13
 *    b20 b21 b22 b23
 *    b30 b31 b32 b33
 *    b40 b41 b42 b43
 *
 * Packed matrix has 2 tiles along rows and 1 tile along columns. So
 * allocated/needed memory for B buffer is (2*4)*(1*6) elements.
 *
 * Packed B matrix looks like as follows:
 *
 * b00 b10 b20 b30 b01 b11 b21 b31 b02 b12 b22 b32 b03 b13 b23 b33 x x x x x \
 * x x x b40 0 0 0 b41 0 0 0 b42 0 0 0 b43 0 0 0 x x x x x x x x
 *
 * * * * * * * * * * * * * * * * * * * *
 *
 * Example 4:
 *    Original B is a 4x7 matrix as follows:
 *    b00 b01 b02 b03 b04 b05 b06
 *    b10 b11 b12 b13 b14 b15 b16
 *    b20 b21 b22 b23 b24 b25 b26
 *    b30 b31 b32 b33 b34 b35 b36
 *
 * Packed matrix has 1 tile along rows and 2 tiles along columns. So
 * allocated/needed memory for B buffer is (1*4)*(2*6) elements.
 *
 * Packed B matrix looks like as follows:
 *
 * b00 b10 b20 b30 b01 b11 b21 b31 b02 b12 b22 b32 b03 b13 b23 b33 b04 b14
 * b24 b34 b05 b15 b25 b35 | b06 b16 b26 b36 x x x x x x x x x x x x x x x x x \
 * x x x
 *
 * * * * * * * * * * * * * * * * * * * *
 *
 * Example 5:
 *    Original B is a 5x7 matrix as follows:
 *    b00 b01 b02 b03 b04 b05 b06
 *    b10 b11 b12 b13 b14 b15 b16
 *    b20 b21 b22 b23 b24 b25 b26
 *    b30 b31 b32 b33 b34 b35 b36
 *    b40 b41 b42 b43 b44 b45 b46
 *
 * Packed matrix has 2 tiles along rows and 2 tiles along columns. So
 * allocated/needed memory for B buffer is (2*4)*(2*6) elements.
 *
 * Packed B matrix looks like as follows:
 *
 * b00 b10 b20 b30 b01 b11 b21 b31 b02 b12 b22 b32 b03 b13 b23 b33 b04 b14 \
 * b24 b34 b05 b15 b25 b35 | b06 b16 b26 b36 x x x x x x x x x x x x x x x x x \
 * x x x | b40 0 0 0 b41 0 0 0 b42 0 0 0 b43 0 0 0 b44 0 0 0 b45 0 0 0 | b46 0 \
 * 0 0 x x x x x x x x x x x x
 *
 * The kernel expects the B matrix to be packed in the way mentioned above for
 * correct operation.
 */

namespace fbgemm {

template <typename T, typename accT>
PackBMatrix<T, accT>::PackBMatrix(
    matrix_op_t trans,
    int32_t nRow,
    int32_t nCol,
    const T* smat,
    int32_t ld,
    inpType* pmat,
    int groups,
    const BlockingFactors* params)
    : PackMatrix<PackBMatrix<T, accT>, T, accT>(
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
  if (params) {
    BaseType::brow_ = params->KCB;
    BaseType::bcol_ = params->NCB;
    row_interleave_ = params->ROW_INTERLEAVE;
  } else {
    const inst_set_t isa = fbgemmInstructionSet();
    switch (isa) {
      case inst_set_t::avx512_vnni:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_) =
            PackingTraits<T, accT, inst_set_t::avx512_vnni>::
              getMatrixPackBParams();
        break;

      case inst_set_t::avx512_vnni_ymm:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_) =
            PackingTraits<T, accT, inst_set_t::avx512_vnni_ymm>::
              getMatrixPackBParams();
        break;

      case inst_set_t::avx512:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_) =
            PackingTraits<T, accT, inst_set_t::avx512>::getMatrixPackBParams();
        break;

      case inst_set_t::avx512_ymm:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_) =
            PackingTraits<T, accT, inst_set_t::avx512_ymm>::
              getMatrixPackBParams();
        break;

      case inst_set_t::avx2:
        std::tie(BaseType::brow_, BaseType::bcol_, row_interleave_) =
            PackingTraits<T, accT, inst_set_t::avx2>::getMatrixPackBParams();
        break;

      default:
        assert(0 && "unknown architecure");
        throw std::runtime_error("unknown architecure");
    }
  }

  if (BaseType::numRows() % groups != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(groups) +
        " does not divide numRows = " + std::to_string(BaseType::numRows()));
  }

  // blocking for one group
  block_type_t block{
      0, BaseType::numRows() / BaseType::numGroups(), 0, BaseType::numCols()};
  BaseType::packedBlock(block);
  if (!pmat) {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = static_cast<T*>(fbgemmAlignedAlloc(
        64,
        BaseType::numGroups() * BaseType::blockRows() * BaseType::brow_ *
            BaseType::blockCols() * BaseType::bcol_ * sizeof(T)));
  }
  pack(block, params);
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::pack_unpack_(
    const block_type_t& block,
    T* unpack_buf,
    T* pack_buf,
    bool ispack,
    const BlockingFactors* params) {
  assert((BaseType::blockRowSize() % row_interleave_) == 0);
  assert((block.row_start % BaseType::blockRowSize()) == 0);
  assert((block.col_start % BaseType::blockColSize()) == 0);

  // When T is char *, type-based alias analysis (TBAA) cannot prove
  // that `unpack_buf` and `pack_buf` do not alias `block` (because
  // char * is the one exception to the C++ strict aliasing rule), so the
  // compiler would have to re-load these attributes from `block` on
  // every loop iteration for correctness. We know better, so let's
  // help the compiler out by doing the loads ourselves into
  // constants.
  const auto blockRowStart = block.row_start;
  const auto blockRowSize = block.row_size;
  const auto blockColStart = block.col_start;
  const auto blockColSize = block.col_size;

  BaseType::packedBlock(block);
  bool tr = (trans_ == matrix_op_t::Transpose);
  for (int g = 0; g < BaseType::numGroups(); ++g) {
    T* pack_buf_cur = pack_buf +
        g * BaseType::packedBufferSize(blockRowSize, blockColSize, params);
    for (int i = blockRowStart; i < blockRowStart + blockRowSize; ++i) {
      int r_offset = ((i / BaseType::blockRowSize()) * BaseType::blockCols()) *
              (BaseType::blockRowSize() * BaseType::blockColSize()) +
          (i % BaseType::blockRowSize() / row_interleave_) *
              BaseType::blockColSize() * row_interleave_ +
          i % row_interleave_;

      int c_start_offset = (blockColStart / BaseType::blockColSize()) *
              BaseType::blockRowSize() * BaseType::blockColSize() +
          (blockColStart % BaseType::blockColSize()) * row_interleave_;

      int c_idx_offset = 0;
      int c_blk_offset = 0;
      for (int j = blockColStart; j < blockColStart + blockColSize; ++j) {
        // int c_offset = (j / BaseType::blockColSize()) *
        //         BaseType::blockRowSize() * BaseType::blockColSize() +
        //     (j % BaseType::blockColSize()) * row_interleave_;
        // 1. Loop invariant hoisting (move block offset calculation out of
        // inner loop); 2. Strength reduction (change modulus in inner loop to
        // an increment + rollover).
        int c_offset = c_start_offset +
            c_blk_offset * BaseType::blockRowSize() * BaseType::blockColSize() +
            c_idx_offset * row_interleave_;

        if (ispack) {
          pack_buf_cur[r_offset + c_offset] = tr
              ? unpack_buf[i + (g * blockColSize + j) * ld_]
              : unpack_buf[(g * blockRowSize + i) * ld_ + j];
        } else {
          T* unpack_buf_cur = tr
              ? &(unpack_buf[i + (g * blockColSize + j) * ld_])
              : &(unpack_buf[(g * blockRowSize + i) * ld_ + j]);
          *unpack_buf_cur = pack_buf_cur[r_offset + c_offset];
        }

        c_idx_offset++;
        if (c_idx_offset == BaseType::blockColSize()) {
          c_idx_offset = 0;
          c_blk_offset++;
        }
      }
    }
    if (ispack) {
      // fill the remaining with zero.
      // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
      for (int i = blockRowStart + blockRowSize;
           i < (blockRowStart + blockRowSize + row_interleave_ - 1) /
               row_interleave_ * row_interleave_;
           ++i) {
        int r_offset =
            ((i / BaseType::blockRowSize()) * BaseType::blockCols()) *
                (BaseType::blockRowSize() * BaseType::blockColSize()) +
            (i % BaseType::blockRowSize() / row_interleave_) *
                BaseType::blockColSize() * row_interleave_ +
            i % row_interleave_;
        for (int j = blockColStart; j < blockColStart + blockColSize;
             j++) {
          int c_offset = (j / BaseType::blockColSize()) *
                  BaseType::blockRowSize() * BaseType::blockColSize() +
              (j % BaseType::blockColSize()) * row_interleave_;

          int out_idx = r_offset + c_offset;
          pack_buf_cur[out_idx] = 0;
        }
      }
    }
  } // for each group
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::pack(
    const block_type_t& block,
    const BlockingFactors* params) {
  pack_unpack_(block, const_cast<T*>(smat_), BaseType::getBuf(), true, params);
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::unpack(
    T* origin_buf,
    const BlockingFactors* params) {
  block_type_t blockB{
      BaseType::packedRowStart(),
      BaseType::numPackedRows(),
      BaseType::packedColStart(),
      BaseType::numPackedCols()};
  pack_unpack_(blockB, origin_buf, BaseType::getBuf(), false, params);
}

template <typename T, typename accT>
int32_t PackBMatrix<T, accT>::addr(int32_t r, int32_t c) const {
  int32_t block_row_id = r / BaseType::blockRowSize();
  int32_t brow_offset = (block_row_id * BaseType::blockCols()) *
      (BaseType::blockRowSize() * BaseType::blockColSize());

  int32_t block_col_id = c / BaseType::blockColSize();
  int32_t bcol_offset =
      block_col_id * BaseType::blockRowSize() * BaseType::blockColSize();
  int32_t block_offset = brow_offset + bcol_offset;
  int32_t inblock_offset = (r % BaseType::blockRowSize() / row_interleave_) *
          BaseType::blockColSize() * row_interleave_ +
      (c % BaseType::blockColSize()) * row_interleave_ + r % row_interleave_;

  int32_t index = block_offset + inblock_offset;

  return index;
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::printPackedMatrix(
    std::string name,
    const BlockingFactors* params) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;
  std::cout << "block size:"
            << "[" << BaseType::blockRowSize() << ", "
            << BaseType::blockColSize() << "]" << std::endl;

  for (int g = 0; g < BaseType::numGroups(); ++g) {
    T* out = BaseType::getBuf() +
        g *
            BaseType::packedBufferSize(
                BaseType::numPackedRows(), BaseType::numPackedCols(), params);
    std::cout << "group: " << g << std::endl;
    for (auto nr = 0; nr < BaseType::blockRows(); ++nr) {
      auto rows = (nr == BaseType::blockRows() - 1) ? BaseType::lastBrow()
                                                    : BaseType::blockRowSize();
      for (auto nc = 0; nc < BaseType::blockCols(); ++nc) {
        std::cout << "block:" << nr << ", " << nc << std::endl;
        auto cols = (nc == BaseType::blockCols() - 1)
            ? BaseType::lastBcol()
            : BaseType::blockColSize();
        for (auto r = 0; r < (rows + row_interleave_ - 1) / row_interleave_;
             ++r) {
          for (auto c = 0; c < cols * row_interleave_; ++c) {
            T val =
                out[nr * BaseType::blockCols() * BaseType::blockRowSize() *
                        BaseType::blockColSize() +
                    nc * BaseType::blockRowSize() * BaseType::blockColSize() +
                    r * BaseType::blockColSize() * row_interleave_ + c];
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
    }
  }
}

template <typename T, typename accT>
bool PackBMatrix<T, accT>::metaEquals(const PackBMatrix<T, accT>& that) const {
  if (BaseType::numRows() != that.numRows() ||
      BaseType::numCols() != that.numCols() ||
      BaseType::blockRowSize() != that.blockRowSize() ||
      BaseType::blockColSize() != that.blockColSize() ||
      BaseType::blockRows() != that.blockRows() ||
      BaseType::blockCols() != that.blockCols() ||
      BaseType::numPackedRows() != that.numPackedRows() ||
      BaseType::numPackedCols() != that.numPackedCols() ||
      trans_ != that.trans_ || BaseType::numGroups() != that.numGroups() ||
      row_interleave_ != that.row_interleave_) {
    return false;
  }

  return true;
}

template <typename T, typename accT>
bool PackBMatrix<T, accT>::equals(const PackBMatrix<T, accT>& that) const {
  if (!metaEquals(that)) {
    return false;
  }

  for (int i = 0; i < this->numRows(); ++i) {
    for (int j = 0; j < this->numCols(); ++j) {
      if (this->buf_[addr(i, j)] != that.buf_[that.addr(i, j)]) {
        return false;
      }
    }
  }

  return true;
}

template class PackBMatrix<int8_t, int32_t>;
template class PackBMatrix<int8_t, int16_t>;
} // namespace fbgemm

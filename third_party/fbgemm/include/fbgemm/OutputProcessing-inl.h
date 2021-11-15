/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

template <typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int memCopy<outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<outT, inT>::value,
      "input and output data type must be of same type");
  // only copy if destination is not the same as source
  if (out + block.row_start * ld_out + block.col_start != inp) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      memcpy(
          out + block.col_start + i * ld_out,
          inp + (i - block.row_start) * ld_in,
          block.col_size * sizeof(inT));
    }
  }
  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}

template <typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int DoSpmdmOnInpBuffer<outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  assert(B_csc_.NumOfCols() % groups_ == 0);
  int n_per_group = B_csc_.NumOfCols() / groups_;
  int g = block.col_start / n_per_group;
  B_csc_.SpMDM(block, A_ + g * B_csc_.NumOfRows(), lda_, true, inp, ld_in);
  return nextop_.template f<instSet>(out, inp, block, ld_out, ld_in);
}

template <typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int DoSConvOnInpBuffer<outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  B_csc_.SparseConv(conv_p_, block, A_, A_zero_point_, true, inp, ld_in);
  return nextop_.template f<instSet>(out, inp, block, ld_out, ld_in);
}

template <
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    typename BIAS_TYPE,
    typename outT,
    typename inT,
    typename nextOPType>
template <inst_set_t instSet>
inline int
ReQuantizeOutput<FUSE_RELU, Q_GRAN, BIAS_TYPE, outT, inT, nextOPType>::f(
    outT* out,
    const inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<inT, int32_t>::value,
      "input data type must be of int32_t type");
  int ncol_per_group = ncols_ / groups_;
  assert(
      block.col_size <= ncol_per_group &&
      "ReQuantizeOutput should be called at most 1 group at a time.");
  int g = block.col_start / ncol_per_group;
  if (instSet == inst_set_t::anyarch || !std::is_same<outT, uint8_t>::value) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        inT raw = inp[(i - block.row_start) * ld_in + (j - block.col_start)];
        if (Aq_zero_point_) {
          raw -= Aq_zero_point_ * q_col_offsets_[j];
        }
        int Bq_zero_point_idx;
        if (Q_GRAN == QuantizationGranularity::TENSOR) {
          Bq_zero_point_idx = 0;
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          Bq_zero_point_idx = g;
        } else if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          Bq_zero_point_idx = j;
        } else {
          assert(false && "unknown quantization granularity");
        }
        if (q_row_offsets_) {
          raw -= q_row_offsets_[i - block.row_start] *
              Bq_zero_point_[Bq_zero_point_idx];
        }
        float raw_f;
        if (bias_) {
          if (std::is_same<BIAS_TYPE, float>::value) {
            raw_f = raw;
            raw_f += bias_[j] / act_times_w_scale_[Bq_zero_point_idx];
          } else {
            raw += bias_[j];
            raw_f = raw;
          }
        } else {
          raw_f = raw;
        }

        float ab = raw_f * C_multiplier_[Bq_zero_point_idx];
        long rounded = std::lrintf(ab) + C_zero_point_;

        out[i * ld_out + j] = std::max(
            FUSE_RELU ? static_cast<long>(C_zero_point_) : 0l,
            std::min(255l, rounded));
      }
    }
  } else if (instSet == inst_set_t::avx2 || instSet == inst_set_t::avx512) {
    bool b_symmetric =
        (Q_GRAN == QuantizationGranularity::TENSOR && Bq_zero_point_[0] == 0) ||
        q_row_offsets_ == nullptr;

    requantizationParams_t<BIAS_TYPE> r = {Aq_zero_point_,
                                           Bq_zero_point_,
                                           C_zero_point_,
                                           C_multiplier_,
                                           q_row_offsets_,
                                           q_col_offsets_,
                                           bias_,
                                           ncols_,
                                           groups_,
                                           act_times_w_scale_};

    if (Aq_zero_point_ == 0) {
      if (b_symmetric) {
        if (bias_ == nullptr) {
          requantizeOutputProcessingAvx2<true, true, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeOutputProcessingAvx2<true, true, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      } else {
        if (bias_ == nullptr) {
          requantizeOutputProcessingAvx2<true, false, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeOutputProcessingAvx2<true, false, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      }
    } else {
      if (b_symmetric) {
        if (bias_ == nullptr) {
          requantizeOutputProcessingAvx2<false, true, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeOutputProcessingAvx2<false, true, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      } else {
        if (bias_ == nullptr) {
          requantizeOutputProcessingAvx2<
              false,
              false,
              Q_GRAN,
              false,
              FUSE_RELU>(out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeOutputProcessingAvx2<false, false, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      }
    }
  } else {
    assert(0 && "Not supported yet");
  }
  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}

template <
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    typename outT,
    typename inT,
    typename nextOPType>
template <inst_set_t instSet>
inline int ReQuantizeForFloat<FUSE_RELU, Q_GRAN, outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<int32_t, inT>::value,
      "input data type is of not expected type");
  static_assert(
      std::is_same<float, outT>::value,
      "output data type is of not expected type");
  int ncol_per_group = ncols_ / groups_;
  assert(
      block.col_size <= ncol_per_group &&
      "ReQuantizeOutput should be called at most 1 group at a time.");
  int g = block.col_start / ncol_per_group;
  if (instSet == inst_set_t::anyarch || !std::is_same<outT, float>::value) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        inT raw = inp[(i - block.row_start) * ld_in + j - block.col_start];
        if (Aq_zero_point_) {
          raw -= Aq_zero_point_ * q_col_offsets_[j];
        }
        int Bq_zero_point_idx;
        if (Q_GRAN == QuantizationGranularity::TENSOR) {
          Bq_zero_point_idx = 0;
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          Bq_zero_point_idx = g;
        } else if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          Bq_zero_point_idx = j;
        } else {
          assert(false && "unknown quantization granularity");
        }
        if (q_row_offsets_) {
          raw -= q_row_offsets_[i - block.row_start] *
              Bq_zero_point_[Bq_zero_point_idx];
        }
        float res = raw * Aq_scale_ * Bq_scale_[Bq_zero_point_idx];
        if (bias_) {
          res += bias_[j];
        }
        out[i * ld_out + j] = res;
        if (FUSE_RELU) {
          out[i * ld_out + j] = std::max<outT>(0.0f, out[i * ld_out + j]);
        }
      }
    }
  } else if (instSet == inst_set_t::avx2 || instSet == inst_set_t::avx512) {
    bool b_symmetric =
        (Q_GRAN == QuantizationGranularity::TENSOR && Bq_zero_point_[0] == 0) ||
        q_row_offsets_ == nullptr;

    requantizationForFloatParams_t r = {Aq_zero_point_,
                                        Bq_zero_point_,
                                        Aq_scale_,
                                        Bq_scale_,
                                        q_row_offsets_,
                                        q_col_offsets_,
                                        bias_,
                                        ncols_,
                                        groups_};

    if (Aq_zero_point_ == 0) {
      if (b_symmetric) {
        if (bias_ == nullptr) {
          requantizeForFloatAvx2<true, true, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeForFloatAvx2<true, true, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      } else {
        if (bias_ == nullptr) {
          requantizeForFloatAvx2<true, false, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeForFloatAvx2<true, false, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      }
    } else {
      if (b_symmetric) {
        if (bias_ == nullptr) {
          requantizeForFloatAvx2<false, true, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeForFloatAvx2<false, true, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      } else {
        if (bias_ == nullptr) {
          requantizeForFloatAvx2<false, false, Q_GRAN, false, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        } else {
          requantizeForFloatAvx2<false, false, Q_GRAN, true, FUSE_RELU>(
              out, inp, block, ld_out, ld_in, r);
        }
      }
    }
  } else {
    assert(0 && "Not supported yet");
  }

  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}

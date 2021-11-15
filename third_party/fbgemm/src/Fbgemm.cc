/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/Fbgemm.h"
#include <cpuinfo.h>
#include <functional>
#include <stdexcept>
#include "./ExecuteKernel.h"

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double packing_time = 0.0;
double computing_time = 0.0;
double run_time = 0.0;
#endif

namespace fbgemm {

template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename cT,
    typename processOutputType>
void fbgemmPacked(
    PackMatrix<
        packingAMatrix,
        typename packingAMatrix::inpType,
        typename packingAMatrix::accType>& packA,
    PackMatrix<
        packingBMatrix,
        typename packingBMatrix::inpType,
        typename packingBMatrix::accType>& packB,
    cT* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params) {
  static_assert(
      std::is_same<
          typename packingAMatrix::accType,
          typename packingBMatrix::accType>::value,
      "Accumulation type of both matrices should be the same");

  // Run time CPU detection
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if ((!fbgemmHasAvx512VnniSupport() && !fbgemmHasAvx512Support() &&
       !fbgemmHasAvx2Support())) {
    assert(0 && "unknown architecure");
    throw std::runtime_error("unknown architecure");
  }

  int MCB;
  int KCB;
  int MR;

  if (blocking_params) {
    MCB = blocking_params->MCB;
    KCB = blocking_params->KCB;
    MR = blocking_params->MR;
  } else {
    const inst_set_t isa = fbgemmInstructionSet();
    switch (isa) {
      case inst_set_t::avx512_vnni:
        std::tie(MCB, KCB, MR) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_vnni>::getCacheBlockParams();
        break;

      case inst_set_t::avx512_vnni_ymm:
        std::tie(MCB, KCB, MR) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_vnni_ymm>::getCacheBlockParams();
        break;

      case inst_set_t::avx512:
        std::tie(MCB, KCB, MR) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512>::getCacheBlockParams();
        break;

      case inst_set_t::avx512_ymm:
        std::tie(MCB, KCB, MR) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_ymm>::getCacheBlockParams();
        break;

      case inst_set_t::avx2:
        std::tie(MCB, KCB, MR) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx2>::getCacheBlockParams();
        break;

      default:
        assert(0 && "unknown architecure");
        throw std::runtime_error("unknown architecure");
    }
  }

  if (!packB.isPrePacked()) {
    throw std::runtime_error("B matrix must be prepacked");
  }
  int G = packA.numGroups();
  if (G != packB.numGroups()) {
    throw std::runtime_error(
        "A.groups = " + std::to_string(G) + " and B.groups = " +
        std::to_string(packB.numGroups()) + " are not the same");
  }

  int MDim = packA.numRows();
  int KDimPerGroup = packB.numRows() / G;
  int NDim = packB.numCols();

  int kBlocks = (KDimPerGroup + KCB - 1) / KCB;

  // remainders
  int _kc = KDimPerGroup % KCB;

  int kc, mc;

  block_type_t blockA{0, 0, 0, 0};

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_very_start,
      t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

  thread_type_t th_info =
      fbgemmGetThreadPartition(G, MDim, NDim, thread_id, num_threads);
  // if (thread_id == 0)
  //   std::cout << ", " << th_info.toString();

  int g_begin, g_end, i_begin, i_end;

  // Calculate the begin and end index along the group dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, G, g_begin, g_end);
  // Calculate the begin and end index along the m dimension
  fbgemmPartition1DBlocked(
      th_info.m_thread_id, th_info.m_num_threads, MDim, MR, i_begin, i_end);

  for (int g = g_begin; g < g_end; ++g) {
    ExecuteKernel<packingAMatrix, packingBMatrix, cT, processOutputType>
        exeKernelObj(
            packA,
            packB,
            C,
            C_buffer,
            ldc,
            outProcess,
            th_info,
            blocking_params);
    for (int i = i_begin; i < i_end; i += MCB) { // i is the element index
      mc = std::min(i_end - i, MCB);
      for (int kb = 0; kb < kBlocks; ++kb) { // kb is the block index
        kc = (kb != kBlocks - 1 || _kc == 0) ? KCB : _kc;
        // pack A matrix
        blockA = {i, mc, g * KDimPerGroup + kb * KCB, kc};
        packA.pack(blockA);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 t_end - t_start)
                 .count();
        packing_time += (dt);
        t_start = std::chrono::high_resolution_clock::now();
#endif

        exeKernelObj.execute(g * kBlocks + kb);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 t_end - t_start)
                 .count();
        computing_time += (dt);
        t_start = std::chrono::high_resolution_clock::now();
#endif
      }
    }
  } // for each group

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  run_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif
}

template <int SPATIAL_DIM>
bool fbgemmOptimizedGConv(const conv_param_t<SPATIAL_DIM>& conv_p) {
  if (SPATIAL_DIM == 1)
    return false;

  int C_per_G = conv_p.IC / conv_p.G;
  int K_per_G = conv_p.OC / conv_p.G;

  int G_together = PackWeightMatrixForGConv<int8_t, int32_t, SPATIAL_DIM>::
      numOfGroupsTogether(conv_p);

  auto areEqual = [](int a, int b) { return a == b; };

  return (C_per_G == K_per_G) &&
      (C_per_G == 2 || C_per_G == 4 || C_per_G == 8 || C_per_G == 16) &&
      (conv_p.G >= G_together) &&

      std::all_of(
             conv_p.K.begin(),
             conv_p.K.end(),
             std::bind(areEqual, std::placeholders::_1, 3)) &&

      std::all_of(
             conv_p.pad.begin(),
             conv_p.pad.end(),
             std::bind(areEqual, std::placeholders::_1, 1)) &&

      std::all_of(
             conv_p.dilation.begin(),
             conv_p.dilation.end(),
             std::bind(areEqual, std::placeholders::_1, 1)) &&

      // Height/Width strides should be the same and
      // should be either 1 or 2
      // Temporal stride can be anything.
      (std::all_of(
           conv_p.stride.begin() + SPATIAL_DIM - 2,
           conv_p.stride.end(),
           std::bind(areEqual, std::placeholders::_1, 1)) ||
       std::all_of(
           conv_p.stride.begin() + SPATIAL_DIM - 2,
           conv_p.stride.end(),
           std::bind(areEqual, std::placeholders::_1, 2))) &&
      !conv_p.transposed;
}

template FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<1>& conv_p);
template FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<2>& conv_p);
template FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<3>& conv_p);

bool fbgemmSupportedCPU() {
  return (cpuinfo_initialize() && fbgemmHasAvx2Support());
}

////////////////////////////////////////////////////////////////////////////////
// Added for Windows DLL for implicit template parameter instantiation
template class FBGEMM_API memCopy<std::int32_t, std::int32_t>;
template class FBGEMM_API DoNothing<std::int32_t, std::int32_t>;
template class FBGEMM_API DoNothing<float, float>;
template class FBGEMM_API DoNothing<std::uint8_t, std::uint8_t>;
template class FBGEMM_API
    ReQuantizeForFloat<false, QuantizationGranularity::TENSOR>;
template class FBGEMM_API
    ReQuantizeForFloat<false, QuantizationGranularity::GROUP>;
template class FBGEMM_API
    ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL>;
template class FBGEMM_API
    ReQuantizeForFloat<true, QuantizationGranularity::TENSOR>;
template class FBGEMM_API
    ReQuantizeForFloat<true, QuantizationGranularity::GROUP>;
template class FBGEMM_API
    ReQuantizeForFloat<true, QuantizationGranularity::OUT_CHANNEL>;

#define INSTANTIATE_BASE(FNAME, RELU, Q_GRAN) \
  template class FBGEMM_API                   \
      FNAME<std::uint8_t, std::int32_t, ReQuantizeOutput<RELU, Q_GRAN>>;

#define INSTANTIATE_Q_GRAN(FNAME, RELU)                           \
  INSTANTIATE_BASE(FNAME, RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BASE(FNAME, RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BASE(FNAME, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_RELU(FNAME)     \
  INSTANTIATE_Q_GRAN(FNAME, false)  \
  INSTANTIATE_Q_GRAN(FNAME, true)

INSTANTIATE_RELU(DoSpmdmOnInpBuffer)
INSTANTIATE_RELU(DoSConvOnInpBuffer)

#undef INSTANTIATE_RELU
#undef INSTANTIATE_Q_GRAN
#undef INSTANTIATE_BASE

template class FBGEMM_API DoSpmdmOnInpBuffer<
    float,
    std::int32_t,
    ReQuantizeForFloat<false, QuantizationGranularity::TENSOR>>;

#define INSTANTIATE_BASE(RELU, Q_GRAN, BIAS_TYPE) \
  template class FBGEMM_API ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>;

#define INSTANTIATE_BIAS_T(RELU, Q_GRAN)       \
  INSTANTIATE_BASE(RELU, Q_GRAN, std::int32_t) \
  INSTANTIATE_BASE(RELU, Q_GRAN, float)

#define INSTANTIATE_Q_GRAN(RELU)                             \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::OUT_CHANNEL)

INSTANTIATE_Q_GRAN(false)
INSTANTIATE_Q_GRAN(true)

#undef INSTANTIATE_Q_GRAN
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

// ReQuantizeOutput
#define INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, BIAS_TYPE)    \
  template FBGEMM_API void fbgemmPacked(                            \
      PackMatrix<PACK_A<uint8_t, ACC_T>, uint8_t, ACC_T>& packA,    \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      uint8_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, Q_GRAN) \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, float)  \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, int32_t)

#define INSTANTIATE_Q_GRANS(PACK_A, ACC_T, RELU)                            \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_RELU(PACK_A, ACC_T)      \
  INSTANTIATE_Q_GRANS(PACK_A, ACC_T, false)  \
  INSTANTIATE_Q_GRANS(PACK_A, ACC_T, true)

#define INSTANTIATE_ACC_T(PACK_A)    \
  INSTANTIATE_RELU(PACK_A, int32_t)  \
  INSTANTIATE_RELU(PACK_A, int16_t)

INSTANTIATE_ACC_T(PackAMatrix)
INSTANTIATE_ACC_T(PackAWithRowOffset)

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_RELU
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, BIAS_TYPE) \
  template FBGEMM_API void fbgemmPacked(                              \
      PackMatrix<                                                     \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,               \
          uint8_t,                                                    \
          ACC_T>& packA,                                              \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB,   \
      uint8_t* C,                                                     \
      int32_t* C_buffer,                                              \
      uint32_t ldc,                                                   \
      const ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,    \
      int thread_id,                                                  \
      int num_threads,                                                \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(ACC_T, RELU, SPATIAL_DIM, Q_GRAN) \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, float)  \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, int32_t)

#define INSTANTIATE_Q_GRANS(ACC_T, RELU, SPATIAL_DIM)             \
  INSTANTIATE_BIAS_T(                                             \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BIAS_T(                                             \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BIAS_T(                                             \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 1)        \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 2)        \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 3)

#define INSTANTIATE_RELU(ACC_T)          \
  INSTANTIATE_SPATIAL_DIM(ACC_T, false)  \
  INSTANTIATE_SPATIAL_DIM(ACC_T, true)

INSTANTIATE_RELU(int32_t)
INSTANTIATE_RELU(int16_t)

#undef INSTANTIATE_RELU
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeForFloat
#define INSTANTIATE_BASE(PACK_A, RELU, Q_GRAN)                          \
  template FBGEMM_API void fbgemmPacked(                                \
      PackMatrix<PACK_A<uint8_t, int32_t>, uint8_t, int32_t>& packA,    \
      PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB, \
      float* C,                                                         \
      int32_t* C_buffer,                                                \
      uint32_t ldc,                                                     \
      const ReQuantizeForFloat<RELU, Q_GRAN>& outProcess,               \
      int thread_id,                                                    \
      int num_threads,                                                  \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(PACK_A, RELU)                          \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_RELU(PACK_A)      \
  INSTANTIATE_Q_GRANS(PACK_A, false)  \
  INSTANTIATE_Q_GRANS(PACK_A, true)

INSTANTIATE_RELU(PackAWithRowOffset)
INSTANTIATE_RELU(PackAWithQuantRowOffset);

#undef INSTANTIATE_RELU
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN)          \
  template FBGEMM_API void fbgemmPacked(                            \
      PackMatrix<                                                   \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,             \
          uint8_t,                                                  \
          ACC_T>& packA,                                            \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      float* C,                                                     \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const ReQuantizeForFloat<RELU, Q_GRAN>& outProcess,           \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(ACC_T, RELU, SPATIAL_DIM)                          \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BASE(                                                            \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 1)        \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 2)        \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 3)

#define INSTANTIATE_RELU(ACC_T)          \
  INSTANTIATE_SPATIAL_DIM(ACC_T, false)  \
  INSTANTIATE_SPATIAL_DIM(ACC_T, true)

INSTANTIATE_RELU(int32_t)
INSTANTIATE_RELU(int16_t)

#undef INSTANTIATE_RELU
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template FBGEMM_API void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<false>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

////////////////////////////////////////////////////////////////////////////////
// DoSpmdmOnInpBuffer
#define INSTANTIATE_BASE(PACK_A, RELU, Q_GRAN)                          \
  template FBGEMM_API void fbgemmPacked(                                \
      PackMatrix<PACK_A<uint8_t, int16_t>, uint8_t, int16_t>& packA,    \
      PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB, \
      uint8_t* C,                                                       \
      int32_t* C_buffer,                                                \
      uint32_t ldc,                                                     \
      const DoSpmdmOnInpBuffer<                                         \
          uint8_t,                                                      \
          int32_t,                                                      \
          ReQuantizeOutput<RELU, Q_GRAN>>& outProcess,                  \
      int thread_id,                                                    \
      int num_threads,                                                  \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(PACK_A, RELU)                          \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_RELU(PACK_A)      \
  INSTANTIATE_Q_GRANS(PACK_A, false)  \
  INSTANTIATE_Q_GRANS(PACK_A, true)

INSTANTIATE_RELU(PackAMatrix)
INSTANTIATE_RELU(PackAWithRowOffset)

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE
#undef INSTANTIATE_RELU

#define INSTANTIATE_BASE(RELU, Q_GRAN)                                        \
  template FBGEMM_API void fbgemmPacked(                                      \
      PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>& packA, \
      PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,       \
      uint8_t* C,                                                             \
      int32_t* C_buffer,                                                      \
      uint32_t ldc,                                                           \
      const DoSConvOnInpBuffer<                                               \
          uint8_t,                                                            \
          int32_t,                                                            \
          ReQuantizeOutput<RELU, Q_GRAN>>& outProcess,                        \
      int thread_id,                                                          \
      int num_threads,                                                        \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::TENSOR)  \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::GROUP)   \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::OUT_CHANNEL)

INSTANTIATE_Q_GRANS(false)
INSTANTIATE_Q_GRANS(true)

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template FBGEMM_API void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoSpmdmOnInpBuffer<float, int32_t, ReQuantizeForFloat<false>>&
        outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

////////////////////////////////////////////////////////////////////////////////
// memCopy
#define INSTANTIATE_BASE(PACK_A, ACC_T)                             \
  template FBGEMM_API void fbgemmPacked(                            \
      PackMatrix<PACK_A<uint8_t, ACC_T>, uint8_t, ACC_T>& packA,    \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      int32_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const memCopy<>& outProcess,                                  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_ACC_T(PACK_A)   \
  INSTANTIATE_BASE(PACK_A, int32_t) \
  INSTANTIATE_BASE(PACK_A, int16_t)

INSTANTIATE_ACC_T(PackAMatrix)
INSTANTIATE_ACC_T(PackAWithRowOffset)

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, SPATIAL_DIM)                        \
  template FBGEMM_API void fbgemmPacked(                            \
      PackMatrix<                                                   \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,             \
          uint8_t,                                                  \
          ACC_T>& packA,                                            \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      int32_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const memCopy<>& outProcess,                                  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_SPATIAL_DIM(ACC_T) \
  INSTANTIATE_BASE(ACC_T, 1)           \
  INSTANTIATE_BASE(ACC_T, 2)           \
  INSTANTIATE_BASE(ACC_T, 3)

INSTANTIATE_SPATIAL_DIM(int32_t)
INSTANTIATE_SPATIAL_DIM(int16_t)

#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BASE

template FBGEMM_API void fbgemmPacked(
    PackMatrix<PackAWithQuantRowOffset<uint8_t, int32_t>, uint8_t, int32_t>&
        packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

template FBGEMM_API void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

} // namespace fbgemm

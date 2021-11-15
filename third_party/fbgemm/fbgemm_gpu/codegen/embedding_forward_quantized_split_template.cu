/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"
#include "codegen/embedding_common.h"

namespace nbit {

using namespace at;
using namespace fbgemm_gpu;

// Keep in sync with EmbeddingLocation in split_table_batched_embeddings_ops.py
enum {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
};

__forceinline__ __host__ __device__ uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b) * b;
}


__forceinline__ __host__ __device__ uint32_t div_round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

__host__ __device__ inline int32_t unpadded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim * 4; }
    if (weight_ty == SparseType::FP16) { return dim * 2; }
    if (weight_ty == SparseType::INT8) { return dim + 4; }
    if (weight_ty == SparseType::INT4) { return dim / 2 + 4; }
    if (weight_ty == SparseType::INT2) { return dim / 4 + 4; }
    return 0;
}

__host__ __device__ inline int32_t padded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty);
  return round_up(r, 16);
}

// "Effective" number of elements in the row when we include the row-wise quantization parameters.
__device__ inline int32_t padded_D(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim; }
    if (weight_ty == SparseType::FP16) { return dim; }
    if (weight_ty == SparseType::INT8) { return dim + 4; }
    if (weight_ty == SparseType::INT4) { return dim + 8; }
    if (weight_ty == SparseType::INT2) { return dim + 16; }
    return 0;
}

struct __align__(32) float8 {
  __host__ __device__ float8() {}
  float4 vals[2];
};

struct __align__(8) half4 {
  __host__ __device__ half4() {}
  half2 vals[2];
};

struct __align__(16) half8 {
  __host__ __device__ half8() {}
  half2 vals[4];
};

__device__ __forceinline__ float4 make_zero_float4() {
  return make_float4(0, 0, 0, 0);
}

__device__ __forceinline__ float8 make_zero_float8() {
  float8 t;
  t.vals[0] = make_float4(0, 0, 0, 0);
  t.vals[1] = make_float4(0, 0, 0, 0);
  return t;
}

__device__ __forceinline__ float2 make_zero_float2() {
  return make_float2(0, 0);
}

__device__ __forceinline__ half8 to_half8(float8 v) {
  half8 t;
  t.vals[0] = __float22half2_rn(make_float2(v.vals[0].x, v.vals[0].y));
  t.vals[1] = __float22half2_rn(make_float2(v.vals[0].z, v.vals[0].w));
  t.vals[2] = __float22half2_rn(make_float2(v.vals[1].x, v.vals[1].y));
  t.vals[3] = __float22half2_rn(make_float2(v.vals[1].z, v.vals[1].w));
  return t;
}

__device__ __forceinline__ half4 to_half4(float4 v) {
  half4 t;
  t.vals[0] = __float22half2_rn(make_float2(v.x, v.y));
  t.vals[1] = __float22half2_rn(make_float2(v.z, v.w));
  return t;
}

__device__ __forceinline__ __half2 to_half2(float2 v) {
  return __float22half2_rn(v);
}

__device__ __forceinline__ __half to_half(float v) {
  return __float2half_rn(v);
}

__forceinline__ __device__ __half2 hfma2(const __half2 a, const __half2 b, const __half2 c) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  return __hfma2(a, b, c);
#else
  float2 fa, fb, fc;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fc = __half22float2(c);
  fc.x = fa.x * fb.x + fc.x;
  fc.y = fa.y * fb.y + fc.y;
  return __float22half2_rn(fc);
#endif
}

__forceinline__ __device__ half hmul(half a, half b) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  return __hmul(a, b);
#else
  return __float2half(__half2float(a) * __half2float(b));
#endif
}

// Reinterpret a  pair of uint16_t (packed into a uint32_t) as half2, and multiply by rhs.
__device__ __forceinline__ __half2 hmul_short2(uint32_t lhs, __half rhs) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  #ifndef __HALF2_TO_UI
  // cuda_fp16.hpp
  #define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
  #endif
  #ifndef __HALF2_TO_CUI
  // cuda_fp16.hpp
  #define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))
  #endif
  __half2 ret;
  __half2 rhsp = make_half2(rhs, rhs);
  asm("mul.f16x2 %0, %1, %2;" : "=r"(__HALF2_TO_UI(ret)) : "r"(__HALF2_TO_CUI(lhs)), "r"(__HALF2_TO_CUI(rhsp)));
  return ret;
#else
  #ifndef __HALF2_TO_UI
  // cuda_fp16.hpp
  #define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
  #endif
  __half2 lhs_h2;
  __HALF2_TO_UI(lhs_h2) = lhs;
  float2 fx = __half22float2(lhs_h2);
  float2 fy = __half22float2(make_half2(rhs, rhs));
  float2 fr;
  fr.x = fx.x * fy.x;
  fr.y = fx.y * fy.y;
  return __float22half2_rn(fr);
#endif
}

__forceinline__ __device__ half8  dequantize_permuted_int4(uint32_t packedVals, __half2 shift_scale) {
  half8 res;
  uint32_t v = packedVals;
  // What's going on here, you might ask? We extra out 4-bit pairs of integers as 2xuint16 packed into an int32
  // via the mask operation, and then we convert them to half precision values.
  // As these are all integers in [0, 15], we can actually just interpret the 4-bit integer values as half-precision values.
  // We multiply by 4096 x 4096 to go from the 4-bit representation to the equivalent fp16 value,
  // or alternatively 32768 * 512 (or 32 when we have shifted the 4-bit value up).
  // See e.g. https://gist.github.com/ajtulloch/021254a291a95966bc509db4e34ffeff for a NumPy implementation.
  // We do this dance because:
  // a) doing bitwise operations on each 4-bit value is expensive on the ALU, and 4-bit to half is expensive on the XU.
  // b) doing a 256-entry shared memory LUT on 8-bit pairs is expensive on SMEM throughput.
  // Credit to @jhj.
  res.vals[0] = hmul_short2(v & 0x000F000F, 32768);
  res.vals[1] = hmul_short2(v & 0x00F000F0, 32768);
  v >>= 8;
  res.vals[2] = hmul_short2(v & 0x000F000F, 32768);
  res.vals[3] = hmul_short2(v & 0x00F000F0, 32768);

  res.vals[0] =
     hfma2(res.vals[0], __half2(hmul(shift_scale.x, 512), hmul(shift_scale.x, 512)),
             __half2(shift_scale.y, shift_scale.y));
  res.vals[1] =
    hfma2(res.vals[1], __half2(hmul(shift_scale.x, 32), hmul(shift_scale.x, 32)),
            __half2(shift_scale.y, shift_scale.y));
  res.vals[2] =
   hfma2(res.vals[2], __half2(hmul(shift_scale.x, 512), hmul(shift_scale.x, 512)),
           __half2(shift_scale.y, shift_scale.y));
  res.vals[3] =
   hfma2(res.vals[3], __half2(hmul(shift_scale.x, 32), hmul(shift_scale.x, 32)),
            __half2(shift_scale.y, shift_scale.y));
  return res;
}

__forceinline__ __device__ half4  dequantize_permuted_int8(uint32_t packedVals, __half2 shift_scale) {
  half4 res;
  uint32_t v = packedVals;
  // See comment above, this is a minor variation.
  res.vals[0] = hmul_short2(v & 0x00FF00FF, 32768);
  v >>= 8;
  res.vals[1] = hmul_short2(v & 0x00FF00FF, 32768);
  res.vals[0] =
     hfma2(res.vals[0], __half2(hmul(shift_scale.x, 512), hmul(shift_scale.x, 512)),
             __half2(shift_scale.y, shift_scale.y));
  res.vals[1] =
    hfma2(res.vals[1], __half2(hmul(shift_scale.x, 512), hmul(shift_scale.x, 512)),
            __half2(shift_scale.y, shift_scale.y));
  return res;
}

__forceinline__ __device__ float2 accumulate_fp16(float2 acc, __half2 vals) {
  float2 v = __half22float2(vals);
  acc.x += v.x;
  acc.y += v.y;
  return acc;
}

__forceinline__ __device__ float2 accumulate_weighted_fp16(float2 acc, __half2 vals, float weight) {
  float2 v = __half22float2(vals);
  acc.x = fmaf(v.x, weight, acc.x);
  acc.y = fmaf(v.y, weight, acc.y);
  return acc;
}

__forceinline__ __device__ float accumulate_fp32(float acc, float vals) {
  acc += vals;
  return acc;
}

__forceinline__ __device__ float accumulate_weighted_fp32(float acc, float vals, float weight) {
  return fmaf(vals, weight, acc);
}

__forceinline__ __device__ float8 accumulate_packed_int4(float8 acc,
                                                         uint32_t packedVals,
                                                         __half2 shift_scale) {
  half8 res = dequantize_permuted_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);

  // Twiddle after permutations.
  acc.vals[0].x += v0.x;
  acc.vals[0].y += v1.x;
  acc.vals[0].z += v2.x;
  acc.vals[0].w += v3.x;
  acc.vals[1].x += v0.y;
  acc.vals[1].y += v1.y;
  acc.vals[1].z += v2.y;
  acc.vals[1].w += v3.y;
  return acc;
}

__forceinline__ __device__ float8 accumulate_weighted_packed_int4(float8 acc,
                                                        uint32_t packedVals,
                                                        __half2 shift_scale,
                                                        float weight) {
  half8 res = dequantize_permuted_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);

  // Twiddle after permutations.
  acc.vals[0].x = fmaf(v0.x, weight, acc.vals[0].x);
  acc.vals[0].y = fmaf(v1.x, weight, acc.vals[0].y);
  acc.vals[0].z = fmaf(v2.x, weight, acc.vals[0].z);
  acc.vals[0].w = fmaf(v3.x, weight, acc.vals[0].w);
  acc.vals[1].x = fmaf(v0.y, weight, acc.vals[1].x);
  acc.vals[1].y = fmaf(v1.y, weight, acc.vals[1].y);
  acc.vals[1].z = fmaf(v2.y, weight, acc.vals[1].z);
  acc.vals[1].w = fmaf(v3.y, weight, acc.vals[1].w);
  return acc;
}

__forceinline__ __device__ float4 accumulate_packed_int8(float4 acc,
                                                         uint32_t packedVals,
                                                         __half2 shift_scale) {
  half4 res = dequantize_permuted_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);

  // Twiddle after permutations.
  acc.x += v0.x;
  acc.y += v1.x;
  acc.z += v0.y;
  acc.w += v1.y;
  return acc;
}

__forceinline__ __device__ float4 accumulate_weighted_packed_int8(float4 acc,
                                                                  uint32_t packedVals,
                                                                  __half2 shift_scale,
                                                                  float weight) {
  half4 res = dequantize_permuted_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);

  // Twiddle after permutations.
  acc.x = fmaf(v0.x, weight, acc.x);
  acc.y = fmaf(v1.x, weight, acc.y);
  acc.z = fmaf(v0.y, weight, acc.z);
  acc.w = fmaf(v1.y, weight, acc.w);
  return acc;
}

// ---------------------- start cp.async helpers, copied from CUTLASS

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

// We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// the previous internal intrinsics if they are available.
#if (! defined (__clang__) && defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);
  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#elif (! defined (__clang__) && defined(__CUDA_ARCH__) &&  __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
  return __nvvm_get_smem_pointer(ptr);
#elif defined(__CUDA_ARCH__)
  uint32_t smem_ptr;
  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
#else
    return 0;
#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

__device__ __forceinline__ void cp_async_fence() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
  #endif
}

/// Partial specialization

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
__device__ __forceinline__ void cp_async_wait<0>() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
  #endif
}

/// Partial specialization
template <int SizeInBytes>
__device__ __forceinline__
void cp_async_zfill_cg(void *smem_ptr, void const *global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
    static_assert(SizeInBytes == 16,
    "cp.async only supports CacheOperation::Global when access size is 16B.");

    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);
    asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
    "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
#else
    static_assert(SizeInBytes == 16, "");
    using AccessType = uint4;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    } else {
      AccessType zeros;
      zeros.x = 0;
      zeros.y = 0;
      zeros.z = 0;
      zeros.w = 0;
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
#endif
}


/// Copy with zero fill
template <int SizeInBytes>
__device__ __forceinline__
void cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
    // Make sure the size is supported.
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
            "Size is not supported");

    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);

    asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
    "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
#else
    static_assert(SizeInBytes == 16, "");
    using AccessType = uint4;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    } else {
      AccessType zeros;
      zeros.x = 0;
      zeros.y = 0;
      zeros.z = 0;
      zeros.w = 0;
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
#endif
}

// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void fp32_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output // [B][total_D],
  ) {

  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::FP32) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
  const int32_t D_total = padded_D(D, weight_ty);
  const int32_t D_padding = D_total - D;

  uint32_t warp_idx = threadIdx.y;
  int32_t indices_starts[OutputRowsPerThread];
  int32_t Ls[OutputRowsPerThread];
  int32_t max_Ls = 0;

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    indices_starts[i] = indices_start;
    Ls[i] = indices_end - indices_start;
    max_Ls = max(max_Ls, Ls[i]);
  }

  const uint8_t* __restrict__ weights;
  const auto placement = weights_placements[t];
  if (placement == DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 1;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  float accumulators[OutputRowsPerThread][MaxNum128BRows];

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      accumulators[i][j] = 0.0;
    }
  }
  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);
    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row = valid ? reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx) * D_bytes]) : reinterpret_cast<const uint4*>(&weights[0]);
        cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);
        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}
      }
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}
        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          float v = reinterpret_cast<const float*>(row)[kWarpSize * j + threadIdx.x];
          {% if weighted %}
          accumulators[i][j] = valid ? accumulate_weighted_fp32(accumulators[i][j], v, row_weight) : accumulators[i][j];
          {% else %}
          accumulators[i][j] = valid ? accumulate_fp32(accumulators[i][j], v) : accumulators[i][j];
          {% endif %}

        }
      }
    }
  }
  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
      if (pooling_mode == MEAN && Ls[i] != 0) {
          float inv_L = static_cast<float>(1.0) / static_cast<float>(Ls[i]);
          accumulators[i][j] *= inv_L;
      }
      static_assert(
        std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value,
        "output_t can only be float or half now"
      );
      if (std::is_same<output_t, float>::value) {
        float val = accumulators[i][j];
        if (output_d >= 0 && output_d < D) {
          output[b][D_start + output_d] = val;
        }
      } else if (std::is_same<output_t, at::Half>::value) {
        __half val = to_half(accumulators[i][j]);
        if (output_d >= 0 && output_d < D) {
          *reinterpret_cast<__half*>(&output[b][D_start + output_d]) = val;
        }
      } else {
        // INT8/4: not implemented yet
      }
    }
  }
}

// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void fp16_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output // [B][total_D],
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::FP16) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
  const int32_t D_total = padded_D(D, weight_ty);
  const int32_t D_padding = D_total - D;

  uint32_t warp_idx = threadIdx.y;
  int32_t indices_starts[OutputRowsPerThread];
  int32_t Ls[OutputRowsPerThread];
  int32_t max_Ls = 0;

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    indices_starts[i] = indices_start;
    Ls[i] = indices_end - indices_start;
    max_Ls = max(max_Ls, Ls[i]);
  }

  const uint8_t* __restrict__ weights;
  const auto placement = weights_placements[t];
  if (placement == DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 2;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  float2 accumulators[OutputRowsPerThread][MaxNum128BRows];

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      accumulators[i][j] = make_zero_float2();
    }
  }

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row = valid ? reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx) * D_bytes]) : reinterpret_cast<const uint4*>(&weights[0]);
        cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);

        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}
      }
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          __half2 v = reinterpret_cast<const __half2*>(row)[kWarpSize * j + threadIdx.x];

          {% if weighted %}
          accumulators[i][j] = valid ? accumulate_weighted_fp16(accumulators[i][j], v, row_weight) : accumulators[i][j];
          {% else %}
          accumulators[i][j] = valid ? accumulate_fp16(accumulators[i][j], v) : accumulators[i][j];
          {% endif %}
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));

    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
      if (pooling_mode == MEAN && Ls[i] != 0) {
          float inv_L = static_cast<float>(1.0) / static_cast<float>(Ls[i]);
          accumulators[i][j].x *= inv_L;
          accumulators[i][j].y *= inv_L;
      }
      static_assert(
        std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value,
        "output_t can only be float or half now"
      );
      if (std::is_same<output_t, float>::value) {
        float2 val = accumulators[i][j];
        if (output_d >= 0 && output_d < D) {
          *reinterpret_cast<int2*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int2*>(&val);
        }
      } else if (std::is_same<output_t, at::Half>::value) {
        half2 val = to_half2(accumulators[i][j]);
        if (output_d >= 0 && output_d < D) {
          *reinterpret_cast<int1*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int1*>(&val);
        }
      } else {
        // INT8/4: not implemented yet
      }
    }
  }
}

template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void int_8bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output // [B][total_D],
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::INT8) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
  const int32_t D_total = padded_D(D, weight_ty);
  const int32_t D_padding = D_total - D;

  uint32_t warp_idx = threadIdx.y;
  int32_t indices_starts[OutputRowsPerThread];
  int32_t Ls[OutputRowsPerThread];
  int32_t max_Ls = 0;

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    indices_starts[i] = indices_start;
    Ls[i] = indices_end - indices_start;
    max_Ls = max(max_Ls, Ls[i]);
  }

  const uint8_t* __restrict__ weights;
  const auto placement = weights_placements[t];
  if (placement == DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 4;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  float4 accumulators[OutputRowsPerThread][MaxNum128BRows];

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      accumulators[i][j] = make_zero_float4();
    }
  }

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row = valid ? reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx) * D_bytes]) : reinterpret_cast<const uint4*>(&weights[0]);
        cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);

        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}

      }
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        half2 shift_scale = reinterpret_cast<const half2*>(row)[0];

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          uint32_t v = reinterpret_cast<const uint32_t*>(row)[kWarpSize * j + threadIdx.x];
          {% if weighted %}
          accumulators[i][j] = valid ? accumulate_weighted_packed_int8(accumulators[i][j], v, shift_scale, row_weight) : accumulators[i][j];
          {% else %}
          accumulators[i][j] = valid ? accumulate_packed_int8(accumulators[i][j], v, shift_scale) : accumulators[i][j];
          {% endif %}
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));

    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
      bool aligned_16b = intptr_t(&output[b][D_start + output_d]) % 16 == 0;
      bool aligned_8b = intptr_t(&output[b][D_start + output_d]) % 8 == 0;
      bool aligned_4b = intptr_t(&output[b][D_start + output_d]) % 4 == 0;

      if (pooling_mode == MEAN && Ls[i] != 0) {
          float inv_L = static_cast<float>(1.0) / static_cast<float>(Ls[i]);
          accumulators[i][j].x *= inv_L;
          accumulators[i][j].y *= inv_L;
          accumulators[i][j].z *= inv_L;
          accumulators[i][j].w *= inv_L;
      }
      static_assert(
        std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value,
        "output_t can only be float or half now"
      );
      if (std::is_same<output_t, float>::value) {
        float4 val = accumulators[i][j];
        if (output_d >= 0 && output_d < D) {
          if (aligned_16b) {
            *reinterpret_cast<int4*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int4*>(&val);
          } else if (aligned_8b) {
            auto v = *reinterpret_cast<const int4*>(&val);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 0]) = make_int2(v.x, v.y);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 2]) = make_int2(v.z, v.w);
          } else {
            output[b][D_start + output_d + 0] = val.x;
            output[b][D_start + output_d + 1] = val.y;
            output[b][D_start + output_d + 2] = val.z;
            output[b][D_start + output_d + 3] = val.w;
          }
        }
      } else if (std::is_same<output_t, at::Half>::value) {
        half4 val = to_half4(accumulators[i][j]);
        if (output_d >= 0 && output_d < D) {
          if (aligned_8b) {
            *reinterpret_cast<int2*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int2*>(&val);
          } else if (aligned_4b) {
            auto v = *reinterpret_cast<const int2*>(&val);
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 0]) = v.x;
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 2]) = v.y;
          } else {
            output[b][D_start + output_d + 0] = val.vals[0].x;
            output[b][D_start + output_d + 1] = val.vals[0].y;
            output[b][D_start + output_d + 2] = val.vals[1].x;
            output[b][D_start + output_d + 3] = val.vals[1].y;
          }
        }
      } else {
        // INT8/4: not implemented yet
      }
    }
  }
}

template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void int_4bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output // [B][total_D],
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::INT4) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
  const int32_t D_total = padded_D(D, weight_ty);
  const int32_t D_padding = D_total - D;

  uint32_t warp_idx = threadIdx.y;
  int32_t indices_starts[OutputRowsPerThread];
  int32_t Ls[OutputRowsPerThread];
  int32_t max_Ls = 0;

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    indices_starts[i] = indices_start;
    Ls[i] = indices_end - indices_start;
    max_Ls = max(max_Ls, Ls[i]);
  }

  const uint8_t* __restrict__ weights;
  const auto placement = weights_placements[t];
  if (placement == DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 8;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  float8 accumulators[OutputRowsPerThread][MaxNum128BRows];

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      accumulators[i][j] = make_zero_float8();
    }
  }

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row = valid ? reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx) * D_bytes]) : reinterpret_cast<const uint4*>(&weights[0]);
        cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);

        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}
      }
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        half2 shift_scale = reinterpret_cast<const half2*>(row)[0];

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          uint32_t v = reinterpret_cast<const uint32_t*>(row)[kWarpSize * j + threadIdx.x];
          {% if weighted %}
          accumulators[i][j] = valid ? accumulate_weighted_packed_int4(accumulators[i][j], v, shift_scale, row_weight) : accumulators[i][j];
          {% else %}
          accumulators[i][j] = valid ? accumulate_packed_int4(accumulators[i][j], v, shift_scale) : accumulators[i][j];
          {% endif %}
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));

    #pragma unroll MaxNum128BRows
    for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
      int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
      bool aligned_16b = intptr_t(&output[b][D_start + output_d]) % 16 == 0;
      bool aligned_8b = intptr_t(&output[b][D_start + output_d]) % 8 == 0;
      bool aligned_4b = intptr_t(&output[b][D_start + output_d]) % 4 == 0;

      if (pooling_mode == MEAN && Ls[i] != 0) {
          float inv_L = static_cast<float>(1.0) / static_cast<float>(Ls[i]);
          accumulators[i][j].vals[0].x *= inv_L;
          accumulators[i][j].vals[0].y *= inv_L;
          accumulators[i][j].vals[0].z *= inv_L;
          accumulators[i][j].vals[0].w *= inv_L;
          accumulators[i][j].vals[1].x *= inv_L;
          accumulators[i][j].vals[1].y *= inv_L;
          accumulators[i][j].vals[1].z *= inv_L;
          accumulators[i][j].vals[1].w *= inv_L;
      }
      static_assert(
        std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value,
        "output_t can only be float or half now"
      );
      if (std::is_same<output_t, float>::value) {
        float8 val = accumulators[i][j];
        if (output_d >= 0 && output_d < D) {
          if (aligned_16b) { // 128 bit cache line
            *reinterpret_cast<int4*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int4*>(&(val.vals[0]));
            *reinterpret_cast<int4*>(&output[b][D_start + output_d + 4]) = *reinterpret_cast<const int4*>(&(val.vals[1]));
          } else if (aligned_8b) {
            auto v0 = *reinterpret_cast<const int4*>(&(val.vals[0]));
            auto v1 = *reinterpret_cast<const int4*>(&(val.vals[1]));
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 0]) = make_int2(v0.x, v0.y);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 2]) = make_int2(v0.z, v0.w);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 4]) = make_int2(v1.x, v1.y);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 6]) = make_int2(v1.z, v1.w);
          } else {
            output[b][D_start + output_d + 0] = val.vals[0].x;
            output[b][D_start + output_d + 1] = val.vals[0].y;
            output[b][D_start + output_d + 2] = val.vals[0].z;
            output[b][D_start + output_d + 3] = val.vals[0].w;
            output[b][D_start + output_d + 4] = val.vals[1].x;
            output[b][D_start + output_d + 5] = val.vals[1].y;
            output[b][D_start + output_d + 6] = val.vals[1].z;
            output[b][D_start + output_d + 7] = val.vals[1].w;
          }
        }
      } else if (std::is_same<output_t, at::Half>::value) {
        half8 val = to_half8(accumulators[i][j]);
        if (output_d >= 0 && output_d < D) {
          if (aligned_16b) {
            *reinterpret_cast<int4*>(&output[b][D_start + output_d]) = *reinterpret_cast<const int4*>(&val);
          } else if (aligned_8b) {
            auto v = *reinterpret_cast<const int4*>(&val);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 0]) = make_int2(v.x, v.y);
            *reinterpret_cast<int2*>(&output[b][D_start + output_d + 4]) = make_int2(v.z, v.w);
          } else if (aligned_4b) {
            auto v = *reinterpret_cast<const int4*>(&val);
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 0]) = v.x;
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 2]) = v.y;
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 4]) = v.z;
            *reinterpret_cast<int*>(&output[b][D_start + output_d + 6]) = v.w;
          } else {
            output[b][D_start + output_d + 0] = val.vals[0].x;
            output[b][D_start + output_d + 1] = val.vals[0].y;
            output[b][D_start + output_d + 2] = val.vals[1].x;
            output[b][D_start + output_d + 3] = val.vals[1].y;
            output[b][D_start + output_d + 4] = val.vals[2].x;
            output[b][D_start + output_d + 5] = val.vals[2].y;
            output[b][D_start + output_d + 6] = val.vals[3].x;
            output[b][D_start + output_d + 7] = val.vals[3].y;
          }
        }
      } else {
        // INT8/4: not implemented yet
      }
    }
  }
}

__device__ inline uint32_t pruned_hash_function(uint32_t h) {
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__global__ void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor64<int32_t, 2, RestrictPtrTraits> hash_table,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_table_offsets,
    int32_t B,
    int32_t T,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dense_indices) {
    // uint32_t capacity = hash_table.size(0);
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;
    if (b_t >= B * T) {
        return;
    }
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;

    int64_t table_start = hash_table_offsets[t];
    int64_t table_end = hash_table_offsets[t + 1];
    int64_t capacity = table_end - table_start;

    if (capacity == 0) {
      // No pruning applied on the indices associated with this table.
      for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
        dense_indices[indices_start + l] = indices[indices_start + l];
      }
      return;
    }

    uint32_t subwarp_id = threadIdx.x / 4;
    uint32_t subwarp_tid = threadIdx.x % 4;
    uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
    for (int32_t l_start = 0; l_start + subwarp_id < L; l_start += kWarpSize / 4) {
        int32_t idx = indices[indices_start + l_start + subwarp_id];
        uint32_t slot_start = pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
        while (true) {
            uint32_t slot = (slot_start + subwarp_tid) % capacity;
            int2 val = *reinterpret_cast<const int2*>(&hash_table[table_start + static_cast<int64_t>(slot)][0]);
            int32_t slot_sparse_idx = val.x;
            int32_t slot_dense_idx = val.y;

            bool found = false;
            bool empty = false;
            if (slot_sparse_idx == -1) {
                empty = true;
            } else if (slot_sparse_idx == idx) {
                found = true;
                dense_indices[indices_start + l_start + subwarp_id] = slot_dense_idx;
            }
            if (__any_sync(subwarp_mask, found)) {
                break;
            } else if (__any_sync(subwarp_mask, empty)) {
                dense_indices[indices_start + l_start + subwarp_id] = -1;
                break;
            }
            slot_start += 4;
        }
    }
}

{% if not weighted %}
__global__ void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> index_remappings,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> index_remappings_offsets,
    int32_t B,
    int32_t T,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dense_indices) {
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;
  if (b_t >= B * T) {
      return;
  }
  int32_t indices_start = offsets[t * B + b];
  int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  int64_t index_remappings_start = index_remappings_offsets[t];
  int64_t index_remappings_end = index_remappings_offsets[t + 1];
  int64_t capacity = index_remappings_end - index_remappings_start;

  for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
    int32_t idx = indices[indices_start + l];
    dense_indices[indices_start + l] = capacity ? index_remappings[index_remappings_start + idx] : idx;
  }
}
{% endif %}

}

at::Tensor int_nbit_split_embedding_codegen_forward_{{ wdesc }}_cuda(
    at::Tensor dev_weights,
    at::Tensor uvm_weights,
    at::Tensor weights_placements,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    at::Tensor indice_weights,
    {% endif %}
    int64_t output_dtype,
    int64_t unused
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);

    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(max_int2_D == 0);

    at::Tensor output;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16);
    if (o_dtype == SparseType::FP32) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else if (o_dtype == SparseType::FP16) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kHalf));
    }

    if (B == 0) {
      return output;
    }

    using index_t = int32_t;

    // launch 4-bit kernel
    constexpr int32_t kWarpsPerBlock = 4;

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::int_4bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(nbit::kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "int4_split_embedding_codegen_forward_kernel", ([&] {
      if (max_int4_D > 0) {
        auto max_int4_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int4_D, SparseType::INT4), 128);
        TORCH_CHECK(max_int4_128b_rows <= 4);
        if (max_int4_128b_rows > 0) {
          X(2, 8, 0, 1);
        }
        if (max_int4_128b_rows > 1) {
          X(2, 4, 1, 2);
        }
        if (max_int4_128b_rows > 2) {
          X(1, 4, 2, 4);
        }
      }
    }));
    #undef X


    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::int_8bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(nbit::kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "int8_split_embedding_codegen_forward_kernel", ([&] {
      if (max_int8_D > 0) {
        auto max_int8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int8_D, SparseType::INT8), 128);
        TORCH_CHECK(max_int8_128b_rows <= 8);
        if (max_int8_128b_rows > 0) {
          X(2, 8, 0, 1);
        }
        if (max_int8_128b_rows > 1) {
          X(2, 4, 1, 2);
        }
        if (max_int8_128b_rows > 2) {
          X(2, 4, 2, 4);
        }
        if (max_int8_128b_rows > 4) {
          X(2, 4, 4, 8);
        }
      }
    }));
    #undef X

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::fp16_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(nbit::kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "fp16_split_embedding_codegen_forward_kernel", ([&] {
      if (max_float16_D > 0) {
        auto max_fp16_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float16_D, SparseType::FP16), 128);
        TORCH_CHECK(max_fp16_128b_rows <= 16);
        if (max_fp16_128b_rows > 0) {
          X(2, 8, 0, 2);
        }
        if (max_fp16_128b_rows > 2) {
          X(2, 8, 2, 4);
        }
        if (max_fp16_128b_rows > 4) {
          X(2, 4, 4, 8);
        }
        if (max_fp16_128b_rows > 8) {
          X(2, 2, 8, 16);
        }
      }
    }));
    #undef X

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::fp32_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(nbit::kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "fp32_split_embedding_codegen_forward_kernel", ([&] {
      if (max_float32_D > 0) {
        auto max_fp32_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float32_D, SparseType::FP32), 128);
        TORCH_CHECK(max_fp32_128b_rows <= 32);
        // FP32 is used for numerical validations and tiny embeddings tables.
        // We haven't carefully tuned the perf of FP32 embeddings.
        X(1, 1, 0, 32);
      }
    }));
    #undef X

    // TODO: 2-bit kernels.
    return output;
}

at::Tensor pruned_hashmap_lookup_{{ wdesc }}_cuda(
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor hash_table,
    at::Tensor hash_table_offsets) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(indices.get_device());
    auto dense_indices = at::empty_like(indices);
    int32_t T = hash_table_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(hash_table.size(0) < std::numeric_limits<int32_t>::max());
    constexpr size_t kForwardMaxThreads = 256;
    nbit::int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel<<<
        nbit::div_round_up(B * T + 1, kForwardMaxThreads / nbit::kWarpSize),
        dim3(nbit::kWarpSize, kForwardMaxThreads / nbit::kWarpSize),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            hash_table.packed_accessor64<int32_t, 2, at::RestrictPtrTraits>(),
            hash_table_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            B,
            T,
            dense_indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense_indices;
}

{% if not weighted %}
at::Tensor pruned_array_lookup_cuda(
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor index_remappings,
    at::Tensor index_remappings_offsets) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  auto dense_indices = at::empty_like(indices);
  int32_t T = index_remappings_offsets.size(0) - 1;
  TORCH_CHECK(
      (offsets.size(0) - 1) % T == 0,
      "offsets.size() - 1 is not divisible by T! offsets.size: ",
      offsets.size(0),
      "T: ",
      T
  );
  int32_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0, "offsets.size(): ", offsets.size(0), ", T: ", T, ", B: ", B);
  TORCH_CHECK(index_remappings.size(0) < std::numeric_limits<int64_t>::max());
  TORCH_CHECK(indices.dim() == 1, "Tensor dim: ", indices.dim());
  TORCH_CHECK(offsets.dim() == 1, "Tensor dim: ", offsets.dim());
  TORCH_CHECK(index_remappings.dim() == 1, "Tensor dim: ", index_remappings.dim());
  TORCH_CHECK(index_remappings_offsets.dim() == 1, "Tensor dim: ", index_remappings_offsets.dim());
  TORCH_CHECK(dense_indices.dim() == 1, "Tensor dim: ", dense_indices.dim());
  constexpr size_t kForwardMaxThreads = 256;
  nbit::int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel<<<
      nbit::div_round_up(offsets.size(0), kForwardMaxThreads / nbit::kWarpSize),
      dim3(nbit::kWarpSize, kForwardMaxThreads / nbit::kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          index_remappings.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          index_remappings_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
          B,
          T,
          dense_indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dense_indices;
}
{% endif %}

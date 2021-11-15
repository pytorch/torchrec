/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

template <typename scalar_t>
DEVICE_INLINE fbgemm_gpu::Vec4T<scalar_t> vec4_acc(
    fbgemm_gpu::Vec4T<scalar_t> lhs,
    fbgemm_gpu::Vec4T<scalar_t> rhs) {
  fbgemm_gpu::Vec4T<scalar_t> s;
  s.acc.x = lhs.acc.x + rhs.acc.x;
  s.acc.y = lhs.acc.y + rhs.acc.y;
  s.acc.z = lhs.acc.z + rhs.acc.z;
  s.acc.w = lhs.acc.w + rhs.acc.w;
  return s;
}

inline at::Tensor asynchronous_complete_cumsum(at::Tensor t_in) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(t_in.dim() == 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", ([&] {
        AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", ([&] {
        AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  return t_out;
}

template <typename index_t>
__global__ void linearize_index_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices) {
  int32_t T = hash_size_cumsum.size(0) - 1;
  int32_t B = (offsets.size(0) - 1) / T;
  int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b = b_t % B;
  int32_t t = b_t / B;
  bool valid = t < T;

  index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
  index_t indices_start = valid ? offsets[t * B + b] : -1;
  int32_t L = valid ? offsets[t * B + b + 1] - indices_start : 0;
  int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
    index_t indices_start_warp = __shfl_sync(0xFFFFFFFF, indices_start, j);
    int32_t b_t_warp = __shfl_sync(0xFFFFFFFF, b_t, j);
    int32_t L_warp = __shfl_sync(0xFFFFFFFF, L, j);
    index_t hash_offset_warp = __shfl_sync(0xFFFFFFFF, hash_offset, j);
    for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
      index_t idx = __ldg(&indices[indices_start_warp + i]);
      infos[indices_start_warp + i] = b_t_warp;
      linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
    }
  }
}

class FixedDivisor {
 public:
  explicit FixedDivisor(const int32_t d) : d_(d) {
    CalcSignedMagic();
  }

  /// Calculates `q = n / d`.
  DEVICE_INLINE int32_t Div(const int32_t n) const {
    // In lieu of a mulhi instruction being available, perform the
    // work in uint64
    return (int32_t)((magic_ * (uint64_t)n) >> shift_);
  }

  /// Calculates `r = n % d`.
  DEVICE_INLINE int32_t Mod(const int32_t n) const {
    return n - d_ * Div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  DEVICE_INLINE void DivMod(const int32_t n, int32_t* q, int32_t* r) const {
    *q = Div(n);
    *r = n - d_ * *q;
  }

 private:
  // Calculates magic multiplicative value and shift amount for calculating `q =
  // n / d` for signed 32-bit integers.
  // Implementation taken from Hacker's Delight section 10.
  void CalcSignedMagic() {
    if (d_ == 1) {
      magic_ = UINT64_C(0x1) << 32;
      shift_ = 32;
      return;
    }

    const uint32_t two31 = UINT32_C(0x80000000);
    const uint32_t ad = std::abs(d_);
    const uint32_t t = two31 + ((uint32_t)d_ >> 31);
    const uint32_t anc = t - 1 - t % ad; // Absolute value of nc.
    uint32_t p = 31; // Init. p.
    uint32_t q1 = two31 / anc; // Init. q1 = 2**p/|nc|.
    uint32_t r1 = two31 - q1 * anc; // Init. r1 = rem(2**p, |nc|).
    uint32_t q2 = two31 / ad; // Init. q2 = 2**p/|d|.
    uint32_t r2 = two31 - q2 * ad; // Init. r2 = rem(2**p, |d|).
    uint32_t delta = 0;
    do {
      ++p;
      q1 <<= 1; // Update q1 = 2**p/|nc|.
      r1 <<= 1; // Update r1 = rem(2**p, |nc|).
      if (r1 >= anc) { // (Must be an unsigned comparison here).
        ++q1;
        r1 -= anc;
      }
      q2 <<= 1; // Update q2 = 2**p/|d|.
      r2 <<= 1; // Update r2 = rem(2**p, |d|).
      if (r2 >= ad) { // (Must be an unsigned comparison here).
        ++q2;
        r2 -= ad;
      }
      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    int32_t magic = q2 + 1;
    if (d_ < 0) {
      magic = -magic;
    }
    shift_ = p;
    magic_ = (uint64_t)(uint32_t)magic;
  }
  int32_t d_ = 1;
  uint64_t magic_;
  int shift_;
};


DEVICE_INLINE int64_t gpuAtomicIncrement(int64_t* p) {
  static_assert(
      sizeof(int64_t) == sizeof(unsigned long long),
      "expected int64_t to be unsigned long long");
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long int*>(p),
      static_cast<unsigned long long int>(1)));
}

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/layout_transform_ops.cuh"
#include "fbgemm_gpu/permute_pooled_embedding_ops.h"

namespace fbgemm {
at::Tensor permute_pooled_embs_gpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(pooled_embs.get_device());
  // We couldn't pass the "pooled_embs.is_contiguous()" check in the backward
  // passs after D22767058. TODO: optimize and make sure pooled_embs is
  // contiguous.
  auto pooled_embs_contiguous = pooled_embs.contiguous();
  const int64_t B = pooled_embs_contiguous.size(0);
  const int64_t T = permute_list.numel();
  const int64_t dim_sum = pooled_embs_contiguous.size(1);
  // inv_permute_list is not being used so it's not checked here.
  TORCH_CHECK(pooled_embs_contiguous.device() == offset_dim_list.device());
  TORCH_CHECK(pooled_embs_contiguous.device() == permute_list.device());
  TORCH_CHECK(pooled_embs_contiguous.device() == inv_offset_dim_list.device());
  TORCH_CHECK(offset_dim_list.numel() == permute_list.numel() + 1);
  TORCH_CHECK(offset_dim_list.numel() == inv_offset_dim_list.numel());
  at::Tensor permuted_pooled_embs = at::empty_like(pooled_embs_contiguous);

  // This kernel is moving D elements per warp.
  // We are launching ( div_round_up(T, warp_per_block), B ) blocks.
  // The grid z dimension is also used by B in case it's greater than 65535.
  const int32_t warp_per_block =
      fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize;
  const int32_t max_grid_dim_y =
      32768; // The CUDA maximum is 65535, not a power of 2.
  const dim3 threads(fbgemm_gpu::kMaxThreads);
  const dim3 blocks(
      fbgemm_gpu::div_round_up(T, warp_per_block),
      std::min(static_cast<int32_t>(B), max_grid_dim_y),
      (B + max_grid_dim_y - 1) / max_grid_dim_y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      pooled_embs_contiguous.type(), "permute_pooled_embeddings", ([&] {
        permute_pooled_embs_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                pooled_embs_contiguous.data_ptr<scalar_t>(),
                offset_dim_list.data_ptr<int64_t>(),
                permute_list.data_ptr<int64_t>(),
                inv_offset_dim_list.data_ptr<int64_t>(),
                permuted_pooled_embs.data_ptr<scalar_t>(),
                B,
                T,
                dim_sum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  return permuted_pooled_embs;
}
} // namespace fbgemm

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

using namespace at;
using namespace fbgemm_gpu;

// Freeing host/uvm memory with cudaFree[Host] requires a cuda context.
// If a uvm tensor is released from an arbitrary thread without a context
// then cuda helpfully create a new default context on the default device.
// If we have not used the default device before in this process cuda
// needs to also allocate a device context. However creating a device
// context requires device resources and may fail with out of memory error
// causing  cudaFree[Host] to fail with out of memory error.
// The solution is simply to remember the device from the allocation context
// and set the correct device in the thread before calling cudaFree[Host]

namespace {
struct CUDAManagedContext {
  void* ptr_;
  int cuda_device_;

  CUDAManagedContext(void* ptr, int cuda_device)
      : ptr_(ptr), cuda_device_(cuda_device){};

  ~CUDAManagedContext() {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(cuda_device_);
    AT_CUDA_CHECK(cudaFree(ptr_));
  }

  static void release(void* ptr) {
    delete static_cast<CUDAManagedContext*>(ptr);
  }
};

// Keep a reference to the UVM memory allocation from the associated
// CPU Tensor to prevent lifetime issues (use after free)
struct CUDAManagedCpuContext {
  Storage storage_;

  CUDAManagedCpuContext(Storage storage) : storage_(std::move(storage)){};

  static void release(void* ptr) {
    delete static_cast<CUDAManagedCpuContext*>(ptr);
  }
};

// Get the default strides from the input Tensor dimensions
std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }
  return strides;
}
} // namespace

// Allocate the ATen Tensor with unified managed memory (UVM)
Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  auto strides = defaultStrides(sizes);
  size_t size_bytes =
      at::detail::computeStorageNbytes(sizes, strides, self.dtype().itemsize());
  void* ptr;
  AT_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes));
  // User hints with "preferred location": Here the kernel will page fault
  // and generate direct mapping to data on the CPU.
  AT_CUDA_CHECK(cudaMemAdvise(
      ptr, size_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  // User hints with "accessed by": GPU will establish direct mapping of data
  // in CPU memory, no page faults will be generated
  AT_CUDA_CHECK(cudaMemAdvise(
      ptr, size_bytes, cudaMemAdviseSetAccessedBy, at::cuda::current_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      size_bytes,
      at::DataPtr(
          ptr,
          new CUDAManagedContext(ptr, self.get_device()),
          &CUDAManagedContext::release,
          {at::DeviceType::CUDA, self.device().index()}),
      nullptr, /* allocator */
      /*resizable=*/false);
  return at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
}

// Check if a tensor is allocated with UVM or host-mapped memory
bool is_uvm_tensor(Tensor t) {
  if (t.device().is_cpu()) {
    return false;
  }
  auto deleter = t.storage().data_ptr().get_deleter();
  return deleter == &CUDAManagedContext::release;
}

// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(Tensor t) {
  TORCH_CHECK(is_uvm_tensor(t));
  // Don't copy the storage - just keep a reference to the original storage
  auto storage = Storage(
      Storage::use_byte_size_t(),
      t.storage().nbytes(),
      at::DataPtr(
          t.data_ptr(),
          new CUDAManagedCpuContext(t.storage()),
          &CUDAManagedCpuContext::release,
          {at::DeviceType::CPU}),
      nullptr, /* allocator */
      /*resizable=*/false);
  return at::empty({0}, t.options().device(Device::Type::CPU))
      .set_(std::move(storage), 0, t.sizes(), t.strides());
}

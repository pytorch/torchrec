/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <nvml.h>

#include <algorithm>


using namespace at;

#define NVML_CHECK(fn)                  \
  do {                                  \
    nvmlReturn_t ret = (fn);            \
    TORCH_CHECK((ret) == NVML_SUCCESS); \
  } while (0)

using Node = int64_t;
using Links = int64_t;
template <typename T>
using AdjacencyMatrix = std::function<T(Node, Node)>;

AdjacencyMatrix<Links> get_nvlink_matrix() {
  auto world_size = at::cuda::getNumGPUs();
  NVML_CHECK(nvmlInit());

  // Note that NVML uses a different numbering method to CUDA runtime,
  // so we need to learn the mapping by using the bus ID.
  uint32_t device_count;
  NVML_CHECK(nvmlDeviceGetCount(&device_count));

  std::map<std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE>, Node>
      pci_bus_ids;
  std::unordered_map<Node, uint32_t> cuda_device_to_nvml_device;

  for (const auto i : c10::irange(device_count)) {
    nvmlDevice_t handle;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &handle));
    nvmlPciInfo_t pci_info;
    NVML_CHECK(nvmlDeviceGetPciInfo(handle, &pci_info));
    std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE> pci_bus_id;
    std::copy(
        &pci_info.busId[0],
        &pci_info.busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
        pci_bus_id.data());
    int32_t node = 0;
    auto err = cudaDeviceGetByPCIBusId(&node, pci_bus_id.data());
    if (err == cudaSuccess) {
      pci_bus_ids.insert({pci_bus_id, node});
      cuda_device_to_nvml_device.insert({node, i});
    } else {
      // flush the last error - this can occur when e.g. we set
      // CUDA_VISIBLE_DEVICES to a subset of the available GPUs in the system.
      cudaGetLastError();
    }
  }

  std::vector<Links> links(world_size * world_size);
  for (const auto i : c10::irange(world_size)) {
    nvmlDevice_t handle;
    NVML_CHECK(
        nvmlDeviceGetHandleByIndex(cuda_device_to_nvml_device[i], &handle));
    for (const auto link : c10::irange(NVML_NVLINK_MAX_LINKS)) {
      nvmlEnableState_t is_active;
      auto nvmlRet = nvmlDeviceGetNvLinkState(handle, link, &is_active);
      if (nvmlRet == NVML_ERROR_INVALID_ARGUMENT ||
          nvmlRet == NVML_ERROR_NOT_SUPPORTED) {
        continue;
      }
      if (is_active != NVML_FEATURE_ENABLED) {
        continue;
      }
      nvmlPciInfo_t pci_info;
      NVML_CHECK(nvmlDeviceGetNvLinkRemotePciInfo(handle, link, &pci_info));
      std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE> pci_bus_id;
      std::copy(
          &pci_info.busId[0],
          &pci_info.busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
          pci_bus_id.data());
      auto dst = pci_bus_ids.find(pci_bus_id);
      if (dst != pci_bus_ids.end()) {
        auto j = dst->second;
        links[i * world_size + j] += 1;
      }
    }
  }

  return [=](Node i, Node j) { return links[i * world_size + j]; };
}

// Hilariously unoptimized, but algorithmic correctness matters more here, and
// we only do it once.
AdjacencyMatrix<Node> get_intermediate_node(AdjacencyMatrix<Links> links) {
  auto world_size = at::cuda::getNumGPUs();
  auto intermediate_node = [&](Node i, Node j) {
    if (i == j) {
      return std::vector<Node>{-1};
    }
    if (links(i, j) != 0) {
      return std::vector<Node>{-1};
    }

    std::vector<std::pair<Node, Links>> paths;
    for (const auto k : c10::irange(world_size)) {
      if (k != i && k != j && links(i, k) != 0 && links(k, j) != 0) {
        paths.push_back({k, links(i, k) + links(k, j)});
      }
    }
    if (paths.empty()) {
      LOG(WARNING)
          << "Expect very bad performance for p2p copies, we are going via sys path for GPU "
          << i << " -> GPU " << j;
      return std::vector<Node>{-1};
    }
    auto mp = std::max_element(
                  paths.begin(),
                  paths.end(),
                  [](std::pair<Node, Links> a, std::pair<Node, Links> b) {
                    return a.second < b.second;
                  })
                  ->second;
    std::vector<Node> candidates;
    for (const auto& p : paths) {
      if (p.second == mp) {
        candidates.push_back(p.first);
      }
    }
    return candidates;
  };

  std::vector<Node> assignments(world_size * world_size);
  // Use a two-phase assignment protocol as the greedy approach
  // can lead to unbalanced usage.
  std::unordered_map<Node, int64_t> uses;
  for (const auto i : c10::irange(world_size)) {
    for (const auto j : c10::irange(world_size)) {
      auto ims = intermediate_node(i, j);
      if (ims.size() == 1) {
        auto v = ims.front();
        if (v != -1) {
          uses[v] += 1;
        }
        assignments[i * world_size + j] = v;
      }
    }
  }

  for (const auto i : c10::irange(world_size)) {
    for (const auto j : c10::irange(world_size)) {
      auto ims = intermediate_node(i, j);
      if (ims.size() > 1) {
        auto v = *std::min_element(ims.begin(), ims.end(), [&](Node a, Node b) {
          return uses[a] < uses[b];
        });
        uses[v] += 1;
        assignments[i * world_size + j] = v;
      }
    }
  }
  if (std::any_of(assignments.begin(), assignments.end(), [](Node n) {
        return n != -1;
      })) {
    auto tensor = at::from_blob(
        assignments.data(),
        {world_size, world_size},
        at::TensorOptions().dtype(at::kLong));
    LOG(INFO) << "Detected a multi-hop NVLink configuration: \n" << tensor;
    return [=](Node i, Node j) { return assignments[i * world_size + j]; };
  } else {
    return [](Node, Node) { return -1; };
  }
}
namespace {
Tensor cat_dim_1(
    std::vector<Tensor> tensors,
    int batch_size,
    at::Device output_device) {
  if (tensors.size() == 0) {
    return at::empty({0}, at::TensorOptions().device(output_device));
  }
  int64_t total_dim_1 = 0;
  std::vector<int64_t> cumulative_dims;
  cumulative_dims.push_back(0);
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dim() == 2);
    TORCH_CHECK(t.size(0) == batch_size);
    total_dim_1 += t.size(-1);
    cumulative_dims.push_back(total_dim_1);
  }

  auto* prop = at::cuda::getCurrentDeviceProperties();
  auto output = at::empty(
      {batch_size, total_dim_1},
      tensors.front().options().device(output_device));
  TORCH_CHECK(output.stride(0) * output.element_size() <= static_cast<int64_t>(prop->memPitch));

  std::vector<at::cuda::CUDAEvent> copy_begin_events(tensors.size());
  std::vector<at::cuda::CUDAEvent> copy_completion_events(tensors.size());

  Node dst_device_id = output_device.index();
  static auto intermediate_nodes = get_intermediate_node(get_nvlink_matrix());
  // Do the intermediate copies, if required by our multi-hop config.
  for (auto& ten: tensors) {
    Node src_device_id = ten.device().index();
    auto intermediate_node = intermediate_nodes(src_device_id, dst_device_id);
    if (intermediate_node != -1) {
      ten = ten.to(at::Device(at::kCUDA, intermediate_node));
    }
  }

  // synchronize source streams and launch copies on source stream.
  for (const auto i : c10::irange(tensors.size())) {
    auto src = tensors[i];
    if (src.device() != output.device()) {
      auto dst = output.slice(1, cumulative_dims[i], cumulative_dims[i + 1]);

      at::Device dst_device = dst.device();
      at::Device src_device = src.device();
      at::cuda::CUDAGuard device_guard(src_device);
      // We always perform the copy on the source device, using the current
      // stream on the source device, and we fully synchronize on both src and
      // dst's current streams for completion of the copy. We have to explicitly
      // do this for non-contig copies. This mimics the behavior of cross-device
      // cudaMemcpyAsync on the default stream.

      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(src_device.index());
      // This is a cross-device copy on the src current stream and dst current
      // stream. We perform a two-way barrier between both devices' streams
      // before the copy. This ensures that any write-after-write and
      // write-after-read dependencies on the destination side are handled, so
      // that no one is operating on the dst memory when we perform the copy.
      // src waits on dst barrier (src already waits on src)
      auto& dst_ready = copy_begin_events[i];
      device_guard.set_device(dst_device);
      dst_ready.record(at::cuda::getCurrentCUDAStream(dst_device.index()));
      device_guard.set_device(src_device);
      dst_ready.block(copy_stream);
      // on source device, launch memcpy.
      AT_CUDA_CHECK(cudaMemcpy2DAsync(
          dst.data_ptr(),
          dst.stride(0) * dst.element_size(),
          src.data_ptr(),
          src.stride(0) * dst.element_size(),
          src.size(1) * src.element_size(),
          src.size(0),
          cudaMemcpyDeviceToDevice,
          copy_stream));
    }
  }

  // Do the same-GPU cases.
  for (const auto i : c10::irange(tensors.size())) {
    auto src = tensors[i];
    if (src.device() == output.device()) {
      auto dst = output.slice(1, cumulative_dims[i], cumulative_dims[i + 1]);
      at::Device src_device = src.device();
      // single device memcpy, not that src_device == dst_device.
      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(src_device.index());
      AT_CUDA_CHECK(cudaMemcpy2DAsync(
          dst.data_ptr(),
          dst.stride(0) * dst.element_size(),
          src.data_ptr(),
          src.stride(0) * src.element_size(),
          src.size(1) * src.element_size(),
          src.size(0),
          cudaMemcpyDeviceToDevice,
          copy_stream));
    }
  }
  // wait for cross-device copies to complete.
  for (const auto i : c10::irange(tensors.size())) {
    auto src = tensors[i];
    if (src.device() != output.device()) {
      auto dst = output.slice(1, cumulative_dims[i], cumulative_dims[i + 1]);
      at::Device dst_device = dst.device();
      at::Device src_device = src.device();
      // Still on src_device, record stream event
      at::cuda::CUDAGuard device_guard(src_device);
      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(src_device.index());

      auto& src_ready = copy_completion_events[i];
      src_ready.record(copy_stream);

      device_guard.set_device(dst_device);
      src_ready.block(at::cuda::getCurrentCUDAStream(dst_device.index()));
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}
} // namespace

// TODO: Add device arg.
Tensor merge_pooled_embeddings(
    std::vector<Tensor> pooled_embeddings,
    int64_t batch_size,
    at::Device target_device) {
  static std::once_flag flag;
  std::call_once(flag, []() {
    for (const auto i : c10::irange(at::cuda::getNumGPUs())) {
      for (const auto j : c10::irange(at::cuda::getNumGPUs())) {
        if (i != j) {
          at::cuda::CUDAGuard g(i);
          const auto err = cudaDeviceEnablePeerAccess(j, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            // ignore and clear the error if access was already enabled
            cudaGetLastError();
          } else {
            AT_CUDA_CHECK(err);
          }
        }
      }
    }
  });

  at::cuda::CUDAGuard g(target_device);

  TORCH_CHECK(!pooled_embeddings.empty());
  return cat_dim_1(pooled_embeddings, batch_size, target_device);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "merge_pooled_embeddings(Tensor[] pooled_embeddings, int batch_size, Device target_device) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "merge_pooled_embeddings",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(merge_pooled_embeddings)));
}

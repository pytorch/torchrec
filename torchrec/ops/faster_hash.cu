/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Dispatch.h> // @manual
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cuda.h> // @manual
#include <torch/torch.h> // @manual
#include <ctime>

#define TORBOREC_CUDA
#include "torchrec/ops/common_utils.cuh"

namespace torch::torchrec::operators {

using at::Tensor;
using namespace torch::torchrec::turborec;

namespace {

static constexpr int32_t kDefaultTensor = -1;
static constexpr int64_t kMaxIdentityNum = INT32_MAX;
static constexpr int64_t kMaxHours = INT32_MAX;
static constexpr int64_t kSecondsInHour = 60 * 60;

template <typename T>
__device__ __inline__ T CAS(T* data, T cmp, T val) {
  return atomicCAS(data, cmp, val);
}

template <>
__device__ __inline__ int64_t
CAS<int64_t>(int64_t* data, int64_t cmp, int64_t val) {
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long*>(data),
      static_cast<unsigned long long>(cmp),
      static_cast<unsigned long long>(val)));
}

template <int32_t METADATA_COUNT>
__device__ __inline__ void update_metadata(
    int32_t* /* metadata */,
    int64_t /* output_index */,
    int32_t /* metadata_val */) {
  static_assert(METADATA_COUNT != 1);
  // no op.
}

template <>
__device__ __inline__ void update_metadata<1>(
    int32_t* metadata,
    int64_t output_index,
    int32_t metadata_val) {
  atomicMax(metadata + output_index, metadata_val);
}

template <int32_t METADATA_COUNT>
__device__ __inline__ void update_metadata_lru(
    int32_t* /* metadata */,
    int64_t /* output_index */,
    int32_t /* val */,
    int32_t* /* process_lock */) {
  static_assert(METADATA_COUNT != 1);
  // no-op
}

template <>
__device__ __inline__ void update_metadata_lru<1>(
    int32_t* metadata,
    int64_t output_index,
    int32_t val,
    int32_t* process_lock) {
  // These should be atomic as we release process lock as last step
  atomicExch(metadata + output_index, val);
  // Release process lock from index
  atomicExch(process_lock + output_index, kDefaultTensor);
}

template <int32_t METADATA_COUNT, typename TIdentity>
__device__ __inline__ int64_t check_min(
    int32_t /* process_index */,
    int32_t* /* metadata */,
    int64_t min_index,
    int64_t /* output_index */,
    int64_t /* offset */,
    int32_t& /* min_hours */,
    int32_t* /* process_lock */,
    PackedTensorAccessor64<TIdentity, 2> /* identities */,
    TIdentity& /* min_slot_identity */,
    int32_t /* eviction_threshold */,
    std::enable_if_t<METADATA_COUNT == 0>* = nullptr) {
  static_assert(METADATA_COUNT == 0);
  // For inference, we keep the same min_index until the ID is found.
  return min_index;
}

template <int32_t METADATA_COUNT, typename TIdentity>
__device__ __inline__ int64_t check_min(
    int32_t process_index,
    int32_t* metadata,
    int64_t min_index,
    int64_t output_index,
    int64_t offset,
    int32_t& min_hours,
    int32_t* process_lock,
    PackedTensorAccessor64<TIdentity, 2> identities,
    TIdentity& min_slot_identity,
    int32_t eviction_threshold,
    std::enable_if_t<METADATA_COUNT == 1>* = nullptr) {
  static_assert(METADATA_COUNT == 1);
  // There could be a case, one id has already occupy the slot,
  // and last update hour is not written yet, while the other id checking the
  // slot for min index, then it would '-1' in this case, hence we need to
  // wait.
  auto insert_idx = output_index + offset;
  int32_t last_seen = kDefaultTensor;
  while (true) {
    last_seen =
        atomicCAS(metadata + insert_idx, kDefaultTensor, kDefaultTensor);
    if (last_seen != kDefaultTensor) {
      break;
    }
  }

  // only check those expired slots
  if (eviction_threshold > last_seen && min_hours > last_seen) {
    // Try to lock index for thread
    auto old_pid =
        atomicCAS(process_lock + insert_idx, kDefaultTensor, process_index);
    if (old_pid == kDefaultTensor) {
      // Index locked for this thread
      // Check if value is still same and not updated by other thread
      if (last_seen == *(metadata + insert_idx)) {
        if (min_index != -1) {
          // Release lock on previous min_index
          atomicCAS(
              process_lock + min_index + offset, process_index, kDefaultTensor);
        }
        // Update min_index to current
        min_index = output_index;
        min_hours = last_seen;
        min_slot_identity = identities[insert_idx][0];
      } else {
        // Value updated by other thread. Release lock on this index
        atomicCAS(process_lock + insert_idx, process_index, kDefaultTensor);
      }
    }
  }
  return min_index;
}

template <int32_t METADATA_COUNT>
__device__ __inline__ bool check_evict(
    int32_t* /* metadata */,
    int64_t /* output_index */,
    int32_t /* eviction_threshold */) {
  static_assert(METADATA_COUNT != 1);
  return false;
}

template <>
__device__ __inline__ bool check_evict<1>(
    int32_t* metadata,
    int64_t output_index,
    int32_t eviction_threshold) {
  // In rare case, one id may have already occupied the slot but its metadata
  // has not been written yet, while the other id checking the slot's eviction
  // status. Therefore, wait until the metadata is not -1.
  int32_t identity_metadata = kDefaultTensor;
  while (true) {
    identity_metadata =
        atomicCAS(metadata + output_index, kDefaultTensor, kDefaultTensor);
    if (identity_metadata != kDefaultTensor) {
      break;
    }
  }

  return eviction_threshold > identity_metadata;
}

template <bool READONLY, typename TIdentity>
__device__ __inline__ bool check_and_maybe_update_slot(
    TIdentity* identities_slot,
    TIdentity identity,
    TIdentity& old_value,
    std::enable_if_t<READONLY == true>* = nullptr) {
  static_assert(READONLY);
  old_value = *identities_slot;
  if (old_value == identity) {
    return true;
  }
  return false;
}

template <bool READONLY, typename TIdentity>
__device__ __inline__ bool check_and_maybe_update_slot(
    TIdentity* identities_slot,
    TIdentity identity,
    TIdentity& old_value,
    std::enable_if_t<READONLY == false>* = nullptr) {
  static_assert(!READONLY);
  old_value =
      CAS(identities_slot, static_cast<TIdentity>(kDefaultTensor), identity);
  if ((old_value == identity) ||
      (old_value == static_cast<TIdentity>(kDefaultTensor))) {
    return true;
  }
  return false;
}

template <bool CIRCULAR_PROBE, typename TIdentity>
__device__ __inline__ int64_t get_identity_slot(
    PackedTensorAccessor64<TIdentity, 2> identities,
    TIdentity identity,
    int64_t output_index,
    int64_t offset,
    int64_t modulo,
    int64_t max_probe) {
  while (max_probe-- > 0) {
    auto insert_idx = output_index + offset;
    auto current_slot_identity = identities[insert_idx][0];
    if (current_slot_identity == kDefaultTensor) {
      // Hits end but still don't find, don't disable eviction.
      return -1;
    } else if (current_slot_identity == identity) {
      // there is identity in probing distance, we shouldn't evict.
      return output_index;
    }

    output_index =
        next_output_index<CIRCULAR_PROBE>(output_index, modulo, max_probe);
  }

  // Nothing found, don't disable eviction.
  return -1;
}

template <
    int32_t EVICTION_POLICY,
    bool DISABLE_FALLBACK,
    int32_t HASH_IDENTITY,
    int32_t METADATA_COUNT,
    bool CIRCULAR_PROBE,
    bool READONLY,
    typename TInput,
    typename TIdentity>
__global__ void process_item_zch(
    const PackedTensorAccessor64<TInput, 1> input,
    PackedTensorAccessor64<int64_t, 1> output,
    int64_t* evict_slots,
    PackedTensorAccessor64<TIdentity, 2> identities,
    int64_t modulo,
    int64_t max_probe,
    int32_t cur_hour,
    const int64_t* const local_sizes,
    const int64_t* const offsets,
    int32_t* metadata,
    const int32_t* const input_metadata,
    int32_t eviction_threshold,
    int32_t* /* process_lock */,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const int32_t* const opt_in_rands,
    TORCH_DSA_KERNEL_ARGS,
    std::enable_if_t<EVICTION_POLICY == 0>* = nullptr) {
  static_assert(EVICTION_POLICY == 0);

  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  // NOLINTNEXTLINE: Implicitly casting
  auto total_items = input.size(0);
  for (int32_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < total_items;
       // NOLINTNEXTLINE: Implicitly casting
       process_index += blockDim.x * gridDim.x) {
    auto item = input[process_index];
    if (local_sizes != nullptr) {
      modulo = local_sizes[process_index];
    }
    int64_t offset = 0;
    if (offsets != nullptr) {
      offset = offsets[process_index];
    }
    // for backward compatibility: previous implementation assigns cur_hour
    // to metadata
    int32_t metadata_val =
        input_metadata != nullptr ? input_metadata[process_index] : cur_hour;

    auto hash = murmur_hash3_2x64(static_cast<uint64_t>(item), 0, 0);
    auto opt_in_block_size =
        opt_in_prob == -1 ? modulo : modulo - num_reserved_slots;
    auto output_index =
        static_cast<int64_t>(hash % opt_in_block_size); // Local idx
    TIdentity identity;

    if constexpr (HASH_IDENTITY == 1) {
      identity = static_cast<TIdentity>(
          murmur_hash3_2x64(
              static_cast<uint64_t>(item),
              0x17, // seed
              0) %
          kMaxIdentityNum);
    } else if (HASH_IDENTITY == 2) {
      identity = static_cast<TIdentity>(item % kMaxIdentityNum);
    } else {
      identity = item;
    }

    // probing.
    auto max_probe_local = max_probe;
    TIdentity old_value = kDefaultTensor;

    // In eviction mode. We might run into case that an ID has already
    // had a slot, in position hash(id) + 50 due to probing.
    // Now between hash(id) to hash(id) + 50 has an expiration slot, during
    // next look up of this id, we will expire that slot and put this id in
    // that slot, if this id is very popular id, this id will start from
    // ground zero, and it's not ideal.
    // Our solution to solve this is to quickly check all probing location,
    // see if our id has already existed, if existed, then we don't need
    // eviction. Has to note, there might be very rare cases the id slot got
    // evicted, it's OK, then we will colide this once with hash(id), and next
    // time, it would pick up an expired slot.
    // Also, we don't need lock here. As we are readonly here and other
    // concurrent write should have no impact on us.
    int64_t identity_slot = get_identity_slot<CIRCULAR_PROBE, TIdentity>(
        identities,
        identity,
        output_index,
        offset,
        opt_in_block_size,
        max_probe);

    bool opt_in = true;
    if (identity_slot == -1 && opt_in_rands != nullptr &&
        opt_in_rands[process_index] >= opt_in_prob) {
      // ID with rand value > opt_in_prob will not be accepted and will
      // instead be assigned to one of the reserved slots.
      opt_in = false;
      output_index =
          opt_in_block_size + static_cast<int64_t>(hash % num_reserved_slots);
      update_metadata<METADATA_COUNT>(
          metadata, output_index + offset, metadata_val);
    }

    while (max_probe_local-- > 0 && opt_in) {
      auto insert_idx = output_index + offset;
      if (check_and_maybe_update_slot<READONLY, TIdentity>(
              &identities[insert_idx][0], identity, old_value)) {
        update_metadata<METADATA_COUNT>(metadata, insert_idx, metadata_val);
        break;
      }

      if (identity_slot == -1 &&
          check_evict<METADATA_COUNT>(
              metadata, insert_idx, eviction_threshold)) {
        auto current_slot_value =
            CAS<TIdentity>(&identities[insert_idx][0], old_value, identity);
        if (current_slot_value == old_value || current_slot_value == identity) {
          evict_slots[process_index] = insert_idx;
          update_metadata<METADATA_COUNT>(metadata, insert_idx, metadata_val);
          break;
        }
      }

      output_index = next_output_index<CIRCULAR_PROBE>(
          output_index,
          opt_in_block_size, // only probe within the opt-in block
          max_probe_local);
    }

    // can't find a slot (all slot full after probing), collide
    if (max_probe_local < 0) {
      if constexpr (DISABLE_FALLBACK) {
        output_index = -1;
        offset = 0;
      } else {
        output_index = opt_in_prob == -1 ? static_cast<int64_t>(hash % modulo)
                                         : opt_in_block_size +
                static_cast<int64_t>(hash % num_reserved_slots);
      }
    }

    output[process_index] = output_index + offset;
  }
}

template <
    int32_t EVICTION_POLICY,
    bool DISABLE_FALLBACK,
    int32_t HASH_IDENTITY,
    int32_t METADATA_COUNT,
    bool CIRCULAR_PROBE,
    bool READONLY,
    typename TInput,
    typename TIdentity>
__global__ void process_item_zch(
    const PackedTensorAccessor64<TInput, 1> input,
    PackedTensorAccessor64<int64_t, 1> output,
    int64_t* evict_slots,
    PackedTensorAccessor64<TIdentity, 2> identities,
    int64_t modulo,
    int64_t max_probe,
    int32_t cur_hour,
    const int64_t* const local_sizes,
    const int64_t* const offsets,
    int32_t* metadata,
    const int32_t* const input_metadata,
    int32_t eviction_threshold,
    int32_t* process_lock,
    int64_t /* opt_in_prob */,
    int64_t /* num_reserved_slots */,
    const int32_t* const /* opt_in_rands */,
    TORCH_DSA_KERNEL_ARGS,
    std::enable_if_t<EVICTION_POLICY == 1>* = nullptr) {
  static_assert(EVICTION_POLICY == 1);

  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  // NOLINTNEXTLINE: Implicitly casting
  auto total_items = input.size(0);

  for (int32_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < total_items;
       // NOLINTNEXTLINE: Implicitly casting
       process_index += blockDim.x * gridDim.x) {
    auto item = input[process_index];
    if (local_sizes != nullptr) {
      modulo = local_sizes[process_index];
    }
    int64_t offset = 0;
    if (offsets != nullptr) {
      offset = offsets[process_index];
    }
    int32_t metadata_val =
        input_metadata != nullptr ? input_metadata[process_index] : cur_hour;
    auto hash = murmur_hash3_2x64(static_cast<uint64_t>(item), 0, 0);
    auto output_index = static_cast<int64_t>(hash % modulo); // Local idx
    TIdentity identity;

    if constexpr (HASH_IDENTITY == 1) {
      identity = static_cast<TIdentity>(
          murmur_hash3_2x64(
              static_cast<uint64_t>(item),
              0x17, // seed
              0) %
          kMaxIdentityNum);
    } else if (HASH_IDENTITY == 2) {
      identity = static_cast<TIdentity>(item % kMaxIdentityNum);
    } else {
      identity = item;
    }

    // probing.
    auto max_probe_local = max_probe;
    TIdentity old_value = kDefaultTensor;

    int64_t min_index = -1; // local_index; initially set it as -1
    int32_t min_hours = kMaxHours;
    // tracks the existing value of canddiate slot may be evicted during
    // probing;
    TIdentity min_slot_identity = kDefaultTensor;
    while (max_probe_local-- > 0) {
      auto insert_idx = output_index + offset;
      if (check_and_maybe_update_slot<READONLY, TIdentity>(
              &identities[insert_idx][0], identity, old_value)) {
        update_metadata_lru<METADATA_COUNT>(
            metadata, insert_idx, metadata_val, process_lock);
        break;
      }

      min_index = check_min<METADATA_COUNT, TIdentity>(
          process_index,
          metadata,
          min_index,
          output_index,
          offset,
          min_hours,
          process_lock,
          identities,
          min_slot_identity,
          eviction_threshold);

      output_index = next_output_index<CIRCULAR_PROBE>(
          output_index, modulo, max_probe_local);
    }

    if (max_probe_local < 0) {
      if (min_index == -1) {
        // Can't find a min slot due to identities completing for slots in
        // probing distance; This case should not be hit frequently. Cases like:
        //  1. Hashes are concentrated in a probing distance
        //  2. Probing distance is too small
        //  3. in eval mode, can't find the identity in probing distance
        if constexpr (DISABLE_FALLBACK) {
          output_index = -1;
          offset = 0;
          output[process_index] = output_index + offset;
          return;
        } else {
          // collide
          output_index = static_cast<int64_t>(hash % modulo);
        }
      } else {
        // find an expire slot to evict
        output_index = min_index;
        // do evict only in training mode
        // directly return output_index in eval mode (readonly = True)
        if constexpr (!READONLY) {
          auto insert_idx = output_index + offset;
          CAS<TIdentity>(
              &identities[insert_idx][0], min_slot_identity, identity);
          update_metadata_lru<METADATA_COUNT>(
              metadata, insert_idx, metadata_val, process_lock);
          evict_slots[process_index] = insert_idx;
        }
      }
    }
    output[process_index] = output_index + offset;
  }
}

} // namespace

template <typename TInput, typename TIdentity>
void _zero_collision_hash_cuda(
    Tensor& output,
    Tensor& evict_slots,
    const Tensor& input,
    Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    int64_t cur_hour,
    bool readonly,
    bool support_evict,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    int32_t hash_identity,
    const std::optional<Tensor>& metadata,
    bool disable_fallback,
    const std::optional<Tensor>& input_metadata,
    int64_t eviction_threshold,
    int64_t eviction_policy,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const std::optional<Tensor>& opt_in_rands) {
  constexpr int64_t kThreads = 256L;
  auto block_size = kThreads;
  auto grid_size = std::min(
      (input.numel() + block_size - 1) / block_size,
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);
  int64_t modulo = identities.size(0);

  // auxiliary data structure to lock each slot
  std::optional<Tensor> process_lock;
  if (eviction_policy == 1 && metadata.has_value()) {
    process_lock = at::full(
        {modulo, 1},
        kDefaultTensor,
        c10::TensorOptions().dtype(at::kInt).device(metadata->device()));
  }
#define INVOKE_KERNEL(                                                        \
    EVICTION_POLICY,                                                          \
    DISABLE_FALLBACK,                                                         \
    HASH_IDENTITY,                                                            \
    METADATA_COUNT,                                                           \
    CIRCULAR_PROBE,                                                           \
    READONLY)                                                                 \
  {                                                                           \
    TORCH_DSA_KERNEL_LAUNCH(                                                  \
        (process_item_zch<                                                    \
            EVICTION_POLICY,                                                  \
            DISABLE_FALLBACK,                                                 \
            HASH_IDENTITY,                                                    \
            METADATA_COUNT,                                                   \
            CIRCULAR_PROBE,                                                   \
            READONLY,                                                         \
            TInput,                                                           \
            TIdentity>),                                                      \
        grid_size,                                                            \
        block_size,                                                           \
        0,                                                                    \
        at::cuda::getCurrentCUDAStream(),                                     \
        input.packed_accessor64<TInput, 1>(),                                 \
        output.packed_accessor64<int64_t, 1>(),                               \
        support_evict ? evict_slots.data_ptr<int64_t>() : nullptr,            \
        identities.packed_accessor64<TIdentity, 2>(),                         \
        modulo,                                                               \
        max_probe,                                                            \
        static_cast<int32_t>(cur_hour),                                       \
        local_sizes.has_value() ? local_sizes->data_ptr<int64_t>() : nullptr, \
        offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr,         \
        metadata.has_value() ? metadata->data_ptr<int32_t>() : nullptr,       \
        input_metadata.has_value() ? input_metadata->data_ptr<int32_t>()      \
                                   : nullptr,                                 \
        static_cast<int32_t>(eviction_threshold),                             \
        process_lock.has_value() ? process_lock->data_ptr<int32_t>()          \
                                 : nullptr,                                   \
        opt_in_prob,                                                          \
        num_reserved_slots,                                                   \
        opt_in_rands.has_value() ? opt_in_rands->data_ptr<int32_t>()          \
                                 : nullptr);                                  \
  }

#define INVOKE_KERNEL_EVICT_POLICY(                                            \
    DISABLE_FALLBACK, HASH_IDENTITY, METADATA_COUNT, CIRCULAR_PROBE, READONLY) \
  {                                                                            \
    if (eviction_policy == 0) {                                                \
      INVOKE_KERNEL(                                                           \
          0,                                                                   \
          DISABLE_FALLBACK,                                                    \
          HASH_IDENTITY,                                                       \
          METADATA_COUNT,                                                      \
          CIRCULAR_PROBE,                                                      \
          READONLY);                                                           \
    } else {                                                                   \
      INVOKE_KERNEL(                                                           \
          1,                                                                   \
          DISABLE_FALLBACK,                                                    \
          HASH_IDENTITY,                                                       \
          METADATA_COUNT,                                                      \
          CIRCULAR_PROBE,                                                      \
          READONLY);                                                           \
    }                                                                          \
  }

#define INVOKE_HASH_IDENTITY(                                             \
    HASH_IDENTITY, METADATA_COUNT, CIRCULAR_PROBE, READONLY)              \
  {                                                                       \
    if (disable_fallback) {                                               \
      INVOKE_KERNEL_EVICT_POLICY(                                         \
          true, HASH_IDENTITY, METADATA_COUNT, CIRCULAR_PROBE, READONLY)  \
    } else {                                                              \
      INVOKE_KERNEL_EVICT_POLICY(                                         \
          false, HASH_IDENTITY, METADATA_COUNT, CIRCULAR_PROBE, READONLY) \
    }                                                                     \
  }

#define INVOKE_KERNEL_METADATA_COUNT(METADATA_COUNT, CIRCULAR_PROBE, READONLY) \
  {                                                                            \
    if (hash_identity == 1) {                                                  \
      INVOKE_HASH_IDENTITY(1, METADATA_COUNT, CIRCULAR_PROBE, READONLY);       \
    } else if (hash_identity == 2) {                                           \
      INVOKE_HASH_IDENTITY(2, METADATA_COUNT, CIRCULAR_PROBE, READONLY);       \
    } else {                                                                   \
      INVOKE_HASH_IDENTITY(0, METADATA_COUNT, CIRCULAR_PROBE, READONLY);       \
    }                                                                          \
  }

#define INVOKE_KERNEL_CIRCULAR_PROBE(CIRCULAR_PROBE, READONLY)   \
  {                                                              \
    if (support_evict) {                                         \
      INVOKE_KERNEL_METADATA_COUNT(1, CIRCULAR_PROBE, READONLY); \
    } else {                                                     \
      INVOKE_KERNEL_METADATA_COUNT(0, CIRCULAR_PROBE, READONLY); \
    }                                                            \
  }

#define INVOKE_KERNEL_READ_ONLY(READONLY)            \
  {                                                  \
    if (circular_probe) {                            \
      INVOKE_KERNEL_CIRCULAR_PROBE(true, READONLY);  \
    } else {                                         \
      INVOKE_KERNEL_CIRCULAR_PROBE(false, READONLY); \
    }                                                \
  }

  if (readonly) {
    INVOKE_KERNEL_READ_ONLY(true);
  } else {
    INVOKE_KERNEL_READ_ONLY(false);
  }

#undef INVOKE_KERNEL_READ_ONLY
#undef INVOKE_KERNEL_CIRCULAR_PROBE
#undef INVOKE_KERNEL_METADATA_COUNT
#undef INVOKE_HASH_IDENTITY
#undef INVOKE_KERNEL
} // namespace torch::torchrec::operators

std::tuple<Tensor, Tensor> zero_collision_hash_cuda(
    const Tensor& input,
    Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    int64_t exp_hours, // to be deprecated
    bool readonly,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    const std::optional<Tensor>& metadata,
    bool output_on_uvm,
    bool disable_fallback,
    bool _modulo_identity_DPRECATED,
    const std::optional<Tensor>& input_metadata,
    int64_t eviction_threshold,
    int64_t eviction_policy,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const std::optional<Tensor>& opt_in_rands) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(identities.dim() == 2);

  int32_t hash_identity = _modulo_identity_DPRECATED ? 1 : 2;
  if (identities.dtype() == input.dtype()) {
    hash_identity = 0;
  }
  if (input.dtype() == torch::kInt32) {
    TORCH_CHECK(identities.dtype() == torch::kInt32);
  }

  if (input_metadata.has_value()) {
    TORCH_CHECK(exp_hours == -1);
    TORCH_CHECK(input_metadata->size(0) == input.size(0));
    TORCH_CHECK(eviction_threshold != -1);
    TORCH_CHECK(eviction_policy == 0 || eviction_policy == 1);
  }
  if (eviction_threshold != -1) {
    TORCH_CHECK(eviction_policy == 0 || eviction_policy == 1);
    TORCH_CHECK(input_metadata.has_value());
  }

  std::time_t now_c = time(nullptr);
  auto hours = static_cast<int64_t>(now_c) / kSecondsInHour;
  auto cur_hour = hours % kMaxHours;

  if (exp_hours > 0) {
    TORCH_CHECK(!input_metadata.has_value());
    TORCH_CHECK(eviction_threshold == -1);

    // for backward compatibility: previous implementation uses cur_hour -
    // exp_hours as threshold
    // note the eviction criteria is the same: eviction_threshold >
    // identity_metadata (last-seen hour)
    eviction_threshold = cur_hour - exp_hours;
  }

  bool support_evict =
      is_eviction_enabled(readonly, eviction_threshold, eviction_policy);

  TORCH_CHECK(
      !support_evict || metadata.has_value(),
      "support_evict=",
      support_evict,
      "metadata is null");
  TORCH_CHECK(
      support_evict || !metadata.has_value(),
      "support_evict=",
      support_evict,
      "metadata is not null");

  if (metadata.has_value()) {
    TORCH_CHECK(metadata->dim() == 2);
    TORCH_CHECK(metadata->is_cuda());
    TORCH_CHECK(metadata->size(0) == identities.size(0));
  }
  // offsets and local_sizes are null in training; not null during
  // inference/eval
  if (local_sizes.has_value()) {
    TORCH_CHECK(local_sizes->is_cuda());
    TORCH_CHECK(input.numel() == local_sizes->numel());
  }
  if (offsets.has_value()) {
    TORCH_CHECK(offsets->is_cuda());
    TORCH_CHECK(input.numel() == offsets->numel());
  }
  if (opt_in_prob != -1) {
    TORCH_CHECK(opt_in_prob > 0 && opt_in_prob < 100);
    TORCH_CHECK(num_reserved_slots > 0);
  }
  if (num_reserved_slots != -1) {
    TORCH_CHECK(opt_in_prob != -1);
  }
  if (opt_in_rands.has_value()) {
    TORCH_CHECK(opt_in_prob != -1);
    TORCH_CHECK(opt_in_rands->size(0) == input.size(0));
    TORCH_CHECK(opt_in_rands->dtype() == torch::kInt32);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  int64_t output_size = input.size(0);
  c10::TensorOptions options;

  if (output_on_uvm) {
    options =
        c10::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(
            true);
  } else {
    options = c10::TensorOptions().dtype(at::kLong).device(input.device());
  }

  Tensor output = at::empty({output_size}, options);

  // evict_slots will contains the index to be evcited, '-1' will be ignored.
  Tensor evict_slots;
  if (support_evict) {
    evict_slots = at::full(
        {output_size},
        static_cast<int64_t>(kDefaultTensor),
        c10::TensorOptions().dtype(at::kLong).device(input.device()));
  }

  if (output_size == 0) {
    return {output, evict_slots};
  }

  AT_DISPATCH_INTEGER_TYPES(
      input.scalar_type(), "zero_collision_hash_input", input_t, [&]() {
        AT_DISPATCH_INTEGER_TYPES(
            identities.scalar_type(),
            "zero_collision_hash_identity",
            identity_t,
            [&]() {
              _zero_collision_hash_cuda<input_t, identity_t>(
                  output,
                  evict_slots,
                  input,
                  identities,
                  max_probe,
                  circular_probe,
                  cur_hour,
                  readonly,
                  support_evict,
                  local_sizes,
                  offsets,
                  hash_identity,
                  metadata,
                  disable_fallback,
                  input_metadata,
                  eviction_threshold,
                  eviction_policy,
                  opt_in_prob,
                  num_reserved_slots,
                  opt_in_rands);
            });
      });

  if (support_evict) {
    evict_slots = std::get<0>(torch::_unique(
        evict_slots.masked_select(evict_slots != kDefaultTensor)));
  }
  if (output_on_uvm) {
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }
  return {output, evict_slots};
}

// Register operators
TORCH_LIBRARY_IMPL(torchrec, CUDA, m) {
  m.impl(
      "zero_collision_hash",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(zero_collision_hash_cuda)));
}

} // namespace torch::torchrec::operators

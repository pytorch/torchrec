/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Parallel.h> // @manual
#include <torch/csrc/jit/runtime/static/ops.h> // @manual
#include <torch/library.h> // @manual
#include <torch/torch.h> // @manual
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "torchrec/ops/common_utils.cuh"

/* Inference ONLY op */

#define FASTER_HASH_CPU_INTRO_OP_PARALLEL 0

namespace torch::torchrec::operators {

using at::Tensor;
using namespace torch::torchrec::turborec;

namespace {
static constexpr int32_t kDefaultTensor = -1;
static constexpr int64_t kMaxIdentityNum = INT32_MAX;

template <
    bool DISABLE_FALLBACK,
    int32_t HASH_IDENTITY,
    bool CIRCULAR_PROBE,
    bool HAS_OFFSET,
    typename TInput,
    typename TIdentity>
void process_item_zch(
    const PackedTensorAccessor64<TInput, 1>& input,
    PackedTensorAccessor64<int64_t, 1> output,
    const PackedTensorAccessor64<TIdentity, 2>& identities,
    int64_t modulo,
    int64_t max_probe,
    const int64_t* const local_sizes,
    const int64_t* const offsets,
    int64_t opt_in_prob,
    int64_t num_reserved_slots) {
  // Do we need multi-threading here considering prediction are already
  // multi-threaded over requests?

  int64_t total_items = input.size(0);

#ifdef FASTER_HASH_CPU_INTRO_OP_PARALLEL
  at::parallel_for(
      0,
      total_items,
      FASTER_HASH_CPU_INTRO_OP_PARALLEL,
      [&](int64_t t_begin, int64_t t_end) {
#else
  int64_t t_begin = 0;
  int64_t t_end = total_items;
#endif
        for (auto process_index = t_begin; process_index < t_end;
             ++process_index) {
          auto item = input[process_index];
          int64_t offset = 0;
          if constexpr (HAS_OFFSET) {
            modulo = local_sizes[process_index];
            offset = offsets[process_index];
          }

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
          } else if constexpr (HASH_IDENTITY == 2) {
            identity = static_cast<TIdentity>(item % kMaxIdentityNum);
          } else {
            identity = item;
          }

          auto max_probe_local = max_probe;
          while (max_probe_local-- > 0) {
            auto insert_idx = output_index + offset;
            auto current_slot_identity = identities[insert_idx][0];
            // Inference treat empty slot (kDefaultTensor) as collision and
            // continue next probe
            if (current_slot_identity == identity) {
              break;
            }

            output_index = next_output_index<CIRCULAR_PROBE>(
                output_index,
                opt_in_block_size, // only probe within the opt-in block
                max_probe_local);
          }

          // can't find a slot (all slot full after probing)
          if (max_probe_local < 0) {
            if constexpr (DISABLE_FALLBACK) {
              output_index = -1;
              offset = 0;
            } else {
              output_index = opt_in_prob == -1
                  ? static_cast<int64_t>(hash % modulo)
                  : opt_in_block_size +
                      static_cast<int64_t>(hash % num_reserved_slots);
            }
          }

          output[process_index] = output_index + offset;
        }
#ifdef FASTER_HASH_CPU_INTRO_OP_PARALLEL
      });
#endif
}

template <typename TInput, typename TIdentity>
void _zero_collision_hash_cpu_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& identities,
    int64_t max_probe,
    const bool circular_probe,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    int32_t hash_identity,
    bool disable_fallback,
    int64_t opt_in_prob,
    int64_t num_reserved_slots) {
  int64_t modulo = identities.size(0);
  auto* local_sizes_ptr =
      local_sizes.has_value() ? local_sizes->data_ptr<int64_t>() : nullptr;
  auto* offsets_ptr =
      offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr;

#define INVOKE_KERNEL(                                           \
    DISABLE_FALLBACK, HASH_IDENTITY, CIRCULAR_PROBE, HAS_OFFSET) \
  {                                                              \
    process_item_zch<                                            \
        DISABLE_FALLBACK,                                        \
        HASH_IDENTITY,                                           \
        CIRCULAR_PROBE,                                          \
        HAS_OFFSET,                                              \
        TInput,                                                  \
        TIdentity>(                                              \
        input.packed_accessor64<TInput, 1>(),                    \
        output.packed_accessor64<int64_t, 1>(),                  \
        identities.packed_accessor64<TIdentity, 2>(),            \
        modulo,                                                  \
        max_probe,                                               \
        local_sizes_ptr,                                         \
        offsets_ptr,                                             \
        opt_in_prob,                                             \
        num_reserved_slots);                                     \
  }

#define INVOKE_HASH_IDENTITY(HASH_IDENTITY, CIRCULAR_PROBE, HAS_OFFSET) \
  {                                                                     \
    if (disable_fallback) {                                             \
      INVOKE_KERNEL(true, HASH_IDENTITY, CIRCULAR_PROBE, HAS_OFFSET)    \
    } else {                                                            \
      INVOKE_KERNEL(false, HASH_IDENTITY, CIRCULAR_PROBE, HAS_OFFSET)   \
    }                                                                   \
  }

#define INVOKE_KERNEL_CIRCULAR_PROBE(CIRCULAR_PROBE, HAS_OFFSET) \
  {                                                              \
    if (hash_identity == 1) {                                    \
      INVOKE_HASH_IDENTITY(1, CIRCULAR_PROBE, HAS_OFFSET);       \
    }                                                            \
    if (hash_identity == 2) {                                    \
      INVOKE_HASH_IDENTITY(2, CIRCULAR_PROBE, HAS_OFFSET);       \
    } else {                                                     \
      INVOKE_HASH_IDENTITY(0, CIRCULAR_PROBE, HAS_OFFSET);       \
    }                                                            \
  }

#define INVOKE_KERNEL_HAS_OFFSET(HAS_OFFSET)           \
  {                                                    \
    if (circular_probe) {                              \
      INVOKE_KERNEL_CIRCULAR_PROBE(true, HAS_OFFSET);  \
    } else {                                           \
      INVOKE_KERNEL_CIRCULAR_PROBE(false, HAS_OFFSET); \
    }                                                  \
  }

  if (local_sizes_ptr != nullptr) {
    INVOKE_KERNEL_HAS_OFFSET(true);
  } else {
    INVOKE_KERNEL_HAS_OFFSET(false);
  }

#undef INVOKE_KERNEL_HAS_OFFSET
#undef INVOKE_KERNEL_CIRCULAR_PROBE
#undef INVOKE_HASH_IDENTITY
#undef INVOKE_KERNEL
}

} // namespace

std::tuple<Tensor, Tensor> zero_collision_hash_meta(
    const Tensor& input,
    Tensor& /* identities */,
    int64_t /* max_probe */,
    bool /* circular_probe */,
    int64_t /* exp_hours */,
    bool /* readonly */,
    const std::optional<Tensor>& /* local_sizes */,
    const std::optional<Tensor>& /* offsets */,
    const std::optional<Tensor>& /* metadata */,
    bool /* output_on_uvm */,
    bool /* disable_fallback */,
    bool /* _modulo_identity_DPRECATED */,
    const std::optional<Tensor>& /* input_metadata */,
    int64_t /* eviction_threshold */,
    int64_t /* eviction_policy */,
    int64_t /* opt_in_prob */,
    int64_t /* num_reserved_slots */,
    const std::optional<Tensor>& /* opt_in_rands */) {
  auto out =
      at::zeros_symint({input.sym_numel()}, input.options().dtype(at::kLong));
  auto evcit_slots = at::zeros_symint({0}, input.options());
  return {input, evcit_slots};
}

std::tuple<Tensor, Tensor> create_zch_buffer_cpu(
    const int64_t size,
    bool support_evict,
    std::optional<at::Device> device,
    bool long_type) {
  Tensor metadata;
  auto identity = at::full(
      {size, 1},
      kDefaultTensor,
      c10::TensorOptions()
          .dtype(long_type ? at::kLong : at::kInt)
          .device(device));
  if (support_evict) {
    metadata = at::full(
        {size, 1},
        kDefaultTensor,
        c10::TensorOptions().dtype(at::kInt).device(device));
  }
  return {identity, metadata};
}

void zero_collision_hash_cpu_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    bool disable_fallback,
    bool _modulo_identity_DPRECATED,
    int64_t opt_in_prob,
    int64_t num_reserved_slots) {
  TORCH_CHECK(output.is_cpu());
  TORCH_CHECK(output.dtype() == torch::kInt64);

  TORCH_CHECK(input.is_cpu());
  TORCH_CHECK(identities.dim() == 2);

  int hash_identity = _modulo_identity_DPRECATED ? 2 : 1;
  if (identities.dtype() == input.dtype()) {
    hash_identity = 0;
  }
  if (input.dtype() == torch::kInt32) {
    TORCH_CHECK(identities.dtype() == torch::kInt32);
  }

  if (local_sizes.has_value()) {
    TORCH_CHECK(local_sizes->is_cpu());
    TORCH_CHECK(input.numel() == local_sizes->numel());
  }
  if (offsets.has_value()) {
    TORCH_CHECK(offsets->is_cpu());
    TORCH_CHECK(input.numel() == offsets->numel());
  }
  if (opt_in_prob != -1) {
    TORCH_CHECK(opt_in_prob > 0 && opt_in_prob < 100);
    TORCH_CHECK(num_reserved_slots > 0);
  }
  if (num_reserved_slots != -1) {
    TORCH_CHECK(opt_in_prob != -1);
  }

  AT_DISPATCH_INTEGER_TYPES(
      input.scalar_type(), "zero_collision_hash_input", input_t, [&]() {
        AT_DISPATCH_INTEGER_TYPES(
            identities.scalar_type(),
            "zero_collision_hash_identity",
            identity_t,
            [&]() {
              _zero_collision_hash_cpu_out<input_t, identity_t>(
                  output,
                  input,
                  identities,
                  max_probe,
                  circular_probe,
                  local_sizes,
                  offsets,
                  hash_identity,
                  disable_fallback,
                  opt_in_prob,
                  num_reserved_slots);
            });
      });
}

std::tuple<Tensor, Tensor> zero_collision_hash_cpu(
    const Tensor& input,
    Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    int64_t exp_hours,
    bool readonly,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    const std::optional<Tensor>& metadata,
    bool /* output_on_uvm */,
    bool disable_fallback,
    bool _modulo_identity_DPRECATED,
    const std::optional<Tensor>& input_metadata,
    int64_t eviction_threshold,
    int64_t /* eviction_policy */,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const std::optional<Tensor>& opt_in_rands) {
  TORCH_CHECK(exp_hours == -1);
  TORCH_CHECK(readonly);
  TORCH_CHECK(metadata.has_value() == false);
  TORCH_CHECK(input_metadata.has_value() == false);
  TORCH_CHECK(eviction_threshold == -1);
  TORCH_CHECK(opt_in_rands.has_value() == false);

  int64_t output_size = input.size(0);
  c10::TensorOptions options =
      c10::TensorOptions().dtype(at::kLong).device(input.device());
  Tensor output = at::empty({output_size}, options);

  // evict_slots will contains the index to be evcited, '-1' will be ignored.
  Tensor evict_slots;

  if (output_size == 0) {
    return {output, evict_slots};
  }

  zero_collision_hash_cpu_out(
      output,
      input,
      identities,
      max_probe,
      circular_probe,
      local_sizes,
      offsets,
      disable_fallback,
      _modulo_identity_DPRECATED,
      opt_in_prob,
      num_reserved_slots);

  return {output, evict_slots};
}

TORCH_LIBRARY_FRAGMENT(torchrec, m) {
  // Create identities buffer. As we need everything to be -1.
  // One could also create themsleves, as long as follow the protocol:
  // 1. all value should be -1.
  // 2. the tensor should be two dimensions.
  // 3. if support evict, need two columns, otherwise, just one column.
  //
  // Args:
  //   size: define identities tensor size.
  //   support_evict: whether we support evict.
  //
  // Result:
  //   Tuple[tensor, tensor] for identities and metadata.
  //      identity: Shape (D, 2) with size(1) = 1
  //      metadata: Shape (D, 2) with size(1) = 1
  //
  // For other examples, consult the unittests.
  m.def(
      "create_zch_buffer("
      "int size, "
      "bool support_evict=False,"
      "Device? device=None,"
      "bool long_type=False"
      ") -> (Tensor, Tensor)");
  // Default impl
  m.impl("create_zch_buffer", TORCH_FN(create_zch_buffer_cpu));

  // technically this is not zero collision, but low collision. Trade-off
  // between probing speed. (Setting probes to a large value and a larger
  // identities tensor size could make it zero collision.)
  //
  // Here we have a few features:
  // 1. probing to find next available slot for hash collision to reduce
  // collision.
  // 2. non circular probing - as this will be used in local rank, and later in
  // publish stage, we will combine all local rank as a global tensor, hence non
  // circular probing could make sure probing logic problems.
  // 3. eviction - a slot could be evited if it's not been used for a while.
  // 4. readonly mode - use for inference, in inference, we don't need atomic
  // operation as everything are readonly.
  //
  // Args:
  //   input: ids to find slots. Shape (D)
  //   identities: a tensor which stores identities for ids. Shape (D, 1).
  //   max_probe: max probing, reach max will fall back to original hash
  //              position. recommend use 128.
  //   circular_probe: when hitting end of identities tensor, circular to
  //              beginning of identities tensor to find slots or not.
  //   exp_hours (to be deprecated): how many hours without any updates
  //              considering as slot for eviction. setting as -1 means
  //              disabling eviction.
  //   readonly: enable readonly mode or not. Perf will be much faster.
  //   local_sizes: local size for each chunk. Used to recover the index in
  //              sharded case.
  //   offsets: offsets for each chunk. Used to recover the index in sharded
  //              case.
  //   disable_fallback: the fallback behavior when an ID does not exist. If
  //              true, -1 is returned, which indicates it fails to find a
  //              position for this ID. If false, the position of the first
  //              probe is returned.
  //   input_metadata: the metadata for each individual ID. It will become the
  //              metadata of the slot if the ID is accepted to that slot. While
  //              it is often used to represent an ID's TTL, the meaning can
  //              vary.
  //   eviction_threshold: the threshold selected for eviction. Kernel makes an
  //              eviction decision based on the existing metadata associated
  //              with slots and the eviction threshold.
  //   eviction_policy: the kernel based on the eviction policy.
  //              0: No eviction or TTL based eviction.
  //              1: LRU based eviction timestamped on the hour.
  //   opt_in_prob: the probability of a new ID being opted in (valid range: 1
  //              to 99). If -1, all new IDs are opted in (100%).
  //   num_reserved_slots: the number of slots reserved (located in the tail)
  //              for IDs that are not opted in. A non-zero value is required
  //              when opt-in is enabled. -1 indicates no reserved slots (100%
  //              opt-in). If the size of embedding table is x, and
  //              num_reserved_slots is y, then the size of the opt-in block
  //              will be (x - y).
  //   opt_in_rands: the random numbers used to determine whether incoming IDs
  //              should be accepted when opt-in is enabled. Its generated by
  //              caller of the kernel and its size needs to be identical to the
  //              input size. Each new ID will be accepted only if its rand
  //              number is less than opt_in_prob.
  // Result:
  //   identities index tensor: the slots found for the ids. Shape (D)
  //   evict slots: the index to identities tensor, indicating which slots got
  //                evicted. note, need to remove '-1' index.
  //
  // For other examples, consult the unittests.
  m.def(
      "zero_collision_hash("
      "Tensor input, "
      "Tensor identities, "
      "int max_probe, "
      "bool circular_probe=False, "
      "int exp_hours=-1, "
      "bool readonly=False, "
      "Tensor? local_sizes=None, "
      "Tensor? offsets=None, "
      "Tensor? metadata=None, "
      "bool output_on_uvm=False, "
      "bool disable_fallback=False, "
      "bool _modulo_identity_DPRECATED=False, "
      "Tensor? input_metadata=None, "
      "int eviction_threshold=-1, "
      "int eviction_policy=0, "
      "int opt_in_prob=-1, "
      "int num_reserved_slots=-1, "
      "Tensor? opt_in_rands=None "
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torchrec, CPU, m) {
  m.impl(
      "create_zch_buffer",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(create_zch_buffer_cpu)));

  m.impl(
      "zero_collision_hash",
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(zero_collision_hash_cpu)));
}

TORCH_LIBRARY_IMPL(torchrec, Meta, m) {
  m.impl(
      "zero_collision_hash",
      torch::dispatch(
          c10::DispatchKey::Meta, TORCH_FN(zero_collision_hash_meta)));
}

} // namespace torch::torchrec::operators

namespace torch::jit {

using at::Tensor;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    torchrec::operators::zero_collision_hash_cpu,
    sparse_zero_collision_hash_cpu,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("sparse::zero_collision_hash("
                                    "Tensor input, "
                                    "Tensor identities, "
                                    "int max_probe, "
                                    "bool circular_probe=False, "
                                    "int exp_hours=-1, "
                                    "bool readonly=False, "
                                    "Tensor? local_sizes=None, "
                                    "Tensor? offsets=None, "
                                    "Tensor? metadata=None, "
                                    "bool output_on_uvm=False, "
                                    "bool disable_fallback=False, "
                                    "bool _modulo_identity_DPRECATED=False, "
                                    "Tensor? input_metadata=None, "
                                    "int eviction_threshold=-1, "
                                    "int eviction_policy=0, "
                                    "int opt_in_prob=-1, "
                                    "int num_reserved_slots=-1, "
                                    "Tensor? opt_in_rands=None"
                                    ") -> (Tensor, Tensor)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        const auto& identities = p_node->Input(1).toTensor();
        const auto max_probe = p_node->Input(2).toInt();
        const auto circular_probe = p_node->Input(3).toBool();

        const auto& local_sizes = p_node->Input(6).toOptional<Tensor>();
        const auto& offsets = p_node->Input(7).toOptional<Tensor>();
        const auto& disable_fallback = p_node->Input(10).to<bool>();
        const auto& _modulo_identity_DPRECATED = p_node->Input(11).to<bool>();
        const auto opt_in_prob = p_node->Input(15).toInt();
        const auto num_reserved_slots = p_node->Input(16).toInt();

        if (p_node->Output(0).isNone()) {
          const at::ScalarType output_type = kLong;
          p_node->Output(0) = torch::jit::create_empty_from(input, output_type);
        }
        auto& out_t = p_node->Output(0).toTensor();
        torchrec::operators::zero_collision_hash_cpu_out(
            out_t,
            input,
            identities,
            max_probe,
            circular_probe,
            local_sizes,
            offsets,
            disable_fallback,
            _modulo_identity_DPRECATED,
            num_reserved_slots,
            opt_in_prob);
      };
    });
} // namespace torch::jit

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

using namespace at;

// Map index to cache_set. h_in: linear_indices; C: #cache_sets.
int64_t host_lxu_cache_slot(int64_t h_in, int64_t C);

// Linearize the indices of all tables to make it be unique
Tensor linearize_cache_indices_cuda(
    Tensor cache_hash_size_cumsum,
    Tensor indices,
    Tensor offsets);

// LRU cache: fetch the rows corresponding to `linear_cache_indices` from
// `weights`, and insert them into the cache at timestep `time_stamp`.
void lru_cache_populate_cuda(
    Tensor weights,
    Tensor hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    bool stochastic_rounding);

// LFU cache: fetch the rows corresponding to `linear_cache_indices` from
// `weights`, and insert them into the cache.
void lfu_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    bool stochastic_rounding);

// Lookup the LRU/LFU cache: find the cache weights location for all indices.
// Look up the slots in the cache corresponding to `linear_cache_indices`, with
// a sentinel value for missing.
Tensor lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state);

// Flush the cache: store the weights from the cache to the backing storage.
void lxu_cache_flush_cuda(
    Tensor uvm_weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    bool stochastic_rounding);

namespace {

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "linearize_cache_indices(Tensor cache_hash_size_cumsum, Tensor indices, Tensor offsets) -> Tensor");
  m.impl(
      "linearize_cache_indices",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(linearize_cache_indices_cuda)));
  m.def(
      "lru_cache_populate(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, bool stochastic_rounding) -> ()");
  m.impl(
      "lru_cache_populate",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(lru_cache_populate_cuda)));
  m.def(
      "lfu_cache_populate(Tensor weights, Tensor cache_hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, Tensor(c!) lfu_state, bool stochastic_rounding) -> ()");
  m.impl(
      "lfu_cache_populate",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(lfu_cache_populate_cuda)));
  m.def(
      "lxu_cache_lookup(Tensor linear_cache_indices, Tensor lxu_cache_state) -> Tensor");
  m.impl(
      "lxu_cache_lookup",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(lxu_cache_lookup_cuda)));
  m.def(
      "lxu_cache_flush(Tensor(a!) uvm_weights, Tensor cache_hash_size_cumsum, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, int total_D, Tensor(b!) lxu_cache_state, Tensor(c!) lxu_cache_weights, bool stochastic_rounding) -> ()");
  m.impl(
      "lxu_cache_flush",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(lxu_cache_flush_cuda)));
  m.def("lxu_cache_slot(int h_in, int C) -> int");
  m.impl(
      "lxu_cache_slot",
      torch::dispatch(
          c10::DispatchKey::CatchAll, TORCH_FN(host_lxu_cache_slot)));
}
} // namespace

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <ostream>
#ifdef FBCODE_CAFFE2
#include <folly/container/Enumerate.h>
#include <folly/container/F14Map.h>
#endif
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>

using namespace at;

Tensor int_nbit_split_embedding_codegen_forward_unweighted_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t output_dtype,
    int64_t unused);

Tensor int_nbit_split_embedding_codegen_forward_weighted_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t output_dtype,
    int64_t unused);

Tensor int_nbit_split_embedding_codegen_lookup_function_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,  // to match the interface of CUDA op using UVM
    Tensor weights_placements,  // to match the interface of CUDA op using UVM
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype) {
  if (!indice_weights) {
    return int_nbit_split_embedding_codegen_forward_unweighted_cpu(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        D_offsets,
        total_D,
        indices,
        offsets,
        pooling_mode,
        output_dtype,
        0);
  }
  return int_nbit_split_embedding_codegen_forward_weighted_cpu(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      indices,
      offsets,
      pooling_mode,
      *indice_weights,
      output_dtype,
      0);
}

void pruned_hashmap_insert_unweighted_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

Tensor pruned_hashmap_lookup_unweighted_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

Tensor pruned_array_lookup_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets);

TORCH_LIBRARY_FRAGMENT(fb, m) {

  m.impl(
      "int_nbit_split_embedding_codegen_lookup_function",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(int_nbit_split_embedding_codegen_lookup_function_cpu)));

  // GPU version of pruned_hashmap needs to use CPU version of
  // pruned_hashmap_insert
  m.def(
      "pruned_hashmap_insert(Tensor indices, Tensor dense_indices, Tensor offsets, Tensor hash_table, Tensor hash_table_offsets) -> ()");
  m.impl(
      "pruned_hashmap_insert",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(pruned_hashmap_insert_unweighted_cpu)));

  // CPU version of hashmap Lookup isn't used. For CPUs, we should use
  // PrunedMapCPU below.
  m.impl(
      "pruned_hashmap_lookup",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(pruned_hashmap_lookup_unweighted_cpu)));

  // CPU version of array lookup.
  m.impl(
      "pruned_array_lookup",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(pruned_array_lookup_cpu)));
}

class PrunedMapCPU : public torch::jit::CustomClassHolder {
 public:
  PrunedMapCPU() {}
  explicit PrunedMapCPU(std::string serialized) {
    torch::serialize::InputArchive archive;
    archive.load_from(serialized.data(), serialized.size());
    Tensor values;
    archive.read(std::string("values"), values);
    Tensor table_offsets;
    archive.read(std::string("table_offsets"), table_offsets);

    auto T = table_offsets.numel() - 1;

    auto values_acc = values.accessor<int32_t, 2>();
    auto table_offsets_acc = table_offsets.accessor<int64_t, 1>();

    maps_.resize(T);
    for (auto t = 0; t < T; ++t) {
      auto& map = maps_[t];
      const auto table_start = table_offsets_acc[t];
      for (auto i = 0; i < values.size(0); ++i) {
        auto slot_sparse_index = values_acc[table_start + i][0];
        auto slot_dense_index = values_acc[table_start + i][1];
        map.emplace(slot_sparse_index, slot_dense_index);
      }
    }
  }
  std::string serialize() const {
    torch::serialize::OutputArchive archive(
        std::make_shared<torch::jit::CompilationUnit>());
    int64_t T = maps_.size();
    auto table_offsets =
        at::empty({T + 1}, at::TensorOptions(at::kCPU).dtype(at::kLong));
    auto table_offsets_acc = table_offsets.accessor<int64_t, 1>();
    table_offsets_acc[0] = 0;
    int64_t N = 0;
    for (auto t = 0; t < T; ++t) {
      N += maps_[t].size();
      table_offsets_acc[t + 1] = N;
    }
    auto values =
        at::empty({N, 2}, at::TensorOptions(at::kCPU).dtype(at::kInt));
    auto values_acc = values.accessor<int32_t, 2>();
    for (auto t = 0; t < maps_.size(); ++t) {
      const auto& map = maps_[t];
      const auto table_start = table_offsets_acc[t];
      TORCH_CHECK(
          map.size() == (table_offsets_acc[t + 1] - table_offsets_acc[t]));
      int index = 0;
      for (const auto& kv : map) {
        values_acc[table_start + index][0] = kv.first;
        values_acc[table_start + index][1] = kv.second;
        index++;
      }
    }
    std::ostringstream oss;
    archive.write(std::string("values"), values);
    archive.write(std::string("table_offsets"), table_offsets);
    archive.save_to(oss);
    return oss.str();
  }

  void insert(Tensor indices, Tensor dense_indices, Tensor offsets, int64_t T) {
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    maps_.resize(T);
    for (int32_t t = 0; t < T; ++t) {
      auto& map = maps_[t];
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t slot_sparse_index = indices_acc[indices_start + l];
          int32_t slot_dense_index = dense_indices_acc[indices_start + l];
          if (slot_dense_index == -1) {
            // -1 means this row has been pruned, do not insert it.
            continue;
          }
          map.emplace(slot_sparse_index, slot_dense_index);
        }
      }
    }
  }

  Tensor lookup(Tensor indices, Tensor offsets) const {
    int32_t T = maps_.size();
    TORCH_CHECK(T > 0);
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(maps_.size() == T);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    for (int32_t t = 0; t < T; ++t) {
      auto& map = maps_[t];
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t slot_sparse_index = indices_acc[indices_start + l];
          auto it = map.find(slot_sparse_index);
          dense_indices_acc[indices_start + l] =
              it != map.end() ? it->second : -1;
        }
      }
    }
    return dense_indices;
  }

 private:
#ifdef FBCODE_CAFFE2
  std::vector<folly::F14FastMap<int32_t, int32_t>> maps_;
#else
  std::vector<std::unordered_map<int32_t, int32_t>> maps_;
#endif
};

static auto PrunedMapCPURegistry =
    torch::class_<PrunedMapCPU>("fb", "PrunedMapCPU")
        .def(torch::init<>())
        .def("insert", &PrunedMapCPU::insert)
        .def("lookup", &PrunedMapCPU::lookup)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<PrunedMapCPU>& self) -> std::string {
              return self->serialize();
            },
            // __setstate__
            [](std::string data) -> c10::intrusive_ptr<PrunedMapCPU> {
              return c10::make_intrusive<PrunedMapCPU>(data);
            });

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/script.h> // @manual
#include <string>
#include <vector>

// key: (value,  weights, lengths, offsets)
using TupleDict = torch::Dict<
    std::string,
    std::tuple<
        torch::Tensor,
        c10::optional<torch::Tensor>,
        torch::Tensor,
        torch::Tensor>>;
using TupleOptionalFields = std::tuple<
    std::vector<std::string>, // keys
    torch::Tensor, // values_
    c10::optional<torch::Tensor>, // weights_
    c10::optional<torch::Tensor>, // lengths_
    c10::optional<torch::Tensor>, // offsets_
    int64_t, // stride_
    c10::optional<std::vector<int64_t>>, // length_per_key_
    c10::optional<std::vector<int64_t>>, // offset_per_key_
    c10::optional<torch::Dict<std::string, int64_t>> // index_per_key_
    >;

class KeyedJaggedTensor : public torch::jit::CustomClassHolder {
 public:
  KeyedJaggedTensor() {}
  KeyedJaggedTensor(
      const std::vector<std::string>& keys,
      const torch::Tensor& values,
      const c10::optional<torch::Tensor>& weights = c10::nullopt,
      const c10::optional<torch::Tensor>& lengths = c10::nullopt,
      const c10::optional<torch::Tensor>& offsets = c10::nullopt,
      const c10::optional<int64_t>& stride = c10::nullopt,
      const c10::optional<std::vector<int64_t>>& length_per_key = c10::nullopt,
      const c10::optional<std::vector<int64_t>>& offset_per_key = c10::nullopt,
      const c10::optional<torch::Dict<std::string, int64_t>>& index_per_key =
          c10::nullopt,
      bool sync = false);

  torch::Tensor offsets();
  torch::Tensor lengths();
  const torch::Tensor values() const;
  c10::optional<torch::Tensor> weights() const;
  int64_t stride() const;
  std::vector<int64_t> length_per_key();
  std::vector<int64_t> offset_per_key();
  torch::Dict<std::string, int64_t> index_per_key();
  TupleOptionalFields all_fields() const;

  TupleOptionalFields to(torch::Device device, bool non_blocking = false);
  void pin_memory();
  void sync();
  TupleDict to_dict();

 private:
  int64_t compute_stride_();

  std::vector<std::string> keys_;
  torch::Tensor values_;
  c10::optional<torch::Tensor> weights_;
  c10::optional<torch::Tensor> lengths_;
  c10::optional<torch::Tensor> offsets_;
  int64_t stride_;
  c10::optional<std::vector<int64_t>> length_per_key_;
  c10::optional<std::vector<int64_t>> offset_per_key_;
  c10::optional<torch::Dict<std::string, int64_t>> index_per_key_;
  c10::optional<TupleDict> jagged_tensor_dict_;
};

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/sparse/jagged_tensor_ops.h"

using namespace torch::indexing;

KeyedJaggedTensor::KeyedJaggedTensor(
    const std::vector<std::string>& keys,
    const torch::Tensor& values,
    const c10::optional<torch::Tensor>& weights,
    const c10::optional<torch::Tensor>& lengths,
    const c10::optional<torch::Tensor>& offsets,
    const c10::optional<int64_t>& stride,
    const c10::optional<std::vector<int64_t>>& _length_per_key,
    const c10::optional<std::vector<int64_t>>& _offset_per_key,
    const c10::optional<torch::Dict<std::string, int64_t>>& _index_per_key,
    bool sync)
    : keys_(keys),
      values_(values),
      weights_(weights),
      lengths_(lengths),
      offsets_(offsets),
      length_per_key_(_length_per_key),
      offset_per_key_(_offset_per_key),
      index_per_key_(_index_per_key) {
  stride_ = stride.has_value() ? *stride : compute_stride_();

  if (sync) {
    length_per_key();
    offset_per_key();
  }
}

const torch::Tensor KeyedJaggedTensor::values() const {
  return values_;
}

torch::Tensor KeyedJaggedTensor::offsets() {
  if (offsets_.has_value()) {
    return *offsets_;
  }
  TORCH_CHECK(
      lengths_.has_value(), "Either offsets_ or lengths_ should have value");
  offsets_ = at::zeros({lengths_->numel() + 1}, lengths_->options());
  offsets_->index_put_({Slice(1, None)}, at::cumsum(*lengths_, 0));
  return *offsets_;
}

torch::Tensor KeyedJaggedTensor::lengths() {
  if (lengths_.has_value()) {
    return *lengths_;
  }
  TORCH_CHECK(
      offsets_.has_value(), "Either offsets_ or lengths_ should have value");
  auto offsets = *offsets_;
  lengths_ = offsets.slice(0, 1, offsets.numel()) -
      offsets.slice(0, 0, offsets.numel() - 1);
  return *lengths_;
}

c10::optional<torch::Tensor> KeyedJaggedTensor::weights() const {
  return weights_;
}

int64_t KeyedJaggedTensor::compute_stride_() {
  if (keys_.size() == 0) {
    return 0;
  }
  if (offsets_.has_value()) {
    return (offsets_->numel() - 1) / keys_.size();
  }
  if (lengths_.has_value()) {
    return lengths_->numel() / keys_.size();
  }
  return 1;
}

std::vector<int64_t> KeyedJaggedTensor::length_per_key() {
  if (length_per_key_.has_value()) {
    return *length_per_key_;
  }
  if (keys_.size() == 0) {
    length_per_key_ = std::vector<int64_t>();
    return length_per_key_.value();
  }
  const auto& length_per_key_tensor =
      torch::sum(lengths().view({-1, stride_}), 1).cpu();
  TORCH_CHECK(
      length_per_key_tensor.dtype() == torch::kInt64,
      "KJT length must be int type");
  const auto& length_per_key_tensor_a =
      length_per_key_tensor.accessor<int64_t, 1>();

  length_per_key_ = std::vector<int64_t>(length_per_key_tensor.numel(), 0);
  for (int i = 0; i < length_per_key_.value().size(); i++) {
    (*length_per_key_)[i] = length_per_key_tensor_a[i];
  }
  return *length_per_key_;
}

std::vector<int64_t> KeyedJaggedTensor::offset_per_key() {
  if (offset_per_key_.has_value()) {
    return *offset_per_key_;
  }
  const auto& tmp_length_per_key = length_per_key();
  offset_per_key_ = std::vector<int64_t>(tmp_length_per_key.size() + 1, 0);
  for (int i = 0; i < tmp_length_per_key.size(); i++) {
    (*offset_per_key_)[i + 1] = (*offset_per_key_)[i] + tmp_length_per_key[i];
  }
  return *offset_per_key_;
}

int64_t KeyedJaggedTensor::stride() const {
  return stride_;
}

// key: (value, weights, lengths, offsets)
TupleDict KeyedJaggedTensor::to_dict() {
  if (jagged_tensor_dict_.has_value()) {
    return *jagged_tensor_dict_;
  }
  TupleDict ret;
  int start_offset = 0, end_offset = 0;

  const auto& tmp_length_per_key = length_per_key();
  const auto& tmp_lengths = lengths();

  for (int i = 0; i < keys_.size(); i++) {
    end_offset = start_offset + tmp_length_per_key[i];

    auto& key = keys_[i];
    const auto& value = values_.slice(0, start_offset, end_offset);
    const auto& lengths = tmp_lengths.slice(0, i * stride_, (i + 1) * stride_);

    c10::optional<torch::Tensor> weights = c10::nullopt;
    if (weights_.has_value()) {
      weights = weights_->slice(0, start_offset, end_offset);
    }
    torch::Tensor offsets = at::zeros({lengths.numel() + 1}, lengths.options());
    offsets.index_put_({Slice(1, None)}, at::cumsum(lengths, 0));
    ret.insert(key, std::make_tuple(value, weights, lengths, offsets));

    start_offset = end_offset;
  }
  jagged_tensor_dict_ = ret;
  return ret;
}

torch::Dict<std::string, int64_t> KeyedJaggedTensor::index_per_key() {
  if (index_per_key_.has_value()) {
    return *index_per_key_;
  }
  index_per_key_ = c10::make_optional<torch::Dict<std::string, int64_t>>(
      torch::Dict<std::string, int64_t>());
  for (int64_t i = 0; i < keys_.size(); i++) {
    index_per_key_->insert(keys_[i], i);
  }
  return *index_per_key_;
}

TupleOptionalFields KeyedJaggedTensor::all_fields() const {
  return std::make_tuple(
      keys_,
      values_,
      weights_,
      lengths_,
      offsets_,
      stride_,
      length_per_key_,
      offset_per_key_,
      index_per_key_);
}

TupleOptionalFields KeyedJaggedTensor::to(
    torch::Device device,
    bool non_blocking) {
  const auto values = values_.to(device, non_blocking);
  c10::optional<torch::Tensor> weights = c10::nullopt;
  if (weights_.has_value()) {
    weights = weights_->to(device, non_blocking);
  }
  c10::optional<torch::Tensor> lengths = c10::nullopt;
  if (lengths_.has_value()) {
    lengths = lengths_->to(device, non_blocking);
  }
  c10::optional<torch::Tensor> offsets = c10::nullopt;
  if (offsets_.has_value()) {
    offsets = offsets_->to(device, non_blocking);
  }
  return std::make_tuple(
      keys_,
      values,
      weights,
      lengths,
      offsets,
      stride_,
      length_per_key_,
      offset_per_key_,
      index_per_key_);
}

void KeyedJaggedTensor::pin_memory() {
  values_ = values_.pin_memory();
  if (weights_.has_value()) {
    weights_ = weights_->pin_memory();
  }
  if (lengths_.has_value()) {
    lengths_ = lengths_->pin_memory();
  }
  if (offsets_.has_value()) {
    offsets_ = offsets_->pin_memory();
  }
}

void KeyedJaggedTensor::sync() {
  length_per_key();
  offset_per_key();
}

//////////////////////////////function wrapper////////////////////////

c10::intrusive_ptr<KeyedJaggedTensor> keyed_jagged_tensor_init(
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
    bool sync = false) {
  return c10::make_intrusive<KeyedJaggedTensor>(
      keys,
      values,
      weights,
      lengths,
      offsets,
      stride,
      length_per_key,
      offset_per_key,
      index_per_key,
      sync);
}

TupleDict to_dict(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->to_dict();
}

const torch::Tensor values(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->values();
}

torch::Tensor offsets(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->offsets();
}

torch::Tensor lengths(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->lengths();
}

c10::optional<torch::Tensor> weights(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->weights();
}

int64_t stride(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->stride();
}

std::vector<int64_t> length_per_key(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->length_per_key();
}

std::vector<int64_t> offset_per_key(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->offset_per_key();
}

torch::Dict<std::string, int64_t> index_per_key(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->index_per_key();
}

TupleOptionalFields all_fields(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return kjt->all_fields();
}

// values, length_per_key, offset_per_key, lengths, stride, weights
std::tuple<
    torch::Tensor,
    std::vector<int64_t>,
    std::vector<int64_t>,
    torch::Tensor,
    int64_t,
    c10::optional<torch::Tensor>>
split_input(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  return std::make_tuple(
      kjt->values(),
      kjt->length_per_key(),
      kjt->offset_per_key(),
      kjt->lengths(),
      kjt->stride(),
      kjt->weights());
}

TupleOptionalFields to(
    const c10::intrusive_ptr<KeyedJaggedTensor>& kjt,
    torch::Device device,
    bool non_blocking = false) {
  return kjt->to(device, non_blocking);
}

void pin_memory(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  kjt->pin_memory();
}

void sync_(const c10::intrusive_ptr<KeyedJaggedTensor>& kjt) {
  kjt->sync();
}

/////////////////////////////ops registry//////////////////////////

TORCH_LIBRARY(torchrec, m) {
  m.class_<KeyedJaggedTensor>("KeyedJaggedTensor")
      .def(torch::init<>())
      .def_pickle(
          [](const c10::intrusive_ptr<KeyedJaggedTensor>& self)
              -> TupleOptionalFields { return self->all_fields(); },
          [](TupleOptionalFields all_fields_)
              -> c10::intrusive_ptr<KeyedJaggedTensor> {
            return c10::make_intrusive<KeyedJaggedTensor>(
                std::get<0>(all_fields_),
                std::get<1>(all_fields_),
                std::get<2>(all_fields_),
                std::get<3>(all_fields_),
                std::get<4>(all_fields_),
                std::get<5>(all_fields_),
                std::get<6>(all_fields_),
                std::get<7>(all_fields_),
                std::get<8>(all_fields_));
          });
  m.def("keyed_jagged_tensor_init", TORCH_FN(keyed_jagged_tensor_init));
  m.def("to_dict", TORCH_FN(to_dict));
  m.def("offsets", TORCH_FN(offsets));
  m.def("values", TORCH_FN(values));
  m.def("lengths", TORCH_FN(lengths));
  m.def("weights", TORCH_FN(weights));
  m.def("stride", TORCH_FN(stride));
  m.def("length_per_key", TORCH_FN(length_per_key));
  m.def("offset_per_key", TORCH_FN(offset_per_key));
  m.def("index_per_key", TORCH_FN(index_per_key));
  m.def("all_fields", TORCH_FN(all_fields));
  m.def("split_input", TORCH_FN(split_input));
  m.def("to", TORCH_FN(to));
  m.def("pin_memory", TORCH_FN(pin_memory));
  m.def("sync", TORCH_FN(sync_));
}

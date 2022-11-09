/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/io.h>

namespace torchrec {

static constexpr std::string_view k_schema_separator = "://";

IO::IO(const std::string& config) {
  auto pos = config.find(k_schema_separator);
  TORCH_CHECK(
      pos != std::string::npos,
      "config string should be schema://cfg_string, cannot find schema");

  std::string schema = config.substr(0, pos);
  std::string rest_cfg = config.substr(pos + k_schema_separator.size());
  auto& reg = IORegistry::Instance();
  provider_ = reg.resolve(schema);
  instance_ = provider_.initialize(rest_cfg.c_str());
}

IO::~IO() {
  if (instance_ == nullptr) {
    return;
  }
  provider_.finalize(instance_);
}

struct FetchContext {
  std::vector<torch::Tensor> tensors;
  std::function<void(std::vector<torch::Tensor>)> on_complete;
  torch::ScalarType scalar_type;
  uint32_t num_optimizer_states;
};

static void on_global_id_fetched(
    void* ctx,
    uint32_t offset,
    uint32_t optimizer_state,
    void* data,
    uint32_t data_len) {
  auto c = reinterpret_cast<FetchContext*>(ctx);
  auto& tensor = c->tensors[offset];
  // non-existed global id
  if (data_len == 0) {
    if (tensor.defined()) {
      tensor = torch::Tensor{};
    }
    return;
  }
  if (!tensor.defined()) {
    size_t elem_size = torch::elementSize(c->scalar_type);
    TORCH_CHECK(data_len % elem_size == 0);
    size_t num_elems = data_len / elem_size;
    tensor = torch::empty(
        {
            static_cast<int64_t>(c->num_optimizer_states),
            static_cast<int64_t>(num_elems),
        },
        c10::TensorOptions().dtype(c->scalar_type));
  }
  void* ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) +
      optimizer_state * data_len);
  memcpy(ptr, data, data_len);
}

static void on_all_fetched(void* ctx) {
  auto c = reinterpret_cast<FetchContext*>(ctx);
  c->on_complete(std::move(c->tensors));
  delete c;
}

void IO::pull(
    const std::string& table_name,
    std::span<const int64_t> global_ids,
    std::span<const int64_t> col_ids,
    uint32_t num_optimizer_states,
    torch::ScalarType type,
    std::function<void(std::vector<torch::Tensor>)> on_fetch_complete) {
  std::unique_ptr<FetchContext> ctx(new FetchContext{
      .on_complete = std::move(on_fetch_complete),
      .scalar_type = type,
      .num_optimizer_states = num_optimizer_states,
  });

  ctx->tensors.resize(
      global_ids.size() * std::max(col_ids.size(), static_cast<size_t>(1)));

  IOPullParameter param{
      .table_name = table_name.c_str(),
      .num_cols = static_cast<uint32_t>(col_ids.size()),
      .num_global_ids = static_cast<uint32_t>(global_ids.size()),
      .col_ids = col_ids.data(),
      .global_ids = global_ids.data(),
      .num_optimizer_states = num_optimizer_states,
      .on_global_id_fetched = on_global_id_fetched,
      .on_all_fetched = on_all_fetched,
  };
  param.on_complete_context = ctx.release();
  provider_.pull(instance_, param);
}

struct PushContext {
  std::function<void()> on_push_complete_;
};

static void OnPushComplete(void* ctx) {
  auto* c = reinterpret_cast<PushContext*>(ctx);
  c->on_push_complete_();
  delete c;
}

void IO::push(
    const std::string& table_name,
    std::span<const int64_t> global_ids,
    std::span<const int64_t> col_ids,
    std::span<const uint32_t> os_ids,
    std::span<const uint8_t> data,
    std::span<const uint64_t> offsets,
    std::function<void()> on_push_complete) {
  std::unique_ptr<PushContext> ctx(new PushContext{
      .on_push_complete_ = std::move(on_push_complete),
  });
  IOPushParameter param{
      .table_name = table_name.c_str(),
      .num_cols = static_cast<uint32_t>(col_ids.size()),
      .num_global_ids = static_cast<uint32_t>(global_ids.size()),
      .col_ids = col_ids.data(),
      .global_ids = global_ids.data(),
      .num_optimizer_states = static_cast<uint32_t>(os_ids.size()),
      .optimizer_state_ids = os_ids.data(),
      .num_offsets = static_cast<uint32_t>(offsets.size()),
      .offsets = offsets.data(),
      .data = data.data(),
      .on_complete_context = ctx.release(),
      .on_push_complete = OnPushComplete,
  };
  provider_.push(instance_, param);
}

} // namespace torchrec

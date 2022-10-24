/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <c10/util/flat_hash_map.h>
#include <dlfcn.h>
#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

namespace tde::details {

struct IOPullParameter {
  const char* table_name;
  uint32_t num_cols;
  uint32_t num_global_ids;
  const int64_t* col_ids;
  const int64_t* global_ids;
  uint32_t num_optimizer_states;
  void* on_complete_context;
  void (*on_global_id_fetched)(
      void* ctx,
      uint32_t offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched)(void* ctx);
};

struct IOPushParameter {
  const char* table_name;
  uint32_t num_cols;
  uint32_t num_global_ids;
  const int64_t* col_ids;
  const int64_t* global_ids;
  uint32_t num_optimizer_states;
  const uint32_t* optimizer_stats_ids;
  uint32_t num_offsets;
  const uint64_t* offsets;
  const void* data;
  void* on_complete_context;
  void (*on_push_complete)(void* ctx);
};

struct IOProvider {
  const char* type;
  void* (*initialize)(const char* cfg);
  void (*pull)(void* instance, IOPullParameter cfg);
  void (*push)(void* instance, IOPushParameter cfg);
  void (*finalize)(void*);
};

class IORegistry {
 public:
  void register_provider(IOProvider provider);
  void register_plugin(const char* filename);
  [[nodiscard]] IOProvider resolve(const std::string& name) const;

  static IORegistry& Instance();

 private:
  IORegistry() = default;
  ska::flat_hash_map<std::string, IOProvider> providers_;
  struct DLCloser {
    void operator()(void* ptr) const;
  };

  using DLPtr = std::unique_ptr<void, DLCloser>;
  std::vector<DLPtr> dls_;
};

} // namespace tde::details

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
  const char* table_name_;
  uint32_t num_cols_;
  uint32_t num_global_ids_;
  const int64_t* col_ids_;
  const int64_t* global_ids_;
  uint32_t num_optimizer_stats_;
  void* on_complete_context_;
  void (*on_global_id_fetched_)(
      void* ctx,
      uint32_t offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched_)(void* ctx);
};

struct IOPushParameter {
  const char* table_name_;
  uint32_t num_cols_;
  uint32_t num_global_ids_;
  const int64_t* col_ids_;
  const int64_t* global_ids_;
  uint32_t num_optimizer_stats_;
  const uint32_t* optimizer_stats_ids_;
  // offsets in bytes
  // data_ptr is 1-D array divided by offsets in bytes.
  // The offsets are jagged and length of offsets array is length + 1.
  //
  // data[global_id * num_cols * num_optimizer_stats_
  //       + col_id * num_optimizer_stats_ + os_id ]
  uint32_t num_offsets_;
  const uint64_t* offsets_;
  const void* data_;
  void* on_complete_context_;
  void (*on_push_complete)(void* ctx);
};

struct IOProvider {
  const char* type_;
  void* (*initialize_)(const char* cfg);
  void (*pull_)(void* instance, IOPullParameter cfg);
  void (*push_)(void* instance, IOPushParameter cfg);
  void (*finalize_)(void*);
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

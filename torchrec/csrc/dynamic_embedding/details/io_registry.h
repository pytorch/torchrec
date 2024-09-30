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
#include <torchrec/csrc/dynamic_embedding/details/io_parameter.h>
#include <memory>
#include <string>
#include <vector>

namespace torchrec {

struct IOProvider {
  const char* type;
  void* (*initialize)(const char* cfg);
  void (*fetch)(void* instance, IOFetchParameter cfg);
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

} // namespace torchrec

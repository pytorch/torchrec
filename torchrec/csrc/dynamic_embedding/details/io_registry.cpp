/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/io_registry.h>

namespace torchrec {

void IORegistry::register_provider(IOProvider provider) {
  std::string type = provider.type;
  auto it = providers_.find(type);
  if (it != providers_.end()) {
    TORCH_WARN("IO provider ", type, " already registered. Ignored this time.");
    return;
  }

  providers_[type] = provider;
}

void IORegistry::register_plugin(const char* filename) {
  DLPtr ptr(dlopen(filename, RTLD_LAZY | RTLD_LOCAL));
  TORCH_CHECK(ptr != nullptr, "cannot load dl ", filename, ", errno ", errno);
  IOProvider provider{};
  auto type_ptr = dlsym(ptr.get(), "IO_type");
  TORCH_CHECK(type_ptr != nullptr, "cannot find IO_type symbol");
  provider.type = *reinterpret_cast<const char**>(type_ptr);

  auto initialize_ptr = dlsym(ptr.get(), "IO_Initialize");
  TORCH_CHECK(initialize_ptr != nullptr, "cannot find IO_Initialize symbol");
  provider.initialize =
      reinterpret_cast<decltype(provider.initialize)>(initialize_ptr);

  auto finalize_ptr = dlsym(ptr.get(), "IO_Finalize");
  TORCH_CHECK(finalize_ptr != nullptr, "cannot find IO_Finalize symbol");
  provider.finalize =
      reinterpret_cast<decltype(provider.finalize)>(finalize_ptr);

  auto pull_ptr = dlsym(ptr.get(), "IO_Pull");
  TORCH_CHECK(pull_ptr != nullptr, "cannot find IO_Pull symbol");
  provider.pull = reinterpret_cast<decltype(provider.pull)>(pull_ptr);

  auto push_ptr = dlsym(ptr.get(), "IO_Push");
  TORCH_CHECK(push_ptr != nullptr, "cannot find IO_Push symbol");
  provider.push = reinterpret_cast<decltype(provider.push)>(push_ptr);

  register_provider(provider);
  dls_.emplace_back(std::move(ptr));
}

void IORegistry::DLCloser::operator()(void* ptr) const {
  if (ptr == nullptr) {
    return;
  }
  TORCH_CHECK(dlclose(ptr) == 0, "cannot close dl library, errno %d", errno);
}

IOProvider IORegistry::resolve(const std::string& name) const {
  auto it = providers_.find(name);
  TORCH_CHECK(
      it != providers_.end(), "IO provider ", name, " is not registered");
  return it->second;
}

IORegistry& IORegistry::Instance() {
  static IORegistry instance;
  return instance;
}

} // namespace torchrec

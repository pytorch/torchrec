#include "tde/details/io_registry.h"
#include "dlfcn.h"
#include "torch/torch.h"
namespace tde::details {

void IORegistry::Register(IOProvider provider) {
  std::string type = provider.type_;
  auto it = providers_.find(type);
  TORCH_CHECK(
      it == providers_.end(), "IO provider %s already registered", type);

  providers_[type] = provider;
}

void IORegistry::RegisterPlugin(const char* filename) {
  DLPtr ptr(dlopen(filename, RTLD_LAZY | RTLD_LOCAL));
  TORCH_CHECK(ptr != nullptr, "cannot load dl %s, errno %d", filename, errno);
  IOProvider provider{};
  auto type_ptr = dlsym(ptr.get(), "IO_type");
  TORCH_CHECK(type_ptr != nullptr, "cannot find IO_type symbol");
  provider.type_ = reinterpret_cast<const char*>(type_ptr);

  auto initialize_ptr = dlsym(ptr.get(), "IO_Initialize");
  TORCH_CHECK(initialize_ptr != nullptr, "cannot find IO_Initialize symbol");
  provider.Initialize =
      reinterpret_cast<decltype(provider.Initialize)>(initialize_ptr);

  auto finalize_ptr = dlsym(ptr.get(), "IO_Finalize");
  TORCH_CHECK(finalize_ptr != nullptr, "cannot find IO_Finalize symbol");
  provider.Finalize =
      reinterpret_cast<decltype(provider.Finalize)>(finalize_ptr);

  auto pull_ptr = dlsym(ptr.get(), "IO_Pull");
  TORCH_CHECK(pull_ptr != nullptr, "cannot find IO_Pull symbol");
  provider.Pull = reinterpret_cast<decltype(provider.Pull)>(pull_ptr);

  Register(provider);
  dls_.emplace_back(std::move(ptr));
}

void IORegistry::DLCloser::operator()(void* ptr) const {
  if (ptr == nullptr) {
    return;
  }
  TORCH_CHECK(dlclose(ptr) == 0, "cannot close dl library, errno %d", errno);
}

IOProvider IORegistry::Resolve(const std::string& name) const {
  auto it = providers_.find(name);
  TORCH_CHECK(it == providers_.end(), "IO provider %s is not registered", name);
  return it->second;
}

IORegistry& IORegistry::Instance() {
  static IORegistry instance;
  return instance;
}

} // namespace tde::details

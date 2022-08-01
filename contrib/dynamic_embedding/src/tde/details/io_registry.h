#pragma once
#include <dlfcn.h>
#include <stdint.h>
#include <memory>
#include <string>
#include <vector>
#include "c10/util/flat_hash_map.h"
namespace tde::details {

struct IOPullParameter {
  const char* table_name_;
  uint32_t num_cols_;
  uint32_t num_global_ids_;
  const int64_t* col_ids_;
  const int64_t* global_ids_;
  uint32_t num_optimizer_stats_;
  int8_t scalar_type_;
  void* on_complete_context_;
  void (*on_global_id_fetched_)(
      void* ctx,
      uint32_t offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched_)(void* ctx);
};

struct IOProvider {
  const char* type_;
  void* (*Initialize)(const char* cfg);
  void (*Pull)(void* instance, IOPullParameter cfg);
  void (*Finalize)(void*);
};

class IORegistry {
 public:
  void Register(IOProvider provider);
  void RegisterPlugin(const char* filename);
  [[nodiscard]] IOProvider Resolve(const std::string& name) const;

  static IORegistry& Instance();
  static void RegisterAllDefaultIOs();

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

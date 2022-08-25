#include <stdint.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>

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

} // namespace tde::details

using namespace tde::details;

class MemoryIO {
 public:
  explicit MemoryIO(const char* cfg) : prefix_(cfg) {}

  void Pull(IOPullParameter param) {
    uint32_t num_global_ids = param.num_global_ids_;
    uint32_t num_cols = param.num_cols_;
    uint32_t num_optimizer_stats = param.num_optimizer_stats_;
    for (uint32_t i = 0; i < num_global_ids; ++i) {
      int64_t gid = param.global_ids_[i];
      for (uint32_t j = 0; j < num_cols; ++j) {
        int64_t cid = param.col_ids_[j];

        uint32_t offset = j + i * num_cols;
        for (uint32_t os_id = 0; os_id < num_optimizer_stats; ++os_id) {
          std::string key = Key(param.table_name_, gid, cid, os_id);
          auto iter = ps_.find(key);

          if (iter == ps_.end()) {
            param.on_global_id_fetched_(
                param.on_complete_context_, offset, os_id, nullptr, 0);
          } else {
            param.on_global_id_fetched_(
                param.on_complete_context_,
                offset,
                os_id,
                iter->second.data(),
                iter->second.size());
          }
        }
      }
    }
    param.on_all_fetched_(param.on_complete_context_);
  }

  void Push(IOPushParameter param) {
    uint32_t num_global_ids = param.num_global_ids_;
    uint32_t num_cols = param.num_cols_;
    uint32_t num_optimizer_stats = param.num_optimizer_stats_;
    for (uint32_t i = 0; i < num_global_ids; ++i) {
      int64_t gid = param.global_ids_[i];
      for (uint32_t j = 0; j < num_cols; ++j) {
        int64_t cid = param.col_ids_[j];
        for (uint32_t k = 0; k < num_optimizer_stats; ++k) {
          uint32_t os_id = param.optimizer_stats_ids_[k];

          uint32_t offset = k + j * num_optimizer_stats +
              i * num_cols * num_optimizer_stats;
          uint64_t beg = param.offsets_[offset];
          uint64_t end = param.offsets_[offset + 1];
          std::string key = Key(param.table_name_, gid, cid, os_id);
          ps_.emplace(
              key,
              std::vector<uint8_t>(
                  reinterpret_cast<const uint8_t*>(param.data_) + beg,
                  reinterpret_cast<const uint8_t*>(param.data_) + end));
        }
      }
    }
    param.on_push_complete(param.on_complete_context_);
  }

 private:
  std::string Key(const char* table_name, int64_t gid, int64_t cid, uint32_t os_id) {
    // should use std::format, sacrifice for lower gcc...
    std::stringstream ss;
    ss << table_name << "_" << gid << "_" << cid << "_" << os_id;
    return ss.str();
  }

  std::string prefix_;
  std::unordered_map<std::string, std::vector<uint8_t>> ps_; 
};

extern "C" {

const char* IO_type = "memory";

void* IO_Initialize(const char* cfg) {
  return new MemoryIO(cfg);
}

void IO_Finalize(void* instance) {
  delete reinterpret_cast<MemoryIO*>(instance);
}

void IO_Pull(void* instance, IOPullParameter cfg) {
  reinterpret_cast<MemoryIO*>(instance)->Pull(cfg);
}

void IO_Push(void* instance, IOPushParameter cfg) {
  reinterpret_cast<MemoryIO*>(instance)->Push(cfg);
}

}

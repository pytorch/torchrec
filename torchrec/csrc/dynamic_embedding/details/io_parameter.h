#include <stdint.h>

namespace torchrec {

using GlobalIDFetchCallback = void (*)(
    void* ctx,
    uint32_t offset,
    uint32_t optimizer_state,
    void* data,
    uint32_t data_len);

struct IOPullParameter {
  const char* table_name;
  uint32_t num_cols;
  uint32_t num_global_ids;
  const int64_t* col_ids;
  const int64_t* global_ids;
  uint32_t num_optimizer_states;
  void* on_complete_context;
  GlobalIDFetchCallback on_global_id_fetched;
  void (*on_all_fetched)(void* ctx);
};

struct IOPushParameter {
  const char* table_name;
  uint32_t num_cols;
  uint32_t num_global_ids;
  const int64_t* col_ids;
  const int64_t* global_ids;
  uint32_t num_optimizer_states;
  const uint32_t* optimizer_state_ids;
  uint32_t num_offsets;
  const uint64_t* offsets;
  const void* data;
  void* on_complete_context;
  void (*on_push_complete)(void* ctx);
};

} // namespace torchrec

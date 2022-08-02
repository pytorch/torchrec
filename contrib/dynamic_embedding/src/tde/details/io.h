#pragma once
#include <cstdint>
#include "tcb/span.hpp"
#include "tde/details/io_registry.h"
#include "tde/details/move_only_function.h"
#include "torch/torch.h"

namespace tde::details {

class IO {
 public:
  explicit IO(const std::string& config);
  ~IO();

  IO(const IO&) = delete;
  IO& operator=(const IO&) = delete;
  IO(IO&&) noexcept = delete;
  IO& operator=(IO&&) noexcept = delete;

  /**
   * Fetch parameter and optimizer states from ParamServer.
   * @param global_ids global ids to fetch
   * @param num_optimizer_states number of optimizer stats to fetch
   * @param type data type
   * @param on_fetch_complete  fetch complete callback. The parameter is
   * a vector, the vector's size is equal to global_ids.size() *
   * max(col_ids.size(), 1). If the parameter server does not contains some
   * parameter, the tensor will be empty. Also, the tensor shape is
   * [num_optimizer_states, embedding_size]. The shape of each global id can be
   * different in some algorithm.
   *
   * @note this method is asynchronous, The col_ids, and global_ids will be
   * copied inside, so it is safe to free col_ids/global_ids before
   * on_fetch_complete.
   */
  void Pull(
      const std::string& table_name,
      tcb::span<const int64_t> global_ids,
      tcb::span<const int64_t> col_ids,
      uint32_t num_optimizer_states,
      torch::ScalarType type,
      MoveOnlyFunction<void(std::vector<torch::Tensor>)> on_fetch_complete);

  /**
   * Push Parameter/Optimizer stats to parameter server.
   * @param table_name
   * @param global_ids
   * @param col_ids empty if no column slices
   * @param os_ids
   * @param data A flatten view of pushing data.
   * data[gid_offset * num_cols * num_os_id + col_offset * num_os_id +
   * os_id_offset]
   *
   * @param offsets
   * @param on_push_complete
   */
  void Push(
      const std::string& table_name,
      tcb::span<const int64_t> global_ids,
      tcb::span<const int64_t> col_ids,
      tcb::span<const uint32_t> os_ids,
      tcb::span<const uint8_t> data,
      tcb::span<const uint64_t> offsets,
      MoveOnlyFunction<void()> on_push_complete);

 private:
  IOProvider provider_{};
  void* instance_{};
};

} // namespace tde::details

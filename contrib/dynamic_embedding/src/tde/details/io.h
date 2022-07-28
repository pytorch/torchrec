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
   * a vector, the vector's size is equal to global_ids.size().
   * If the parameter server does not contains some parameter, the tensor
   * will be empty.
   * Also, the tensor shape is [num_optimizer_states, embedding_size]. The
   * shape of each global id can be different in some algorithm.
   */
  void Pull(
      const std::string& table_name,
      tcb::span<const int64_t> col_ids,
      tcb::span<const int64_t> global_ids,
      uint32_t num_optimizer_states,
      torch::ScalarType type,
      MoveOnlyFunction<void(std::vector<torch::Tensor>)> on_fetch_complete);

  // TODO: Decide push interface
  void Push();

 private:
  IOProvider provider_{};
  void* instance_{};
};

} // namespace tde::details

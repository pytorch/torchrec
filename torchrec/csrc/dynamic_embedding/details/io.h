/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/io_registry.h>
#include <cstdint>
#include <span>

namespace torchrec {

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
   * @param global_ids Global ids to fetch
   * @param col_ids The column id (in term of colum-wise sharding) of
   * the embedding. It will be empty if the embedding is not sharded
   * along the column.
   * @param num_optimizer_states Number of optimizer states to fetch
   * @param type Data type of the embedding.
   * @param on_fetch_complete The complete callback when the fetch complete.
   * The parameter is a vector of tensor, whose size is equal to
   * `global_ids.size() * max(col_ids.size(), 1)`.
   * If the parameter server does not contains some
   * parameter, the tensor will be empty. Also, the tensor shape is
   * [num_optimizer_states, embedding_size]. The shape of each global id can be
   * different in some algorithm.
   *
   * @note This method is asynchronous. The `col_ids`, and `global_ids` will be
   * copied inside, so it is safe to free `col_ids`/`global_ids` before
   * `on_fetch_complete`.
   */
  void fetch(
      const std::string& table_name,
      std::span<const int64_t> global_ids,
      std::span<const int64_t> col_ids,
      uint32_t num_optimizer_states,
      torch::ScalarType type,
      std::function<void(std::vector<torch::Tensor>)> on_fetch_complete);

  /**
   * Push Parameter/Optimizer stats to parameter server.
   * @param global_ids Global ids to push
   * @param col_ids The column id (in term of colum-wise sharding) of
   * the embedding. It will be empty if the embedding is not sharded
   * along the column.
   * @param os_ids The optimizer state id of the corresponding embedding.
   * Its element should be integer within [0, 2].
   * @param data A flatten view of data to push. There will be
   * `global_is.size() * max(col_ids.size(), 1) * os_ids.size()`
   * embedding to push.
   * And to accommodate embedding of different data type, here we convert
   * data into `uint8_t` and the data of the `col_ids[j]` of optimizer state
   * k of `global_id[i]`, is in:
   * `data[begin..end]` where:
   * ```
   * auto x = i * col_ids.size() * os_id.size() + j * os_id.size() + k;
   * auto begin = offsets[x];
   * auto end = offsets[x + 1];
   * ```
   * @param offsets The offset of each embedding in `data`
   * @param on_push_complete The callback when the push finishes.
   */
  void push(
      const std::string& table_name,
      std::span<const int64_t> global_ids,
      std::span<const int64_t> col_ids,
      std::span<const uint32_t> os_ids,
      std::span<const uint8_t> data,
      std::span<const uint64_t> offsets,
      std::function<void()> on_push_complete);

 private:
  IOProvider provider_{};
  void* instance_{};
};

} // namespace torchrec

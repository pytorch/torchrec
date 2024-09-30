/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <hiredis.h>
#include <torchrec/csrc/dynamic_embedding/details/io_parameter.h>
#include <condition_variable>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <thread>

namespace torchrec::redis {

struct Option {
 public:
  std::string host;
  std::string username;
  std::string password;
  uint16_t port{6379};
  uint16_t db{0};
  uint32_t num_io_threads{1};
  std::string prefix;
  uint32_t timeout_ms{10000};
  uint32_t heart_beat_interval_ms{100000};
  uint32_t retry_limit{3};
  uint32_t chunk_size{100};
};

Option parse_option(std::string_view config_str);

namespace helper {
struct ContextDeleter {
  void operator()(void* ctx) {
    if (ctx == nullptr) {
      return;
    }
    redisFree(reinterpret_cast<redisContext*>(ctx));
  }
};
using ContextPtr = std::unique_ptr<redisContext, ContextDeleter>;

struct ReplyDeleter {
  void operator()(void* cmd) {
    if (cmd == nullptr) {
      return;
    }
    freeReplyObject(cmd);
  }
};

using ReplyPtr = std::unique_ptr<redisReply, ReplyDeleter>;

} // namespace helper

class Redis {
 public:
  explicit Redis(Option opt);

  ~Redis();

  void fetch(IOFetchParameter param);

  void push(IOPushParameter param);

 private:
  void start_thread();
  void heartbeat(helper::ContextPtr& connection);
  [[nodiscard]] helper::ContextPtr connect() const;

  void do_fetch(
      uint32_t gid_offset,
      void* fetch_param,
      helper::ContextPtr& connection) const;

  void do_push(
      uint32_t gid_offset,
      void* push_ctx,
      helper::ContextPtr& connection) const;

  void check_status(
      std::string_view label,
      helper::ContextPtr& connection,
      helper::ReplyPtr& reply) const;

  Option opt_;
  std::vector<std::thread> io_threads_;
  std::deque<std::function<void(helper::ContextPtr&)>> jobs_;
  std::condition_variable jobs_not_empty_;
  std::mutex jobs_mutex_;
};

extern "C" {

extern const char* IO_type;

void* IO_Initialize(const char* cfg);
void IO_Finalize(void* instance);
void IO_Fetch(void* instance, IOFetchParameter param);
void IO_Push(void* instance, IOPushParameter param);
}

} // namespace torchrec::redis

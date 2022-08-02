#pragma once
#include <string>
#include <string_view>
#include "hiredis.h"
#include "tde/details/io_registry.h"
#include "tde/details/thread_pool.h"

namespace tde::details::redis_v1 {

struct Option {
 public:
  std::string host_;
  std::string username_;
  std::string password_;
  uint16_t port_{6379};
  uint16_t db_{0};
  uint32_t num_io_threads_{1};
  std::string prefix_;
  uint32_t timeout_ms_{10000};
  uint32_t heart_beat_interval_ms_{100000};
  uint32_t retry_limit_{3};
  uint32_t chunk_size_{100};

  Option() = default;

  static Option Parse(std::string_view config_str) {
    return Option(config_str);
  }

 private:
  Option(std::string_view config_str);
};

namespace redis {
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

} // namespace redis

class RedisV1 {
 public:
  explicit RedisV1(Option opt);

  ~RedisV1();

  void Pull(IOPullParameter param);

  void Push(IOPushParameter param);

 private:
  void StartThread();
  void HeartBeat(redis::ContextPtr& connection);
  [[nodiscard]] redis::ContextPtr Connect() const;

  void DoFetch(
      uint32_t gid_offset,
      void* fetch_param,
      redis::ContextPtr& connection) const;

  void DoPush(
      uint32_t gid_offset,
      void* push_ctx,
      redis::ContextPtr& connection) const;

  void CheckStatus(
      std::string_view label,
      redis::ContextPtr& connection,
      redis::ReplyPtr& reply) const;

  Option opt_;
  std::vector<std::thread> io_threads_;
  std::deque<MoveOnlyFunction<void(redis::ContextPtr&)>> jobs_;
  std::condition_variable jobs_not_empty_;
  std::mutex jobs_mutex_;
};

} // namespace tde::details::redis_v1

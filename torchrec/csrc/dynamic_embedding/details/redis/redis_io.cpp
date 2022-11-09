/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lexy/callback.hpp>
#include <lexy/dsl.hpp>
#include <torchrec/csrc/dynamic_embedding/details/redis/redis_io.h>
#include <torchrec/csrc/dynamic_embedding/details/redis/url.h>
#include <iostream>
#include <variant>

namespace torchrec::redis {

struct NumThreadsOpt {
  uint32_t num_threads_;
};
struct DBOpt {
  uint32_t db_;
};
struct PrefixOpt {
  std::string prefix_;
};

struct TimeoutMsOpt {
  uint32_t timeout_;
};

struct HeartBeatMsOpt {
  uint32_t heartbeat_;
};

struct RetryLimitOpt {
  uint32_t limit_;
};

struct ChunkSizeOpt {
  uint32_t chunk_size_;
};

using OptVar = std::variant<
    NumThreadsOpt,
    DBOpt,
    PrefixOpt,
    TimeoutMsOpt,
    HeartBeatMsOpt,
    RetryLimitOpt,
    ChunkSizeOpt>;

struct OptionSetter {
  void operator()(Option* self, NumThreadsOpt opt) {
    TORCH_CHECK(opt.num_threads_ != 0);
    self->num_io_threads_ = opt.num_threads_;
  }
  void operator()(Option* self, DBOpt opt) {
    self->db_ = opt.db_;
  }
  void operator()(Option* self, PrefixOpt opt) {
    self->prefix_ = std::move(opt.prefix_);
  }
  void operator()(Option* self, TimeoutMsOpt opt) {
    TORCH_CHECK(opt.timeout_ != 0);
    self->timeout_ms_ = opt.timeout_;
  }
  void operator()(Option* self, HeartBeatMsOpt opt) {
    TORCH_CHECK(opt.heartbeat_ != 0);
    self->heart_beat_interval_ms_ = opt.heartbeat_;
  }
  void operator()(Option* self, RetryLimitOpt opt) {
    TORCH_CHECK(opt.limit_ != 0);
    self->retry_limit_ = opt.limit_;
  }
  void operator()(Option* self, ChunkSizeOpt opt) {
    TORCH_CHECK(opt.chunk_size_ != 0);
    self->chunk_size_ = opt.chunk_size_;
  }
};

namespace option_rules {
namespace dsl = lexy::dsl;
struct Integer {
  constexpr static auto rule =
      dsl::integer<uint32_t>(dsl::digits<>.no_leading_zero());
  constexpr static auto value = lexy::construct<uint32_t>;
};

struct NumThreads {
  constexpr static auto rule = LEXY_LIT("num_threads=") >> dsl::p<Integer>;
  constexpr static auto value = lexy::construct<NumThreadsOpt>;
};

struct DB {
  constexpr static auto rule = LEXY_LIT("db=") >> dsl::p<Integer>;
  constexpr static auto value = lexy::construct<DBOpt>;
};

struct Prefix {
  constexpr static auto rule = LEXY_LIT("prefix=") >>
      dsl::capture(dsl::token(dsl::identifier(
          dsl::ascii::alpha_underscore,
          dsl::ascii::alpha_digit_underscore)));
  constexpr static auto value =
      lexy::callback<PrefixOpt>([](auto&& str) -> PrefixOpt {
        return PrefixOpt{std::string(str.data(), str.size())};
      });
};

struct Duration {
  struct UnknownUnit {
    constexpr static auto name = "unknown unit";
  };

  constexpr static auto rule =
      dsl::integer<uint32_t>(dsl::digits<>.no_leading_zero()) >>
      dsl::opt(LEXY_LIT(".") >> dsl::capture(dsl::digits<>)) >>
      (dsl::capture(
           dsl::literal_set(LEXY_LIT("ms"), LEXY_LIT("s"), LEXY_LIT("m"))) |
       dsl::error<UnknownUnit>);

  constexpr static auto value = lexy::callback<uint32_t>(
      [](auto&& dec, std::optional<lexy::lexeme<lexy::_prd>> fp, auto&& unit) {
        double scale;
        auto unit_sv = std::string_view{unit.data(), unit.size()};
        if (unit_sv == "s") {
          scale = 1000;
        } else if (unit_sv == "m") {
          scale = 1000 * 60;
        } else if (unit_sv == "ms") {
          scale = 1;
        } else {
          TORCH_CHECK(false, "unit should in [s, m, ms]");
        }
        double val = dec;
        if (fp.has_value()) {
          double scale_fp = 10.0;
          for (auto& ch : fp.value()) {
            val += (ch - '0') / scale_fp;
            scale_fp *= 10;
          }
        }
        val *= scale;
        auto timeout_ = static_cast<uint32_t>(val);
        return timeout_;
      });
};

struct Timeout {
  constexpr static auto rule = LEXY_LIT("timeout=") >> dsl::p<Duration>;
  constexpr static auto value = lexy::construct<TimeoutMsOpt>;
};

struct HeartBeat {
  constexpr static auto rule = LEXY_LIT("heartbeat=") >> dsl::p<Duration>;
  constexpr static auto value = lexy::construct<HeartBeatMsOpt>;
};

struct RetryLimit {
  constexpr static auto rule = LEXY_LIT("retry_limit=") >> dsl::p<Integer>;
  constexpr static auto value = lexy::construct<RetryLimitOpt>;
};

struct ChunkSize {
  constexpr static auto rule = LEXY_LIT("chunk_size=") >> dsl::p<Integer>;
  constexpr static auto value = lexy::construct<ChunkSizeOpt>;
};

struct UnknownOption {
  constexpr static auto name = "unknown option";
};

struct Option {
  constexpr static auto rule = dsl::p<NumThreads> | dsl::p<DB> |
      dsl::p<Prefix> | dsl::p<Timeout> | dsl::p<HeartBeat> |
      dsl::p<RetryLimit> | dsl::p<ChunkSize> | dsl::error<UnknownOption>;
  constexpr static auto value = lexy::construct<OptVar>;
};

struct Options {
  constexpr static auto rule =
      dsl::list(dsl::p<Option>, dsl::sep(LEXY_LIT("&&")));

  constexpr static auto value = lexy::as_list<std::vector<OptVar>>;
};

} // namespace option_rules

Option::Option(std::string_view config_str) {
  auto url = url_parser::parse_url(config_str);
  if (url.auth_.has_value()) {
    username_ = std::move(url.auth_->username_);
    if (url.auth_->password_.has_value()) {
      password_ = std::move(url.auth_->password_.value());
    }
  }

  host_ = std::move(url.host_);

  if (url.port_.has_value()) {
    port_ = url.port_.value();
  }

  if (url.param_.has_value()) {
    std::ostringstream err_oss_;
    url_parser::ErrorCollector collector{err_oss_};

    auto result = lexy::parse<option_rules::Options>(
        lexy::string_input(url.param_.value()), collector);
    auto err_str = err_oss_.str();

    TORCH_CHECK(
        result.has_value() && err_str.empty(), "parse param error ", err_str);

    for (auto&& opt_var : result.value()) {
      std::visit(
          [this](auto&& opt) {
            OptionSetter setter;
            setter(this, std::move(opt));
          },
          opt_var);
    }
  }
}

Redis::Redis(Option opt) : opt_(std::move(opt)) {
  TORCH_CHECK(opt_.num_io_threads_ != 0, "num_io_threads must not be empty");
  TORCH_CHECK(
      opt_.heart_beat_interval_ms_ != 0,
      "heart beat interval must not be zero.");
  for (size_t i = 0; i < opt_.num_io_threads_; ++i) {
    start_thread();
  }
}

void Redis::start_thread() {
  auto connection = Connect();
  heartbeat(connection);

  io_threads_.emplace_back(
      [connection = std::move(connection), this]() mutable {
        std::chrono::milliseconds heart_beat(opt_.heart_beat_interval_ms_);
        while (true) {
          std::function<void(redis::ContextPtr&)> todo;
          bool heartbeat_timeout;
          {
            std::unique_lock<std::mutex> lock(this->jobs_mutex_);
            heartbeat_timeout = !jobs_not_empty_.wait_for(
                lock, heart_beat, [this] { return !jobs_.empty(); });
            if (!heartbeat_timeout) {
              todo = std::move(jobs_.front());
              jobs_.pop_front();
            }
          }

          if (heartbeat_timeout) {
            heartbeat(connection);
            continue;
          }

          if (!todo) {
            break;
          }
          todo(connection);
        }
      });
}

void Redis::heartbeat(redis::ContextPtr& connection) {
  for (uint32_t retry = 0; retry < opt_.retry_limit_; ++retry) {
    try {
      auto reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
          redisCommand(connection.get(), "PING")));
      TORCH_CHECK(
          reply && reply->type == REDIS_REPLY_STRING,
          "Ping should return string");
      auto rsp = std::string_view(reply->str, reply->len);
      TORCH_CHECK(rsp == "PONG", "ping/pong error");
    } catch (...) {
      // reconnect if heart beat error
      connection = Connect();
    }
  }
}

redis::ContextPtr Redis::Connect() const {
  redis::ContextPtr connection;
  if (opt_.timeout_ms_ == 0) {
    connection =
        redis::ContextPtr(redisConnect(opt_.host_.c_str(), opt_.port_));
  } else {
    struct timeval interval {};
    interval.tv_sec = opt_.timeout_ms_ / 1000;
    interval.tv_usec = opt_.timeout_ms_ % 1000 * 1000;
    connection = redis::ContextPtr(
        redisConnectWithTimeout(opt_.host_.c_str(), opt_.port_, interval));
  }
  TORCH_CHECK(
      !connection->err,
      "connect to %s:%d error occurred %s",
      opt_.host_,
      opt_.port_,
      connection->errstr);

  if (!opt_.password_.empty()) {
    redis::ReplyPtr reply;
    if (opt_.username_.empty()) {
      reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
          redisCommand(connection.get(), "AUTH %s", opt_.password_.c_str())));
    } else {
      reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(redisCommand(
          connection.get(),
          "AUTH %s %s",
          opt_.username_.c_str(),
          opt_.password_.c_str())));
    }
    check_status("auth error", connection, reply);
  }

  if (opt_.db_ != 0) {
    auto reply = redis::ReplyPtr(reinterpret_cast<redisReply*>(
        redisCommand(connection.get(), "SELECT %d", opt_.db_)));
    check_status("select db error", connection, reply);
  }

  return connection;
}

Redis::~Redis() {
  for (uint32_t i = 0; i < opt_.num_io_threads_; ++i) {
    jobs_.emplace_back();
  }
  jobs_not_empty_.notify_all();
  for (auto& th : io_threads_) {
    th.join();
  }
}

static uint32_t CalculateChunkSizeByGlobalIDs(
    uint32_t chunk_size,
    uint32_t num_cols,
    uint32_t num_os) {
  static constexpr uint32_t low = 1;
  return std::max(
      chunk_size / std::max(num_cols, low) / std::max(num_os, low), low);
}

struct RedisPullContext {
  std::atomic<uint32_t> num_complete_ids{0};
  uint32_t chunk_size;
  std::string table_name;
  std::vector<int64_t> global_ids;
  std::vector<int64_t> col_ids;
  uint32_t num_optimizer_states;
  void* on_complete_context;
  void (*on_global_id_fetched)(
      void* ctx,
      uint32_t gid_offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched)(void* ctx);

  explicit RedisPullContext(uint32_t chunk_size, IOPullParameter param)
      : chunk_size(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols,
            param.num_optimizer_states)),
        table_name(param.table_name),
        global_ids(param.global_ids, param.global_ids + param.num_global_ids),
        num_optimizer_states(param.num_optimizer_states),
        on_complete_context(param.on_complete_context),
        on_global_id_fetched(param.on_global_id_fetched),
        on_all_fetched(param.on_all_fetched) {
    if (param.num_cols == 0) {
      col_ids.emplace_back(-1);
    } else {
      col_ids =
          std::vector<int64_t>(param.col_ids, param.col_ids + param.num_cols);
    }
  }
};

void Redis::pull(IOPullParameter param) {
  auto* fetch_param = new RedisPullContext(opt_.chunk_size_, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids;
         i += fetch_param->chunk_size) {
      jobs_.emplace_back([i, fetch_param, this](redis::ContextPtr& connection) {
        do_fetch(i, fetch_param, connection);
      });
    }
  }
  jobs_not_empty_.notify_all();
}

void Redis::do_fetch(
    uint32_t gid_offset,
    void* fetch_param_void,
    redis::ContextPtr& connection) const {
  auto& fetch_param = *reinterpret_cast<RedisPullContext*>(fetch_param_void);

  uint32_t end = std::min(
      gid_offset + fetch_param.chunk_size,
      static_cast<uint32_t>(fetch_param.global_ids.size()));

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = fetch_param.global_ids[i];
      for (uint32_t j = 0; j < fetch_param.col_ids.size(); ++j) {
        auto& col_id = fetch_param.col_ids[j];
        for (uint32_t os_id = 0; os_id < fetch_param.num_optimizer_states;
             ++os_id) {
          callback(i * fetch_param.col_ids.size() + j, gid, col_id, os_id);
        }
      }
    }
  };

  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    redisAppendCommand(
        connection.get(),
        "GET %s_table_%s_gid_%d_cid_%d_osid_%d",
        opt_.prefix_.c_str(),
        fetch_param.table_name.c_str(),
        gid,
        col_id,
        os_id);
  });

  void* reply;
  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    int status = redisGetReply(connection.get(), &reply);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host_,
        opt_.port_);
    auto reply_ptr = redis::ReplyPtr(reinterpret_cast<redisReply*>(reply));

    if (reply_ptr->type == REDIS_REPLY_NIL) {
      fetch_param.on_global_id_fetched(
          fetch_param.on_complete_context, offset, os_id, nullptr, 0);
    } else {
      fetch_param.on_global_id_fetched(
          fetch_param.on_complete_context,
          offset,
          os_id,
          reply_ptr->str,
          reply_ptr->len);
    }
  });

  uint32_t n = end - gid_offset;
  uint32_t target = fetch_param.global_ids.size();

  if (fetch_param.num_complete_ids.fetch_add(n) + n ==
      target) { // last fetch complete
    fetch_param.on_all_fetched(fetch_param.on_complete_context);
    delete &fetch_param;
  }
}

struct RedisPushContext {
  std::atomic<uint32_t> num_complete_ids{0};
  uint32_t chunk_size;
  std::string table_name;
  std::span<const int64_t> global_ids;
  std::vector<int64_t> col_ids;
  std::span<const uint32_t> os_ids;
  std::span<const uint64_t> offsets;
  const void* data;
  void* on_complete_context;
  void (*on_push_complete)(void*);

  RedisPushContext(uint32_t chunk_size, IOPushParameter param)
      : chunk_size(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols,
            param.num_optimizer_states)),
        table_name(param.table_name),
        global_ids(param.global_ids, param.num_global_ids),
        os_ids(param.optimizer_state_ids, param.num_optimizer_states),
        offsets(param.offsets, param.num_offsets),
        data(param.data),
        on_complete_context(param.on_complete_context),
        on_push_complete(param.on_push_complete) {
    if (param.num_cols != 0) {
      col_ids =
          std::vector<int64_t>(param.col_ids, param.col_ids + param.num_cols);
    } else {
      col_ids.emplace_back(-1);
    }
  }
};

void Redis::push(IOPushParameter param) {
  auto* ctx = new RedisPushContext(opt_.chunk_size_, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids; i += ctx->chunk_size) {
      jobs_.emplace_back([i, ctx, this](redis::ContextPtr& connection) {
        do_push(i, ctx, connection);
      });
    }
  }
  jobs_not_empty_.notify_all();
}
void Redis::do_push(
    uint32_t gid_offset,
    void* push_ctx_ptr,
    redis::ContextPtr& connection) const {
  auto& push_ctx = *reinterpret_cast<RedisPushContext*>(push_ctx_ptr);

  uint32_t end = gid_offset + push_ctx.chunk_size;
  if (end > push_ctx.global_ids.size()) {
    end = push_ctx.global_ids.size();
  }

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = push_ctx.global_ids[i];
      for (uint32_t j = 0; j < push_ctx.col_ids.size(); ++j) {
        int64_t cid = push_ctx.col_ids[j];
        for (uint32_t k = 0; k < push_ctx.os_ids.size(); ++k) {
          uint32_t os_id = push_ctx.os_ids[k];

          uint32_t offset = k + j * push_ctx.os_ids.size() +
              i * push_ctx.col_ids.size() * push_ctx.os_ids.size();
          callback(offset, gid, cid, os_id);
        }
      }
    }
  };

  loop([&](uint32_t o, int64_t gid, int64_t cid, uint32_t os_id) {
    uint64_t beg = push_ctx.offsets[o];
    uint64_t end = push_ctx.offsets[o + 1];

    redisAppendCommand(
        connection.get(),
        "SET %s_table_%s_gid_%d_cid_%d_osid_%d %b",
        opt_.prefix_.c_str(),
        push_ctx.table_name.c_str(),
        gid,
        cid,
        os_id,
        reinterpret_cast<const uint8_t*>(push_ctx.data) + beg,
        static_cast<size_t>(end - beg));
  });

  void* replay_ptr;
  loop([&](...) {
    int status = redisGetReply(connection.get(), &replay_ptr);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host_,
        opt_.port_);
    redis::ReplyPtr reply(reinterpret_cast<redisReply*>(replay_ptr));
    check_status("reply should be ok", connection, reply);
  });

  uint32_t n = end - gid_offset;
  uint32_t target = push_ctx.global_ids.size();
  if (push_ctx.num_complete_ids.fetch_add(n) + n == target) {
    push_ctx.on_push_complete(push_ctx.on_complete_context);
    delete &push_ctx;
  }
}
void Redis::check_status(
    std::string_view label,
    redis::ContextPtr& connection,
    redis::ReplyPtr& reply) const {
  TORCH_CHECK(
      connection->err == 0,
      label,
      " connection error: (",
      connection->errstr,
      "), from redis://",
      opt_.host_,
      ":",
      opt_.port_);

  TORCH_CHECK(
      reply->type == REDIS_REPLY_STATUS,
      label,
      " reply should be status, but actual type is ",
      reply->type,
      ". from redis://",
      opt_.host_,
      ":",
      opt_.port_);

  auto status = std::string_view{reply->str, reply->len};
  TORCH_CHECK(
      status == "OK",
      label,
      " reply status should be OK, but actual is ",
      status,
      ". from redis://",
      opt_.host_,
      ":",
      opt_.port_);
}

extern "C" {

const char* IO_type = "redis";

void* IO_Initialize(const char* cfg) {
  auto opt = Option::parse(cfg);
  return new Redis(opt);
}

void IO_Finalize(void* instance) {
  delete reinterpret_cast<Redis*>(instance);
}

void IO_Pull(void* instance, IOPullParameter param) {
  reinterpret_cast<Redis*>(instance)->pull(param);
}

void IO_Push(void* instance, IOPushParameter param) {
  reinterpret_cast<Redis*>(instance)->push(param);
}
}

} // namespace torchrec::redis

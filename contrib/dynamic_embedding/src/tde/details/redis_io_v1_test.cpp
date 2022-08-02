#include "gtest/gtest.h"
#include "tcb/span.hpp"
#include "tde/details/notification.h"
#include "tde/details/redis_io_v1.h"

namespace tde::details::redis_v1 {

TEST(TDE, redis_v1_Option) {
  auto opt = Option::Parse("192.168.3.1:3948/?db=3&&num_threads=2");
  ASSERT_EQ(opt.host_, "192.168.3.1");
  ASSERT_EQ(opt.port_, 3948);
  ASSERT_EQ(opt.db_, 3);
  ASSERT_EQ(opt.num_io_threads_, 2);
  ASSERT_TRUE(opt.prefix_.empty());
}

struct PullContext {
  Notification* notification_;
  std::function<void(uint32_t, uint32_t, void*, uint32_t)> on_data_;
};

TEST(TDE, redis_v1_push_pull) {
  auto opt = Option::Parse("127.0.0.1:6379");
  RedisV1 redis(opt);

  constexpr static int64_t global_ids[] = {1, 3, 4};
  constexpr static uint32_t os_ids[] = {0};
  constexpr static float params[] = {1, 2, 3, 4, 5, 9, 8, 1};
  constexpr static uint64_t offsets[] = {
      0 * sizeof(float),
      2 * sizeof(float),
      4 * sizeof(float),
      6 * sizeof(float),
      8 * sizeof(float)};

  Notification notification;

  IOPushParameter push{
      .table_name_ = "table",
      .num_global_ids_ = sizeof(global_ids) / sizeof(global_ids[0]),
      .global_ids_ = global_ids,
      .num_optimizer_stats_ = sizeof(os_ids) / sizeof(os_ids[0]),
      .optimizer_stats_ids_ = os_ids,
      .num_offsets_ = sizeof(offsets) / sizeof(offsets[0]),
      .offsets_ = offsets,
      .data_ = params,
      .on_complete_context_ = &notification,
      .on_push_complete =
          +[](void* ctx) {
            auto* notification = reinterpret_cast<Notification*>(ctx);
            notification->Done();
          },
  };
  redis.Push(push);

  notification.Wait();

  notification.Clear();

  PullContext ctx{
      .notification_ = &notification,
      .on_data_ =
          [&](uint32_t offset, uint32_t os_id, void* data, uint32_t len) {
            ASSERT_EQ(os_id, 0);
            uint32_t param_len = 2;
            ASSERT_EQ(len, sizeof(float) * param_len);
            auto actual =
                tcb::span<const float>(reinterpret_cast<const float*>(data), 2);

            auto expect = tcb::span<const float>(
                reinterpret_cast<const float*>(&params[offset * param_len]), 2);

            ASSERT_EQ(expect[0], actual[0]);
            ASSERT_EQ(expect[1], actual[1]);
          }};

  IOPullParameter pull{
      .table_name_ = "table",
      .num_global_ids_ = sizeof(global_ids) / sizeof(global_ids[0]),
      .global_ids_ = global_ids,
      .num_optimizer_stats_ = sizeof(os_ids) / sizeof(os_ids[0]),
      .on_complete_context_ = &ctx,
      .on_global_id_fetched_ =
          +[](void* ctx,
              uint32_t offset,
              uint32_t os_id,
              void* data,
              uint32_t len) {
            auto c = reinterpret_cast<PullContext*>(ctx);
            c->on_data_(offset, os_id, data, len);
          },
      .on_all_fetched_ =
          +[](void* ctx) {
            auto c = reinterpret_cast<PullContext*>(ctx);
            c->notification_->Done();
          }};
  redis.Pull(pull);
  notification.Wait();
}
} // namespace tde::details::redis_v1

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/notification.h>
#include <torchrec/csrc/dynamic_embedding/details/redis/redis_io.h>

namespace torchrec::redis {

TEST(TDE, redis_Option) {
  auto opt = Option::parse(
      "192.168.3.1:3948/?db=3&&num_threads=2&&timeout=3s&&chunk_size=3000");
  ASSERT_EQ(opt.host_, "192.168.3.1");
  ASSERT_EQ(opt.port_, 3948);
  ASSERT_EQ(opt.db_, 3);
  ASSERT_EQ(opt.num_io_threads_, 2);
  ASSERT_EQ(opt.chunk_size_, 3000);
  ASSERT_EQ(opt.timeout_ms_, 3000);
  ASSERT_TRUE(opt.prefix_.empty());
}

TEST(TDE, redis_Option_ParseError) {
  ASSERT_ANY_THROW(
      Option::parse("192.168.3.1:3948/?db=3&&no_opt=3000&&num_threads=2"));
  ASSERT_ANY_THROW(Option::parse("192.168.3.1:3948/?timeout=3d"));
}

struct PullContext {
  Notification* notification_;
  std::function<void(uint32_t, uint32_t, void*, uint32_t)> on_data_;
};

TEST(TDE, redis_push_pull) {
  auto opt = Option::parse("127.0.0.1:6379");
  Redis redis(opt);

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
      .table_name = "table",
      .num_global_ids = sizeof(global_ids) / sizeof(global_ids[0]),
      .global_ids = global_ids,
      .num_optimizer_states = sizeof(os_ids) / sizeof(os_ids[0]),
      .optimizer_state_ids = os_ids,
      .num_offsets = sizeof(offsets) / sizeof(offsets[0]),
      .offsets = offsets,
      .data = params,
      .on_complete_context = &notification,
      .on_push_complete =
          +[](void* ctx) {
            auto* notification = reinterpret_cast<Notification*>(ctx);
            notification->done();
          },
  };
  redis.push(push);

  notification.wait();

  notification.clear();

  PullContext ctx{
      .notification_ = &notification,
      .on_data_ =
          [&](uint32_t offset, uint32_t os_id, void* data, uint32_t len) {
            ASSERT_EQ(os_id, 0);
            uint32_t param_len = 2;
            ASSERT_EQ(len, sizeof(float) * param_len);
            auto actual =
                std::span<const float>(reinterpret_cast<const float*>(data), 2);

            auto expect = std::span<const float>(
                reinterpret_cast<const float*>(&params[offset * param_len]), 2);

            ASSERT_EQ(expect[0], actual[0]);
            ASSERT_EQ(expect[1], actual[1]);
          }};

  IOPullParameter pull{
      .table_name = "table",
      .num_global_ids = sizeof(global_ids) / sizeof(global_ids[0]),
      .global_ids = global_ids,
      .num_optimizer_states = sizeof(os_ids) / sizeof(os_ids[0]),
      .on_complete_context = &ctx,
      .on_global_id_fetched =
          +[](void* ctx,
              uint32_t offset,
              uint32_t os_id,
              void* data,
              uint32_t len) {
            auto c = reinterpret_cast<PullContext*>(ctx);
            c->on_data_(offset, os_id, data, len);
          },
      .on_all_fetched =
          +[](void* ctx) {
            auto c = reinterpret_cast<PullContext*>(ctx);
            c->notification_->done();
          }};
  redis.pull(pull);
  notification.wait();
}
} // namespace torchrec::redis

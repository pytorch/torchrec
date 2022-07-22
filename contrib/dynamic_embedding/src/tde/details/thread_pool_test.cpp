#include "gtest/gtest.h"
#include "tde/details/thread_pool.h"

namespace tde::details {

TEST(tde, ThreadPool) {
  ThreadPool pool(1);
  auto fut = pool.Enqueue([]() -> int { return 42; });
  ASSERT_EQ(42, fut.get());

  auto fut2 = pool.Enqueue(
      [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
  fut2.wait();
}

} // namespace tde::details

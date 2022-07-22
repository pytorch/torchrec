#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "tde/details/move_only_function.h"

namespace tde::details {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) noexcept = delete;
  ThreadPool& operator=(ThreadPool&& o) noexcept = delete;

  template <typename Callable>
  auto Enqueue(Callable callable) -> std::future<std::result_of_t<Callable()>> {
    using R = std::result_of_t<Callable()>;
    auto task = std::make_unique<std::packaged_task<R()>>(std::move(callable));
    auto future = task->get_future();
    PushJob([t = std::move(task)] { (*t)(); });
    return future;
  }

 private:
  void ThreadMain();
  void PushJob(MoveOnlyFunction<void()> job);

  std::vector<std::thread> threads_;
  std::deque<MoveOnlyFunction<void()>> jobs_;
  std::mutex jobs_mtx_;
  std::condition_variable not_empty_cv_;
};

} // namespace tde::details

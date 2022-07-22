#include "thread_pool.h"

namespace tde::details {

ThreadPool::ThreadPool(size_t num_threads) {
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this] { ThreadMain(); });
  }
}

ThreadPool::~ThreadPool() {
  for (size_t i = 0; i < threads_.size(); ++i) {
    PushJob({});
  }
  for (auto& th : threads_) {
    th.join();
  }
}

void ThreadPool::ThreadMain() {
  while (true) {
    MoveOnlyFunction<void()> job;
    {
      std::unique_lock<std::mutex> lock(jobs_mtx_);
      not_empty_cv_.wait(lock, [this] { return !jobs_.empty(); });
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    if (!job) {
      break;
    }
    job();
  }
}

void ThreadPool::PushJob(MoveOnlyFunction<void()> job) {
  {
    std::lock_guard<std::mutex> guard(jobs_mtx_);
    jobs_.emplace_back(std::move(job));
  }
  not_empty_cv_.notify_one();
}

} // namespace tde::details

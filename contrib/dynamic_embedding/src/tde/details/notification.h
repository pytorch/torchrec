#pragma once
#include <condition_variable>
#include <mutex>

namespace tde::details {

/**
 * Multi-thread notification
 */
class Notification {
 public:
  void Done();
  void Wait();


 private:
  bool set_{false};
  std::mutex mtx_;
  std::condition_variable cv_;
};

} // namespace tde::details

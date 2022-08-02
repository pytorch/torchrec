#include "notification.h"

namespace tde::details {
void Notification::Done() {
  {
    std::lock_guard<std::mutex> guard(mtx_);
    set_ = true;
  }
  cv_.notify_all();
}
void Notification::Wait() {
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [this] { return set_; });
}
} // namespace tde::details

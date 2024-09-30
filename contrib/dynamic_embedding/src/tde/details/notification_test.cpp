#include <thread>
#include "gtest/gtest.h"
#include "tde/details/notification.h"
namespace tde::details {
TEST(TDE, notification) {
  Notification notification;
  std::thread th([&] { notification.Done(); });
  notification.Wait();
  th.join();
}
} // namespace tde::details

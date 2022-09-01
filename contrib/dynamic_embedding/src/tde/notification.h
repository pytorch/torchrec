#pragma once
#include <torch/torch.h>
#include "tde/details/notification.h"

namespace tde {

class Notification : public torch::CustomClassHolder {
 public:
  void Done() {
    return notification_.Done();
  }
  void Wait() {
    return notification_.Wait();
  }

 private:
  details::Notification notification_;
};

} // namespace tde

#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>

namespace tde {

class IDTransformer : public torch::CustomClassHolder {
 public:
  explicit IDTransformer(int64_t val) : val_(val) {}
  int64_t val() const {
    return val_;
  }

 private:
  int64_t val_;
};

} // namespace tde

#pragma once
#include <torch/torch.h>

namespace tde {

class TensorList : public torch::CustomClassHolder {
  using Container = std::vector<torch::Tensor>;

 public:
  TensorList() = default;

  void push_back(at::Tensor tensor) {
    tensors_.push_back(tensor);
  }
  int64_t size() const {
    return tensors_.size();
  }
  torch::Tensor& operator[](int64_t index) {
    return tensors_[index];
  }

  Container::const_iterator begin() const {
    return tensors_.begin();
  }
  Container::const_iterator end() const {
    return tensors_.end();
  }

 private:
  Container tensors_;
};

} // namespace tde

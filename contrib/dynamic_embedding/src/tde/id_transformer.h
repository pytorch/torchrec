#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/id_transformer_variant.h"

namespace tde {

struct TransformResult : public torch::CustomClassHolder {
  TransformResult(bool success, torch::Tensor ids_to_fetch) :
    success_(success), ids_to_fetch_(ids_to_fetch) {}
  bool success_;
  torch::Tensor ids_to_fetch_;
};

class TensorList: public torch::CustomClassHolder {
 public:
  TensorList() = default;

  void push_back(at::Tensor tensor) { tensors_.push_back(tensor); }
  int64_t size() const { return tensors_.size(); }
  torch::Tensor operator[](int64_t index) { return tensors_[index]; }

 private:
  std::vector<torch::Tensor> tensors_;
};

class IDTransformer : public torch::CustomClassHolder {
 public:
  IDTransformer(int64_t num_embeddings, nlohmann::json json);
  c10::intrusive_ptr<TransformResult> Transform(
      c10::intrusive_ptr<TensorList> global_ids,
      c10::intrusive_ptr<TensorList> cache_ids,
      int64_t time);

  torch::Tensor Evict(int64_t num_to_evict);

 private:
  nlohmann::json json_;
  details::IDTransformer transformer_;

  std::atomic<int64_t> num_ids_to_fetch_;
  std::vector<int64_t> ids_to_fetch_;
};

} // namespace tde

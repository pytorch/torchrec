#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/id_transformer_variant.h"
#include "tde/tensor_list.h"

namespace tde {

struct TransformResult : public torch::CustomClassHolder {
  TransformResult(bool success, torch::Tensor ids_to_fetch)
      : success_(success), ids_to_fetch_(ids_to_fetch) {}
  bool success_;
  torch::Tensor ids_to_fetch_;
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
  details::IDTransformer transformer_;
  std::vector<int64_t> ids_to_fetch_;
};

} // namespace tde

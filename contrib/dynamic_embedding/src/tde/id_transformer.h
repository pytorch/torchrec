#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/id_transformer_variant.h"

namespace tde {

class IDTransformer : public torch::CustomClassHolder {
 public:
  IDTransformer(int64_t num_embeddings, nlohmann::json json);
  std::tuple<int64_t, torch::Tensor> Transform(
      torch::Tensor global_ids,
      torch::Tensor cache_ids,
      int64_t time);

  torch::Tensor Evict(int64_t num_to_evict);

 private:
  nlohmann::json json_;
  details::IDTransformer transformer_;

  std::atomic<int64_t> num_ids_to_fetch_;
  std::vector<int64_t> ids_to_fetch_;
};

} // namespace tde

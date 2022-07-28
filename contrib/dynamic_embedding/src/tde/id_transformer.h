#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "tde/details/mixed_lfu_lru_strategy.h"
#include "tde/details/multithreaded_id_transformer.h"
#include "tde/details/no_fetcher.h"

namespace tde {

class IDTransformer : public torch::CustomClassHolder {
 public:
  IDTransformer(int64_t num_embedding, size_t num_threads);
  int64_t Transform(torch::Tensor global_ids, torch::Tensor cache_ids);

 private:
  using LXURecord = details::MixedLFULRUStrategy::lxu_record_t;

  details::MultiThreadedIDTransformer<details::NaiveIDTransformer<LXURecord>>
      transformer_;
  details::MixedLFULRUStrategy strategy_;
  details::NoFetcher fetcher_;
};

} // namespace tde

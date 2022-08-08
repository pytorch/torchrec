#include "tde/details/io_registry.h"
#include "tde/id_transformer.h"
#include "torch/torch.h"
namespace tde {
TORCH_LIBRARY(tde, m) {
  details::IORegistry::RegisterAllDefaultIOs();
  m.class_<TransformResult>("TransformResult")
      .def_readonly("num_transformed", &TransformResult::num_transformed_)
      .def_readonly("ids_to_fetch", &TransformResult::ids_to_fetch_);

  m.class_<IDTransformer>("IDTransformer")
      .def(torch::init([](int64_t num_embedding, std::string config) {
        nlohmann::json json = nlohmann::json::parse(config);
        return c10::make_intrusive<IDTransformer>(
            num_embedding, std::move(json));
      }))
      .def("transform", &IDTransformer::Transform)
      .def("evict", &IDTransformer::Evict);
}
} // namespace tde

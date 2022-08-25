#include "tde/id_transformer.h"
#include "tde/ps.h"

#include "torch/torch.h"
namespace tde {
TORCH_LIBRARY(tde, m) {
  details::IORegistry::RegisterAllDefaultIOs();

  m.def("register_io", [] (std::string name) {
    details::IORegistry::Instance().RegisterPlugin(name.c_str());
  });

  m.class_<TransformResult>("TransformResult")
      .def_readonly("success", &TransformResult::success_)
      .def_readonly("ids_to_fetch", &TransformResult::ids_to_fetch_);

  m.class_<TensorList>("TensorList")
      .def(torch::init([] () {
        return c10::make_intrusive<TensorList>();
      }))
      .def("append", &TensorList::push_back)
      .def("__len__", &TensorList::size)
      .def("__getitem__", &TensorList::operator[]);

  m.class_<IDTransformer>("IDTransformer")
      .def(torch::init([](int64_t num_embedding, std::string config) {
        nlohmann::json json = nlohmann::json::parse(config);
        return c10::make_intrusive<IDTransformer>(
            num_embedding, std::move(json));
      }))
      .def("transform", &IDTransformer::Transform)
      .def("evict", &IDTransformer::Evict);

  m.class_<LocalShardList>("LocalShardList")
      .def(torch::init([] () {
        return c10::make_intrusive<LocalShardList>();
      }))
      .def("append", &LocalShardList::emplace_back);

  m.class_<PS>("PS")
      .def(torch::init<std::string, c10::intrusive_ptr<LocalShardList>, int64_t, int64_t, std::string>())
      .def("fetch", &PS::Fetch)
      .def("evict", &PS::Evict);

}
} // namespace tde

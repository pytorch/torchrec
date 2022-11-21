/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>

#include <torchrec/csrc/dynamic_embedding/details/io_registry.h>
#include <torchrec/csrc/dynamic_embedding/id_transformer.h>

namespace torchrec {
TORCH_LIBRARY(tde, m) {
  m.def("register_io", [](const std::string& name) {
    IORegistry::Instance().register_plugin(name.c_str());
  });

  m.class_<TransformResult>("TransformResult")
      .def_readonly("success", &TransformResult::success)
      .def_readonly("ids_to_fetch", &TransformResult::ids_to_fetch);

  m.class_<TensorList>("TensorList")
      .def(torch::init([]() { return c10::make_intrusive<TensorList>(); }))
      .def("append", &TensorList::push_back)
      .def("__len__", &TensorList::size)
      .def("__getitem__", &TensorList::operator[]);

  m.class_<IDTransformer>("IDTransformer")
      .def(torch::init([](int64_t num_embedding,
                          const std::string& id_transformer_type,
                          const std::string& lxu_strategy_type,
                          int64_t min_used_freq_power = 5) {
        LXUStrategyVariant strategy(
            lxu_strategy_type, static_cast<uint16_t>(min_used_freq_power));
        IDTransformerVariant transformer(
            std::move(strategy), num_embedding, id_transformer_type);
        return c10::make_intrusive<IDTransformer>(std::move(transformer));
      }))
      .def("transform", &IDTransformer::transform)
      .def("evict", &IDTransformer::evict)
      .def("save", &IDTransformer::save);
}
} // namespace torchrec

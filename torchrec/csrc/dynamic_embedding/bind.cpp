/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>

#include <torchrec/csrc/dynamic_embedding/details/io_registry.h>
#include <torchrec/csrc/dynamic_embedding/id_transformer_wrapper.h>

namespace torchrec {
TORCH_LIBRARY(tde, m) {
  m.def("register_io", [](const std::string& name) {
    IORegistry::Instance().register_plugin(name.c_str());
  });

  m.class_<TransformResult>("TransformResult")
      .def_readonly("success", &TransformResult::success)
      .def_readonly("ids_to_fetch", &TransformResult::ids_to_fetch);

  m.class_<IDTransformerWrapper>("IDTransformer")
      .def(torch::init<int64_t, std::string, std::string, int64_t>())
      .def("transform", &IDTransformerWrapper::transform)
      .def("evict", &IDTransformerWrapper::evict)
      .def("save", &IDTransformerWrapper::save);
}
} // namespace torchrec

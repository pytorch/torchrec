#include "tde/id_transformer.h"
#include "torch/torch.h"
namespace tde {
TORCH_LIBRARY(tde, m) {
  m.class_<IDTransformer>("IDTransformer")
      .def(torch::init<int64_t, int64_t>())
      .def("transform", &IDTransformer::Transform);
}
} // namespace tde

#pragma once

#include <stdint.h>
#include <functional>
#include <optional>

namespace torchrec {

using lxu_record_t = uint32_t;

struct record_t {
  int64_t global_id;
  int64_t cache_id;
  lxu_record_t lxu_record;
};

using iterator_t = std::function<std::optional<record_t>()>;
using update_t =
    std::function<lxu_record_t(std::optional<lxu_record_t>, int64_t, int64_t)>;
using fetch_t = std::function<void(int64_t, int64_t)>;

} // namespace torchrec

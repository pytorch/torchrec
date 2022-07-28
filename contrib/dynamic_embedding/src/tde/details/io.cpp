#include "io.h"

namespace tde::details {

static constexpr std::string_view k_schema_separator = "://";

IO::IO(const std::string& config) {
  auto pos = config.find(k_schema_separator);
  TORCH_CHECK(
      pos != std::string::npos,
      "config string should be schema://cfg_string, cannot find schema");

  std::string schema = config.substr(0, pos);
  std::string rest_cfg = config.substr(pos + k_schema_separator.size());
  auto& reg = IORegistry::Instance();
  provider_ = reg.Resolve(schema);
  instance_ = provider_.Initialize(rest_cfg.c_str());
}

IO::~IO() {
  if (instance_ == nullptr) {
    return;
  }
  provider_.Finalize(instance_);
}

void IO::Push() {}

struct FetchContext {
  std::vector<torch::Tensor> tensors_;
  MoveOnlyFunction<void(std::vector<torch::Tensor>)> on_complete_;
  torch::ScalarType scalar_type_;
  uint32_t num_optimizer_states_;
};

static void OnGlobalIDFetched(
    void* ctx,
    uint32_t offset,
    uint32_t optimizer_state,
    void* data,
    uint32_t data_len) {
  auto c = reinterpret_cast<FetchContext*>(ctx);
  auto& tensor = c->tensors_[offset];
  if (!tensor.defined()) {
    size_t elem_size = torch::elementSize(c->scalar_type_);
    TORCH_CHECK(data_len % elem_size == 0);
    size_t num_elems = data_len / elem_size;
    tensor = torch::empty(
        {
            static_cast<int64_t>(c->num_optimizer_states_),
            static_cast<int64_t>(num_elems),
        },
        c10::TensorOptions().dtype(c->scalar_type_));
  }
  void* ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) +
      optimizer_state * data_len);
  memcpy(ptr, data, data_len);
}

static void OnAllFetched(void* ctx) {
  auto c = reinterpret_cast<FetchContext*>(ctx);
  c->on_complete_(std::move(c->tensors_));
  delete c;
}

void IO::Pull(
    const std::string& table_name,
    tcb::span<const int64_t> col_ids,
    tcb::span<const int64_t> global_ids,
    uint32_t num_optimizer_states,
    torch::ScalarType type,
    MoveOnlyFunction<void(std::vector<torch::Tensor>)> on_fetch_complete) {
  std::unique_ptr<FetchContext> ctx(new FetchContext{
      .on_complete_ = std::move(on_fetch_complete),
      .scalar_type_ = type,
      .num_optimizer_states_ = num_optimizer_states,
  });
  ctx->tensors_.resize(global_ids.size());

  IOPullParameter param{
      .table_name_ = table_name.c_str(),
      .num_cols_ = static_cast<uint32_t>(col_ids.size()),
      .num_global_ids_ = static_cast<uint32_t>(global_ids.size()),
      .col_ids_ = col_ids.data(),
      .global_ids_ = global_ids.data(),
      .num_optimizer_stats_ = num_optimizer_states,
      .scalar_type_ = static_cast<int8_t>(type),
      .on_global_id_fetched_ = OnGlobalIDFetched,
      .on_all_fetched_ = OnAllFetched,
  };
  param.on_complete_context_ = ctx.release();
  provider_.Pull(instance_, param);
}

} // namespace tde::details

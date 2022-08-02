#include "tde/details/id_transformer_variant.h"
#include <vector>

namespace tde::details {

template <typename List>
struct LXUStrategyCreator;

struct CreatorBase {
  const std::string& type_;
  const nlohmann::json& json_;

  CreatorBase(const std::string& t, const nlohmann::json& j)
      : type_(t), json_(j) {}
};

template <typename Head, typename... Tail>
struct LXUStrategyCreator<type_list<Head, Tail...>> : public CreatorBase {
  using CreatorBase::CreatorBase;
  LXUStrategy::Variant operator()() const {
    if (Head::type_ != type_) {
      LXUStrategyCreator<type_list<Tail...>> creator(type_, json_);
      return creator();
    }
    return Head::Create(json_);
  }
};

template <>
struct LXUStrategyCreator<type_list<>> : public CreatorBase {
  using CreatorBase::CreatorBase;
  LXUStrategy::Variant operator()() const {
    TORCH_CHECK(false, "lxu strategy %s is not registered", type_);
  }
};

LXUStrategy::LXUStrategy(const nlohmann::json& json)
    : strategy_((LXUStrategyCreator<LXUStrategies>(json.at("type"), json))()) {}
void LXUStrategy::UpdateTime(uint32_t time) {
  return std::visit(
      [&](auto& strategy) { return strategy.UpdateTime(time); }, strategy_);
}

LXUStrategy::lxu_record_t LXUStrategy::DefaultRecordValue() {
  return std::visit(
      [&](auto& strategy) -> LXUStrategy::lxu_record_t {
        using T = typename std::decay_t<decltype(strategy)>::lxu_record_t;
        return T{};
      },
      strategy_);
}

template <typename LXURecordType, typename List>
struct IDTransformerCreator;

struct IDTransformerCreatorBase : public CreatorBase {
  typedef IDTransformer::Variant (*CreatorFunc)(
      int64_t,
      const nlohmann::json&,
      LXUStrategy::lxu_record_t record);
  CreatorFunc creator_;
  int64_t num_embeddings_;
  IDTransformerCreatorBase(
      CreatorFunc func,
      int64_t num_embeddings,
      const std::string& t,
      const nlohmann::json& j)
      : CreatorBase(t, j), creator_(func), num_embeddings_(num_embeddings) {}
};

template <typename LXURecordType, typename Head, typename... Tail>
struct IDTransformerCreator<LXURecordType, type_list<Head, Tail...>>
    : public IDTransformerCreatorBase {
  using IDTransformerCreatorBase::IDTransformerCreatorBase;

  IDTransformer::Variant CallNext() {
    IDTransformerCreator<LXURecordType, type_list<Tail...>> creator(
        creator_, num_embeddings_, type_, json_);
    return creator();
  }

  IDTransformer::Variant operator()() {
    if (!std::is_same_v<typename Head::lxu_record_t, LXURecordType>) {
      return CallNext();
    }
    if (Head::type_ != type_) {
      return CallNext();
    }
    if constexpr (Head::IsCompose) {
      auto underlying = json_["underlying"];
      auto creator = [&](int64_t num_embeddings) ->
          typename Head::underlying_t {
            auto var = creator_(num_embeddings, underlying, LXURecordType());
            return std::get<typename Head::underlying_t>(std::move(var));
          };
      return Head::Create(num_embeddings_, json_, std::move(creator));
    } else {
      return Head::Create(num_embeddings_, json_);
    }
  }
};

template <typename LXURecordType>
struct IDTransformerCreator<LXURecordType, type_list<>>
    : public IDTransformerCreatorBase {
  using IDTransformerCreatorBase::IDTransformerCreatorBase;
  IDTransformer::Variant operator()() {
    TORCH_CHECK(false, "not support type %s", type_);
  }
};

static IDTransformer::Variant CreateIDTransformer(
    int64_t num_embeddings,
    const nlohmann::json& json,
    LXUStrategy::lxu_record_t record) {
  auto& type = static_cast<const std::string&>(json.at("type"));
  return std::visit(
      [&](auto& r) {
        using T = std::decay_t<decltype(r)>;
        IDTransformerCreator<T, Transformers> creator(
            CreateIDTransformer, num_embeddings, type, json);
        return creator();
      },
      record);
}

IDTransformer::IDTransformer(
    LXUStrategy strategy,
    int64_t num_embeddings,
    const nlohmann::json& json)
    : strategy_(std::move(strategy)),
      var_(CreateIDTransformer(
          num_embeddings,
          json,
          strategy_.DefaultRecordValue())) {}

} // namespace tde::details

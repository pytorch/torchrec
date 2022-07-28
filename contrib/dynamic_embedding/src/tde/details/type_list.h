#pragma once
#include <tuple>
#include <variant>
namespace tde::details {
template <typename... Args>
struct type_list {};

template <typename T, typename U>
struct cons;

template <template <typename...> class List, typename... Ls, typename... Rs>
struct cons<List<Ls...>, List<Rs...>> {
  using type = List<Ls..., Rs...>;
};

template <typename L, typename R>
using cons_t = typename cons<L, R>::type;

template <typename L>
struct to_tuple;

template <template <typename...> class List, typename... Ls>
struct to_tuple<List<Ls...>> {
  using type = std::tuple<Ls...>;
};

template <typename L>
using to_tuple_t = typename to_tuple<L>::type;


template <typename L>
struct to_variant;

template <typename... Args>
struct to_variant<type_list<Args...>> {
  using type = std::variant<Args...>;
};

template <typename L>
using to_variant_t = typename to_variant<L>::type;

} // namespace tde::details

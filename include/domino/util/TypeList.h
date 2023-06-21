#ifndef DOMINO_UTIL_TYPELIST_H_
#define DOMINO_UTIL_TYPELIST_H_

#include <domino/support/TypeTraits.h>
#include <domino/util/IfConstexpr.h>

#include <type_traits>

namespace domino {

template <typename... T> struct false_t : std::false_type {};
template <template <typename> class... T>
struct false_higher_t : std::false_type {};

template <class... Items> struct typelist final {
public:
  typelist() = delete;
};

template <class TypeList> struct size final {
  static_assert(false_t<TypeList>::value,
                "In typelist::size<T>, T must be typelist<...>.");
};
template <class... Types> struct size<typelist<Types...>> final {
  static constexpr size_t value = sizeof...(Types);
};

template <class TypeList> struct to_tuple final {
  static_assert(false_t<TypeList>::value,
                "In typelist::to_tuple<T>, T must be typelist<...>.");
};
template <class... Types> struct to_tuple<typelist<Types...>> final {
  using type = std::tuple<Types...>;
};
template <class TypeList> using to_tuple_t = typename to_tuple<TypeList>::type;

template <class TypeList> struct from_tuple final {
  static_assert(false_t<TypeList>::value,
                "In typelist::from_tuple<T>, T must be typelist<...>.");
};
template <class... Types> struct from_tuple<std::tuple<Types...>> final {
  using type = typelist<Types...>;
};
template <class Tuple> using from_tuple_t = typename from_tuple<Tuple>::type;

template <class... TypeList> struct concat final {
  static_assert(false_t<TypeList...>::value,
                "In typelist::concat<T>, T must be typelist<...>.");
};
template <class... Types1, class... Types2, class... TailLists>
struct concat<typelist<Types1...>, typelist<Types2...>, TailLists...> final {
  using type =
      typename concat<typelist<Types1..., Types2...>, TailLists...>::type;
};
template <class... Types> struct concat<typelist<Types...>> final {
  using type = typelist<Types...>;
};
template <> struct concat<> final {
  using type = typelist<>;
};
template <class... TypeList>
using concat_t = typename concat<TypeList...>::type;

} // namespace domino

#endif // DOMINO_UTIL_TYPELIST_H_
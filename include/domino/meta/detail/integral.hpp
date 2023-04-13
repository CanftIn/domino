#ifndef DOMINO_META_DETAIL_INTEGRAL_HPP_
#define DOMINO_META_DETAIL_INTEGRAL_HPP_

#include <type_traits>

namespace domino {

namespace meta {

template <bool B>
using meta_bool = std::integral_constant<bool, B>;

using meta_true = meta_bool<true>;
using meta_false = meta_bool<false>;

template <class T>
using meta_to_bool = meta_bool<static_cast<bool>(T::value)>;

template <class T>
using meta_not = meta_bool<!(T::value)>;

template <int I>
using meta_int = std::integral_constant<int, I>;

template <std::size_t N>
using meta_size_t = std::integral_constant<std::size_t, N>;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_INTEGRAL_HPP_
#ifndef DOMINO_META_DETAIL_PLUS_HPP_
#define DOMINO_META_DETAIL_PLUS_HPP_

#include <domino/meta/detail/integral.hpp>
#include <type_traits>

namespace domino {

namespace meta {

namespace detail {

template <class... T>
struct meta_plus_impl {
  static const auto _v = (T::value + ... + 0);
  using type =
      std::integral_constant<typename std::remove_const<decltype(_v)>::type,
                             _v>;
};

template <class T1, class... T>
struct meta_plus_impl<T1, T...> {
  static const decltype(T1::value + meta_plus_impl<T...>::type::value) _v =
      T1::value + meta_plus_impl<T...>::type::value;
  using type =
      std::integral_constant<typename std::remove_const<decltype(_v)>::type,
                             _v>;
};

template <class T1, class T2, class T3, class T4, class T5, class T6, class T7,
          class T8, class T9, class T10, class... T>
struct meta_plus_impl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T...> {
  static const decltype(T1::value + T2::value + T3::value + T4::value +
                        T5::value + T6::value + T7::value + T8::value +
                        T9::value + T10::value +
                        meta_plus_impl<T...>::type::value) _v =
      T1::value + T2::value + T3::value + T4::value + T5::value + T6::value +
      T7::value + T8::value + T9::value + T10::value +
      meta_plus_impl<T...>::type::value;
  using type =
      std::integral_constant<typename std::remove_const<decltype(_v)>::type,
                             _v>;
};

}  // namespace detail

template <class... T>
using meta_plus = typename detail::meta_plus_impl<T...>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_PLUS_HPP_
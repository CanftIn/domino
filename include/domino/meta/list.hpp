#ifndef DOMINO_META_LIST_HPP_
#define DOMINO_META_LIST_HPP_

#include <domino/meta/detail/append.hpp>
#include <domino/meta/detail/front.hpp>
#include <domino/meta/detail/integral.hpp>
#include <domino/meta/detail/is_list.hpp>
#include <domino/meta/detail/list.hpp>
#include <domino/meta/detail/rename.hpp>
#include <type_traits>

namespace domino {
  
namespace meta {

// meta_list_c<T, I...>
template <class T, T... I>
using meta_list_c = meta_list<std::integral_constant<T, I>...>;

// meta_is_list<L>
//   in detail/meta_is_list.hpp

// meta_size<L>
namespace detail {

template <class L>
struct meta_size_impl {
  // An error "no type named 'type'" here means that the argument to meta_size
  // is not a list
};

template <template <class...> class L, class... T>
struct meta_size_impl<L<T...>> {
  using type = meta_size_t<sizeof...(T)>;
};

}  // namespace detail

template <class L>
using meta_size = typename detail::meta_size_impl<L>::type;

// meta_empty<L>
template <class L>
using meta_empty = meta_bool<meta_size<L>::value == 0>;

// meta_assign<L1, L2>
namespace detail {

template <class L1, class L2>
struct meta_assign_impl;

template <template <class...> class L1, class... T,
          template <class...> class L2, class... U>
struct meta_assign_impl<L1<T...>, L2<U...>> {
  using type = L1<U...>;
};

}  // namespace detail

template <class L1, class L2>
using meta_assign = typename detail::meta_assign_impl<L1, L2>::type;

// meta_clear<L>
template <class L>
using meta_clear = meta_assign<L, meta_list<>>;

// meta_front<L>
//   in detail/meta_front.hpp

// meta_pop_front<L>
namespace detail {

template <class L>
struct meta_pop_front_impl {
  // An error "no type named 'type'" here means that the argument to
  // meta_pop_front is either not a list, or is an empty list
};

template <template <class...> class L, class T1, class... T>
struct meta_pop_front_impl<L<T1, T...>> {
  using type = L<T...>;
};

}  // namespace detail

template <class L>
using meta_pop_front = typename detail::meta_pop_front_impl<L>::type;

// meta_first<L>
template <class L>
using meta_first = meta_front<L>;

// meta_rest<L>
template <class L>
using meta_rest = meta_pop_front<L>;

// meta_second<L>
namespace detail {

template <class L>
struct meta_second_impl {
  // An error "no type named 'type'" here means that the argument to meta_second
  // is either not a list, or has fewer than two elements
};

template <template <class...> class L, class T1, class T2, class... T>
struct meta_second_impl<L<T1, T2, T...>> {
  using type = T2;
};

}  // namespace detail

template <class L>
using meta_second = typename detail::meta_second_impl<L>::type;

// meta_third<L>
namespace detail {

template <class L>
struct meta_third_impl {
  // An error "no type named 'type'" here means that the argument to meta_third
  // is either not a list, or has fewer than three elements
};

template <template <class...> class L, class T1, class T2, class T3, class... T>
struct meta_third_impl<L<T1, T2, T3, T...>> {
  using type = T3;
};

}  // namespace detail

template <class L>
using meta_third = typename detail::meta_third_impl<L>::type;

// meta_push_front<L, T...>
namespace detail {

template <class L, class... T>
struct meta_push_front_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_push_front is not a list
};

template <template <class...> class L, class... U, class... T>
struct meta_push_front_impl<L<U...>, T...> {
  using type = L<T..., U...>;
};

}  // namespace detail

template <class L, class... T>
using meta_push_front = typename detail::meta_push_front_impl<L, T...>::type;

// meta_push_back<L, T...>
namespace detail {

template <class L, class... T>
struct meta_push_back_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_push_back is not a list
};

template <template <class...> class L, class... U, class... T>
struct meta_push_back_impl<L<U...>, T...> {
  using type = L<U..., T...>;
};

}  // namespace detail

template <class L, class... T>
using meta_push_back = typename detail::meta_push_back_impl<L, T...>::type;

// meta_rename<L, B>
// meta_apply<F, L>
// meta_apply_q<Q, L>
//   in detail/meta_rename.hpp

// meta_replace_front<L, T>
namespace detail {

template <class L, class T>
struct meta_replace_front_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_replace_front is either not a list, or is an empty list
};

template <template <class...> class L, class U1, class... U, class T>
struct meta_replace_front_impl<L<U1, U...>, T> {
  using type = L<T, U...>;
};

}  // namespace detail

template <class L, class T>
using meta_replace_front = typename detail::meta_replace_front_impl<L, T>::type;

// meta_replace_first<L, T>
template <class L, class T>
using meta_replace_first = typename detail::meta_replace_front_impl<L, T>::type;

// meta_replace_second<L, T>
namespace detail {

template <class L, class T>
struct meta_replace_second_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_replace_second is either not a list, or has fewer than two elements
};

template <template <class...> class L, class U1, class U2, class... U, class T>
struct meta_replace_second_impl<L<U1, U2, U...>, T> {
  using type = L<U1, T, U...>;
};

}  // namespace detail

template <class L, class T>
using meta_replace_second =
    typename detail::meta_replace_second_impl<L, T>::type;

// meta_replace_third<L, T>
namespace detail {

template <class L, class T>
struct meta_replace_third_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_replace_third is either not a list, or has fewer than three elements
};

template <template <class...> class L, class U1, class U2, class U3, class... U,
          class T>
struct meta_replace_third_impl<L<U1, U2, U3, U...>, T> {
  using type = L<U1, U2, T, U...>;
};

}  // namespace detail

template <class L, class T>
using meta_replace_third = typename detail::meta_replace_third_impl<L, T>::type;

// meta_transform_front<L, F>
namespace detail {

template <class L, template <class...> class F>
struct meta_transform_front_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_transform_front is either not a list, or is an empty list
};

template <template <class...> class L, class U1, class... U,
          template <class...> class F>
struct meta_transform_front_impl<L<U1, U...>, F> {
  using type = L<F<U1>, U...>;
};

}  // namespace detail

template <class L, template <class...> class F>
using meta_transform_front =
    typename detail::meta_transform_front_impl<L, F>::type;
template <class L, class Q>
using meta_transform_front_q = meta_transform_front<L, Q::template fn>;

// meta_transform_first<L, F>
template <class L, template <class...> class F>
using meta_transform_first =
    typename detail::meta_transform_front_impl<L, F>::type;
template <class L, class Q>
using meta_transform_first_q = meta_transform_first<L, Q::template fn>;

// meta_transform_second<L, F>
namespace detail {

template <class L, template <class...> class F>
struct meta_transform_second_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_transform_second is either not a list, or has fewer than two elements
};

template <template <class...> class L, class U1, class U2, class... U,
          template <class...> class F>
struct meta_transform_second_impl<L<U1, U2, U...>, F> {
  using type = L<U1, F<U2>, U...>;
};

}  // namespace detail

template <class L, template <class...> class F>
using meta_transform_second =
    typename detail::meta_transform_second_impl<L, F>::type;
template <class L, class Q>
using meta_transform_second_q = meta_transform_second<L, Q::template fn>;

// meta_transform_third<L, F>
namespace detail {

template <class L, template <class...> class F>
struct meta_transform_third_impl {
  // An error "no type named 'type'" here means that the first argument to
  // meta_transform_third is either not a list, or has fewer than three elements
};

template <template <class...> class L, class U1, class U2, class U3, class... U,
          template <class...> class F>
struct meta_transform_third_impl<L<U1, U2, U3, U...>, F> {
  using type = L<U1, U2, F<U3>, U...>;
};

}  // namespace detail

template <class L, template <class...> class F>
using meta_transform_third =
    typename detail::meta_transform_third_impl<L, F>::type;
template <class L, class Q>
using meta_transform_third_q = meta_transform_third<L, Q::template fn>;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_LIST_HPP_
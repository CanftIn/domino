#ifndef DOMINO_META_DETAIL_RENAME_HPP_
#define DOMINO_META_DETAIL_RENAME_HPP_

namespace domino {

namespace meta {

namespace detail {

template <class A, template <class...> class B>
struct meta_rename_impl {};

template <template <class...> class A, class... T, template <class...> class B>
struct meta_rename_impl<A<T...>, B> {
  using type = B<T...>;
};

}  // namespace detail

template <class A, template <class...> class B>
using meta_rename = typename detail::meta_rename_impl<A, B>::type;

template <template <class...> class F, class L>
using meta_apply = typename detail::meta_rename_impl<L, F>::type;

template <class Q, class L>
using meta_apply_q = typename detail::meta_rename_impl<L, Q::template fn>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_RENAME_HPP_
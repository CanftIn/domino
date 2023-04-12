#ifndef DOMINO_META_DETAIL_IS_LIST_HPP_
#define DOMINO_META_DETAIL_IS_LIST_HPP_

#include <domino/meta/detail/integral.hpp>

namespace domino {

namespace meta {

namespace detail {

template <class L>
struct meta_is_list_impl {
  using type = meta_false;
};

template <template <class...> class L, class... T>
struct meta_is_list_impl<L<T...>> {
  using type = meta_true;
};

}  // namespace detail

template <class L>
using meta_is_list = typename detail::meta_is_list_impl<L>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_IS_LIST_HPP_
#ifndef DOMINO_META_DETAIL_FRONT_HPP_
#define DOMINO_META_DETAIL_FRONT_HPP_

namespace domino {

namespace meta {

namespace detail {

template <class L>
struct meta_front_impl {};

template <template <class...> class L, class T1, class... T>
struct meta_front_impl<L<T1, T...>> {
  using type = T1;
};

}  // namespace detail

template <class L>
using meta_front = typename detail::meta_front_impl<L>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_FRONT_HPP_
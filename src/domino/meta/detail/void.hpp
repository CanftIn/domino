#ifndef DOMINO_META_DETAIL_VOID_HPP_
#define DOMINO_META_DETAIL_VOID_HPP_

namespace domino {

namespace meta {

namespace detail {

template <class... T>
struct meta_void_impl {
  using type = void;
};
}  // namespace detail

template <class... T>
using meta_void = typename detail::meta_void_impl<T...>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_VOID_HPP_
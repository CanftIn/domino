#ifndef DOMINO_META_DETAIL_MIN_ELEMENT_HPP_
#define DOMINO_META_DETAIL_MIN_ELEMENT_HPP_

#include <domino/meta/detail/fold.hpp>
#include <domino/meta/detail/utility.hpp>
#include <domino/meta/list.hpp>

namespace domino {

namespace meta {

namespace detail {

template <template <class...> class P>
struct select_min {
  template <class T1, class T2>
  using fn = meta_if<P<T1, T2>, T1, T2>;
};

}

template<class L, template<class...> class P> using meta_min_element = meta_fold_q<meta_rest<L>, meta_first<L>, detail::select_min<P>>;
template<class L, class Q> using meta_min_element_q = meta_min_element<L, Q::template fn>;

// meta_max_element<L, P>
namespace detail
{

template<template<class...> class P> struct select_max
{
    template<class T1, class T2> using fn = meta_if<P<T2, T1>, T1, T2>;
};

} // namespace detail

template<class L, template<class...> class P> using meta_max_element = meta_fold_q<meta_rest<L>, meta_first<L>, detail::select_max<P>>;
template<class L, class Q> using meta_max_element_q = meta_max_element<L, Q::template fn>;

}
}

#endif  // DOMINO_META_DETAIL_MIN_ELEMENT_HPP_
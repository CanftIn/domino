#ifndef DOMINO_META_DETAIL_UTILITY_HPP_
#define DOMINO_META_DETAIL_UTILITY_HPP_

#include <domino/meta/detail/fold.hpp>
#include <domino/meta/detail/front.hpp>
#include <domino/meta/detail/integral.hpp>
#include <domino/meta/detail/list.hpp>
#include <domino/meta/detail/rename.hpp>

namespace domino {

namespace meta {

// meta_identity
template <class T>
struct meta_identity {
  using type = T;
};

// meta_identity_t
template <class T>
using meta_identity_t = typename meta_identity<T>::type;

// meta_inherit
template <class... T>
struct meta_inherit : T... {};

// meta_if, meta_if_c
namespace detail {

template <bool C, class T, class... E>
struct meta_if_c_impl {};

template <class T, class... E>
struct meta_if_c_impl<true, T, E...> {
  using type = T;
};

template <class T, class E>
struct meta_if_c_impl<false, T, E> {
  using type = E;
};

}  // namespace detail

template <bool C, class T, class... E>
using meta_if_c = typename detail::meta_if_c_impl<C, T, E...>::type;
template <class C, class T, class... E>
using meta_if =
    typename detail::meta_if_c_impl<static_cast<bool>(C::value), T, E...>::type;

namespace detail {

template <template <class...> class F, class... T>
struct meta_valid_impl {
  template <template <class...> class G, class = G<T...>>
  static meta_true check(int);
  template <template <class...> class>
  static meta_false check(...);

  using type = decltype(check<F>(0));
};

}  // namespace detail

template <template <class...> class F, class... T>
using meta_valid = typename detail::meta_valid_impl<F, T...>::type;

template <class Q, class... T>
using meta_valid_q = meta_valid<Q::template fn, T...>;

// meta_defer
namespace detail {

template <template <class...> class F, class... T>
struct meta_defer_impl {
  using type = F<T...>;
};

struct meta_no_type {};

}  // namespace detail

template <template <class...> class F, class... T>
using meta_defer =
    meta_if<meta_valid<F, T...>, detail::meta_defer_impl<F, T...>,
            detail::meta_no_type>;

// meta_eval_if, meta_eval_if_c
namespace detail {

template <bool C, class T, template <class...> class F, class... U>
struct meta_eval_if_c_impl;

template <class T, template <class...> class F, class... U>
struct meta_eval_if_c_impl<true, T, F, U...> {
  using type = T;
};

template <class T, template <class...> class F, class... U>
struct meta_eval_if_c_impl<false, T, F, U...> : meta_defer<F, U...> {};

}  // namespace detail

template <bool C, class T, template <class...> class F, class... U>
using meta_eval_if_c =
    typename detail::meta_eval_if_c_impl<C, T, F, U...>::type;
template <class C, class T, template <class...> class F, class... U>
using meta_eval_if =
    typename detail::meta_eval_if_c_impl<static_cast<bool>(C::value), T, F,
                                         U...>::type;
template <class C, class T, class Q, class... U>
using meta_eval_if_q =
    typename detail::meta_eval_if_c_impl<static_cast<bool>(C::value), T,
                                         Q::template fn, U...>::type;

// meta_eval_if_not
template <class C, class T, template <class...> class F, class... U>
using meta_eval_if_not = meta_eval_if<meta_not<C>, T, F, U...>;
template <class C, class T, class Q, class... U>
using meta_eval_if_not_q = meta_eval_if<meta_not<C>, T, Q::template fn, U...>;

// meta_eval_or
template <class T, template <class...> class F, class... U>
using meta_eval_or = meta_eval_if_not<meta_valid<F, U...>, T, F, U...>;
template <class T, class Q, class... U>
using meta_eval_or_q = meta_eval_or<T, Q::template fn, U...>;

// meta_valid_and_true
template <template <class...> class F, class... T>
using meta_valid_and_true = meta_eval_or<meta_false, F, T...>;
template <class Q, class... T>
using meta_valid_and_true_q = meta_valid_and_true<Q::template fn, T...>;

// meta_cond

// so elegant; so doesn't work
// template<class C, class T, class... E> using meta_cond = meta_eval_if<C, T,
// meta_cond, E...>;

namespace detail {

template <class C, class T, class... E>
struct meta_cond_impl;

}  // namespace detail

template <class C, class T, class... E>
using meta_cond = typename detail::meta_cond_impl<C, T, E...>::type;

namespace detail {

template <class C, class T, class... E>
using meta_cond_ = meta_eval_if<C, T, meta_cond, E...>;

template <class C, class T, class... E>
struct meta_cond_impl : meta_defer<meta_cond_, C, T, E...> {};

}  // namespace detail

// meta_quote
template <template <class...> class F>
struct meta_quote {
  // the indirection through meta_defer works around the language inability
  // to expand T... into a fixed parameter list of an alias template

  template <class... T>
  using fn = typename meta_defer<F, T...>::type;
};

// meta_quote_trait
template <template <class...> class F>
struct meta_quote_trait {
  template <class... T>
  using fn = typename F<T...>::type;
};

template <class Q, class... T>
using meta_invoke_q = typename Q::template fn<T...>;

// meta_not_fn<P>
template <template <class...> class P>
struct meta_not_fn {
  template <class... T>
  using fn = meta_not<meta_invoke_q<meta_quote<P>, T...>>;
};

template <class Q>
using meta_not_fn_q = meta_not_fn<Q::template fn>;

// meta_compose
namespace detail {

template <class L, class Q>
using meta_compose_helper = meta_list<meta_apply_q<Q, L>>;

}  // namespace detail

template <class... Q>
struct meta_compose_q {
  template <class... T>
  using fn = meta_front<
      meta_fold<meta_list<Q...>, meta_list<T...>, detail::meta_compose_helper>>;
};

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_UTILITY_HPP_
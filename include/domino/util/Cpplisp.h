#ifndef DOMINO_UTIL_CPPLISP_H_
#define DOMINO_UTIL_CPPLISP_H_

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using std::nullptr_t;

namespace domino {

namespace cpplisp {

#define caar(X) car(car(X))
#define cadr(X) car(cdr(X))
#define cdar(X) cdr(car(X))
#define cddr(X) cdr(cdr(X))
#define caaar(X) car(car(car(X)))
#define caadr(X) car(car(cdr(X)))
#define cadar(X) car(cdr(car(X)))
#define cdaar(X) cdr(car(car(X)))
#define caddr(X) car(cdr(cdr(X)))
#define cdadr(X) cdr(car(cdr(X)))
#define cddar(X) cdr(cdr(car(X)))
#define cdddr(X) cdr(cdr(cdr(X)))
#define caaaar(X) car(car(car(car(X))))
#define caaadr(X) car(car(car(cdr(X))))
#define caadar(X) car(car(cdr(car(X))))
#define caaddr(X) car(car(cdr(cdr(X))))
#define cadaar(X) car(cdr(car(car(X))))
#define cadadr(X) car(cdr(car(cdr(X))))
#define caddar(X) car(cdr(cdr(car(X))))
#define cadddr(X) car(cdr(cdr(cdr(X))))
#define cdaaar(X) cdr(car(car(car(X))))
#define cdaadr(X) cdr(car(car(cdr(X))))
#define cdadar(X) cdr(car(cdr(car(X))))
#define cdaddr(X) cdr(car(cdr(cdr(X))))
#define cddaar(X) cdr(cdr(car(car(X))))
#define cddadr(X) cdr(cdr(car(cdr(X))))
#define cdddar(X) cdr(cdr(cdr(car(X))))
#define cddddr(X) cdr(cdr(cdr(cdr(X))))

#define first(X) (car(X))
#define second(X) (car(cdr(X)))
#define third(X) (car(cddr(X)))
#define fourth(X) (car(cdddr(X)))
#define fifth(X) (car(cddddr(X)))
#define sixth(X) (car(cdr(cddddr(X))))
#define seventh(X) (car(cddr(cddddr(X))))
#define eighth(X) (car(cdddr(cddddr(X))))
#define ninth(X) (car(cddddr(cddddr(X))))
#define tenth(X) (car(cdr(cddddr(cddddr(X)))))

#define var auto

template <typename T, typename U>
class Cons;

template <typename T, typename U>
using ConsPtr = std::shared_ptr<Cons<T, U>>;

using nil_t = ConsPtr<nullptr_t, nullptr_t>;

static const nil_t nil = nullptr;

template <typename T, typename U>
class Cons {
 private:
  T _car;
  U _cdr;

 public:
  Cons(T car) : _car(car), _cdr(nil) {}

  Cons(T car, U cdr) : _car(car), _cdr(cdr) {}

  T car() const { return _car; }

  U cdr() const { return _cdr; }

  ConsPtr<T, U> set_car(T car) {
    _car = car;
    return ConsPtr<T, U>(this);
  }

  ConsPtr<T, U> set_cdr(U cdr) {
    _cdr = cdr;
    return ConsPtr<T, U>(this);
  }
};

/// @brief [symbols] cons: T -> U -> Cons<T, U>
/// @tparam T
/// @tparam U
/// @param car
/// @param cdr
/// @return
template <typename T, typename U>
inline auto cons(T&& car, U&& cdr) noexcept {
  return std::make_shared<Cons<T, U>>(std::forward<T>(car),
                                      std::forward<U>(cdr));
}

/// @brief [symbols] list: T -> Cons<T, nil_t>
/// @tparam T
/// @param car
/// @return
template <typename T>
inline auto list(T&& car) {
  return cons(std::forward<T>(car), nil);
}

/// @brief [symbols] list: T -> U -> Cons<T, Cons<U, nil_t>>
/// @tparam T
/// @tparam U
/// @param car
/// @param cadr
/// @return
template <typename T, typename U>
inline auto list(T&& car, U&& cadr) {
  return cons(std::forward<T>(car),
              std::move(cons(std::forward<U>(cadr), nil)));
}

/// @brief [symbols] list: T -> ...Rest -> cons(T, list(Rest...))
/// @tparam T
/// @tparam ...Rest
/// @param car
/// @param ...rest
/// @return
template <typename T, typename... Rest>
inline auto list(T&& car, Rest&&... rest) {
  return cons(std::forward<T>(car),
              std::move(list(std::forward<Rest>(rest)...)));
}

/// @brief [symbols] car: ConsPtr<T, U> -> T
/// @tparam T
/// @tparam U
/// @param c
/// @return
template <typename T, typename U>
inline T car(ConsPtr<T, U> c) {
  return c->car();
}

/// @brief [symbols] cdr: ConsPtr<T, U> -> U
/// @tparam T
/// @tparam U
/// @param c
/// @return
template <typename T, typename U>
inline U cdr(ConsPtr<T, U> c) {
  return c->cdr();
}

template <typename T>
struct _consp : std::false_type {};

template <typename T, typename U>
struct _consp<ConsPtr<T, U>> : std::true_type {};

/// @brief [symbols] consp: (type a) => a -> bool
/// @tparam T
/// @param
/// @return
template <typename T>
inline bool consp(T) {
  return _consp<T>::value;
}

template <typename T>
constexpr bool consp_v = _consp<T>::value;

template <typename... T>
struct _list_t {
  using type = nullptr_t;
};

template <>
struct _list_t<nullptr_t> {
  using type = nil_t;
};

template <typename T>
struct _list_t<T> {
  using type = ConsPtr<T, nil_t>;
};

template <typename T, typename... U>
struct _list_t<T, U...> {
  using type = ConsPtr<T, typename _list_t<U...>::type>;
};

template <typename T>
struct _car_t {};

template <>
struct _car_t<nil_t> {
  using type = nil_t;
};

template <>
struct _car_t<const nil_t&> {
  using type = const nil_t&;
};

template <typename T, typename U>
struct _car_t<ConsPtr<T, U>> {
  using type = T;
};

template <typename T>
struct _cdr_t {};

template <>
struct _cdr_t<nil_t> {
  using type = nil_t;
};

template <>
struct _cdr_t<const nil_t&> {
  using type = const nil_t&;
};

template <typename T, typename U>
struct _cdr_t<ConsPtr<T, U>> {
  using type = U;
};

template <typename T>
struct _listp : std::false_type {};

template <>
struct _listp<nil_t> : std::true_type {};

template <>
struct _listp<nil_t&> : std::true_type {};

template <>
struct _listp<const nil_t&> : std::true_type {};

template <typename T, typename U>
struct _listp<ConsPtr<T, U>> : _listp<U> {};

/// @brief [symbols] listp: (type a) => a -> bool
/// @tparam T
/// @param
/// @return
template <typename T>
inline bool listp(T) {
  return _listp<T>::value;
}

template <typename T>
constexpr bool listp_v = _listp<T>::value;

template <typename T>
struct _list_len {
  static const int value = 0;
};

template <>
struct _list_len<nil_t> {
  static const int value = 0;
};

template <typename T>
struct _list_len<ConsPtr<T, nil_t>> {
  static const int value = 1;
};

template <typename T, typename U>
struct _list_len<ConsPtr<T, U>> {
  static const int value = 1 + _list_len<U>::value;
};

/// @brief [symbols] length: (Squence a) => a -> int
/// @tparam T
/// @tparam U
/// @tparam
/// @param l
/// @return
template <typename T, typename U,
          typename IsProperList = std::enable_if_t<listp_v<ConsPtr<T, U>>>>
inline constexpr int length(ConsPtr<T, U> l) {
  return _list_len<ConsPtr<T, U>>::value;
}

template <std::size_t N, typename T>
struct _nth_t {};

template <typename T, typename U>
struct _nth_t<0, ConsPtr<T, U>> {
  using type = T;
};

template <std::size_t N, typename T, typename U>
struct _nth_t<N, ConsPtr<T, U>> {
  using type = typename _nth_t<N - 1, U>::type;
};

template <std::size_t N>
struct _nth;

template <>
struct _nth<0> {
  template <typename T, typename U>
  static T position(ConsPtr<T, U> lst) {
    return car(lst);
  }
};

template <std::size_t N>
struct _nth {
  template <typename T, typename U>
  static typename _nth_t<N, ConsPtr<T, U>>::type position(ConsPtr<T, U> lst) {
    return _nth<N - 1>::position(cdr(lst));
  }
};

/// @brief [symbols] nth: (Squence a) => int -> a -> b
/// @tparam T
/// @tparam U
/// @tparam WithInListRange
/// @tparam N
/// @param lst
/// @return
template <std::size_t N, typename T, typename U,
          typename WithInListRange = std::enable_if_t<
              (N < _list_len<ConsPtr<T, U>>::value) && listp_v<ConsPtr<T, U>>>>
typename _nth_t<N, ConsPtr<T, U>>::type nth(ConsPtr<T, U> lst) {
  return _nth<N>::position(lst);
}

template <typename T>
struct _nullp : std::false_type {};

template <>
struct _nullp<nil_t> : std::true_type {};

/// @brief [symbols] nullp
/// @tparam T
/// @param
/// @return
template <typename T>
inline bool nullp(T) {
  return _nullp<T>::value;
}

namespace prettyprint {
inline std::ostream& operator<<(std::ostream& os, nil_t) {
  os << "nil";
  return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, ConsPtr<T, const nil_t&> c) {
  os << "(" << car(c) << " . nil)";
  return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, ConsPtr<T, nil_t&> c) {
  os << "(" << car(c) << " . nil)";
  return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, ConsPtr<T, nil_t> c) {
  os << "(" << car(c) << " . nil)";
  return os;
}

template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os, ConsPtr<T, U> c) {
  os << "(" << car(c);
  if (cdr(c)) {
    os << " . ";
    os << cdr(c);
  }
  os << ")";
  return os;
}

template <typename T, typename U>
inline std::string to_string(ConsPtr<T, U> c) {
  std::stringstream ss;
  ss << c;
  return ss.str();
}
}  // namespace prettyprint

template <typename T>
inline bool equals(T a, T b) {
  return a == b;
}

template <typename T, typename U>
inline bool equals(ConsPtr<T, U> lhs, ConsPtr<T, U> rhs) {
  if (car(lhs) == car(rhs)) {
    if (consp(cdr(lhs)) && consp(cdr(rhs))) {
      return equals(cdr(lhs), cdr(rhs));
    }
    return cdr(lhs) == cdr(rhs);
  }
  return false;
}

// single type list-of
template <typename T, typename U>
struct _list_of : std::false_type {};
template <typename T>
struct _list_of<T, ConsPtr<T, nil_t>> : std::true_type {};
template <typename T, typename S>
struct _list_of<T, ConsPtr<T, S>> : _list_of<T, S> {};
template <typename T, typename U>
constexpr bool list_of_v = _list_of<T, U>::value;

template <typename T, typename U>
struct _m_list_of : std::false_type {};
template <typename T>
struct _m_list_of<T, ConsPtr<T, nil_t>> : std::true_type {};
template <typename T, typename U>
struct _m_list_of<T, ConsPtr<U, nil_t>> : _m_list_of<T, U> {};
template <typename T, typename S>
struct _m_list_of<T, ConsPtr<T, S>> : _m_list_of<T, S> {};
template <typename T, typename U, typename S>
struct _m_list_of<T, ConsPtr<U, S>> {
  static const bool value = _m_list_of<T, U>::value && _m_list_of<T, S>::value;
};
template <typename T, typename U>
constexpr bool m_list_of_v = _m_list_of<T, U>::value;

template <typename T, typename U>
struct _append_t {
  using type = nullptr_t;
};
template <typename T, typename U>
struct _append_t<const nil_t&, ConsPtr<T, U>> {
  using type = ConsPtr<T, U>;
};
template <typename T, typename U>
struct _append_t<ConsPtr<T, U>, const nil_t&> {
  using type = ConsPtr<T, U>;
};
template <typename T, typename S, typename Y>
struct _append_t<T, ConsPtr<S, Y>> {
  using type = ConsPtr<T, ConsPtr<S, Y>>;
};
template <typename T, typename S, typename Y>
struct _append_t<ConsPtr<T, const nil_t&>, ConsPtr<S, Y>> {
  using type = ConsPtr<T, ConsPtr<S, Y>>;
};
template <typename T, typename U, typename S, typename Y>
struct _append_t<ConsPtr<T, U>, ConsPtr<S, Y>> {
  using type = ConsPtr<T, typename _append_t<U, ConsPtr<S, Y>>::type>;
};
template <typename S, typename Y>
inline ConsPtr<S, Y> _append(const nil_t&, ConsPtr<S, Y> b) {
  return b;
}
template <typename T, typename U>
inline ConsPtr<T, U> _append(ConsPtr<T, U> a, const nil_t&) {
  return a;
}
template <typename T, typename U, typename S, typename Y,
          typename R = typename _append_t<ConsPtr<T, U>, ConsPtr<S, Y>>::type>
R _append(ConsPtr<T, U> a, ConsPtr<S, Y> b) {
  return cons(car(a), _append(cdr(a), b));
}
template <typename T, typename U, typename S, typename Y,
          typename BothProperLists = std::enable_if_t<listp_v<ConsPtr<T, U>> &&
                                                      listp_v<ConsPtr<S, Y>>>>
auto append(ConsPtr<T, U> a, ConsPtr<S, Y> b) {
  return _append(a, b);
}

template <typename T>
struct _reverse_t {
  using type = nullptr_t;
};
template <>
struct _reverse_t<const nil_t&> {
  using type = const nil_t&;
};
template <typename T>
struct _reverse_t<ConsPtr<T, const nil_t&>> {
  using type = ConsPtr<T, const nil_t&>;
};
template <typename T, typename U>
struct _reverse_t<ConsPtr<T, ConsPtr<U, const nil_t&>>> {
  using type = ConsPtr<U, ConsPtr<T, const nil_t&>>;
};
template <typename T, typename U>
struct _reverse_t<ConsPtr<T, U>> {
  using type = typename _append_t<typename _reverse_t<U>::type,
                                  ConsPtr<T, const nil_t&>>::type;
};
inline nil_t _reverse(const nil_t&) { return nil; }
template <typename T, typename U,
          typename R = typename _reverse_t<ConsPtr<T, U>>::type>
R _reverse(ConsPtr<T, U> lst) {
  return append(_reverse(cdr(lst)), list(car(lst)));
}
template <typename T, typename U,
          typename IsProperList = std::enable_if_t<listp_v<ConsPtr<T, U>>>>
auto reverse(ConsPtr<T, U> lst) {
  return _reverse(lst);
}

template <typename T, typename U, typename... Us>
struct _mapcar_t {};
template <typename F>
struct _mapcar_t<F, const nil_t&> {
  using type = const nil_t&;
};
template <typename F, typename T>
struct _mapcar_t<F, ConsPtr<T, const nil_t&>> {
  using type = ConsPtr<typename std::result_of<F&(T)>::type, const nil_t&>;
};
template <typename F, typename T, typename U>
struct _mapcar_t<F, ConsPtr<T, U>> {
  using type = ConsPtr<typename std::result_of<F&(T)>::type,
                       typename _mapcar_t<F, U>::type>;
};
template <typename F, typename... S>
struct _packed_result_t {
  using type = typename std::result_of<F&(S...)>::type;
};
template <typename F, typename T, typename... S>
struct _mapcar_t<F, ConsPtr<T, const nil_t&>, S...> {
  using type = ConsPtr<
      typename _packed_result_t<F, T, typename _car_t<S>::type...>::type,
      const nil_t&>;
};
template <typename F, typename T, typename U, typename... S>
struct _mapcar_t<F, ConsPtr<T, U>, S...> {
  using type = ConsPtr<
      typename _packed_result_t<F, T, typename _car_t<S>::type...>::type,
      typename _mapcar_t<F, U, typename _cdr_t<S>::type...>::type>;
};
template <typename F, typename T>
auto _mapcar(F fn, ConsPtr<T, const nil_t&> lst) {
  return list(fn(car(lst)));
}
template <typename F, typename T, typename U,
          typename R = typename _mapcar_t<F, ConsPtr<T, U>>::type>
R _mapcar(F fn, ConsPtr<T, U> lst) {
  return cons(fn(car(lst)), _mapcar(fn, cdr(lst)));
}
template <
    typename F, typename T, typename... S,
    typename R = typename _mapcar_t<F, ConsPtr<T, const nil_t&>, S...>::type>
R _mapcar(F fn, ConsPtr<T, const nil_t&> lst, S... rest) {
  return list(fn(car(lst), car(rest)...));
};
template <typename F, typename T, typename U, typename... S,
          typename R = typename _mapcar_t<F, ConsPtr<T, U>, S...>::type>
R _mapcar(F fn, ConsPtr<T, U> lst, S... rest) {
  return cons(fn(car(lst), car(rest)...), _mapcar(fn, cdr(lst), cdr(rest)...));
};
template <typename F, typename T, typename U,
          typename IsProperList = std::enable_if_t<listp_v<ConsPtr<T, U>>>>
auto mapcar(F fn, ConsPtr<T, U> lst) {
  return _mapcar(fn, lst);
}
template <typename F, typename T, typename U, typename... S,
          typename IsProperList = std::enable_if_t<listp_v<ConsPtr<T, U>>>>
auto mapcar(F fn, ConsPtr<T, U> lst, S... rest) {
  return _mapcar(fn, lst, rest...);
}

template <typename R, typename C, bool cp, typename... As>
struct _lambda_type {
  static const bool constp = cp;
  enum { arity = sizeof...(As) };
  using return_type = R;
  using arg_type = typename _list_t<As...>::type;
};
template <typename L>
struct lambda_type : lambda_type<decltype(&L::operator())> {};
template <typename R, typename C, typename... As>
struct lambda_type<R (C::*)(As...)> : _lambda_type<R, C, false, As...> {};
template <typename R, typename C, typename... As>
struct lambda_type<R (C::*)(As...) const> : _lambda_type<R, C, true, As...> {};

template <typename T>
struct _values_t {
  using type = nullptr_t;
};
template <typename T>
struct _values_t<ConsPtr<T, const nil_t&>> {
  using type = ConsPtr<T*, const nil_t&>;
};
template <typename T, typename U>
struct _values_t<ConsPtr<T, U>> {
  using type = ConsPtr<T*, typename _values_t<U>::type>;
};
template <typename T,
          typename S = typename _values_t<ConsPtr<T, const nil_t&>>::type>
void _mvb(ConsPtr<T, const nil_t&> lst, S sym) {
  *(car(sym)) = car(lst);
}
template <typename T, typename U,
          typename S = typename _values_t<ConsPtr<T, U>>::type>
void _mvb(ConsPtr<T, U> lst, S sym) {
  *(car(sym)) = car(lst);
  _mvb(cdr(lst), cdr(sym));
}
template <typename L>
struct get_return : get_return<decltype(&L::operator())> {};
template <typename C, typename... A>
struct get_return<void (C::*)(A...)> {
  template <typename L>
  ConsPtr<bool, nullptr_t> operator()(L fn) {
    fn();
    return cons(false, nullptr);
  }
};
template <typename R, typename C, typename... A>
struct get_return<R (C::*)(A...)> {
  template <typename L>
  ConsPtr<bool, R> operator()(L fn) {
    return cons(true, fn());
  }
};
template <typename C, typename... A>
struct get_return<void (C::*)(A...) const> {
  template <typename L>
  ConsPtr<bool, nullptr_t> operator()(L fn) {
    fn();
    return cons(false, nullptr);
  }
};
template <typename R, typename C, typename... A>
struct get_return<R (C::*)(A...) const> {
  template <typename L>
  ConsPtr<bool, R> operator()(L fn) {
    return cons(true, fn());
  }
};
template <typename T, typename U, typename... Ss,
          typename IsProperList = std::enable_if_t<listp_v<ConsPtr<T, U>>>>
auto multiple_value_bind(ConsPtr<T, U> c, Ss&&... sym) {
  using sym_lst_t = typename _values_t<ConsPtr<T, U>>::type;
  sym_lst_t sym_lst = list(std::forward<Ss>(sym)...);
  auto orig = mapcar([](auto p) { return *p; }, sym_lst);
  _mvb(c, sym_lst);
  return [orig, sym_lst](auto fn) {
    auto retval = get_return<decltype(fn)>()(fn);
    _mvb(orig, sym_lst);
    if (car(retval)) {
      return cdr(retval);
    }
    typename _cdr_t<decltype(retval)>::type ret = 0;
    return ret;
  };
}

}  // namespace cpplisp

}  // namespace domino

#endif  // DOMINO_UTIL_CPPLISP_H_
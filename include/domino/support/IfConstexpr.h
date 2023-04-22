#ifndef DOMINO_UTIL_CPP17_H_
#define DOMINO_UTIL_CPP17_H_

#include <functional>
#include <memory>
#include <type_traits>

namespace domino {

template <typename F, typename... Args>
using invoke_result = typename std::result_of<F && (Args && ...)>;

template <typename F, typename... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

template <typename T>
using is_pod = std::is_pod<T>;

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

template <typename Base, typename Child, typename... Args>
typename std::enable_if<!std::is_array<Base>::value &&
                            !std::is_array<Child>::value &&
                            std::is_base_of<Base, Child>::value,
                        std::unique_ptr<Base>>::type
make_unique_base(Args&&... args) {
  return std::unique_ptr<Base>(new Child(std::forward<Args>(args)...));
}

template <class... B>
using conjunction = std::conjunction<B...>;
template <class... B>
using disjunction = std::disjunction<B...>;
template <bool B>
using bool_constant = std::bool_constant<B>;
template <class B>
using negation = std::negation<B>;

template <typename Functor, typename... Args>
typename std::enable_if<
    std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename domino::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
typename std::enable_if<
    !std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename domino::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

template <class T>
using void_t = std::void_t<T>;

struct _identity final {
  template <class T>
  using type_identity = T;

  template <class T>
  decltype(auto) operator()(T&& arg) {
    return std::forward<T>(arg);
  }
};

template <class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};

template <class Func>
struct function_takes_identity_argument<
    Func, void_t<decltype(std::declval<Func>()(_identity()))>>
    : std::true_type {};
}  // namespace domino

template <bool Condition>
struct _if_constexpr;

template <>
struct _if_constexpr<true> final {
  template <class ThenCallback, class ElseCallback,
            std::enable_if_t<
                domino::function_takes_identity_argument<ThenCallback>::value,
                void*> = nullptr>
  static decltype(auto) call(ThenCallback&& thenCallback,
                             ElseCallback&& /* elseCallback */) {
    return thenCallback(domino::_identity());
  }

  template <class ThenCallback, class ElseCallback,
            std::enable_if_t<
                !domino::function_takes_identity_argument<ThenCallback>::value,
                void*> = nullptr>
  static decltype(auto) call(ThenCallback&& thenCallback,
                             ElseCallback&& /* elseCallback */) {
    return thenCallback();
  }
};

template <>
struct _if_constexpr<false> final {
  template <class ThenCallback, class ElseCallback,
            std::enable_if_t<
                domino::function_takes_identity_argument<ElseCallback>::value,
                void*> = nullptr>
  static decltype(auto) call(ThenCallback&& /* thenCallback */,
                             ElseCallback&& elseCallback) {
    return elseCallback(domino::_identity());
  }

  template <class ThenCallback, class ElseCallback,
            std::enable_if_t<
                !domino::function_takes_identity_argument<ElseCallback>::value,
                void*> = nullptr>
  static decltype(auto) call(ThenCallback&& /* thenCallback */,
                             ElseCallback&& elseCallback) {
    return elseCallback();
  }
};

template <bool Condition, class ThenCallback, class ElseCallback>
decltype(auto) if_constexpr(ThenCallback&& thenCallback,
                            ElseCallback&& elseCallback) {
  return _if_constexpr<Condition>::call(
      static_cast<ThenCallback&&>(thenCallback),
      static_cast<ElseCallback&&>(elseCallback));
}

template <bool Condition, class ThenCallback>
decltype(auto) if_constexpr(ThenCallback&& thenCallback) {
  return if_constexpr<Condition>(static_cast<ThenCallback&&>(thenCallback),
                                 [](auto) {});
}

#endif  // DOMINO_UTIL_CPP17_H_
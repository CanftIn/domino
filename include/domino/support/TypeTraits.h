#ifndef DOMINO_SUPPORT_TYPE_TRAITS_H_
#define DOMINO_SUPPORT_TYPE_TRAITS_H_

#include <functional>
#include <type_traits>
#include <utility>

namespace domino {

template <typename T>
class is_integral_or_enum {
  using UnderlyingT = std::remove_reference_t<T>;

 public:
  static const bool value =
      !std::is_class<UnderlyingT>::value &&
      !std::is_pointer<UnderlyingT>::value &&
      !std::is_floating_point_v<UnderlyingT> &&
      (std::is_enum<UnderlyingT>::value ||
       std::is_convertible_v<UnderlyingT, unsigned long long>);
};

template <typename T, typename Enable = void>
struct add_lvalue_reference_if_not_pointer {
  using type = T &;
};

template <typename T>
struct add_lvalue_reference_if_not_pointer<
    T, std::enable_if_t<std::is_pointer_v<T>>> {
  using type = T;
};

template <typename T, typename Enable = void>
struct add_const_past_pointer {
  using type = const T;
};

template <typename T>
struct add_const_past_pointer<T, std::enable_if_t<std::is_pointer_v<T>>> {
  using type = const std::remove_pointer_t<T> *;
};

template <typename T, typename Enable = void>
struct const_pointer_or_const_ref {
  using type = const T &;
};

template <typename T>
struct const_pointer_or_const_ref<T, std::enable_if_t<std::is_pointer_v<T>>> {
  using type = typename add_const_past_pointer<T>::type;
};

namespace detail {

template <typename T>
union copy_construction_triviality_helper {
  T t;
  copy_construction_triviality_helper() = default;
  copy_construction_triviality_helper(
      const copy_construction_triviality_helper &) = default;
  ~copy_construction_triviality_helper() = default;
};

template <typename T>
union move_construction_triviality_helper {
  T t;
  move_construction_triviality_helper() = default;
  move_construction_triviality_helper(move_construction_triviality_helper &&) =
      default;
  ~move_construction_triviality_helper() = default;
};

template <class T>
union trivial_helper {
  T t;
};

}  // namespace detail

template <typename T>
struct is_trivially_copy_constructible
    : std::is_copy_constructible<
          ::domino::detail::copy_construction_triviality_helper<T>> {};
template <typename T>
struct is_trivially_copy_constructible<T &> : std::true_type {};
template <typename T>
struct is_trivially_copy_constructible<T &&> : std::false_type {};

template <typename T>
struct is_trivially_move_constructible
    : std::is_move_constructible<
          ::domino::detail::move_construction_triviality_helper<T>> {};
template <typename T>
struct is_trivially_move_constructible<T &> : std::true_type {};
template <typename T>
struct is_trivially_move_constructible<T &&> : std::true_type {};

template <typename T>
struct is_copy_assignable {
  template <class F>
  static auto get(F *)
      -> decltype(std::declval<F &>() = std::declval<const F &>(),
                  std::true_type{});
  static std::false_type get(...);
  static constexpr bool value = decltype(get((T *)nullptr))::value;
};

template <typename T>
struct is_move_assignable {
  template <class F>
  static auto get(F *)
      -> decltype(std::declval<F &>() = std::declval<F &&>(), std::true_type{});
  static std::false_type get(...);
  static constexpr bool value = decltype(get((T *)nullptr))::value;
};

template <typename T, class Enable = void>
struct is_equality_comparable : std::false_type {};
template <typename T>
struct is_equality_comparable<
    T, std::enable_if_t<std::is_same_v<
           decltype(std::declval<T &>() == std::declval<T &>()), bool>>>
    : std::true_type {};
template <typename T>
using is_equality_comparable_t = typename is_equality_comparable<T>::type;

template <typename T, class Enable = void>
struct is_hashable : std::false_type {};
template <typename T>
struct is_hashable<
    T, std::enable_if_t<std::is_same_v<
           decltype(std::hash<T>{}(std::declval<T>())), std::size_t>>>
    : std::true_type {};
template <typename T>
using is_hashable_t = typename is_hashable<T>::type;

template <typename T>
struct is_function_type : std::false_type {};
template <typename Ret, class... Args>
struct is_function_type<Ret(Args...)> : std::true_type {};
template <typename T>
using is_function_type_t = typename is_function_type<T>::type;

template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;

namespace detail {

template <typename T>
struct strip_class {};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};
template <typename T>
using strip_class_t = typename strip_class<T>::type;

}  // namespace detail

template <class Functor, class Enable = void>
struct is_functor : std::false_type {};
template <class Functor>
struct is_functor<
    Functor, std::enable_if_t<is_function_type<
                 detail::strip_class_t<decltype(&Functor::operator())>>::value>>
    : std::true_type {};

// lambda_is_stateless<T> is true if the lambda type T is stateless
// (i.e. does not have a closure).
// Example:
//  auto stateless_lambda = [] (int a) {return a;};
//  lambda_is_stateless<decltype(stateless_lambda)> // true
//  auto stateful_lambda = [&] (int a) {return a;};
//  lambda_is_stateless<decltype(stateful_lambda)> // false

namespace detail {

template <class LambdaType, class FuncType>
struct is_stateless_lambda__ final {
  static_assert(!std::is_same<LambdaType, LambdaType>::value,
                "Base case shouldn't be hit");
};
// implementation idea: According to the C++ standard, stateless lambdas are
// convertible to function pointers
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...) const>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...)>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};

// case where LambdaType is not even a functor
template <class LambdaType, class Enable = void>
struct is_stateless_lambda_ final : std::false_type {};
// case where LambdaType is a functor
template <class LambdaType>
struct is_stateless_lambda_<LambdaType,
                            std::enable_if_t<is_functor<LambdaType>::value>>
    : is_stateless_lambda__<LambdaType, decltype(&LambdaType::operator())> {};

}  // namespace detail
template <class T>
using is_stateless_lambda = detail::is_stateless_lambda_<std::decay_t<T>>;

// is_type_condition<C> is true_type if C<...> is a type trait representing a
// condition (i.e. has a constexpr static bool ::value member) Example:
//   is_type_condition<std::is_reference>  // true

template <template <class> class C, class Enable = void>
struct is_type_condition : std::false_type {};
template <template <class> class C>
struct is_type_condition<
    C, std::enable_if_t<std::is_same<
           bool, std::remove_cv_t<decltype(C<int>::value)>>::value>>
    : std::true_type {};

template <class T>
struct is_fundamental : std::is_fundamental<T> {};

}  // namespace domino

#endif  // DOMINO_SUPPORT_TYPE_TRAITS_H_
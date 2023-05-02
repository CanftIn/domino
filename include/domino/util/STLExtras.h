#ifndef DOMINO_UTIL_STLEXTRAS_H_
#define DOMINO_UTIL_STLEXTRAS_H_

#include <domino/util/Iterator.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace domino {

template <typename T, T>
struct SameType;

template <typename T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t = typename domino::remove_cvref<T>::type;

template <typename T, typename Function>
auto transformOptional(const std::optional<T> &O, const Function &F)
    -> std::optional<decltype(F(*O))> {
  if (O) return F(*O);
  return std::nullopt;
}

template <typename T, typename Function>
auto transformOptional(std::optional<T> &&O, const Function &F)
    -> std::optional<decltype(F(*std::move(O)))> {
  if (O) return F(*std::move(O));
  return std::nullopt;
}

template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  intptr_t callable;

  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable *>(callable))(
        std::forward<Params>(params)...);
  }

 public:
  function_ref() = default;
  function_ref(std::nullptr_t) {}

  template <typename Callable>
  function_ref(
      Callable &&callable,
      std::enable_if_t<!std::is_same<remove_cvref_t<Callable>,
                                     function_ref>::value> * = nullptr,
      std::enable_if_t<std::is_void<Ret>::value ||
                       std::is_convertible<decltype(std::declval<Callable>()(
                                               std::declval<Params>()...)),
                                           Ret>::value> * = nullptr)
      : callback(callback_fn<std::remove_reference_t<Callable>>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  Ret operator()(Params... params) const {
    return callback(callable, std::forward<Params>(params)...);
  }

  explicit operator bool() const { return callback; }
};

namespace adl_detail {

using std::begin;

template <typename ContainerTy>
decltype(auto) adl_begin(ContainerTy &&container) {
  return begin(std::forward<ContainerTy>(container));
}

using std::end;

template <typename ContainerTy>
decltype(auto) adl_end(ContainerTy &&container) {
  return end(std::forward<ContainerTy>(container));
}

using std::swap;

template <typename T>
void adl_swap(T &&lhs, T &&rhs) noexcept(noexcept(swap(std::declval<T>(),
                                                       std::declval<T>()))) {
  swap(std::forward<T>(lhs), std::forward<T>(rhs));
}

}  // namespace adl_detail

template <typename ContainerTy>
decltype(auto) adl_begin(ContainerTy &&container) {
  return adl_detail::adl_begin(std::forward<ContainerTy>(container));
}

template <typename ContainerTy>
decltype(auto) adl_end(ContainerTy &&container) {
  return adl_detail::adl_end(std::forward<ContainerTy>(container));
}

template <typename T>
void adl_swap(T &&lhs, T &&rhs) noexcept(
    noexcept(adl_detail::adl_swap(std::declval<T>(), std::declval<T>()))) {
  adl_detail::adl_swap(std::forward<T>(lhs), std::forward<T>(rhs));
}

/// Provide wrappers to std::for_each which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename UnaryFunction>
UnaryFunction for_each(R &&Range, UnaryFunction F) {
  return std::for_each(adl_begin(Range), adl_end(Range), F);
}

/// Provide wrappers to std::all_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool all_of(R &&Range, UnaryPredicate P) {
  return std::all_of(adl_begin(Range), adl_end(Range), P);
}

/// Provide wrappers to std::any_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool any_of(R &&Range, UnaryPredicate P) {
  return std::any_of(adl_begin(Range), adl_end(Range), P);
}

/// Provide wrappers to std::none_of which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
bool none_of(R &&Range, UnaryPredicate P) {
  return std::none_of(adl_begin(Range), adl_end(Range), P);
}

/// Provide wrappers to std::find which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename T>
auto find(R &&Range, const T &Val) {
  return std::find(adl_begin(Range), adl_end(Range), Val);
}

/// Provide wrappers to std::find_if which take ranges instead of having to pass
/// begin/end explicitly.
template <typename R, typename UnaryPredicate>
auto find_if(R &&Range, UnaryPredicate P) {
  return std::find_if(adl_begin(Range), adl_end(Range), P);
}

template <typename R, typename UnaryPredicate>
auto find_if_not(R &&Range, UnaryPredicate P) {
  return std::find_if_not(adl_begin(Range), adl_end(Range), P);
}

/// Provide wrappers to std::remove_if which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename UnaryPredicate>
auto remove_if(R &&Range, UnaryPredicate P) {
  return std::remove_if(adl_begin(Range), adl_end(Range), P);
}

/// Provide wrappers to std::copy_if which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename OutputIt, typename UnaryPredicate>
OutputIt copy_if(R &&Range, OutputIt Out, UnaryPredicate P) {
  return std::copy_if(adl_begin(Range), adl_end(Range), Out, P);
}

//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

namespace callable_detail {

/// Templated storage wrapper for a callable.
///
/// This class is consistently default constructible, copy / move
/// constructible / assignable.
///
/// Supported callable types:
///  - Function pointer
///  - Function reference
///  - Lambda
///  - Function object
template <typename T,
          bool = std::is_function_v<std::remove_pointer_t<remove_cvref_t<T>>>>
class Callable {
  using value_type = std::remove_reference_t<T>;
  using reference = value_type &;
  using const_reference = value_type const &;

  std::optional<value_type> Obj;

  static_assert(!std::is_pointer_v<value_type>,
                "Pointers to non-functions are not callable.");

 public:
  Callable() = default;
  Callable(T const &O) : Obj(std::in_place, O) {}

  Callable(Callable const &Other) = default;
  Callable(Callable &&Other) = default;

  Callable &operator=(Callable const &Other) {
    Obj = std::nullopt;
    if (Other.Obj) Obj.emplace(*Other.Obj);
    return *this;
  }

  Callable &operator=(Callable &&Other) {
    Obj = std::nullopt;
    if (Other.Obj) Obj.emplace(std::move(*Other.Obj));
    return *this;
  }

  template <typename... Pn,
            std::enable_if_t<std::is_invocable_v<T, Pn...>, int> = 0>
  decltype(auto) operator()(Pn &&...Params) {
    return (*Obj)(std::forward<Pn>(Params)...);
  }

  template <typename... Pn,
            std::enable_if_t<std::is_invocable_v<T const, Pn...>, int> = 0>
  decltype(auto) operator()(Pn &&...Params) const {
    return (*Obj)(std::forward<Pn>(Params)...);
  }

  bool valid() const { return Obj != std::nullopt; }
  bool reset() { return Obj = std::nullopt; }

  operator reference() { return *Obj; }
  operator const_reference() const { return *Obj; }
};

// Function specialization.  No need to waste extra space wrapping with a
// std::optional.
template <typename T>
class Callable<T, true> {
  static constexpr bool IsPtr = std::is_pointer_v<remove_cvref_t<T>>;

  using StorageT = std::conditional_t<IsPtr, T, std::remove_reference_t<T> *>;
  using CastT = std::conditional_t<IsPtr, T, T &>;

 private:
  StorageT Func = nullptr;

 private:
  template <typename In>
  static constexpr auto convertIn(In &&I) {
    if constexpr (IsPtr) {
      // Pointer... just echo it back.
      return I;
    } else {
      // Must be a function reference.  Return its address.
      return &I;
    }
  }

 public:
  Callable() = default;

  // Construct from a function pointer or reference.
  template <typename FnPtrOrRef,
            std::enable_if_t<
                !std::is_same_v<remove_cvref_t<FnPtrOrRef>, Callable>, int> = 0>
  Callable(FnPtrOrRef &&F) : Func(convertIn(F)) {}

  template <typename... Pn,
            std::enable_if_t<std::is_invocable_v<T, Pn...>, int> = 0>
  decltype(auto) operator()(Pn &&...Params) const {
    return Func(std::forward<Pn>(Params)...);
  }

  bool valid() const { return Func != nullptr; }
  void reset() { Func = nullptr; }

  operator T const &() const {
    if constexpr (IsPtr) {
      // T is a pointer... just echo it back.
      return Func;
    } else {
      static_assert(std::is_reference_v<T>,
                    "Expected a reference to a function.");
      // T is a function reference... dereference the stored pointer.
      return *Func;
    }
  }
};

}  // namespace callable_detail

template <typename ItTy, typename FuncTy,
          typename ReferenceTy =
              decltype(std::declval<FuncTy>()(*std::declval<ItTy>()))>
class mapped_iterator
    : public iterator_adaptor_base<
          mapped_iterator<ItTy, FuncTy>, ItTy,
          typename std::iterator_traits<ItTy>::iterator_category,
          std::remove_reference_t<ReferenceTy>,
          typename std::iterator_traits<ItTy>::difference_type,
          std::remove_reference_t<ReferenceTy> *, ReferenceTy> {
 public:
  mapped_iterator() = default;
  mapped_iterator(ItTy U, FuncTy F)
      : mapped_iterator::iterator_adaptor_base(std::move(U)), F(std::move(F)) {}

  ItTy getCurrent() { return this->I; }

  const FuncTy &getFunction() const { return F; }

  ReferenceTy operator*() const { return F(*this->I); }

 private:
  callable_detail::Callable<FuncTy> F{};
};

template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(ItTy I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(std::move(I), std::move(F));
}

template <class ContainerTy, class FuncTy>
auto map_range(ContainerTy &&C, FuncTy F) {
  return make_range(map_iterator(C.begin(), F), map_iterator(C.end(), F));
}

template <typename T, typename... Ts>
using is_one_of = std::disjunction<std::is_same<T, Ts>...>;

namespace detail {
template <class, template <class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
};
template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};
}  // end namespace detail

/// Detects if a given trait holds for some set of arguments 'Args'.
/// For example, the given trait could be used to detect if a given type
/// has a copy assignment operator:
///   template<class T>
///   using has_copy_assign_t = decltype(std::declval<T&>()
///                                                 = std::declval<const T&>());
///   bool fooHasCopyAssign = is_detected<has_copy_assign_t, FooClass>::value;
template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;

template <typename R, typename OutputIt>
OutputIt copy(R &&Range, OutputIt Out) {
  return std::copy(adl_begin(Range), adl_end(Range), Out);
}

/// Provide wrappers to std::replace_copy_if which take ranges instead of having
/// to pass begin/end explicitly.
template <typename R, typename OutputIt, typename UnaryPredicate, typename T>
OutputIt replace_copy_if(R &&Range, OutputIt Out, UnaryPredicate P,
                         const T &NewValue) {
  return std::replace_copy_if(adl_begin(Range), adl_end(Range), Out, P,
                              NewValue);
}

/// Provide wrappers to std::replace_copy which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename OutputIt, typename T>
OutputIt replace_copy(R &&Range, OutputIt Out, const T &OldValue,
                      const T &NewValue) {
  return std::replace_copy(adl_begin(Range), adl_end(Range), Out, OldValue,
                           NewValue);
}

/// Provide wrappers to std::move which take ranges instead of having to
/// pass begin/end explicitly.
template <typename R, typename OutputIt>
OutputIt move(R &&Range, OutputIt Out) {
  return std::move(adl_begin(Range), adl_end(Range), Out);
}

/// Wrapper function around std::find to detect if an element exists
/// in a container.
template <typename R, typename E>
bool is_contained(R &&Range, const E &Element) {
  return std::find(adl_begin(Range), adl_end(Range), Element) != adl_end(Range);
}

template <typename T>
constexpr bool is_contained(std::initializer_list<T> Set, T Value) {
  for (T V : Set)
    if (V == Value) return true;
  return false;
}

/// This class provides various trait information about a callable object.
///   * To access the number of arguments: Traits::num_args
///   * To access the type of an argument: Traits::arg_t<Index>
///   * To access the type of the result:  Traits::result_t
template <typename T, bool isClass = std::is_class<T>::value>
struct function_traits : public function_traits<decltype(&T::operator())> {};

/// Overload for class function types.
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const, false> {
  /// The number of arguments to this function.
  enum { num_args = sizeof...(Args) };

  /// The result type of this function.
  using result_t = ReturnType;

  /// The type of an argument to this function.
  template <size_t Index>
  using arg_t = std::tuple_element_t<Index, std::tuple<Args...>>;
};
/// Overload for class function types.
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...), false>
    : public function_traits<ReturnType (ClassType::*)(Args...) const> {};
/// Overload for non-class function types.
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType (*)(Args...), false> {
  /// The number of arguments to this function.
  enum { num_args = sizeof...(Args) };

  /// The result type of this function.
  using result_t = ReturnType;

  /// The type of an argument to this function.
  template <size_t i>
  using arg_t = std::tuple_element_t<i, std::tuple<Args...>>;
};
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType (*const)(Args...), false>
    : public function_traits<ReturnType (*)(Args...)> {};
/// Overload for non-class function type references.
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType (&)(Args...), false>
    : public function_traits<ReturnType (*)(Args...)> {};

}  // namespace domino

#endif  // DOMINO_UTIL_STLEXTRAS_H_
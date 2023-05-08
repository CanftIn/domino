#ifndef DOMINO_UTIL_STLEXTRAS_H_
#define DOMINO_UTIL_STLEXTRAS_H_

#include <domino/util/Iterator.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace domino {
class StringRef;

template <typename T, T>
struct SameType;

namespace adl_detail {

using std::begin;

template <typename RangeT>
constexpr auto begin_impl(RangeT &&range)
    -> decltype(begin(std::forward<RangeT>(range))) {
  return begin(std::forward<RangeT>(range));
}

using std::end;

template <typename RangeT>
constexpr auto end_impl(RangeT &&range)
    -> decltype(end(std::forward<RangeT>(range))) {
  return end(std::forward<RangeT>(range));
}

using std::swap;

template <typename T>
constexpr void swap_impl(T &&lhs,
                         T &&rhs) noexcept(noexcept(swap(std::declval<T>(),
                                                         std::declval<T>()))) {
  swap(std::forward<T>(lhs), std::forward<T>(rhs));
}

using std::size;

template <typename RangeT>
constexpr auto size_impl(RangeT &&range)
    -> decltype(size(std::forward<RangeT>(range))) {
  return size(std::forward<RangeT>(range));
}

}  // end namespace adl_detail

/// Returns the begin iterator to \p range using `std::begin` and
/// function found through Argument-Dependent Lookup (ADL).
template <typename RangeT>
constexpr auto adl_begin(RangeT &&range)
    -> decltype(adl_detail::begin_impl(std::forward<RangeT>(range))) {
  return adl_detail::begin_impl(std::forward<RangeT>(range));
}

/// Returns the end iterator to \p range using `std::end` and
/// functions found through Argument-Dependent Lookup (ADL).
template <typename RangeT>
constexpr auto adl_end(RangeT &&range)
    -> decltype(adl_detail::end_impl(std::forward<RangeT>(range))) {
  return adl_detail::end_impl(std::forward<RangeT>(range));
}

/// Swaps \p lhs with \p rhs using `std::swap` and functions found through
/// Argument-Dependent Lookup (ADL).
template <typename T>
constexpr void adl_swap(T &&lhs, T &&rhs) noexcept(
    noexcept(adl_detail::swap_impl(std::declval<T>(), std::declval<T>()))) {
  adl_detail::swap_impl(std::forward<T>(lhs), std::forward<T>(rhs));
}

/// Returns the size of \p range using `std::size` and functions found through
/// Argument-Dependent Lookup (ADL).
template <typename RangeT>
constexpr auto adl_size(RangeT &&range)
    -> decltype(adl_detail::size_impl(std::forward<RangeT>(range))) {
  return adl_detail::size_impl(std::forward<RangeT>(range));
}

namespace detail {

template <typename RangeT>
using IterOfRange = decltype(adl_begin(std::declval<RangeT &>()));

template <typename RangeT>
using ValueOfRange =
    std::remove_reference_t<decltype(*adl_begin(std::declval<RangeT &>()))>;

}  // end namespace detail

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

/// An STL-style algorithm similar to std::for_each that applies a second
/// functor between every pair of elements.
///
/// This provides the control flow logic to, for example, print a
/// comma-separated list:
/// \code
///   interleave(names.begin(), names.end(),
///              [&](StringRef name) { os << name; },
///              [&] { os << ", "; });
/// \endcode
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor,
          typename = std::enable_if_t<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end) return;
  each_fn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    each_fn(*begin);
  }
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor,
          typename = std::enable_if_t<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>>
inline void interleave(const Container &c, UnaryFunctor each_fn,
                       NullaryFunctor between_fn) {
  interleave(c.begin(), c.end(), each_fn, between_fn);
}

/// Overload of interleave for the common case of string separator.
template <typename Container, typename UnaryFunctor, typename StreamT,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, StreamT &os, UnaryFunctor each_fn,
                       const StringRef &separator) {
  interleave(c.begin(), c.end(), each_fn, [&] { os << separator; });
}
template <typename Container, typename StreamT,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, StreamT &os,
                       const StringRef &separator) {
  interleave(
      c, os, [&](const T &a) { os << a; }, separator);
}

template <typename Container, typename UnaryFunctor, typename StreamT,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, StreamT &os,
                            UnaryFunctor each_fn) {
  interleave(c, os, each_fn, ", ");
}
template <typename Container, typename StreamT,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, StreamT &os) {
  interleaveComma(c, os, [&](const T &a) { os << a; });
}

namespace detail {

using std::declval;

// We have to alias this since inlining the actual type at the usage site
// in the parameter list of iterator_facade_base<> below ICEs MSVC 2017.
template <typename... Iters>
struct ZipTupleType {
  using type = std::tuple<decltype(*declval<Iters>())...>;
};

template <typename ZipType, typename ReferenceTupleType, typename... Iters>
using zip_traits = iterator_facade_base<
    ZipType,
    std::common_type_t<
        std::bidirectional_iterator_tag,
        typename std::iterator_traits<Iters>::iterator_category...>,
    // ^ TODO: Implement random access methods.
    ReferenceTupleType,
    typename std::iterator_traits<
        std::tuple_element_t<0, std::tuple<Iters...>>>::difference_type,
    // ^ FIXME: This follows boost::make_zip_iterator's assumption that all
    // inner iterators have the same difference_type. It would fail if, for
    // instance, the second field's difference_type were non-numeric while the
    // first is.
    ReferenceTupleType *, ReferenceTupleType>;

template <typename ZipType, typename ReferenceTupleType, typename... Iters>
struct zip_common : public zip_traits<ZipType, ReferenceTupleType, Iters...> {
  using Base = zip_traits<ZipType, ReferenceTupleType, Iters...>;
  using IndexSequence = std::index_sequence_for<Iters...>;
  using value_type = typename Base::value_type;

  std::tuple<Iters...> iterators;

 protected:
  template <size_t... Ns>
  value_type deref(std::index_sequence<Ns...>) const {
    return value_type(*std::get<Ns>(iterators)...);
  }

  template <size_t... Ns>
  void tup_inc(std::index_sequence<Ns...>) {
    (++std::get<Ns>(iterators), ...);
  }

  template <size_t... Ns>
  void tup_dec(std::index_sequence<Ns...>) {
    (--std::get<Ns>(iterators), ...);
  }

  template <size_t... Ns>
  bool test_all_equals(const zip_common &other,
                       std::index_sequence<Ns...>) const {
    return ((std::get<Ns>(this->iterators) == std::get<Ns>(other.iterators)) &&
            ...);
  }

 public:
  zip_common(Iters &&...ts) : iterators(std::forward<Iters>(ts)...) {}

  value_type operator*() const { return deref(IndexSequence{}); }

  ZipType &operator++() {
    tup_inc(IndexSequence{});
    return static_cast<ZipType &>(*this);
  }

  ZipType &operator--() {
    static_assert(Base::IsBidirectional,
                  "All inner iterators must be at least bidirectional.");
    tup_dec(IndexSequence{});
    return static_cast<ZipType &>(*this);
  }

  /// Return true if all the iterator are matching `other`'s iterators.
  bool all_equals(zip_common &other) {
    return test_all_equals(other, IndexSequence{});
  }
};

template <typename... Iters>
struct zip_first : zip_common<zip_first<Iters...>,
                              typename ZipTupleType<Iters...>::type, Iters...> {
  using zip_common<zip_first, typename ZipTupleType<Iters...>::type,
                   Iters...>::zip_common;

  bool operator==(const zip_first &other) const {
    return std::get<0>(this->iterators) == std::get<0>(other.iterators);
  }
};

template <typename... Iters>
struct zip_shortest
    : zip_common<zip_shortest<Iters...>, typename ZipTupleType<Iters...>::type,
                 Iters...> {
  using zip_common<zip_shortest, typename ZipTupleType<Iters...>::type,
                   Iters...>::zip_common;

  bool operator==(const zip_shortest &other) const {
    return any_iterator_equals(other, std::index_sequence_for<Iters...>{});
  }

 private:
  template <size_t... Ns>
  bool any_iterator_equals(const zip_shortest &other,
                           std::index_sequence<Ns...>) const {
    return ((std::get<Ns>(this->iterators) == std::get<Ns>(other.iterators)) ||
            ...);
  }
};

/// Helper to obtain the iterator types for the tuple storage within `zippy`.
template <template <typename...> class ItType, typename TupleStorageType,
          typename IndexSequence>
struct ZippyIteratorTuple;

/// Partial specialization for non-const tuple storage.
template <template <typename...> class ItType, typename... Args,
          std::size_t... Ns>
struct ZippyIteratorTuple<ItType, std::tuple<Args...>,
                          std::index_sequence<Ns...>> {
  using type = ItType<decltype(adl_begin(
      std::get<Ns>(declval<std::tuple<Args...> &>())))...>;
};

/// Partial specialization for const tuple storage.
template <template <typename...> class ItType, typename... Args,
          std::size_t... Ns>
struct ZippyIteratorTuple<ItType, const std::tuple<Args...>,
                          std::index_sequence<Ns...>> {
  using type = ItType<decltype(adl_begin(
      std::get<Ns>(declval<const std::tuple<Args...> &>())))...>;
};

template <template <typename...> class ItType, typename... Args>
class zippy {
 private:
  std::tuple<Args...> storage;
  using IndexSequence = std::index_sequence_for<Args...>;

 public:
  using iterator = typename ZippyIteratorTuple<ItType, decltype(storage),
                                               IndexSequence>::type;
  using const_iterator =
      typename ZippyIteratorTuple<ItType, const decltype(storage),
                                  IndexSequence>::type;
  using iterator_category = typename iterator::iterator_category;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using pointer = typename iterator::pointer;
  using reference = typename iterator::reference;
  using const_reference = typename const_iterator::reference;

  zippy(Args &&...args) : storage(std::forward<Args>(args)...) {}

  const_iterator begin() const { return begin_impl(IndexSequence{}); }
  iterator begin() { return begin_impl(IndexSequence{}); }
  const_iterator end() const { return end_impl(IndexSequence{}); }
  iterator end() { return end_impl(IndexSequence{}); }

 private:
  template <size_t... Ns>
  const_iterator begin_impl(std::index_sequence<Ns...>) const {
    return const_iterator(adl_begin(std::get<Ns>(storage))...);
  }
  template <size_t... Ns>
  iterator begin_impl(std::index_sequence<Ns...>) {
    return iterator(adl_begin(std::get<Ns>(storage))...);
  }

  template <size_t... Ns>
  const_iterator end_impl(std::index_sequence<Ns...>) const {
    return const_iterator(adl_end(std::get<Ns>(storage))...);
  }
  template <size_t... Ns>
  iterator end_impl(std::index_sequence<Ns...>) {
    return iterator(adl_end(std::get<Ns>(storage))...);
  }
};

}  // end namespace detail

/// zip iterator for two or more iteratable types. Iteration continues until the
/// end of the *shortest* iteratee is reached.
template <typename T, typename U, typename... Args>
detail::zippy<detail::zip_shortest, T, U, Args...> zip(T &&t, U &&u,
                                                       Args &&...args) {
  return detail::zippy<detail::zip_shortest, T, U, Args...>(
      std::forward<T>(t), std::forward<U>(u), std::forward<Args>(args)...);
}

/// zip iterator that assumes that all iteratees have the same length.
/// In builds with assertions on, this assumption is checked before the
/// iteration starts.
template <typename T, typename U, typename... Args>
detail::zippy<detail::zip_first, T, U, Args...> zip_equal(T &&t, U &&u,
                                                          Args &&...args) {
  assert(all_equal({range_size(t), range_size(u), range_size(args)...}) &&
         "Iteratees do not have equal length");
  return detail::zippy<detail::zip_first, T, U, Args...>(
      std::forward<T>(t), std::forward<U>(u), std::forward<Args>(args)...);
}

/// zip iterator that, for the sake of efficiency, assumes the first iteratee to
/// be the shortest. Iteration continues until the end of the first iteratee is
/// reached. In builds with assertions on, we check that the assumption about
/// the first iteratee being the shortest holds.
template <typename T, typename U, typename... Args>
detail::zippy<detail::zip_first, T, U, Args...> zip_first(T &&t, U &&u,
                                                          Args &&...args) {
  assert(range_size(t) <= std::min({range_size(u), range_size(args)...}) &&
         "First iteratee is not the shortest");

  return detail::zippy<detail::zip_first, T, U, Args...>(
      std::forward<T>(t), std::forward<U>(u), std::forward<Args>(args)...);
}

}  // namespace domino

#endif  // DOMINO_UTIL_STLEXTRAS_H_
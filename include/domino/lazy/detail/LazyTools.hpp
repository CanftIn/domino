#ifndef DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_
#define DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_

#include <iterator>
#include <tuple>

namespace domino::lazy {

#define LAZY_HAS_CONCEPTS
#ifdef LAZY_HAS_CONCEPTS
template <class I>
concept BasicIterable = requires(I i) {
  { std::begin(i) } -> std::input_or_output_iterator;
  { std::end(i) } -> std::input_or_output_iterator;
};

template <class I>
concept BidirectionalIterable = requires(I i) {
  { std::begin(i) } -> std::bidirectional_iterator;
  { std::end(i) } -> std::bidirectional_iterator;
};

template <class I>
concept Arithmetic = std::is_arithmetic_v<I>;

#define LAZY_CONCEPT_ARITHMETIC Arithmetic
#define LAZY_CONCEPT_INTEGRAL std::integral
#define LAZY_CONCEPT_INVOCABLE std::invocable
#define LAZY_CONCEPT_ITERABLE BasicIterable
#define LAZY_CONCEPT_ITERATOR std::input_or_output_iterator
#define LAZY_CONCEPT_BIDIRECTIONAL_ITERATOR std::bidirectional_iterator
#define LAZY_CONCEPT_BIDIRECTIONAL_ITERABLE BidirectionalIterable

#endif  // LAZY_HAS_CONCEPTS

namespace internal {

template <class>
struct AlwaysFalse : std::false_type {};

[[noreturn]] inline void assertionFail(const char* file, const int line,
                                       const char* func, const char* message) {
  std::fprintf(stderr,
               "%s:%d assertion failed in function '%s' with message:\n\t%s\n",
               file, line, func, message);
  std::terminate();
}

#define DOMINO_LAZY_ASSERT(CONDITION, MSG) \
  ((CONDITION) ? ((void)0) : (assertionFail(__FILE__, __LINE__, __func__, MSG)))

template <class Iterable>
constexpr auto begin(Iterable&& c) noexcept
    -> decltype(std::forward<Iterable>(c).begin()) {
  return std::forward<Iterable>(c).begin();
}

template <class Iterable>
constexpr auto end(Iterable&& c) noexcept
    -> decltype(std::forward<Iterable>(c).end()) {
  return std::forward<Iterable>(c).end();
}

template <class T, size_t N>
constexpr auto begin(T (&array)[N]) noexcept -> T* {
  return std::begin(array);
}

template <class T, size_t N>
constexpr auto end(T (&array)[N]) noexcept -> T* {
  return std::end(array);
}

template <size_t...>
struct IndexSequence {};

template <size_t N, size_t... Rest>
struct IndexSequenceHelper : public IndexSequenceHelper<N - 1, N - 1, Rest...> {
};

template <size_t... Next>
struct IndexSequenceHelper<0, Next...> {
  using Type = IndexSequence<Next...>;
};

template <size_t N>
using MakeIndexSequence = typename IndexSequenceHelper<N>::Type;

template <class T>
using Decay = typename std::decay_t<T>;

template <size_t I, class T>
using TupleElement = typename std::tuple_element_t<I, T>;

template <class Iterable>
using IterTypeFromIterable =
    decltype(begin(std::forward<Iterable>(std::declval<Iterable>())));

template <class Iterator>
using ValueType = typename std::iterator_traits<Iterator>::value_type;

template <class Iterator>
using RefType = typename std::iterator_traits<Iterator>::reference;

template <class Iterator>
using PointerType = typename std::iterator_traits<Iterator>::pointer;

template <class Iterator>
using DiffType = typename std::iterator_traits<Iterator>::difference_type;

template <class Iterator>
using IterCat = typename std::iterator_traits<Iterator>::iterator_category;

template <class Function, class... Args>
using FunctionReturnType =
    decltype(std::declval<Function>()(std::declval<Args>()...));

template <class Iterable>
using ValueTypeIterable =
    typename std::iterator_traits<IterTypeFromIterable<Iterable>>::value_type;

template <class Iterable>
using DiffTypeIterable = typename std::iterator_traits<
    IterTypeFromIterable<Iterable>>::difference_type;

template <bool B>
struct EnableIfImpl {};

template <>
struct EnableIfImpl<true> {
  template <class T>
  using type = T;
};

template <bool B, class T = void>
using EnableIf = typename EnableIfImpl<B>::template type<T>;

template <bool B>
struct ConditionalImpl;

template <>
struct ConditionalImpl<true> {
  template <class IfTrue, class /* IfFalse */>
  using type = IfTrue;
};

template <>
struct ConditionalImpl<false> {
  template <class /* IfTrue */, class IfFalse>
  using type = IfFalse;
};

template <bool B, class IfTrue, class IfFalse>
using Conditional = typename ConditionalImpl<B>::template type<IfTrue, IfFalse>;

template <class T, class U, class... Vs>
struct IsAllSame
    : std::integral_constant<bool, std::is_same<T, U>::value &&
                                       IsAllSame<U, Vs...>::value> {};

template <class T, class U>
struct IsAllSame<T, U> : std::is_same<T, U> {};

template <class IterTag>
struct IsBidirectionalTag
    : std::is_convertible<IterTag, std::bidirectional_iterator_tag> {};

template <class Iterator>
struct IsBidirectional : IsBidirectionalTag<IterCat<Iterator>> {};

template <class Iterator>
struct IsForward
    : std::is_convertible<IterCat<Iterator>, std::forward_iterator_tag> {};

template <class IterTag>
struct IsRandomAccessTag
    : std::is_convertible<IterTag, std::random_access_iterator_tag> {};

template <class Iterator>
struct IsRandomAccess : IsRandomAccessTag<IterCat<Iterator>> {};

template <class Iter>
auto sizeHint(Iter first, Iter last) -> DiffType<Iter> {
  if constexpr (IsRandomAccess<Iter>::value) {
    return std::distance(std::move(first), std::move(last));
  } else {
    return 0;
  }
}

}  // namespace internal

}  // namespace domino::lazy

#endif  // DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_
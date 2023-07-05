#ifndef DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_
#define DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_

#include <iterator>

namespace domino::lazy {

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

}  // namespace internal

}  // namespace domino::lazy

#endif  // DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_
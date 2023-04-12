#include <gtest/gtest.h>

#include <domino/meta/detail/utility.hpp>

using domino::meta::meta_invoke_q;

template <class...>
struct X {};

template <template <class...> class F, class... T>
using Y = X<F<T>...>;

template <class Q, class... T>
using Z = X<meta_invoke_q<Q, T>...>;

template <class T, class U>
struct P {};

template <class T, class U>
using first = T;

TEST(Meta, meta_quote) {
  using domino::meta::meta_identity_t;
  using domino::meta::meta_quote;

  {
    using Q = meta_quote<meta_identity_t>;

    EXPECT_TRUE((std::is_same<meta_invoke_q<Q, void>, void>::value));
    EXPECT_TRUE((std::is_same<meta_invoke_q<Q, int[]>, int[]>::value));
  }

  {
    using Q = meta_quote<P>;

    EXPECT_TRUE(
        (std::is_same<meta_invoke_q<Q, void, void>, P<void, void>>::value));
    EXPECT_TRUE((std::is_same<meta_invoke_q<Q, char[], int[]>,
                              P<char[], int[]>>::value));
  }

  {
    using Q = meta_quote<first>;

    EXPECT_TRUE((std::is_same<meta_invoke_q<Q, void, int>, void>::value));
    EXPECT_TRUE((std::is_same<meta_invoke_q<Q, char[], int[]>, char[]>::value));
  }
}
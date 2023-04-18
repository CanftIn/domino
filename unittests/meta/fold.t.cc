#include <gtest/gtest.h>

#include <domino/meta/detail/fold.hpp>
#include <domino/meta/list.hpp>

struct X1 {};
struct X2 {};
struct X3 {};
struct X4 {};

template <class T1, class T2>
struct F {};

TEST(Fold, meta_fold) {
  using namespace domino::meta;

  {
    EXPECT_TRUE((std::is_same<meta_fold<meta_list<>, void, F>, void>::value));
    EXPECT_TRUE((std::is_same<meta_fold<meta_list<X1>, void, F>, F<void, X1>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<meta_list<X1, X2>, void, F>,
                              F<F<void, X1>, X2>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<meta_list<X1, X2, X3>, void, F>,
                              F<F<F<void, X1>, X2>, X3>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<meta_list<X1, X2, X3, X4>, void, F>,
                              F<F<F<F<void, X1>, X2>, X3>, X4>>::value));
  }

  {
    EXPECT_TRUE((std::is_same<meta_fold<std::tuple<>, void, F>, void>::value));
    EXPECT_TRUE((std::is_same<meta_fold<std::tuple<X1>, void, F>, F<void, X1>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<std::tuple<X1, X2>, void, F>,
                              F<F<void, X1>, X2>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<std::tuple<X1, X2, X3>, void, F>,
                              F<F<F<void, X1>, X2>, X3>>::value));
    EXPECT_TRUE((std::is_same<meta_fold<std::tuple<X1, X2, X3, X4>, void, F>,
                              F<F<F<F<void, X1>, X2>, X3>, X4>>::value));
  }
}
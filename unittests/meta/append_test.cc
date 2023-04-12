#include <gtest/gtest.h>

#include <domino/meta/detail/utility.hpp>
#include <domino/meta/list.hpp>
#include <type_traits>
#include <tuple>
#include <utility>

struct X1 {};
struct X2 {};
struct X3 {};
struct X4 {};
struct X5 {};
struct X6 {};

TEST(Meta, list_append) {
  using domino::meta::meta_list;
  using domino::meta::meta_append;

  using L1 = meta_list<char[1], char[1]>;
  using L2 = meta_list<char[2], char[2]>;
  using L3 = meta_list<char[3], char[3]>;
  using L4 = meta_list<char[4], char[4]>;

  EXPECT_TRUE((std::is_same<meta_append<>, meta_list<>>::value));
  EXPECT_TRUE((std::is_same<meta_append<L1>, meta_list<char[1], char[1]>>::value));
  EXPECT_TRUE((std::is_same<meta_append<L1, L2>, meta_list<char[1], char[1], char[2], char[2]>>::value));
  EXPECT_TRUE((std::is_same<meta_append<L1, L2, L3>, meta_list<char[1], char[1], char[2], char[2], char[3], char[3]>>::value));
  EXPECT_TRUE((std::is_same<meta_append<L1, L2, L3, L4>, meta_list<char[1], char[1], char[2], char[2], char[3], char[3], char[4], char[4]>>::value));

  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>>, std::tuple<>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>>, std::tuple<X1>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>, std::tuple<X2>>, std::tuple<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>, std::tuple<X2>, std::tuple<X3>>, std::tuple<X1, X2, X3>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>, std::tuple<X2>, std::tuple<X3>, std::tuple<X4>>, std::tuple<X1, X2, X3, X4>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>, std::tuple<X2>, std::tuple<X3>, std::tuple<X4>, std::tuple<X5>>, std::tuple<X1, X2, X3, X4, X5>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::tuple<X1>, std::tuple<X2>, std::tuple<X3>, std::tuple<X4>, std::tuple<X5>, std::tuple<X6>>, std::tuple<X1, X2, X3, X4, X5, X6>>::value));

  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<>>, std::tuple<>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>>, std::tuple<X1>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>, std::tuple<X2>>, std::tuple<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>, std::tuple<X2>, meta_list<X3>>, std::tuple<X1, X2, X3>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>, std::tuple<X2>, meta_list<X3>, std::tuple<X4>>, std::tuple<X1, X2, X3, X4>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>, std::tuple<X2>, meta_list<X3>, std::tuple<X4>, meta_list<X5>>, std::tuple<X1, X2, X3, X4, X5>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, meta_list<X1>, std::tuple<X2>, meta_list<X3>, std::tuple<X4>, meta_list<X5>, std::tuple<X6>>, std::tuple<X1, X2, X3, X4, X5, X6>>::value));

  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::pair<X1, X2>>, std::tuple<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::pair<X1, X2>, std::pair<X3, X4>>, std::tuple<X1, X2, X3, X4>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::tuple<>, std::pair<X1, X2>, std::pair<X3, X4>, std::pair<X5, X6>>, std::tuple<X1, X2, X3, X4, X5, X6>>::value));

  EXPECT_TRUE((std::is_same<meta_append<std::pair<X1, X2>>, std::pair<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::pair<X1, X2>, meta_list<>>, std::pair<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::pair<X1, X2>, meta_list<>, meta_list<>>, std::pair<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::pair<X1, X2>, meta_list<>, meta_list<>, meta_list<>>, std::pair<X1, X2>>::value));
  EXPECT_TRUE((std::is_same<meta_append<std::pair<X1, X2>, meta_list<>, meta_list<>, meta_list<>, meta_list<>>, std::pair<X1, X2>>::value));
}
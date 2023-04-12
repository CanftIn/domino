#include <gtest/gtest.h>

#include <domino/meta/list.hpp>
#include <type_traits>

TEST(Meta, list_c) {
  using domino::meta::meta_int;
  using domino::meta::meta_list;
  using domino::meta::meta_list_c;

  EXPECT_TRUE((std::is_same<meta_list_c<int>, meta_list<>>::value));
  EXPECT_TRUE((std::is_same<meta_list_c<int, 1>, meta_list<meta_int<1>>>::value));
  EXPECT_TRUE((std::is_same<meta_list_c<int, 1, 3>,
                            meta_list<meta_int<1>, meta_int<3>>>::value));
  EXPECT_TRUE((std::is_same<meta_list_c<int, 1, 3, 5>,
                            meta_list<meta_int<1>, meta_int<3>, meta_int<5>>>::value));
}
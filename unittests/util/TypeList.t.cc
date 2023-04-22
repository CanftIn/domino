#include <domino/util/TypeList.h>
#include <gtest/gtest.h>

#include <type_traits>

using namespace domino;

class MyClass {};

TEST(TypeList, size) {
  EXPECT_TRUE((0 == size<typelist<>>::value));
  EXPECT_TRUE((1 == size<typelist<int>>::value));
  EXPECT_TRUE((3 == size<typelist<int, float&, const MyClass&&>>::value));
}

TEST(TypeList, to_tuple) {
  EXPECT_TRUE((
      std::is_same<std::tuple<int, float&, const MyClass&&>,
                   to_tuple_t<typelist<int, float&, const MyClass&&>>>::value));
  EXPECT_TRUE((std::is_same<std::tuple<>, to_tuple_t<typelist<>>>::value));
}

TEST(TypeList, from_tuple) {
  EXPECT_TRUE((std::is_same<
               typelist<int, float&, const MyClass&&>,
               from_tuple_t<std::tuple<int, float&, const MyClass&&>>>::value));
  EXPECT_TRUE((std::is_same<typelist<>, from_tuple_t<std::tuple<>>>::value));
}

TEST(TypeList, concat) {
  EXPECT_TRUE((std::is_same<typelist<>, concat_t<>>::value));
  EXPECT_TRUE((std::is_same<typelist<>, concat_t<typelist<>>>::value));
  EXPECT_TRUE(
      (std::is_same<typelist<>, concat_t<typelist<>, typelist<>>>::value));
  EXPECT_TRUE((std::is_same<typelist<int>, concat_t<typelist<int>>>::value));
  EXPECT_TRUE((
      std::is_same<typelist<int>, concat_t<typelist<int>, typelist<>>>::value));
  EXPECT_TRUE((
      std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>>>::value));
  EXPECT_TRUE(
      (std::is_same<typelist<int>,
                    concat_t<typelist<>, typelist<int>, typelist<>>>::value));
  EXPECT_TRUE((std::is_same<typelist<int, float&>,
                            concat_t<typelist<int>, typelist<float&>>>::value));
  EXPECT_TRUE(
      (std::is_same<
          typelist<int, float&>,
          concat_t<typelist<>, typelist<int, float&>, typelist<>>>::value));
  EXPECT_TRUE((std::is_same<typelist<int, float&, const MyClass&&>,
                            concat_t<typelist<>, typelist<int, float&>,
                                     typelist<const MyClass&&>>>::value));
}
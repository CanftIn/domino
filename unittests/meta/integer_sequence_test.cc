#include <gtest/gtest.h>

#include <domino/meta/detail/integer_sequence.hpp>

using namespace domino::meta;

TEST(IntegerSequence, make_integer_sequence) {
  EXPECT_TRUE((std::is_same<make_integer_sequence<int, 0>,
                            integer_sequence<int>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<int, 1>,
                            integer_sequence<int, 0>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<int, 2>,
                            integer_sequence<int, 0, 1>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<int, 3>,
                            integer_sequence<int, 0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<int, 4>,
                            integer_sequence<int, 0, 1, 2, 3>>::value));
}
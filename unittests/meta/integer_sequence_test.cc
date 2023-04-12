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

  EXPECT_TRUE((std::is_same<make_integer_sequence<char, 0>,
                            integer_sequence<char>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<char, 1>,
                            integer_sequence<char, 0>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<char, 2>,
                            integer_sequence<char, 0, 1>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<char, 3>,
                            integer_sequence<char, 0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<char, 4>,
                            integer_sequence<char, 0, 1, 2, 3>>::value));

  EXPECT_TRUE((std::is_same<make_integer_sequence<std::size_t, 0>,
                            integer_sequence<std::size_t>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<std::size_t, 1>,
                            integer_sequence<std::size_t, 0>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<std::size_t, 2>,
                            integer_sequence<std::size_t, 0, 1>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<std::size_t, 3>,
                            integer_sequence<std::size_t, 0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<make_integer_sequence<std::size_t, 4>,
                            integer_sequence<std::size_t, 0, 1, 2, 3>>::value));
}

TEST(IntegerSequence, make_index_sequence) {
  EXPECT_TRUE((std::is_same<make_index_sequence<0>,
                            integer_sequence<std::size_t>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<1>,
                            integer_sequence<std::size_t, 0>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<2>,
                            integer_sequence<std::size_t, 0, 1>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<3>,
                            integer_sequence<std::size_t, 0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<4>,
                            integer_sequence<std::size_t, 0, 1, 2, 3>>::value));

  EXPECT_TRUE((std::is_same<make_index_sequence<0>, index_sequence<>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<1>, index_sequence<0>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<2>, index_sequence<0, 1>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<3>, index_sequence<0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<make_index_sequence<4>, index_sequence<0, 1, 2, 3>>::value));
}

TEST(IntegerSequence, index_sequence_for) {
  EXPECT_TRUE((std::is_same<index_sequence_for<>, index_sequence<>>::value));
  EXPECT_TRUE((std::is_same<index_sequence_for<void>, index_sequence<0>>::value));
  EXPECT_TRUE((std::is_same<index_sequence_for<void, void>,
                            index_sequence<0, 1>>::value));
  EXPECT_TRUE((std::is_same<index_sequence_for<void, void, void>,
                            index_sequence<0, 1, 2>>::value));
  EXPECT_TRUE((std::is_same<index_sequence_for<void, void, void, void>,
                            index_sequence<0, 1, 2, 3>>::value));
}
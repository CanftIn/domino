#include <domino/util/Array.h>
#include <gtest/gtest.h>

using namespace domino;

TEST(Array, fill) {
  Array<int, 3> A;
  A.fill(1);
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(1, A[1]);
  EXPECT_EQ(1, A[2]);
}

TEST(Array, swap) {
  Array<int, 3> A;
  A.fill(1);
  Array<int, 3> B;
  B.fill(2);
  A.swap(B);
  EXPECT_EQ(2, A[0]);
  EXPECT_EQ(2, A[1]);
  EXPECT_EQ(2, A[2]);
  EXPECT_EQ(1, B[0]);
  EXPECT_EQ(1, B[1]);
  EXPECT_EQ(1, B[2]);
}

TEST(Array, swap_empty) {
  Array<int, 3> A{{3, 3, 3}};
  Array<int, 2> B{{2, 2}};
  A.swap(B);
  EXPECT_EQ(2, A[0]);
  EXPECT_EQ(2, A[1]);
  EXPECT_EQ(3, A[2]);
}

TEST(Array, operator) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);
  EXPECT_EQ(3, A[2]);
}

TEST(Array, operator_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);
  EXPECT_EQ(3, A[2]);
}

TEST(Array, at) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.at(0));
  EXPECT_EQ(2, A.at(1));
  EXPECT_EQ(3, A.at(2));
}

TEST(Array, at_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.at(0));
  EXPECT_EQ(2, A.at(1));
  EXPECT_EQ(3, A.at(2));
}

TEST(Array, front) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.front());
}

TEST(Array, front_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.front());
}

TEST(Array, back) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(3, A.back());
}

TEST(Array, back_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(3, A.back());
}

TEST(Array, data) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.data()[0]);
  EXPECT_EQ(2, A.data()[1]);
  EXPECT_EQ(3, A.data()[2]);
}

TEST(Array, data_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, A.data()[0]);
  EXPECT_EQ(2, A.data()[1]);
  EXPECT_EQ(3, A.data()[2]);
}

TEST(Array, begin) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, *A.begin());
}

TEST(Array, begin_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, *A.begin());
}

TEST(Array, end) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(3, *(A.end() - 1));
}

TEST(Array, end_const) {
  const Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(3, *(A.end() - 1));
}

TEST(Array, cbegin) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(1, *A.cbegin());
}

TEST(Array, cend) {
  Array<int, 3> A{{1, 2, 3}};
  EXPECT_EQ(3, *(A.cend() - 1));
}

TEST(Array, equals) {
  EXPECT_TRUE((Array<int, 0>{{}} == Array<int, 0>{{}}));
  EXPECT_TRUE((Array<int, 3>{{2, 3, 4}} == Array<int, 3>{{2, 3, 4}}));
  EXPECT_TRUE(!(Array<int, 3>{{2, 3, 4}} == Array<int, 3>{{1, 3, 4}}));
  EXPECT_TRUE(!(Array<int, 3>{{2, 3, 4}} == Array<int, 3>{{2, 1, 4}}));
  EXPECT_TRUE(!(Array<int, 3>{{2, 3, 4}} == Array<int, 3>{{2, 3, 1}}));
}

TEST(Array, notequals) {
  EXPECT_TRUE(!(Array<int, 0>{{}} != Array<int, 0>{{}}));
  EXPECT_TRUE(!(Array<int, 3>{{2, 3, 4}} != Array<int, 3>{{2, 3, 4}}));
  EXPECT_TRUE((Array<int, 3>{{2, 3, 4}} != Array<int, 3>{{1, 3, 4}}));
  EXPECT_TRUE((Array<int, 3>{{2, 3, 4}} != Array<int, 3>{{2, 1, 4}}));
  EXPECT_TRUE((Array<int, 3>{{2, 3, 4}} != Array<int, 3>{{2, 3, 1}}));
}

TEST(Array, lessthan) {
  EXPECT_TRUE(!(Array<int, 0>{{}} < Array<int, 0>{{}}));
  EXPECT_TRUE(!(Array<int, 1>{{2}} < Array<int, 1>{{1}}));
  EXPECT_TRUE((Array<int, 1>{{1}} < Array<int, 1>{{2}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{2, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{0, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{1, 3, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{1, 1, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{1, 2, 4}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} < Array<int, 3>{{1, 2, 2}}));
}

TEST(Array, greaterthan) {
  EXPECT_TRUE(!(Array<int, 0>{{}} > Array<int, 0>{{}}));
  EXPECT_TRUE(!(Array<int, 1>{{1}} > Array<int, 1>{{2}}));
  EXPECT_TRUE((Array<int, 1>{{2}} > Array<int, 1>{{1}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{2, 2, 3}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{0, 2, 3}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 3, 3}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 1, 3}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 4}} > Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 2}} > Array<int, 3>{{1, 2, 3}}));
}

TEST(Array, lessequals) {
  EXPECT_TRUE((Array<int, 0>{{}} <= Array<int, 0>{{}}));
  EXPECT_TRUE(!(Array<int, 1>{{2}} <= Array<int, 1>{{1}}));
  EXPECT_TRUE((Array<int, 1>{{1}} <= Array<int, 1>{{2}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{2, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{0, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{1, 3, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{1, 1, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{1, 2, 4}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 3}} <= Array<int, 3>{{1, 2, 2}}));
}

TEST(Array, greaterequals) {
  EXPECT_TRUE((Array<int, 0>{{}} >= Array<int, 0>{{}}));
  EXPECT_TRUE(!(Array<int, 1>{{1}} >= Array<int, 1>{{2}}));
  EXPECT_TRUE((Array<int, 1>{{2}} >= Array<int, 1>{{1}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 3}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{2, 2, 3}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{0, 2, 3}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 3, 3}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 1, 3}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE((Array<int, 3>{{1, 2, 4}} >= Array<int, 3>{{1, 2, 3}}));
  EXPECT_TRUE(!(Array<int, 3>{{1, 2, 2}} >= Array<int, 3>{{1, 2, 3}}));
}

TEST(Array, tail) {
  EXPECT_TRUE((Array<int, 2>{{3, 4}} == tail(Array<int, 3>{{2, 3, 4}})));
  EXPECT_TRUE((Array<int, 0>{{}} == tail(Array<int, 1>{{3}})));
}

TEST(Array, prepend) {
  EXPECT_TRUE((Array<int, 3>{{2, 3, 4}} == prepend(2, Array<int, 2>{{3, 4}})));
  EXPECT_TRUE((Array<int, 1>{{3}} == prepend(3, Array<int, 0>{{}})));
}

TEST(Array, to_array) {
  constexpr int obj2[3] = {3, 5, 6};
  EXPECT_TRUE((Array<int, 3>{{3, 5, 6}} == to_array(obj2)));
  EXPECT_TRUE((Array<int, 3>{{3, 5, 6}} == to_array<int, 3>({3, 5, 6})));
}

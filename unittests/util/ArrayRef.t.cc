#include <domino/util/ArrayRef.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

using namespace domino;

TEST(ArrayRefTest, DropBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(TheNumbers, AR1.size() - 1);
  EXPECT_TRUE(AR1.drop_back().equals(AR2));
}

TEST(ArrayRefTest, DropFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(&TheNumbers[2], AR1.size() - 2);
  EXPECT_TRUE(AR1.drop_front(2).equals(AR2));
}

TEST(ArrayRefTest, DropWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_while([](const int &N) { return N % 2 == 1; }));

  EXPECT_EQ(AR1, AR1.drop_while([](const int &N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(),
            AR1.drop_while([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, DropUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_until([](const int &N) { return N % 2 == 0; }));

  EXPECT_EQ(ArrayRef<int>(),
            AR1.drop_until([](const int &N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.drop_until([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.end() - 1, 1);
  EXPECT_TRUE(AR1.take_back().equals(AR2));
}

TEST(ArrayRefTest, TakeFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.data(), 2);
  EXPECT_TRUE(AR1.take_front(2).equals(AR2));
}

TEST(ArrayRefTest, TakeWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_while([](const int &N) { return N % 2 == 1; }));

  EXPECT_EQ(ArrayRef<int>(),
            AR1.take_while([](const int &N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.take_while([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_until([](const int &N) { return N % 2 == 0; }));

  EXPECT_EQ(AR1, AR1.take_until([](const int &N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(),
            AR1.take_until([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, Equals) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  ArrayRef<int> AR1(A1);
  EXPECT_TRUE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({8, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({2, 4, 5, 6, 6, 7, 8, 1}));
  EXPECT_FALSE(AR1.equals({0, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 42, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({42, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 42}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8, 9}));

  ArrayRef<int> AR1a = AR1.drop_back();
  EXPECT_TRUE(AR1a.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1a.equals({1, 2, 3, 4, 5, 6, 7, 8}));

  ArrayRef<int> AR1b = AR1a.slice(2, 4);
  EXPECT_TRUE(AR1b.equals({3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({2, 3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({3, 4, 5, 6, 7}));
}

TEST(ArrayRefTest, EmptyEquals) {
  EXPECT_TRUE(ArrayRef<unsigned>() == ArrayRef<unsigned>());
}

TEST(ArrayRefTest, ConstConvert) {
  int buf[4];
  for (int i = 0; i < 4; ++i) buf[i] = i;

  static int *A[] = {&buf[0], &buf[1], &buf[2], &buf[3]};
  ArrayRef<const int *> a((ArrayRef<int *>(A)));
  a = ArrayRef<int *>(A);
}

static std::vector<int> ReturnTest12() { return {1, 2}; }
static void ArgTest12(ArrayRef<int> A) {
  EXPECT_EQ(2U, A.size());
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);
}

TEST(ArrayRefTest, InitializerList) {
  std::initializer_list<int> init_list = {0, 1, 2, 3, 4};
  ArrayRef<int> A = init_list;
  for (int i = 0; i < 5; ++i) EXPECT_EQ(i, A[i]);

  std::vector<int> B = ReturnTest12();
  A = B;
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);

  ArgTest12({1, 2});
}

TEST(ArrayRefTest, EmptyInitializerList) {
  ArrayRef<int> A = {};
  EXPECT_TRUE(A.empty());

  A = {};
  EXPECT_TRUE(A.empty());
}

TEST(ArrayRefTest, ArrayRef) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // A copy is expected for non-const ArrayRef (thin copy)
  ArrayRef<int> AR1(A1);
  const ArrayRef<int> &AR1Ref = ArrayRef(AR1);
  EXPECT_NE(&AR1, &AR1Ref);
  EXPECT_TRUE(AR1.equals(AR1Ref));

  // A copy is expected for non-const ArrayRef (thin copy)
  const ArrayRef<int> AR2(A1);
  const ArrayRef<int> &AR2Ref = ArrayRef(AR2);
  EXPECT_NE(&AR2Ref, &AR2);
  EXPECT_TRUE(AR2.equals(AR2Ref));
}
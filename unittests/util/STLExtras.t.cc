#include <domino/util/STLExtras.h>
#include <gtest/gtest.h>

#include "MoveOnly.h"

using namespace domino;

template <typename T>
class STLExtrasRemoveCVRefTest : public ::testing::Test {};

using STLExtrasRemoveCVRefTestTypes = ::testing::Types<
    // clang-format off
    std::pair<int, int>,
    std::pair<int &, int>,
    std::pair<const int, int>,
    std::pair<volatile int, int>,
    std::pair<const volatile int &, int>,
    std::pair<int *, int *>,
    std::pair<int *const, int *>,
    std::pair<const int *, const int *>,
    std::pair<int *&, int *>
    // clang-format on
    >;

TYPED_TEST_SUITE(STLExtrasRemoveCVRefTest, STLExtrasRemoveCVRefTestTypes, );

TYPED_TEST(STLExtrasRemoveCVRefTest, RemoveCVRef) {
  using From = typename TypeParam::first_type;
  using To = typename TypeParam::second_type;
  EXPECT_TRUE(
      (std::is_same<typename domino::remove_cvref<From>::type, To>::value));
}

TYPED_TEST(STLExtrasRemoveCVRefTest, RemoveCVRefT) {
  using From = typename TypeParam::first_type;
  EXPECT_TRUE((std::is_same<typename domino::remove_cvref<From>::type,
                            domino::remove_cvref_t<From>>::value));
}

TEST(TransformTest, Transform) {
  std::optional<int> A;

  std::optional<int> B =
      domino::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_FALSE(B.has_value());

  A = 3;
  std::optional<int> C =
      domino::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(4, *C);
}

TEST(TranformTest, MoveTransform) {
  std::optional<MoveOnly> A;

  MoveOnly::ResetCounts();
  std::optional<int> B = domino::transformOptional(
      std::move(A), [&](const MoveOnly &M) { return M.Val + 2; });
  EXPECT_FALSE(B.has_value());
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);

  A = MoveOnly(5);
  MoveOnly::ResetCounts();
  std::optional<int> C = domino::transformOptional(
      std::move(A), [&](const MoveOnly &M) { return M.Val + 2; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(7, *C);
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

namespace {

std::pair<int, int> apply_to_pair(function_ref<int(int)> func,
                                  const std::pair<int, int> &p) {
  return {func(p.first), func(p.second)};
}

int square(int x) { return x * x; }

}  // namespace

TEST(FunctionRefTest, FunctionRef) {
  auto Lambda = [](int N) { return N + 1; };
  function_ref<int(int)> F(Lambda);
  EXPECT_EQ(4, F(3));

  auto square_func = std::bind(square, std::placeholders::_1);
  function_ref<int(int)> square_func_ref = square_func;
  EXPECT_EQ(900, square_func_ref(30));

  std::vector<std::pair<int, int>> pairs = {{1, 2}, {3, 4}, {5, 6}};

  for (const auto &p : pairs) {
    auto result = apply_to_pair([](int x) { return x * x * x; }, p);
    EXPECT_EQ(p.first * p.first * p.first, result.first);
    EXPECT_EQ(p.second * p.second * p.second, result.second);
  }
}
#include <domino/util/IfConstexpr.h>

#include <gtest/gtest.h>

namespace {

using namespace domino;

struct Empty {};
struct HasFoo {
  int foo() const { return 1; }
};

TEST(IfConstexpr, Basic) {
  int i = 0;
  IfConstexpr<false>([&](const auto &t) { i = t.foo(); }, Empty{});
  EXPECT_EQ(i, 0);

  IfConstexpr<false>([&](const auto &t) { i = t.foo(); }, HasFoo{});
  EXPECT_EQ(i, 0);

  IfConstexpr<true>([&](const auto &t) { i = t.foo(); }, HasFoo{});
  EXPECT_EQ(i, 1);
}

TEST(IfConstexprElse, Basic) {
  EXPECT_EQ(IfConstexprElse<false>([&](const auto &t) { return t.foo(); },
                                   [&](const auto &) { return 2; }, Empty{}),
            2);

  EXPECT_EQ(IfConstexprElse<false>([&](const auto &t) { return t.foo(); },
                                   [&](const auto &) { return 2; }, HasFoo{}),
            2);

  EXPECT_EQ(IfConstexprElse<true>([&](const auto &t) { return t.foo(); },
                                  [&](const auto &) { return 2; }, HasFoo{}),
            1);
}

struct HasFooRValue {
  int foo() && { return 1; }
};
struct RValueFunc {
  void operator()(HasFooRValue &&t) && { *i = std::move(t).foo(); }

  int *i = nullptr;
};

TEST(IfConstexpr, RValues) {
  int i = 0;
  RValueFunc func = {&i};
  IfConstexpr<false>(std::move(func), HasFooRValue{});
  EXPECT_EQ(i, 0);

  func = RValueFunc{&i};
  IfConstexpr<true>(std::move(func), HasFooRValue{});
  EXPECT_EQ(i, 1);
}

} // namespace
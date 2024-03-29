#include <domino/support/Errc.h>
#include <domino/support/ErrorOr.h>
#include <gtest/gtest.h>

#include <memory>

using namespace domino;

ErrorOr<int> t1() { return 1; }
ErrorOr<int> t2() { return errc::invalid_argument; }

TEST(ErrorOr, SimpleValue) {
  ErrorOr<int> a = t1();
  EXPECT_TRUE(!!a);
  EXPECT_EQ(1, *a);

  ErrorOr<int> b = a;
  EXPECT_EQ(1, *b);

  a = t2();
  EXPECT_FALSE(a);
  EXPECT_EQ(a.getError(), errc::invalid_argument);
}

ErrorOr<std::unique_ptr<int>> t3() { return std::unique_ptr<int>(new int(3)); }

TEST(ErrorOr, Types) {
  int x;
  ErrorOr<int&> a(x);
  *a = 42;
  EXPECT_EQ(42, x);

  // Move only types.
  EXPECT_EQ(3, **t3());
}

struct B {};
struct D : B {};

TEST(ErrorOr, Covariant) {
  ErrorOr<B*> b(ErrorOr<D*>(nullptr));
  b = ErrorOr<D*>(nullptr);

  ErrorOr<std::unique_ptr<B> > b1(ErrorOr<std::unique_ptr<D> >(nullptr));
  b1 = ErrorOr<std::unique_ptr<D> >(nullptr);

  ErrorOr<std::unique_ptr<int>> b2(ErrorOr<int *>(nullptr));
  ErrorOr<int *> b3(nullptr);
  ErrorOr<std::unique_ptr<int>> b4(b3);
}

TEST(ErrorOr, Comparison) {
  ErrorOr<int> x(errc::no_such_file_or_directory);
  EXPECT_EQ(x, errc::no_such_file_or_directory);
}

TEST(ErrorOr, ImplicitConversion) {
  ErrorOr<std::string> x("string literal");
  EXPECT_TRUE(!!x);
  EXPECT_TRUE((*x) == "string literal");
}

TEST(ErrorOr, ImplicitConversionCausesMove) {
  struct Source {};
  struct Destination {
    Destination(const Source&) {}
    Destination(Source&&) = delete;
  };
  Source s;
  ErrorOr<Destination> x = s;
  EXPECT_TRUE(!!x);
}

TEST(ErrorOr, ImplicitConversionNoAmbiguity) {
  struct CastsToErrorCode {
    CastsToErrorCode() = default;
    CastsToErrorCode(std::error_code) {}
    operator std::error_code() { return errc::invalid_argument; }
  } casts_to_error_code;
  ErrorOr<CastsToErrorCode> x1(casts_to_error_code);
  ErrorOr<CastsToErrorCode> x2 = casts_to_error_code;
  ErrorOr<CastsToErrorCode> x3 = {casts_to_error_code};
  ErrorOr<CastsToErrorCode> x4{casts_to_error_code};
  ErrorOr<CastsToErrorCode> x5(errc::no_such_file_or_directory);
  ErrorOr<CastsToErrorCode> x6 = errc::no_such_file_or_directory;
  ErrorOr<CastsToErrorCode> x7 = {errc::no_such_file_or_directory};
  ErrorOr<CastsToErrorCode> x8{errc::no_such_file_or_directory};
  EXPECT_TRUE(!!x1);
  EXPECT_TRUE(!!x2);
  EXPECT_TRUE(!!x3);
  EXPECT_TRUE(!!x4);
  EXPECT_FALSE(x5);
  EXPECT_FALSE(x6);
  EXPECT_FALSE(x7);
  EXPECT_FALSE(x8);
}
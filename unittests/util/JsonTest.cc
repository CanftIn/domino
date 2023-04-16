#include <domino/util/Json.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

using namespace domino;

TEST(Json, string) {
  EXPECT_EQ(Json::Load("\"12\"").dump(), "\"12\"");
  EXPECT_EQ(Json::Load("\"\"").dump(), "\"\"");
  EXPECT_EQ(Json::Load("\"Hello\"").dump(), "\"Hello\"");
  EXPECT_EQ(Json::Load("\"Hello\nWorld\"").dump(), "\"Hello\\nWorld\"");
  EXPECT_EQ(Json::Load("\" \\ / \b \f \n \r \t\"").dump(),
            "\" \\\\/ \\b \\f \\n \\r \\t\"");
}

TEST(Json, number_double) {
  EXPECT_EQ(Json::Load("0.0").dump(), "0.000000");
  EXPECT_EQ(Json::Load("0.0").dump(), "0.000000");
  EXPECT_EQ(Json::Load("0.0").dump(), "0.000000");
  EXPECT_EQ(Json::Load("1.0").dump(), "1.000000");
  EXPECT_EQ(Json::Load("-1.0").dump(), "-1.000000");
  EXPECT_EQ(Json::Load("1.5").dump(), "1.500000");
  EXPECT_EQ(Json::Load("-1.5").dump(), "-1.500000");
  EXPECT_EQ(Json::Load("3.1416").dump(), "3.141600");
}

TEST(Json, number_exponent) {
  EXPECT_EQ(Json::Load("1E9").dump(), "1000000000.000000");
  EXPECT_EQ(Json::Load("1e9").dump(), "1000000000.000000");
  EXPECT_EQ(Json::Load("1E+10").dump(), "10000000000.000000");
  EXPECT_EQ(Json::Load("1E-10").dump(), "0.000000");
  EXPECT_EQ(Json::Load("-1E10").dump(), "-10000000000.000000");
  EXPECT_EQ(Json::Load("-1e10").dump(), "-10000000000.000000");
  EXPECT_EQ(Json::Load("-1E+10").dump(), "-10000000000.000000");
  EXPECT_EQ(Json::Load("-1E-10").dump(), "-0.000000");
  EXPECT_EQ(Json::Load("1E-4").dump(), "0.000100");
  EXPECT_EQ(Json::Load("-1E-4").dump(), "-0.000100");
  EXPECT_EQ(Json::Load("1.234E+10").dump(), "12340000000.000000");
  EXPECT_EQ(Json::Load("1.234E-10").dump(), "0.000000");
}

TEST(Json, object) {
  EXPECT_EQ(Json::Load("{\"a\":1,\"b\":2}").dump(),
            "{\n  \"a\" : 1,\n  \"b\" : 2\n}");
}
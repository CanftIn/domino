#include <domino/util/StringRef.h>
#include <gtest/gtest.h>

using namespace domino;

TEST(StringRefTest, Construction) {
  EXPECT_EQ("", StringRef());
  EXPECT_EQ("hello", StringRef("hello"));
  EXPECT_EQ("hello", StringRef("hello world", 5));
  EXPECT_EQ("hello", StringRef(std::string("hello")));
  EXPECT_EQ("hello", StringRef(std::string_view("hello")));
}
#include <domino/util/StringSwitch.h>
#include <gtest/gtest.h>

using namespace domino;

TEST(StringSwitchTest, Case) {
  auto Translate = [](StringRef S) {
    return domino::StringSwitch<int>(S)
        .Case("0", 0)
        .Case("1", 1)
        .Case("2", 2)
        .Case("3", 3)
        .Case("4", 4)
        .Case("5", 5)
        .Case("6", 6)
        .Case("7", 7)
        .Case("8", 8)
        .Case("9", 9)
        .Case("A", 10)
        .Case("B", 11)
        .Case("C", 12)
        .Case("D", 13)
        .Case("E", 14)
        .Case("F", 15)
        .Default(-1);
  };
  EXPECT_EQ(1, Translate("1"));
  EXPECT_EQ(2, Translate("2"));
  EXPECT_EQ(11, Translate("B"));
  EXPECT_EQ(-1, Translate("b"));
  EXPECT_EQ(-1, Translate(""));
  EXPECT_EQ(-1, Translate("Test"));
}
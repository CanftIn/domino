#include <domino/support/filesystem/UniqueID.h>
#include <gtest/gtest.h>

using namespace domino;
using namespace domino::sys::fs;

TEST(FSUniqueIDTest, construct) {
  EXPECT_EQ(20u, UniqueID(20, 10).getDevice());
  EXPECT_EQ(10u, UniqueID(20, 10).getFile());
}

TEST(FSUniqueIDTest, equals) {
  EXPECT_EQ(UniqueID(20, 10), UniqueID(20, 10));
  EXPECT_NE(UniqueID(20, 20), UniqueID(20, 10));
  EXPECT_NE(UniqueID(10, 10), UniqueID(20, 10));
}

TEST(FSUniqueIDTest, less) {
  EXPECT_FALSE(UniqueID(20, 2) < UniqueID(20, 2));
  EXPECT_FALSE(UniqueID(20, 3) < UniqueID(20, 2));
  EXPECT_FALSE(UniqueID(30, 2) < UniqueID(20, 2));
  EXPECT_FALSE(UniqueID(30, 2) < UniqueID(20, 40));
  EXPECT_TRUE(UniqueID(20, 2) < UniqueID(20, 3));
  EXPECT_TRUE(UniqueID(20, 2) < UniqueID(30, 2));
  EXPECT_TRUE(UniqueID(20, 40) < UniqueID(30, 2));
}

#include <domino/util/to_string.h>
#include <gtest/gtest.h>

using domino::to_string;

TEST(ToString, base_type) {
  EXPECT_EQ(to_string(10), "10");
  EXPECT_EQ(to_string(true), "1");
  EXPECT_EQ(to_string('a'), "a");
  EXPECT_EQ(to_string(1.2), "1.2");
  EXPECT_EQ(to_string("abc"), "abc");
}

TEST(ToString, locale_en_US_int_to_string) {
  std::locale::global(std::locale("en_US.UTF-8"));
  EXPECT_EQ(to_string(1000000), "1000000");
}

TEST(ToString, locale_en_US_floating_point_to_string) {
  std::locale::global(std::locale("en_US.UTF-8"));
  EXPECT_EQ(to_string(1.5), "1.5");
  EXPECT_EQ(to_string(1.5f), "1.5");
  EXPECT_EQ(to_string(1.5L), "1.5");
}

TEST(ToString, empty_vector_to_string) {
  std::vector<int> l;
  EXPECT_EQ(to_string(l), "[]");
}

TEST(ToString, single_item_vector_to_string) {
  std::vector<int> l;
  l.push_back(100);
  EXPECT_EQ(to_string(l), "[100]");
}

TEST(ToString, multiple_item_vector_to_string) {
  std::vector<int> l;
  l.push_back(100);
  l.push_back(150);
  EXPECT_EQ(to_string(l), "[100, 150]");
}

TEST(ToString, empty_map_to_string) {
  std::map<int, std::string> m;
  EXPECT_EQ(to_string(m), "{}");
}

TEST(ToString, single_item_map_to_string) {
  std::map<int, std::string> m;
  m[12] = "abc";
  EXPECT_EQ(to_string(m), "{12: abc}");
}

TEST(ToString, multi_item_map_to_string) {
  std::map<int, std::string> m;
  m[12] = "abc";
  m[31] = "xyz";
  EXPECT_EQ(to_string(m), "{12: abc, 31: xyz}");
}

TEST(ToString, empty_set_to_string) {
  std::set<char> s;
  EXPECT_EQ(to_string(s), "{}");
}

TEST(ToString, single_item_set_to_string) {
  std::set<char> s;
  s.insert('c');
  EXPECT_EQ(to_string(s), "{c}");
}

TEST(ToString, multi_item_set_to_string) {
  std::set<char> s;
  s.insert('a');
  s.insert('z');
  EXPECT_EQ(to_string(s), "{a, z}");
}

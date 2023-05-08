#include <domino/util/StringRef.h>
#include <domino/util/ToString.h>
#include <gtest/gtest.h>

using namespace domino;

TEST(StringRefTest, Construction) {
  EXPECT_EQ("", StringRef());
  EXPECT_EQ("hello", StringRef("hello"));
  EXPECT_EQ("hello", StringRef("hello world", 5));
  EXPECT_EQ("hello", StringRef(std::string("hello")));
  EXPECT_EQ("hello", StringRef(std::string_view("hello")));
}

TEST(StringRefTest, Conversion) {
  EXPECT_EQ("hello", std::string(StringRef("hello")));
  EXPECT_EQ("hello", std::string_view(StringRef("hello")));
}

TEST(StringRefTest, EmptyInitializerList) {
  StringRef S = {};
  EXPECT_TRUE(S.empty());

  S = {};
  EXPECT_TRUE(S.empty());
}

TEST(StringRefTest, Iteration) {
  StringRef S("hello");
  const char *p = "hello";
  for (const char *it = S.begin(), *ie = S.end(); it != ie; ++it, ++p)
    EXPECT_EQ(*it, *p);
}

TEST(StringRefTest, StringOps) {
  const char *p = "hello";
  EXPECT_EQ(p, StringRef(p, 0).data());
  EXPECT_TRUE(StringRef().empty());
  EXPECT_EQ((size_t)5, StringRef("hello").size());
  EXPECT_GT(0, StringRef("aab").compare("aad"));
  EXPECT_EQ(0, StringRef("aab").compare("aab"));
  EXPECT_LT(0, StringRef("aab").compare("aaa"));
  EXPECT_GT(0, StringRef("aab").compare("aabb"));
  EXPECT_LT(0, StringRef("aab").compare("aa"));
  EXPECT_LT(0, StringRef("\xFF").compare("\1"));

  EXPECT_EQ(-1, StringRef("AaB").compare_insensitive("aAd"));
  EXPECT_EQ(0, StringRef("AaB").compare_insensitive("aab"));
  EXPECT_EQ(1, StringRef("AaB").compare_insensitive("AAA"));
  EXPECT_EQ(-1, StringRef("AaB").compare_insensitive("aaBb"));
  EXPECT_EQ(-1, StringRef("AaB").compare_insensitive("bb"));
  EXPECT_EQ(1, StringRef("aaBb").compare_insensitive("AaB"));
  EXPECT_EQ(1, StringRef("bb").compare_insensitive("AaB"));
  EXPECT_EQ(1, StringRef("AaB").compare_insensitive("aA"));
  EXPECT_EQ(1, StringRef("\xFF").compare_insensitive("\1"));

  EXPECT_EQ(-1, StringRef("aab").compare_numeric("aad"));
  EXPECT_EQ(0, StringRef("aab").compare_numeric("aab"));
  EXPECT_EQ(1, StringRef("aab").compare_numeric("aaa"));
  EXPECT_EQ(-1, StringRef("aab").compare_numeric("aabb"));
  EXPECT_EQ(1, StringRef("aab").compare_numeric("aa"));
  EXPECT_EQ(-1, StringRef("1").compare_numeric("10"));
  EXPECT_EQ(0, StringRef("10").compare_numeric("10"));
  EXPECT_EQ(0, StringRef("10a").compare_numeric("10a"));
  EXPECT_EQ(1, StringRef("2").compare_numeric("1"));
  EXPECT_EQ(0, StringRef("domino_v1i64_ty").compare_numeric("domino_v1i64_ty"));
  EXPECT_EQ(1, StringRef("\xFF").compare_numeric("\1"));
  EXPECT_EQ(1, StringRef("V16").compare_numeric("V1_q0"));
  EXPECT_EQ(-1, StringRef("V1_q0").compare_numeric("V16"));
  EXPECT_EQ(-1, StringRef("V8_q0").compare_numeric("V16"));
  EXPECT_EQ(1, StringRef("V16").compare_numeric("V8_q0"));
  EXPECT_EQ(-1, StringRef("V1_q0").compare_numeric("V8_q0"));
  EXPECT_EQ(1, StringRef("V8_q0").compare_numeric("V1_q0"));
}

TEST(StringRefTest, Operators) {
  EXPECT_EQ("", StringRef());
  EXPECT_TRUE(StringRef("aab") < StringRef("aad"));
  EXPECT_FALSE(StringRef("aab") < StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") <= StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") <= StringRef("aaa"));
  EXPECT_TRUE(StringRef("aad") > StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") > StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") >= StringRef("aab"));
  EXPECT_FALSE(StringRef("aaa") >= StringRef("aab"));
  EXPECT_EQ(StringRef("aab"), StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") == StringRef("aac"));
  EXPECT_FALSE(StringRef("aab") != StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") != StringRef("aac"));
  EXPECT_EQ('a', StringRef("aab")[1]);
}

TEST(StringRefTest, Substr) {
  StringRef Str("hello");
  EXPECT_EQ("lo", Str.substr(3));
  EXPECT_EQ("", Str.substr(100));
  EXPECT_EQ("hello", Str.substr(0, 100));
  EXPECT_EQ("o", Str.substr(4, 10));
}

TEST(StringRefTest, Slice) {
  StringRef Str("hello");
  EXPECT_EQ("l", Str.slice(2, 3));
  EXPECT_EQ("ell", Str.slice(1, 4));
  EXPECT_EQ("llo", Str.slice(2, 100));
  EXPECT_EQ("", Str.slice(2, 1));
  EXPECT_EQ("", Str.slice(10, 20));
}

TEST(StringRefTest, Split) {
  StringRef Str("hello");
  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")), Str.split('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")), Str.split('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")), Str.split('h'));
  EXPECT_EQ(std::make_pair(StringRef("he"), StringRef("lo")), Str.split('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")), Str.split('o'));

  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")), Str.rsplit('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")), Str.rsplit('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")), Str.rsplit('h'));
  EXPECT_EQ(std::make_pair(StringRef("hel"), StringRef("o")), Str.rsplit('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")), Str.rsplit('o'));

  EXPECT_EQ(std::make_pair(StringRef("he"), StringRef("o")), Str.rsplit("ll"));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")), Str.rsplit("h"));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")), Str.rsplit("o"));
  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")),
            Str.rsplit("::"));
  EXPECT_EQ(std::make_pair(StringRef("hel"), StringRef("o")), Str.rsplit("l"));
}

TEST(StringRefTest, Split2) {
  SmallVector<StringRef, 5> parts;
  SmallVector<StringRef, 5> expected;

  expected.push_back("ab");
  expected.push_back("c");
  StringRef(",ab,,c,").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("");
  expected.push_back("ab");
  expected.push_back("");
  expected.push_back("c");
  expected.push_back("");
  StringRef(",ab,,c,").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("");
  StringRef("").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  StringRef("").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  StringRef(",").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("");
  expected.push_back("");
  StringRef(",").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("b");
  StringRef("a,b").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  // Test MaxSplit
  expected.clear();
  parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("");
  expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("");
  expected.push_back("b");
  expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("b");
  expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("b");
  expected.push_back("c");
  StringRef("a,,b,c").split(parts, ',', 3, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("");
  StringRef().split(parts, ",", 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back(StringRef());
  StringRef("").split(parts, ",", 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  StringRef("").split(parts, ",", 0, false);
  EXPECT_TRUE(parts == expected);
  StringRef().split(parts, ",", 0, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("");
  expected.push_back("b");
  expected.push_back("c,d");
  StringRef("a,,b,c,d").split(parts, ",", 3, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("");
  StringRef().split(parts, ',', 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back(StringRef());
  StringRef("").split(parts, ',', 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  StringRef("").split(parts, ',', 0, false);
  EXPECT_TRUE(parts == expected);
  StringRef().split(parts, ',', 0, false);
  EXPECT_TRUE(parts == expected);

  expected.clear();
  parts.clear();
  expected.push_back("a");
  expected.push_back("");
  expected.push_back("b");
  expected.push_back("c,d");
  StringRef("a,,b,c,d").split(parts, ',', 3, true);
  EXPECT_TRUE(parts == expected);
}

TEST(StringRefTest, Trim) {
  StringRef Str0("hello");
  StringRef Str1(" hello ");
  StringRef Str2("  hello  ");
  StringRef Str3("\t\n\v\f\r  hello  \t\n\v\f\r");

  EXPECT_EQ(StringRef("hello"), Str0.rtrim());
  EXPECT_EQ(StringRef(" hello"), Str1.rtrim());
  EXPECT_EQ(StringRef("  hello"), Str2.rtrim());
  EXPECT_EQ(StringRef("\t\n\v\f\r  hello"), Str3.rtrim());
  EXPECT_EQ(StringRef("hello"), Str0.ltrim());
  EXPECT_EQ(StringRef("hello "), Str1.ltrim());
  EXPECT_EQ(StringRef("hello  "), Str2.ltrim());
  EXPECT_EQ(StringRef("hello  \t\n\v\f\r"), Str3.ltrim());
  EXPECT_EQ(StringRef("hello"), Str0.trim());
  EXPECT_EQ(StringRef("hello"), Str1.trim());
  EXPECT_EQ(StringRef("hello"), Str2.trim());
  EXPECT_EQ(StringRef("hello"), Str3.trim());

  EXPECT_EQ(StringRef("ello"), Str0.trim("hhhhhhhhhhh"));

  EXPECT_EQ(StringRef(""), StringRef("").trim());
  EXPECT_EQ(StringRef(""), StringRef(" ").trim());
  EXPECT_EQ(StringRef("\0", 1), StringRef(" \0 ", 3).trim());
  EXPECT_EQ(StringRef("\0\0", 2), StringRef("\0\0", 2).trim());
  EXPECT_EQ(StringRef("x"), StringRef("\0\0x\0\0", 5).trim('\0'));
}

TEST(StringRefTest, StartsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.starts_with(""));
  EXPECT_TRUE(Str.starts_with("he"));
  EXPECT_FALSE(Str.starts_with("helloworld"));
  EXPECT_FALSE(Str.starts_with("hi"));
}

TEST(StringRefTest, StartsWithInsensitive) {
  StringRef Str("heLLo");
  EXPECT_TRUE(Str.starts_with_insensitive(""));
  EXPECT_TRUE(Str.starts_with_insensitive("he"));
  EXPECT_TRUE(Str.starts_with_insensitive("hell"));
  EXPECT_TRUE(Str.starts_with_insensitive("HELlo"));
  EXPECT_FALSE(Str.starts_with_insensitive("helloworld"));
  EXPECT_FALSE(Str.starts_with_insensitive("hi"));
}

TEST(StringRefTest, ConsumeFront) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.consume_front(""));
  EXPECT_EQ("hello", Str);
  EXPECT_TRUE(Str.consume_front("he"));
  EXPECT_EQ("llo", Str);
  EXPECT_FALSE(Str.consume_front("lloworld"));
  EXPECT_EQ("llo", Str);
  EXPECT_FALSE(Str.consume_front("lol"));
  EXPECT_EQ("llo", Str);
  EXPECT_TRUE(Str.consume_front("llo"));
  EXPECT_EQ("", Str);
  EXPECT_FALSE(Str.consume_front("o"));
  EXPECT_TRUE(Str.consume_front(""));
}

TEST(StringRefTest, ConsumeFrontInsensitive) {
  StringRef Str("heLLo");
  EXPECT_TRUE(Str.consume_front_insensitive(""));
  EXPECT_EQ("heLLo", Str);
  EXPECT_FALSE(Str.consume_front("HEl"));
  EXPECT_EQ("heLLo", Str);
  EXPECT_TRUE(Str.consume_front_insensitive("HEl"));
  EXPECT_EQ("Lo", Str);
  EXPECT_FALSE(Str.consume_front_insensitive("loworld"));
  EXPECT_EQ("Lo", Str);
  EXPECT_FALSE(Str.consume_front_insensitive("ol"));
  EXPECT_EQ("Lo", Str);
  EXPECT_TRUE(Str.consume_front_insensitive("lo"));
  EXPECT_EQ("", Str);
  EXPECT_FALSE(Str.consume_front_insensitive("o"));
  EXPECT_TRUE(Str.consume_front_insensitive(""));
}

TEST(StringRefTest, EndsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.ends_with(""));
  EXPECT_TRUE(Str.ends_with("lo"));
  EXPECT_FALSE(Str.ends_with("helloworld"));
  EXPECT_FALSE(Str.ends_with("worldhello"));
  EXPECT_FALSE(Str.ends_with("so"));
}

TEST(StringRefTest, EndsWithInsensitive) {
  StringRef Str("heLLo");
  EXPECT_TRUE(Str.ends_with_insensitive(""));
  EXPECT_TRUE(Str.ends_with_insensitive("lo"));
  EXPECT_TRUE(Str.ends_with_insensitive("LO"));
  EXPECT_TRUE(Str.ends_with_insensitive("ELlo"));
  EXPECT_FALSE(Str.ends_with_insensitive("helloworld"));
  EXPECT_FALSE(Str.ends_with_insensitive("hi"));
}

TEST(StringRefTest, ConsumeBack) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.consume_back(""));
  EXPECT_EQ("hello", Str);
  EXPECT_TRUE(Str.consume_back("lo"));
  EXPECT_EQ("hel", Str);
  EXPECT_FALSE(Str.consume_back("helhel"));
  EXPECT_EQ("hel", Str);
  EXPECT_FALSE(Str.consume_back("hle"));
  EXPECT_EQ("hel", Str);
  EXPECT_TRUE(Str.consume_back("hel"));
  EXPECT_EQ("", Str);
  EXPECT_FALSE(Str.consume_back("h"));
  EXPECT_TRUE(Str.consume_back(""));
}

TEST(StringRefTest, ConsumeBackInsensitive) {
  StringRef Str("heLLo");
  EXPECT_TRUE(Str.consume_back_insensitive(""));
  EXPECT_EQ("heLLo", Str);
  EXPECT_FALSE(Str.consume_back("lO"));
  EXPECT_EQ("heLLo", Str);
  EXPECT_TRUE(Str.consume_back_insensitive("lO"));
  EXPECT_EQ("heL", Str);
  EXPECT_FALSE(Str.consume_back_insensitive("helhel"));
  EXPECT_EQ("heL", Str);
  EXPECT_FALSE(Str.consume_back_insensitive("hle"));
  EXPECT_EQ("heL", Str);
  EXPECT_TRUE(Str.consume_back_insensitive("hEl"));
  EXPECT_EQ("", Str);
  EXPECT_FALSE(Str.consume_back_insensitive("h"));
  EXPECT_TRUE(Str.consume_back_insensitive(""));
}

TEST(StringRefTest, Find) {
  StringRef Str("helloHELLO");
  StringRef LongStr("hellx xello hell ello world foo bar hello HELLO");

  struct {
    StringRef Str;
    char C;
    std::size_t From;
    std::size_t Pos;
    std::size_t InsensitivePos;
  } CharExpectations[] = {
      {Str, 'h', 0U, 0U, 0U},
      {Str, 'e', 0U, 1U, 1U},
      {Str, 'l', 0U, 2U, 2U},
      {Str, 'l', 3U, 3U, 3U},
      {Str, 'o', 0U, 4U, 4U},
      {Str, 'L', 0U, 7U, 2U},
      {Str, 'z', 0U, StringRef::npos, StringRef::npos},
  };

  struct {
    StringRef Str;
    domino::StringRef S;
    std::size_t From;
    std::size_t Pos;
    std::size_t InsensitivePos;
  } StrExpectations[] = {
      {Str, "helloword", 0, StringRef::npos, StringRef::npos},
      {Str, "hello", 0, 0U, 0U},
      {Str, "ello", 0, 1U, 1U},
      {Str, "zz", 0, StringRef::npos, StringRef::npos},
      {Str, "ll", 2U, 2U, 2U},
      {Str, "ll", 3U, StringRef::npos, 7U},
      {Str, "LL", 2U, 7U, 2U},
      {Str, "LL", 3U, 7U, 7U},
      {Str, "", 0U, 0U, 0U},
      {LongStr, "hello", 0U, 36U, 36U},
      {LongStr, "foo", 0U, 28U, 28U},
      {LongStr, "hell", 2U, 12U, 12U},
      {LongStr, "HELL", 2U, 42U, 12U},
      {LongStr, "", 0U, 0U, 0U}};

  for (auto &E : CharExpectations) {
    EXPECT_EQ(E.Pos, E.Str.find(E.C, E.From));
    EXPECT_EQ(E.InsensitivePos, E.Str.find_insensitive(E.C, E.From));
    EXPECT_EQ(E.InsensitivePos, E.Str.find_insensitive(toupper(E.C), E.From));
  }

  for (auto &E : StrExpectations) {
    EXPECT_EQ(E.Pos, E.Str.find(E.S, E.From));
    EXPECT_EQ(E.InsensitivePos, E.Str.find_insensitive(E.S, E.From));
    EXPECT_EQ(E.InsensitivePos, E.Str.find_insensitive(E.S.upper(), E.From));
  }

  EXPECT_EQ(3U, Str.rfind('l'));
  EXPECT_EQ(StringRef::npos, Str.rfind('z'));
  EXPECT_EQ(StringRef::npos, Str.rfind("helloworld"));
  EXPECT_EQ(0U, Str.rfind("hello"));
  EXPECT_EQ(1U, Str.rfind("ello"));
  EXPECT_EQ(StringRef::npos, Str.rfind("zz"));

  EXPECT_EQ(8U, Str.rfind_insensitive('l'));
  EXPECT_EQ(8U, Str.rfind_insensitive('L'));
  EXPECT_EQ(StringRef::npos, Str.rfind_insensitive('z'));
  EXPECT_EQ(StringRef::npos, Str.rfind_insensitive("HELLOWORLD"));
  EXPECT_EQ(5U, Str.rfind("HELLO"));
  EXPECT_EQ(6U, Str.rfind("ELLO"));
  EXPECT_EQ(StringRef::npos, Str.rfind("ZZ"));

  EXPECT_EQ(2U, Str.find_first_of('l'));
  EXPECT_EQ(1U, Str.find_first_of("el"));
  EXPECT_EQ(StringRef::npos, Str.find_first_of("xyz"));

  Str = "hello";
  EXPECT_EQ(1U, Str.find_first_not_of('h'));
  EXPECT_EQ(4U, Str.find_first_not_of("hel"));
  EXPECT_EQ(StringRef::npos, Str.find_first_not_of("hello"));

  EXPECT_EQ(3U, Str.find_last_not_of('o'));
  EXPECT_EQ(1U, Str.find_last_not_of("lo"));
  EXPECT_EQ(StringRef::npos, Str.find_last_not_of("helo"));
}

TEST(StringRefTest, Count) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.count('l'));
  EXPECT_EQ(1U, Str.count('o'));
  EXPECT_EQ(0U, Str.count('z'));
  EXPECT_EQ(0U, Str.count("helloworld"));
  EXPECT_EQ(1U, Str.count("hello"));
  EXPECT_EQ(1U, Str.count("ello"));
  EXPECT_EQ(0U, Str.count("zz"));
  EXPECT_EQ(0U, Str.count(""));

  StringRef OverlappingAbba("abbabba");
  EXPECT_EQ(1U, OverlappingAbba.count("abba"));
  StringRef NonOverlappingAbba("abbaabba");
  EXPECT_EQ(2U, NonOverlappingAbba.count("abba"));
  StringRef ComplexAbba("abbabbaxyzabbaxyz");
  EXPECT_EQ(2U, ComplexAbba.count("abba"));
}

TEST(StringRefTest, EditDistance) {
  StringRef Hello("hello");
  EXPECT_EQ(2U, Hello.edit_distance("hill"));

  StringRef Industry("industry");
  EXPECT_EQ(6U, Industry.edit_distance("interest"));

  StringRef Soylent("soylent green is people");
  EXPECT_EQ(19U, Soylent.edit_distance("people soiled our green"));
  EXPECT_EQ(26U, Soylent.edit_distance("people soiled our green",
                                       /* allow replacements = */ false));
  EXPECT_EQ(9U, Soylent.edit_distance("people soiled our green",
                                      /* allow replacements = */ true,
                                      /* max edit distance = */ 8));
  EXPECT_EQ(53U, Soylent.edit_distance("people soiled our green "
                                       "people soiled our green "
                                       "people soiled our green "));
}

TEST(StringRefTest, EditDistanceInsensitive) {
  StringRef Hello("HELLO");
  EXPECT_EQ(2U, Hello.edit_distance_insensitive("hill"));
  EXPECT_EQ(0U, Hello.edit_distance_insensitive("hello"));

  StringRef Industry("InDuStRy");
  EXPECT_EQ(6U, Industry.edit_distance_insensitive("iNtErEsT"));
}

TEST(StringRefTest, Drop) {
  StringRef Test("StringRefTest::Drop");

  StringRef Dropped = Test.drop_front(5);
  EXPECT_EQ(Dropped, "gRefTest::Drop");

  Dropped = Test.drop_back(5);
  EXPECT_EQ(Dropped, "StringRefTest:");

  Dropped = Test.drop_front(0);
  EXPECT_EQ(Dropped, Test);

  Dropped = Test.drop_back(0);
  EXPECT_EQ(Dropped, Test);

  Dropped = Test.drop_front(Test.size());
  EXPECT_TRUE(Dropped.empty());

  Dropped = Test.drop_back(Test.size());
  EXPECT_TRUE(Dropped.empty());
}

TEST(StringRefTest, Take) {
  StringRef Test("StringRefTest::Take");

  StringRef Taken = Test.take_front(5);
  EXPECT_EQ(Taken, "Strin");

  Taken = Test.take_back(5);
  EXPECT_EQ(Taken, ":Take");

  Taken = Test.take_front(Test.size());
  EXPECT_EQ(Taken, Test);

  Taken = Test.take_back(Test.size());
  EXPECT_EQ(Taken, Test);

  Taken = Test.take_front(0);
  EXPECT_TRUE(Taken.empty());

  Taken = Test.take_back(0);
  EXPECT_TRUE(Taken.empty());
}

TEST(StringRefTest, FindIf) {
  StringRef Punct("Test.String");
  StringRef NoPunct("ABCDEFG");
  StringRef Empty;

  auto IsPunct = [](char c) { return ::ispunct(c); };
  auto IsAlpha = [](char c) { return ::isalpha(c); };
  EXPECT_EQ(4U, Punct.find_if(IsPunct));
  EXPECT_EQ(StringRef::npos, NoPunct.find_if(IsPunct));
  EXPECT_EQ(StringRef::npos, Empty.find_if(IsPunct));

  EXPECT_EQ(4U, Punct.find_if_not(IsAlpha));
  EXPECT_EQ(StringRef::npos, NoPunct.find_if_not(IsAlpha));
  EXPECT_EQ(StringRef::npos, Empty.find_if_not(IsAlpha));
}

TEST(StringRefTest, TakeWhileUntil) {
  StringRef Test("String With 1 Number");

  StringRef Taken = Test.take_while([](char c) { return ::isdigit(c); });
  EXPECT_EQ("", Taken);

  Taken = Test.take_until([](char c) { return ::isdigit(c); });
  EXPECT_EQ("String With ", Taken);

  Taken = Test.take_while([](char c) { return true; });
  EXPECT_EQ(Test, Taken);

  Taken = Test.take_until([](char c) { return true; });
  EXPECT_EQ("", Taken);

  Test = "";
  Taken = Test.take_while([](char c) { return true; });
  EXPECT_EQ("", Taken);
}

TEST(StringRefTest, DropWhileUntil) {
  StringRef Test("String With 1 Number");

  StringRef Taken = Test.drop_while([](char c) { return ::isdigit(c); });
  EXPECT_EQ(Test, Taken);

  Taken = Test.drop_until([](char c) { return ::isdigit(c); });
  EXPECT_EQ("1 Number", Taken);

  Taken = Test.drop_while([](char c) { return true; });
  EXPECT_EQ("", Taken);

  Taken = Test.drop_until([](char c) { return true; });
  EXPECT_EQ(Test, Taken);

  StringRef EmptyString = "";
  Taken = EmptyString.drop_while([](char c) { return true; });
  EXPECT_EQ("", Taken);
}

TEST(StringRefTest, StringLiteral) {
  constexpr StringRef StringRefs[] = {"Foo", "Bar"};
  EXPECT_EQ(StringRef("Foo"), StringRefs[0]);
  EXPECT_EQ(3u, (std::integral_constant<size_t, StringRefs[0].size()>::value));
  EXPECT_EQ(false,
            (std::integral_constant<bool, StringRefs[0].empty()>::value));
  EXPECT_EQ(StringRef("Bar"), StringRefs[1]);

  constexpr StringLiteral Strings[] = {"Foo", "Bar"};
  EXPECT_EQ(StringRef("Foo"), Strings[0]);
  EXPECT_EQ(3u, (std::integral_constant<size_t, Strings[0].size()>::value));
  EXPECT_EQ(false, (std::integral_constant<bool, Strings[0].empty()>::value));
  EXPECT_EQ(StringRef("Bar"), Strings[1]);
}

TEST(StringRefTest, GTestPrinter) {
  EXPECT_EQ("foo", to_string(StringRef("foo")));
}

TEST(StringRefTest, LFLineEnding) {
  constexpr StringRef Cases[] = {"\nDoggo\nPupper", "Floofer\n", "Woofer"};
  EXPECT_EQ(StringRef("\n"), Cases[0].detectEOL());
  EXPECT_EQ(StringRef("\n"), Cases[1].detectEOL());
  EXPECT_EQ(StringRef("\n"), Cases[2].detectEOL());
}

TEST(StringRefTest, CRLineEnding) {
  constexpr StringRef Cases[] = {"\rDoggo\rPupper", "Floofer\r", "Woo\rfer\n"};
  EXPECT_EQ(StringRef("\r"), Cases[0].detectEOL());
  EXPECT_EQ(StringRef("\r"), Cases[1].detectEOL());
  EXPECT_EQ(StringRef("\r"), Cases[2].detectEOL());
}

TEST(StringRefTest, CRLFLineEnding) {
  constexpr StringRef Cases[] = {"\r\nDoggo\r\nPupper", "Floofer\r\n",
                                 "Woofer\r\nSubWoofer\n"};
  EXPECT_EQ(StringRef("\r\n"), Cases[0].detectEOL());
  EXPECT_EQ(StringRef("\r\n"), Cases[1].detectEOL());
  EXPECT_EQ(StringRef("\r\n"), Cases[2].detectEOL());
}

TEST(StringRefTest, LFCRLineEnding) {
  constexpr StringRef Cases[] = {"\n\rDoggo\n\rPupper", "Floofer\n\r",
                                 "Woofer\n\rSubWoofer\n"};
  EXPECT_EQ(StringRef("\n\r"), Cases[0].detectEOL());
  EXPECT_EQ(StringRef("\n\r"), Cases[1].detectEOL());
  EXPECT_EQ(StringRef("\n\r"), Cases[2].detectEOL());
}
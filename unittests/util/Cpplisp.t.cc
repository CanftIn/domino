#include <iostream>
#include <typeinfo>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <domino/util/Cpplisp.h>

TEST(Cpplisp, cons) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, 2);
  var cons_2 = cons("foo", "bar");
  var cons_3 = cons(1, "foo");

  EXPECT_EQ(consp(cons_1), true);
  EXPECT_EQ(consp(cons_2), true);
  EXPECT_EQ(consp(cons_3), true);
}

TEST(Cpplisp, listp_and_consp) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, 2);
  var cons_2 = cons("foo", "bar");
  var cons_3 = cons(1, "foo");

  var null_cons = cons(nullptr, nullptr);
  var cons_4 = cons(3, nil);
  var cons_5 = cons("foo", nil);
  var cons_6 = cons(4, cons(nullptr, nullptr));
  var cons_7 = cons("foo", null_cons);

  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  EXPECT_EQ(listp(cons_1), false);
  EXPECT_EQ(listp(cons_2), false);
  EXPECT_EQ(listp(cons_3), false);
  EXPECT_EQ(consp(cons_1), true);
  EXPECT_EQ(consp(cons_2), true);
  EXPECT_EQ(consp(cons_3), true);
  EXPECT_EQ(listp(cons_4), true);
  EXPECT_EQ(listp(cons_5), true);
  EXPECT_EQ(listp(cons_6), true);
  EXPECT_EQ(listp(cons_7), true);
  EXPECT_EQ(listp(ls_1), true);
  EXPECT_EQ(listp(ls_2), true);
}

TEST(Cpplisp, length) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, cons("foo", nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  var lt = list(1, "bar", std::vector<int>{1, 2, 3}, std::string("foo"));

  EXPECT_EQ(length(cons_1), 2);
  EXPECT_EQ(length(ls_1), 3);
  EXPECT_EQ(length(ls_2), 3);
  EXPECT_EQ(length(lt), 4);
}

TEST(Cpplisp, car) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, cons("foo", nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  var lt = list(1, "bar", std::vector<int>{1, 2, 3}, std::string("foo"));

  EXPECT_EQ(car(cons_1), 1);
  EXPECT_EQ(cadr(cons_1), "foo");
}

TEST(Cpplisp, nth) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, cons("foo", nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  var lt = list(1, "bar", std::vector<int>{1, 2, 3}, std::string("foo"));

  EXPECT_EQ(nth<0>(cons_1), 1);
  EXPECT_EQ(nth<0>(lt), 1);
}

TEST(Cpplisp, nullp) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, cons("foo", nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  var lt = list(1, "bar", std::vector<int>{1, 2, 3}, std::string("foo"));

  EXPECT_EQ(nullp(nil), true);
  EXPECT_EQ(nullp(cddr(cons_1)), true);
  EXPECT_EQ(nullp(cddr(lt)), false);
  EXPECT_EQ(nullp(cddddr(lt)), true);
}

// TODO: need more test
TEST(Cpplisp, prettyprint) {
  using namespace domino::cpplisp;
  using namespace domino::cpplisp::prettyprint;

  var cons_1 = cons(1, cons("foo", nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  //var lt = list(1, "bar", std::vector<int>{1, 2, 3}, std::string("foo"));

  EXPECT_EQ(to_string(cons_1), "(1 . (foo . nil))");
  EXPECT_EQ(to_string(ls_1), "(1 . (2 . (3 . nil)))");
}

TEST(Cpplisp, equals) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, cons("foo", nil));
  var cons_2 = cons(1, 2);
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "bar", "foo");

  EXPECT_EQ(equals(ls_1, cons(1, cons(2, cons(3, nil)))), true);
  EXPECT_EQ(equals(cons_2, cons(1, 2)), true);
}

TEST(Cpplisp, append) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, nil);
  var cons_2 = cons("foo", cons(2.0, nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(1, "foo", 2.0);

  EXPECT_EQ(listp(cons_1), true);
  EXPECT_EQ(listp(cons_2), true);
  var app_ret = append(cons_1, cons_2);
  EXPECT_EQ(domino::cpplisp::prettyprint::to_string(app_ret),
            "(1 . (foo . (2 . nil)))");
  EXPECT_EQ(equals(append(cons_1, cons_2), ls_2), true);
}

TEST(Cpplisp, reverse) {
  using namespace domino::cpplisp;

  var cons_1 = cons(1, nil);
  var cons_2 = cons("foo", cons(2.0, nil));
  var ls_1 = list(1, 2, 3);
  var ls_2 = list(2.0, "foo");

  EXPECT_EQ(listp(cons_1), true);
  EXPECT_EQ(listp(cons_2), true);
  var app_ret = reverse(cons_2);
  EXPECT_EQ(domino::cpplisp::prettyprint::to_string(app_ret),
            "(2 . (foo . nil))");
  EXPECT_EQ(equals(reverse(cons_2), ls_2), true);
}

TEST(Cpplisp, mapcar) {
  using namespace domino::cpplisp;

  var ls_1 = list(1, 2, 3);
  //var ls_2 = list(2.0, "foo");
  var list_abc = list("a", "b", "c");
  var list_efg = list("e", "f", "g");

  var ls_3 = mapcar([](auto n) { return n + 1; }, ls_1);
  var ls_4 = mapcar([](auto n, auto s) { return std::to_string(n) + s; }, 
                    ls_1, list_abc);
  var ls_5 = mapcar([](auto n, auto s1, auto s2) { return std::to_string(n) + s1 + s2; }, 
                    ls_1, list_abc, list_efg);
  EXPECT_EQ(domino::cpplisp::prettyprint::to_string(ls_3),
            "(2 . (3 . (4 . nil)))");
  EXPECT_EQ(domino::cpplisp::prettyprint::to_string(ls_4),
            "(1a . (2b . (3c . nil)))");
  EXPECT_EQ(domino::cpplisp::prettyprint::to_string(ls_5),
            "(1ae . (2bf . (3cg . nil)))");
}

TEST(Cpplisp, multiple_value_bind) {
  using namespace domino::cpplisp;

  var ls_1 = list(1, 2, 3);
  var ls_2 = list(2.0, "foo");
  var list_abc = list("a", "b", "c");
  var list_efg = list("e", "f", "g");

  int v1, v2, v3;
  var mvb_bind = multiple_value_bind(ls_1, &v1, &v2, &v3);
  var ls_3 = mvb_bind([=] () { return list(v1, v2, v3); });
  //std::cout << ls_3 << std::endl;

  std::string v4;
  var ls_4 = multiple_value_bind(append(ls_1, list((std::string)"foo")), &v1, &v2, &v3, &v4)([=] () {
    //std::cout << "v1: " << v1 << " v2: " << v2 << " v3: " << v3 << " v4: " << v4 << std::endl;
    return list(v4, v1, v2, v3);
  });
  //std::cout << ls_4 << std::endl;
}
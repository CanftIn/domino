#include <domino/support/Casting.h>
#include <gtest/gtest.h>

namespace domino {

struct bar {
  bar() {}
  struct foo *baz();
  struct foo *caz();
  struct foo *daz();
  struct foo *naz();

 private:
  bar(const bar &);
};

struct foo {
  foo(const bar &) {}
  void ext() const;
};

template <>
struct isa_impl<foo, bar> {
  static inline bool doit(const bar &Val) {
    //std::cout << "Classof: " << &Val << "\n";
    return true;
  }
};

struct base {
  virtual ~base() {}
};

struct derived : public base {
  static bool classof(const base *B) { return true; }
};

}  // namespace domino

using namespace domino;

TEST(CastingTest, isa) {
  bar B;
  bar &B1 = B;
  const bar *B2 = &B;
  const bar &B3 = B1;
  const bar *const B4 = B2;

  base b1;
  derived d2;

  EXPECT_TRUE(isa<foo>(B1));
  EXPECT_TRUE(isa<foo>(B2));
  EXPECT_TRUE(isa<foo>(B3));
  EXPECT_TRUE(isa<foo>(B4));

  EXPECT_TRUE(isa<base>(d2));
}
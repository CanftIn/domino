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
    // std::cout << "Classof: " << &Val << "\n";
    return true;
  }
};

struct Base {
  bool IsDerived;
  Base(bool IsDerived = false) : IsDerived(IsDerived) {}
};

struct Derived : Base {
  Derived() : Base(true) {}
  static bool classof(const Base *B) { return B->IsDerived; }
};

class PTy {
  Base *B;

 public:
  PTy(Base *B) : B(B) {}
  explicit operator bool() const { return get(); }
  Base *get() const { return B; }
};

template <>
struct simplify_type<PTy> {
  typedef Base *SimpleType;
  static SimpleType getSimplifiedValue(PTy &P) { return P.get(); }
};
template <>
struct simplify_type<const PTy> {
  typedef Base *SimpleType;
  static SimpleType getSimplifiedValue(const PTy &P) { return P.get(); }
};

}  // namespace domino

using namespace domino;

TEST(CastingTest, isa) {
  bar B;
  bar &B1 = B;
  const bar *B2 = &B;
  const bar &B3 = B1;
  const bar *const B4 = B2;

  Base b1;
  Derived d2;

  EXPECT_TRUE(isa<foo>(B1));
  EXPECT_TRUE(isa<foo>(B2));
  EXPECT_TRUE(isa<foo>(B3));
  EXPECT_TRUE(isa<foo>(B4));

  EXPECT_TRUE(isa<Base>(d2));
}

// Some objects.
Base B;
Derived D;

// Mutable "smart" pointers.
PTy MN(nullptr);
PTy MB(&B);
PTy MD(&D);

// Const "smart" pointers.
const PTy CN(nullptr);
const PTy CB(&B);
const PTy CD(&D);

TEST(CastingTest, smart_isa) {
  EXPECT_TRUE(!isa<Derived>(MB));
  EXPECT_TRUE(!isa<Derived>(CB));
  EXPECT_TRUE(isa<Derived>(MD));
  EXPECT_TRUE(isa<Derived>(CD));
}
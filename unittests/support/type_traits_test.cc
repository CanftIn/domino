#include "domino/support/type_traits.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace triviality {

template <typename T, bool IsTriviallyCopyConstructible,
          bool IsTriviallyMoveConstructible>
void TrivialityTester() {
  static_assert(domino::is_trivially_copy_constructible<T>::value ==
                    IsTriviallyCopyConstructible,
                "Mismatch in expected trivial copy construction!");
  static_assert(domino::is_trivially_move_constructible<T>::value ==
                    IsTriviallyMoveConstructible,
                "Mismatch in expected trivial move construction!");
}

}  // namespace triviality

struct X {};
struct Y {
  Y(const Y &);
};
struct Z {
  Z(const Z &);
  Z(Z &&);
};
struct A {
  A(const A &) = default;
  A(A &&);
};
struct B {
  B(const B &);
  B(B &&) = default;
};

TEST(Triviality, Tester) {
  using namespace triviality;

  TrivialityTester<int, true, true>();
  TrivialityTester<void *, true, true>();
  TrivialityTester<int &, true, true>();
  TrivialityTester<int &&, false, true>();

  TrivialityTester<X, true, true>();
  TrivialityTester<Y, false, false>();
  TrivialityTester<Z, false, false>();
  TrivialityTester<A, true, false>();
  TrivialityTester<B, false, true>();

  TrivialityTester<Z &, true, true>();
  TrivialityTester<A &, true, true>();
  TrivialityTester<B &, true, true>();
  TrivialityTester<Z &&, false, true>();
  TrivialityTester<A &&, false, true>();
  TrivialityTester<B &&, false, true>();
}
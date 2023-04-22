#include <domino/support/TypeTraits.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

using namespace domino;

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

class NotEqualityComparable {};
class EqualityComparable {};

inline bool operator==(const EqualityComparable &, const EqualityComparable &) {
  return false;
}

TEST(TypeTraits, is_equality_comparable) {
  EXPECT_TRUE((!is_equality_comparable<NotEqualityComparable>::value));
  EXPECT_TRUE((is_equality_comparable<EqualityComparable>::value));
  EXPECT_TRUE((is_equality_comparable<int>::value));
}

class NotHashable {};
class Hashable {};

namespace std {

template <>
struct hash<Hashable> final {
  size_t operator()(const Hashable &) { return 0; }
};

}  // namespace std

TEST(TypeTraits, is_hashable) {
  EXPECT_TRUE((is_hashable<int>::value));
  EXPECT_TRUE((is_hashable<Hashable>::value));
  EXPECT_TRUE((!is_hashable<NotHashable>::value));
}

class MyClass {};
struct Functor {
  void operator()() {}
};
auto lambda = []() {};

// func() and func__ just exists to silence a compiler warning about lambda
// being unused
bool func() {
  lambda();
  return true;
}
bool func__ = func();

TEST(TypeTraits, is_function_type) {
  EXPECT_TRUE((is_function_type<void()>::value));
  EXPECT_TRUE((is_function_type<int()>::value));
  EXPECT_TRUE((is_function_type<MyClass()>::value));
  EXPECT_TRUE((is_function_type<void(MyClass)>::value));
  EXPECT_TRUE((is_function_type<void(int)>::value));
  EXPECT_TRUE((is_function_type<void(void *)>::value));
  EXPECT_TRUE((is_function_type<int()>::value));
  EXPECT_TRUE((is_function_type<int(MyClass)>::value));
  EXPECT_TRUE((is_function_type<int(const MyClass &)>::value));
  EXPECT_TRUE((is_function_type<int(MyClass &&)>::value));
  EXPECT_TRUE((is_function_type < MyClass && () > ::value));
  EXPECT_TRUE((is_function_type < MyClass && (MyClass &&) > ::value));
  EXPECT_TRUE((is_function_type<const MyClass &(int, float, MyClass)>::value));

  EXPECT_TRUE((!is_function_type<void>::value));
  EXPECT_TRUE((!is_function_type<int>::value));
  EXPECT_TRUE((!is_function_type<MyClass>::value));
  EXPECT_TRUE((!is_function_type<void *>::value));
  EXPECT_TRUE((!is_function_type<const MyClass &>::value));
  EXPECT_TRUE((!is_function_type<MyClass &&>::value));

  EXPECT_TRUE((!is_function_type<void (*)()>::value,
               "function pointers aren't plain functions"));
  EXPECT_TRUE(
      (!is_function_type<Functor>::value, "Functors aren't plain functions"));
  EXPECT_TRUE((!is_function_type<decltype(lambda)>::value,
               "Lambdas aren't plain functions"));
}

template <class T>
class Single {};
template <class T1, class T2>
class Double {};
template <class... T>
class Multiple {};

TEST(TypeTraits, is_instantiation_of) {
  EXPECT_TRUE((is_instantiation_of<Single, Single<void>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<MyClass>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<int>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<void *>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<int *>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<const MyClass &>>::value));
  EXPECT_TRUE((is_instantiation_of<Single, Single<MyClass &&>>::value));
  EXPECT_TRUE((is_instantiation_of<Double, Double<int, void>>::value));
  EXPECT_TRUE(
      (is_instantiation_of<Double, Double<const int &, MyClass *>>::value));
  EXPECT_TRUE((is_instantiation_of<Multiple, Multiple<>>::value));
  EXPECT_TRUE((is_instantiation_of<Multiple, Multiple<int>>::value));
  EXPECT_TRUE((is_instantiation_of<Multiple, Multiple<MyClass &, int>>::value));
  EXPECT_TRUE((
      is_instantiation_of<Multiple, Multiple<MyClass &, int, MyClass>>::value));
  EXPECT_TRUE(
      (is_instantiation_of<Multiple,
                           Multiple<MyClass &, int, MyClass, void *>>::value));

  EXPECT_TRUE((!is_instantiation_of<Single, Double<int, int>>::value));
  EXPECT_TRUE((!is_instantiation_of<Single, Double<int, void>>::value));
  EXPECT_TRUE((!is_instantiation_of<Single, Multiple<int>>::value));
  EXPECT_TRUE((!is_instantiation_of<Double, Single<int>>::value));
  EXPECT_TRUE((!is_instantiation_of<Double, Multiple<int, int>>::value));
  EXPECT_TRUE((!is_instantiation_of<Double, Multiple<>>::value));
  EXPECT_TRUE((!is_instantiation_of<Multiple, Double<int, int>>::value));
  EXPECT_TRUE((!is_instantiation_of<Multiple, Single<int>>::value));
}

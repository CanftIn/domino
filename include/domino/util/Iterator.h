#ifndef DOMINO_UTIL_ITERATOR_H_
#define DOMINO_UTIL_ITERATOR_H_

#include <domino/util/IteratorRange.h>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace domino {

template <typename DerivedT, typename IteratorCategoryT, typename T,
          typename DifferenceTypeT = std::ptrdiff_t, typename PointerT = T *,
          typename ReferenceT = T &>
class iterator_facade_base {
 public:
  using iterator_category = IteratorCategoryT;
  using value_type = T;
  using difference_type = DifferenceTypeT;
  using pointer = PointerT;
  using reference = ReferenceT;

 protected:
  enum {
    IsRandomAccess =
        std::is_base_of_v<std::random_access_iterator_tag, IteratorCategoryT>,
    IsBidirectional =
        std::is_base_of_v<std::bidirectional_iterator_tag, IteratorCategoryT>,
  };

  class ReferenceProxy {
    friend iterator_facade_base;

    DerivedT I;

    ReferenceProxy(DerivedT I) : I(std::move(I)) {}

   public:
    operator ReferenceT() const { return *I; }
  };
  class PointerProxy {
    friend iterator_facade_base;

    ReferenceT R;

    template <typename RefT>
    PointerProxy(RefT &&R) : R(std::forward<RefT>(R)) {}

   public:
    PointerT operator->() const { return &R; }
  };

 public:
  DerivedT operator+(DifferenceTypeT n) const {
    static_assert(std::is_base_of<iterator_facade_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
    static_assert(
        IsRandomAccess,
        "The '+' operator is only defined for random access iterators.");
    DerivedT tmp = *static_cast<const DerivedT *>(this);
    tmp += n;
    return tmp;
  }
  friend DerivedT operator+(DifferenceTypeT n, const DerivedT &i) {
    static_assert(
        IsRandomAccess,
        "The '+' operator is only defined for random access iterators.");
    return i + n;
  }
  DerivedT operator-(DifferenceTypeT n) const {
    static_assert(
        IsRandomAccess,
        "The '-' operator is only defined for random access iterators.");
    DerivedT tmp = *static_cast<const DerivedT *>(this);
    tmp -= n;
    return tmp;
  }

  DerivedT &operator++() {
    static_assert(std::is_base_of<iterator_facade_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
    return static_cast<DerivedT *>(this)->operator+=(1);
  }
  DerivedT operator++(int) {
    DerivedT tmp = *static_cast<DerivedT *>(this);
    ++*static_cast<DerivedT *>(this);
    return tmp;
  }
  DerivedT &operator--() {
    static_assert(
        IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    return static_cast<DerivedT *>(this)->operator-=(1);
  }
  DerivedT operator--(int) {
    static_assert(
        IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    DerivedT tmp = *static_cast<DerivedT *>(this);
    --*static_cast<DerivedT *>(this);
    return tmp;
  }

#ifndef __cpp_impl_three_way_comparison
  bool operator!=(const DerivedT &RHS) const {
    return !(static_cast<const DerivedT &>(*this) == RHS);
  }
#endif

  bool operator>(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) < RHS) &&
           !(static_cast<const DerivedT &>(*this) == RHS);
  }
  bool operator<=(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) > RHS);
  }
  bool operator>=(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) < RHS);
  }

  PointerProxy operator->() const {
    return static_cast<const DerivedT *>(this)->operator*();
  }
  ReferenceProxy operator[](DifferenceTypeT n) const {
    static_assert(IsRandomAccess,
                  "Subscripting is only defined for random access iterators.");
    return static_cast<const DerivedT *>(this)->operator+(n);
  }
};

template <
    typename DerivedT, typename WrappedIteratorT,
    typename IteratorCategoryT =
        typename std::iterator_traits<WrappedIteratorT>::iterator_category,
    typename T = typename std::iterator_traits<WrappedIteratorT>::value_type,
    typename DifferenceTypeT =
        typename std::iterator_traits<WrappedIteratorT>::difference_type,
    typename PointerT = std::conditional_t<
        std::is_same<T, typename std::iterator_traits<
                            WrappedIteratorT>::value_type>::value,
        typename std::iterator_traits<WrappedIteratorT>::pointer, T *>,
    typename ReferenceT = std::conditional_t<
        std::is_same<T, typename std::iterator_traits<
                            WrappedIteratorT>::value_type>::value,
        typename std::iterator_traits<WrappedIteratorT>::reference, T &>>
class iterator_adaptor_base
    : public iterator_facade_base<DerivedT, IteratorCategoryT, T,
                                  DifferenceTypeT, PointerT, ReferenceT> {
  using BaseT = typename iterator_adaptor_base::iterator_facade_base;

 protected:
  WrappedIteratorT I;

  iterator_adaptor_base() = default;

  explicit iterator_adaptor_base(WrappedIteratorT u) : I(std::move(u)) {
    static_assert(std::is_base_of<iterator_adaptor_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
  }

  const WrappedIteratorT &wrapped() const { return I; }

 public:
  using difference_type = DifferenceTypeT;

  DerivedT &operator+=(difference_type n) {
    static_assert(
        BaseT::IsRandomAccess,
        "The '+=' operator is only defined for random access iterators.");
    I += n;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT &operator-=(difference_type n) {
    static_assert(
        BaseT::IsRandomAccess,
        "The '-=' operator is only defined for random access iterators.");
    I -= n;
    return *static_cast<DerivedT *>(this);
  }
  using BaseT::operator-;
  difference_type operator-(const DerivedT &RHS) const {
    static_assert(
        BaseT::IsRandomAccess,
        "The '-' operator is only defined for random access iterators.");
    return I - RHS.I;
  }

  // explicitly provide ++ and -- rather than letting the facade
  // forward to += because WrappedIteratorT might not support +=.
  using BaseT::operator++;
  DerivedT &operator++() {
    ++I;
    return *static_cast<DerivedT *>(this);
  }
  using BaseT::operator--;
  DerivedT &operator--() {
    static_assert(
        BaseT::IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    --I;
    return *static_cast<DerivedT *>(this);
  }

  friend bool operator==(const iterator_adaptor_base &LHS,
                         const iterator_adaptor_base &RHS) {
    return LHS.I == RHS.I;
  }
  friend bool operator<(const iterator_adaptor_base &LHS,
                        const iterator_adaptor_base &RHS) {
    static_assert(
        BaseT::IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return LHS.I < RHS.I;
  }

  ReferenceT operator*() const { return *I; }
};

template <typename WrappedIteratorT,
          typename T = std::remove_reference_t<
              decltype(**std::declval<WrappedIteratorT>())>>
struct pointee_iterator
    : iterator_adaptor_base<
          pointee_iterator<WrappedIteratorT, T>, WrappedIteratorT,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category,
          T> {
  pointee_iterator() = default;
  template <typename U>
  pointee_iterator(U &&u)
      : pointee_iterator::iterator_adaptor_base(std::forward<U &&>(u)) {}

  T &operator*() const { return **this->I; }
};

template <typename RangeT, typename WrappedIteratorT =
                               decltype(std::begin(std::declval<RangeT>()))>
iterator_range<pointee_iterator<WrappedIteratorT>> make_pointee_range(
    RangeT &&Range) {
  using PointeeIteratorT = pointee_iterator<WrappedIteratorT>;
  return make_range(PointeeIteratorT(std::begin(std::forward<RangeT>(Range))),
                    PointeeIteratorT(std::end(std::forward<RangeT>(Range))));
}

template <typename WrappedIteratorT,
          typename T = decltype(&*std::declval<WrappedIteratorT>())>
class pointer_iterator
    : public iterator_adaptor_base<
          pointer_iterator<WrappedIteratorT, T>, WrappedIteratorT,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category,
          T> {
  mutable T Ptr;

 public:
  pointer_iterator() = default;

  explicit pointer_iterator(WrappedIteratorT u)
      : pointer_iterator::iterator_adaptor_base(std::move(u)) {}

  T &operator*() const { return Ptr = &*this->I; }
};

template <typename RangeT, typename WrappedIteratorT =
                               decltype(std::begin(std::declval<RangeT>()))>
iterator_range<pointer_iterator<WrappedIteratorT>> make_pointer_range(
    RangeT &&Range) {
  using PointerIteratorT = pointer_iterator<WrappedIteratorT>;
  return make_range(PointerIteratorT(std::begin(std::forward<RangeT>(Range))),
                    PointerIteratorT(std::end(std::forward<RangeT>(Range))));
}

template <typename WrappedIteratorT,
          typename T1 = std::remove_reference_t<
              decltype(**std::declval<WrappedIteratorT>())>,
          typename T2 = std::add_pointer_t<T1>>
using raw_pointer_iterator =
    pointer_iterator<pointee_iterator<WrappedIteratorT, T1>, T2>;

}  // namespace domino

#endif  // DOMINO_UTIL_ITERATOR_H_
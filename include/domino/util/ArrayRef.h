#ifndef DOMINO_UTIL_ARRAYREF_H_
#define DOMINO_UTIL_ARRAYREF_H_

#include <domino/util/SmallVector.h>
#include <domino/util/STLExtras.h>

#include <array>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <type_traits>
#include <vector>

namespace domino {

template <typename T>
class ArrayRef final {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = const_pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iteartor = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

 private:
  const T* Data = nullptr;

  size_type Length;

  void debugCheckNullptrInvariant() const {
    assert((Data != nullptr) ||
           (Length == 0) &&
               "created ArrayRef with nullptr and non-zero length!");
  }

 public:
  ArrayRef() = default;

  ArrayRef(std::nullopt_t) {}

  ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  constexpr ArrayRef(const T* Data, size_t Length)
      : Data(Data), Length(Length) {
    debugCheckNullptrInvariant();
  }

  constexpr ArrayRef(const T* Begin, const T* End)
      : Data(Begin), Length(End - Begin) {
    debugCheckNullptrInvariant();
  }

  template <typename U>
  ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    debugCheckNullptrInvariant();
  }

  template <typename Container,
            typename = std::enable_if_t<std::is_same<
                std::remove_const_t<decltype(std::declval<Container>().data())>,
                T*>::value>>
  ArrayRef(const Container& C) : Data(C.data()), Length(C.size()) {
    debugCheckNullptrInvariant();
  }

  template <typename A>
  ArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {}

  template <size_t N>
  constexpr ArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  template <size_t N>
  constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

  constexpr ArrayRef(const std::initializer_list<T>& Vec)
      : Data(Vec.begin() == Vec.end() ? (T*)nullptr : Vec.begin()),
        Length(Vec.size()) {}

  /// Construct an ArrayRef<const T*> from ArrayRef<T*>. This uses SFINAE to
  /// ensure that only ArrayRefs of pointers can be converted.
  template <typename U>
  ArrayRef(
      const ArrayRef<U*>& A,
      std::enable_if_t<std::is_convertible_v<U* const*, T const*>>* = nullptr)
      : Data(A.data()), Length(A.size()) {}

  /// Construct an ArrayRef<const T*> from a SmallVector<T*>. This is
  /// templated in order to avoid instantiating SmallVectorTemplateCommon<T>
  /// whenever we copy-construct an ArrayRef.
  template <typename U, typename DummyT>
  ArrayRef(
      const SmallVectorTemplateCommon<U*, DummyT>& Vec,
      std::enable_if_t<std::is_convertible_v<U* const*, T const*>>* = nullptr)
      : Data(Vec.data()), Length(Vec.size()) {}

  /// Construct an ArrayRef<const T*> from std::vector<T*>. This uses SFINAE
  /// to ensure that only vectors of pointers can be converted.
  template <typename U, typename A>
  ArrayRef(
      const std::vector<U*, A>& Vec,
      std::enable_if_t<std::is_convertible_v<U* const*, T const*>>* = nullptr)
      : Data(Vec.data()), Length(Vec.size()) {}

  constexpr iterator begin() const { return Data; }
  constexpr iterator end() const { return Data + Length; }

  constexpr const_iterator cbegin() const { return Data; }
  constexpr const_iterator cend() const { return Data + Length; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  const_reverse_iteartor crbegin() const {
    return const_reverse_iteartor(cend());
  }
  const_reverse_iteartor crend() const {
    return const_reverse_iteartor(cbegin());
  }

  constexpr bool empty() const { return Length == 0; }
  constexpr const T* data() const { return Data; }
  constexpr size_t size() const { return Length; }

  const T& front() const {
    assert(!empty() && "Cannot get the front of an empty ArrayRef");
    return Data[0];
  }

  const T& back() const {
    assert(!empty() && "Cannot get the back of an empty ArrayRef");
    return Data[Length - 1];
  }

  constexpr bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Chop off the first N elements of the array, and keep M
  /// elements in the array.
  ArrayRef<T> slice(size_t N, size_t M) const {
    assert(N + M <= size() && "Invalid slice range");
    return ArrayRef<T>(data() + N, M);
  }

  ArrayRef<T> slice(size_t N) const { return slice(N, size() - N); }

  ArrayRef<T> drop_front(size_t N = 1) const { return slice(N, size() - N); }

  ArrayRef<T> drop_back(size_t N = 1) const { return slice(0, size() - N); }

  template <class PredicateT>
  ArrayRef<T> drop_while(PredicateT Pred) const {
    return ArrayRef<T>(find_if_not(*this, Pred), end());
  }

  template <class PredicateT>
  ArrayRef<T> drop_until(PredicateT Pred) const {
    return ArrayRef<T>(find_if(*this, Pred), end());
  }

  /// Return a copy of *this with only the first \p N elements.
  ArrayRef<T> take_front(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_back(size() - N);
  }

  /// Return a copy of *this with only the last \p N elements.
  ArrayRef<T> take_back(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_front(size() - N);
  }

  /// Return the first N elements of this Array that satisfy the given
  /// predicate.
  template <class PredicateT>
  ArrayRef<T> take_while(PredicateT Pred) const {
    return ArrayRef<T>(begin(), find_if_not(*this, Pred));
  }

  /// Return the first N elements of this Array that don't satisfy the
  /// given predicate.
  template <class PredicateT>
  ArrayRef<T> take_until(PredicateT Pred) const {
    return ArrayRef<T>(begin(), find_if(*this, Pred));
  }

  constexpr const T& operator[](size_t Index) const {
    assert(Index < Length && "Invalid index!");
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(Data, Data + Length); }

  operator std::vector<T>() const { return vec(); }
};

/// Deduction guide to construct an ArrayRef from a single element.
template <typename T>
ArrayRef(const T& OneElt) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a pointer and length
template <typename T>
ArrayRef(const T* data, size_t length) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a range
template <typename T>
ArrayRef(const T* data, const T* end) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a SmallVector
template <typename T>
ArrayRef(const SmallVectorImpl<T>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a SmallVector
template <typename T, unsigned N>
ArrayRef(const SmallVector<T, N>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a std::vector
template <typename T>
ArrayRef(const std::vector<T>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a std::array
template <typename T, std::size_t N>
ArrayRef(const std::array<T, N>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from an ArrayRef (const)
template <typename T>
ArrayRef(const ArrayRef<T>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from an ArrayRef
template <typename T>
ArrayRef(ArrayRef<T>& Vec) -> ArrayRef<T>;

/// Deduction guide to construct an ArrayRef from a C array.
template <typename T, size_t N>
ArrayRef(const T (&Arr)[N]) -> ArrayRef<T>;

/// Construct an ArrayRef from a single element.
template <typename T>
ArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

/// Construct an ArrayRef from a pointer and length.
template <typename T>
ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return ArrayRef<T>(data, length);
}

/// Construct an ArrayRef from a range.
template <typename T>
ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return ArrayRef<T>(begin, end);
}

/// Construct an ArrayRef from a SmallVector.
template <typename T>
ArrayRef<T> makeArrayRef(const SmallVectorImpl<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a SmallVector.
template <typename T, unsigned N>
ArrayRef<T> makeArrayRef(const SmallVector<T, N>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a std::vector.
template <typename T>
ArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a std::array.
template <typename T, std::size_t N>
ArrayRef<T> makeArrayRef(const std::array<T, N>& Arr) {
  return Arr;
}

/// Construct an ArrayRef from an ArrayRef (no-op) (const)
template <typename T>
ArrayRef<T> makeArrayRef(const ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from an ArrayRef (no-op)
template <typename T>
ArrayRef<T>& makeArrayRef(ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a C array.
template <typename T, size_t N>
ArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return ArrayRef<T>(Arr);
}

template <typename T>
bool operator==(domino::ArrayRef<T> a1, domino::ArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(domino::ArrayRef<T> a1, domino::ArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, domino::ArrayRef<T> a2) {
  return domino::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, domino::ArrayRef<T> a2) {
  return !domino::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(domino::ArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(domino::ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(domino::ArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(domino::ArrayRef<T>(a2));
}

using IntArrayRef = ArrayRef<int64_t>;

}  // namespace domino

#endif  // DOMINO_UTIL_ARRAYREF_H_
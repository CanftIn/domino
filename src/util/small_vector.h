#ifndef DOMINO_UTIL_SMALL_VECTOR_H_
#define DOMINO_UTIL_SMALL_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <ostream>
#include <type_traits>
#include <utility>

#include "macros.h"

namespace domino {

template <class Size_T>
class SmallVectorBase {
 protected:
  void* BeginX;
  Size_T Size = 0, Capacity;

  static constexpr size_t SizeTypeMax() {
    return std::numeric_limits<Size_T>::max();
  }

  SmallVectorBase(void* FirstEl, size_t TotalCapacity)
      : BeginX(FirstEl), Capacity(TotalCapacity) {}

  void mallocForGrow(size_t MinSize, size_t TSize, size_t& NewCapacity);

  void growPod(void* FirstEl, size_t MinSize, size_t TSize);

 public:
  SmallVectorBase() = delete;

  size_t size() const { return Size; }

  size_t capacity() const { return Capacity; }

  bool empty() const { return !Size; }

  void setSize(size_t N) {
    assert(N <= capacity());
    Size = N;
  }
};

template <class T>
using SmallVectorSizeType =
    typename std::conditional<sizeof(T) < 4 && sizeof(void*) >= 8, uint64_t,
                              uint32_t>::type;

template <class T, typename = void>
struct SmallVectorAlignmentAndSize {
  alignas(SmallVectorBase<SmallVectorSizeType<T>>) char Base[sizeof(
      SmallVectorBase<SmallVectorSizeType<T>>)];
  alignas(T) char FirstEl[sizeof(T)];
};

template <typename T, typename = void>
class SmallVectorTemplateCommon
    : public SmallVectorBase<SmallVectorSizeType<T>> {
  using Base = SmallVectorBase<SmallVectorSizeType<T>>;

  void* getFirstEl() const {
    return const_cast<void*>(reinterpret_cast<const void*>(
        reinterpret_cast<const char*>(this) +
        offsetof(SmallVectorAlignmentAndSize<T>, FirstEl)));
  }

 protected:
  SmallVectorTemplateCommon(size_t Size) : Base(getFirstEl(), Size) {}

  void growPod(size_t MinSize, size_t TSize) {
    Base::growPod(getFirstEl(), MinSize, TSize);
  }

  bool isSmall() const { return this->BeginX == getFirstEl(); }

  void resetToSmall() {
    this->BeginX = getFirstEl();
    this->Size = this->Capacity = 0;
  }

  bool isReferenceToRange(const void* V, const void* First,
                          const void* Last) const {
    std::less<> LessThan;
    return !LessThan(V, First) && LessThan(V, Last);
  }

  bool isReferenceToStorage(const void* V) {
    return isReferenceToRange(V, this->begin(), this->end());
  }

  bool isRangeInStorage(const void* First, const void* Last) const {
    std::less<> LessThan;
    return !LessThan(First, this->begin()) && !LessThan(Last, First) &&
           !LessThan(this->end(), Last);
  }

  bool isSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    // If the element is not in inline storage, it is safe to reference.
    if (DOMINO_LIKELY(!isReferenceToStorage(Elt))) return true;

    // If the element is in inline storage and the new size is smaller or equal to the current size,
    // it is safe to reference if the element's address is less than the current end.
    if (NewSize <= this->size()) return Elt < this->begin() + NewSize;

    // If the element is in inline storage and the new size is larger than the current size,
    // it is safe to reference if the new size is smaller than or equal to the current capacity.
    return NewSize <= this->capacity();
  }

  void assertSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    (void)Elt;
    (void)NewSize;
    assert(isSafeToReferenceAfterResize(Elt, NewSize) &&
           "Attempting to reference an element of the vector in an operation "
           "that invalidates it");
  }

  void assertSafeToAdd(const void* Elt, size_t N = 1) {
    this->assertSafeToReferenceAfterResize(Elt, this->size() + N);
  }

  void assertSafeToReferenceAfterClear(const T* From, const T* To) {
    if (From == To)
      return;
    this->assertSafeToReferenceAfterResize(From, 0);
    this->assertSafeToReferenceAfterResize(To - 1, 0);
  }

  template <class ItTy,
            std::enable_if_t<!std::is_same<std::remove_const_t<ItTy>, T*>::value,
                             bool> = false>
  void assertSafeToReferenceAfterClear(ItTy, ItTy) {}

  void assertSafeToAddRange(const T* From, const T* To) {
    if (From == To)
      return;
    this->assertSafeToAdd(From, To - From);
    this->assertSafeToAdd(To - 1, To - From);
  }

  template <class ItTy,
            std::enable_if_t<!std::is_same<std::remove_const_t<ItTy>, T*>::value,
                             bool> = false>
  void assertSafeToAddRange(ItTy, ItTy) {}

 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;

  using Base::capacity;
  using Base::empty;
  using Base::size;

  iterator begin() {
    return (iterator)this->BeginX;
  }

  const_iterator begin() const {
    return (const_iterator)this->BeginX;
  }

  iterator end() {
    return begin() + size();
  }

  const_iterator end() const {
    return begin() + size();
  }

  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  size_type size_in_bytes() const {
    return size() * sizeof(T);
  }

  size_type max_size() const {
    return std::min(this->SizeTypeMax(), size_type(-1) / sizeof(T));
  }

  size_t capacity_in_bytes() const {
    return capacity() * sizeof(T);
  }

  pointer data() {
    return pointer(begin());
  }

  const_pointer data() const {
    return const_pointer(begin());
  }

  reference at(size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }

  const_reference at(size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference operator[](size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }

  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference operator[](size_type idx) {
    assert()
  }
};

}  // namespace domino

#endif  // DOMINO_UTIL_SMALL_VECTOR_H_

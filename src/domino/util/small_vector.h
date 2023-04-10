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

  void set_size(size_t N) {
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

    // If the element is in inline storage and the new size is smaller or equal
    // to the current size, it is safe to reference if the element's address is
    // less than the current end.
    if (NewSize <= this->size()) return Elt < this->begin() + NewSize;

    // If the element is in inline storage and the new size is larger than the
    // current size, it is safe to reference if the new size is smaller than or
    // equal to the current capacity.
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
    if (From == To) return;
    this->assertSafeToReferenceAfterResize(From, 0);
    this->assertSafeToReferenceAfterResize(To - 1, 0);
  }

  template <class ItTy, std::enable_if_t<
                            !std::is_same<std::remove_const_t<ItTy>, T*>::value,
                            bool> = false>
  void assertSafeToReferenceAfterClear(ItTy, ItTy) {}

  void assertSafeToAddRange(const T* From, const T* To) {
    if (From == To) return;
    this->assertSafeToAdd(From, To - From);
    this->assertSafeToAdd(To - 1, To - From);
  }

  template <class ItTy, std::enable_if_t<
                            !std::is_same<std::remove_const_t<ItTy>, T*>::value,
                            bool> = false>
  void assertSafeToAddRange(ItTy, ItTy) {}

  template <class U>
  static const T* reserveForParamAndGetAddressImpl(U* This, const T& Elt,
                                                   size_t N) {
    size_t NewSize = This->size() + N;
    if (DOMINO_LIKELY(NewSize <= This->capacity())) return &Elt;

    bool ReferencesStorage = false;
    int64_t Index = -1;
    if (!U::TakesParamByValue) {
      if (DOMINO_UNLIKELY(This->isReferenceToStorage(&Elt))) {
        ReferencesStorage = true;
        Index = &Elt - This->begin();
      }
    }
    This->grow(NewSize);
    return ReferencesStorage ? This->begin() + Index : &Elt;
  }

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

  iterator begin() { return (iterator)this->BeginX; }

  const_iterator begin() const { return (const_iterator)this->BeginX; }

  iterator end() { return begin() + size(); }

  const_iterator end() const { return begin() + size(); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  size_type size_in_bytes() const { return size() * sizeof(T); }

  size_type max_size() const {
    return std::min(this->SizeTypeMax(), size_type(-1) / sizeof(T));
  }

  size_t capacity_in_bytes() const { return capacity() * sizeof(T); }

  pointer data() { return pointer(begin()); }

  const_pointer data() const { return const_pointer(begin()); }

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

  reference front() {
    assert(!empty());
    return begin()[0];
  }

  const_reference front() const {
    assert(!empty());
    return begin()[0];
  }

  reference back() {
    assert(!empty());
    return end()[-1];
  }

  const_reference back() const {
    assert(!empty());
    return end()[-1];
  }
};

template <typename T, bool = (std::is_trivially_copy_constructible<T>::value) &&
                             (std::is_trivially_move_constructible<T>::value) &&
                             (std::is_trivially_destructible<T>::value)>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  static constexpr bool TakesParamByValue = false;
  using ValueParamT = const T&;

  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T* S, T* E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Move the range [I, E) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(std::make_move_iterator(I),
                            std::make_move_iterator(E), Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It2 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  // Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

  T* mallocForGrow(size_t MinSize, size_t& NewCapacity) {
    return static_cast<T*>(
        SmallVectorBase<SmallVectorSizeType<T>>::mallocForGrow(
            MinSize, sizeof(T), NewCapacity));
  }

  void moveElementsForGrow(T* NewElts);

  void takeAllocationForGrow(T* NewElts, size_t NewCapacity);

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
    return this->reserveForParamAndGetAddressImpl(this, Elt, N);
  }

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  static T&& forward_value_param(T&& V) { return std::move(V); }

  static const T& forward_value_param(const T& V) { return V; }

  void growAndAssign(size_t NumElts, const T& Elt) {
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(NumElts, NewCapacity);
    std::uninitialized_fill_n(NewElts, NumElts, Elt);
    this->destory_range(this->begin(), this->end());
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(0, NewCapacity);
    ::new ((void*)(NewElts + this->size())) T(std::forward<ArgTypes>(Args)...);
    moveElementsForGrow(NewElts);
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(this->size() + 1);
    return this->back();
  }

 public:
  void push_back(const T& Elt) {
    const T* EltPtr = reserveForParamAndGetAddress(Elt);
    ::new ((void*)this->end()) T(*EltPtr);
    this->set_size(this->size() + 1);
  }

  void push_back(T&& Elt) {
    T* EltPtr = reserveForParamAndGetAddress(Elt);
    ::new ((void*)this->end()) T(::std::move(*EltPtr));
    this->set_size(this->size() + 1);
  }

  void pop_back() {
    this->set_size(this->size() - 1);
    this->end()->~T();
  }
};

template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::grow(size_t MinSize) {
  size_t NewCapacity = 0;
  T* NewElts = mallocForGrow(MinSize, NewCapacity);
  moveElementsForGrow(NewElts);
  takeAllocationForGrow(NewElts, NewCapacity);
}

template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::moveElementsForGrow(
    T* NewElts) {
  this->uninitialized_move(this->begin(), this->end(), NewElts);
  destory_range(this->begin(), this->end());
}

template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::takeAllocationForGrow(
    T* NewElts, size_t NewCapacity) {
  if (!this->isSmall()) free(this->begin());

  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  static constexpr bool TakesParamByValue = sizeof(T) <= 2 * sizeof(void*);

  using ValueParamT =
      typename std::conditional<TakesParamByValue, T, const T&>::type;

  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T*, T*) {}

  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It2 E, It2 Dest) {
    uninitialized_copy(I, E, Dest);
  }

  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It2 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1* I, T2* E, T2* Dest,
      std::enable_if_t<std::is_same<typename std::remove_const<T1>::type,
                                    T2>::value>* = nullptr) {
    if (I != E) memcpy(reinterpret_cast<void*>(Dest), I, (E - I) * sizeof(T));
  }

  void grow(size_t MinSize = 0) { this->growPod(MinSize, sizeof(T)); }

  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
    return this->reserveForParamAndGetAddressImpl(this, Elt, N);
  }

  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  static ValueParamT forward_value_param(ValueParamT V) { return V; }

  void growAndAssign(size_t NumElts, T Elt) {
    this->set_size(0);
    this->grow(NumElts);
    std::uninitialized_fill_n(this->begin(), NumElts, Elt);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    push_back(T(std::forward<ArgTypes>(Args)...));
    return this->back();
  }

 public:
  void push_back(ValueParamT Elt) {
    const T* EltPtr = reserveForParamAndGetAddress(Elt);
    memcpy(reinterpret_cast<void*>(this->end()), EltPtr, sizeof(T));
    this->set_size(this->size() + 1);
  }

  void pop_back() { this->set_size(this->size() - 1); }
};

template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T> {
  using SuperClass = SmallVectorTemplateBase<T>;

 public:
  using iterator = typename SuperClass::iterator;
  using const_iterator = typename SuperClass::const_iterator;
  using reference = typename SuperClass::reference;
  using size_type = typename SuperClass::size_type;

 protected:
  using SmallVectorTemplateBase<T>::TakesParamByValue;
  using ValueParamT = typename SuperClass::ValueParamT;

  explicit SmallVectorImpl(unsigned N) : SmallVectorTemplateBase<T>(N) {}

 public:
  SmallVectorImpl(const SmallVectorImpl&) = delete;

  ~SmallVectorImpl() {
    if (!this->isSmall()) free(this->begin());
  }

  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->Size = 0;
  }

 private:
  template <bool ForOverwrite>
  void resizeImpl(size_type N) {
    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
    } else if (N > this->size()) {
      this->reserve(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I) {
        if (ForOverwrite)
          new (&*I) T;
        else
          new (&*I) T();
      }
      this->set_size(N);
    }
  }

 public:
  void resize(size_type N) { resizeImpl<false>(N); }

  /// Like resize, but \ref T is POD, the new values won't be initialized.
  void resize_for_overwrite(size_type N) { resizeImpl<true>(N); }

  void resize(size_type N, ValueParamT NV) {
    if (N == this->size()) return;

    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
      return;
    }

    // N > this->size(). Defer to append.
    this->append(N - this->size(), NV);
  }

  void reserve(size_type N) {
    if (this->capacity() < N) this->grow(N);
  }

  void pop_back_n(size_type NumItems) {
    assert(this->size() >= NumItems);
    this->destroy_range(this->end() - NumItems, this->end());
    this->set_size(this->size() - NumItems);
  }

  [[nodiscard]] T pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(SmallVectorImpl& RHS);

  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>>
  void append(in_iter in_start, in_iter in_end) {
    this->assertSafeToAddRange(in_start, in_end);
    size_type NumInputs = std::distance(in_start, in_end);
    this->reserve(this->size() + NumInputs);
    this->uninitialized_copy(in_start, in_end, this->end());
    this->set_size(this->size() + NumInputs);
  }

  void append(size_type NumInputs, ValueParamT Elt) {
    const T* EltPtr = this->reserveForParamAndGetAddress(Elt, NumInputs);
    std::uninitialized_fill_n(this->end(), NumInputs, *EltPtr);
    this->set_size(this->size() + NumInputs);
  }

  void append(std::initializer_list<T> IL) {
    append(IL.begin(), IL.end());
  }

  void append(const SmallVectorImpl& RHS) {
    append(RHS.begin(), RHS.end());
  }

  
};

}  // namespace domino

#endif  // DOMINO_UTIL_SMALL_VECTOR_H_

#ifndef DOMINO_UTIL_SMALL_VECTOR_H_
#define DOMINO_UTIL_SMALL_VECTOR_H_

#include <domino/util/macros.h>

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

  void* mallocForGrow(size_t MinSize, size_t TSize, size_t& NewCapacity);

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

[[noreturn]] static void report_size_overflow(size_t MinSize, size_t MaxSize);
static void report_size_overflow(size_t MinSize, size_t MaxSize) {
  std::string Reason = "SmallVector unable to grow. Requested capacity (" +
                       std::to_string(MinSize) +
                       ") is larger than maximum value for size type (" +
                       std::to_string(MaxSize) + ")";
  throw std::length_error(Reason);
}

[[noreturn]] static void report_at_maximum_capacity(size_t MaxSize);
static void report_at_maximum_capacity(size_t MaxSize) {
  std::string Reason =
      "SmallVector capacity unable to grow. Already at maximum size " +
      std::to_string(MaxSize);
  throw std::length_error(Reason);
}

template <class Size_T>
static size_t getNewCapacity(size_t MinSize, size_t TSize, size_t OldCapacity) {
  constexpr size_t MaxSize = std::numeric_limits<Size_T>::max();

  if (MinSize > MaxSize) report_size_overflow(MinSize, MaxSize);

  if (OldCapacity == MaxSize) report_at_maximum_capacity(MaxSize);

  size_t NewCapacity = 2 * OldCapacity + 1;
  return std::min(std::max(NewCapacity, MinSize), MaxSize);
}

template <class Size_T>
void* SmallVectorBase<Size_T>::mallocForGrow(size_t MinSize, size_t TSize,
                                             size_t& NewCapacity) {
  NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
  auto Result = std::malloc(NewCapacity * TSize);
  if (Result == nullptr) throw std::bad_alloc();
  return Result;
}

template <class Size_T>
void SmallVectorBase<Size_T>::growPod(void* FirstEl, size_t MinSize,
                                      size_t TSize) {
  size_t NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
  void* NewElts = nullptr;
  if (BeginX == FirstEl) {
    NewElts = std::malloc(NewCapacity * TSize);
    if (NewElts == nullptr) throw std::bad_alloc();

    memcpy(NewElts, this->BeginX, size() * TSize);
  } else {
    NewElts = std::realloc(this->BeginX, NewCapacity * TSize);
    if (NewElts == nullptr) throw std::bad_alloc();
  }
  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

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
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
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
    this->destroy_range(this->begin(), this->end());
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
  destroy_range(this->begin(), this->end());
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
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    uninitialized_copy(I, E, Dest);
  }

  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1* I, T1* E, T2* Dest,
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

  void append(std::initializer_list<T> IL) { append(IL.begin(), IL.end()); }

  void append(const SmallVectorImpl& RHS) { append(RHS.begin(), RHS.end()); }

  void assign(size_type NumElts, ValueParamT Elt) {
    if (NumElts > this->capacity()) {
      this->growAndAssign(NumElts, Elt);
      return;
    }
    std::fill_n(this->begin(), std::min(NumElts, this->size()), Elt);
    if (NumElts > this->size())
      std::uninitialized_fill_n(this->end(), NumElts - this->size(), Elt);
    else if (NumElts < this->size())
      this->destroy_range(this->begin() + NumElts, this->end());
    this->set_size(NumElts);
  }

  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>>
  void assign(in_iter in_start, in_iter in_end) {
    this->assertSafeToReferenceAfterClear(in_start, in_end);
    clear();
    append(in_start, in_end);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  void assign(const SmallVectorImpl& RHS) { assign(RHS.begin(), RHS.end()); }

  iterator erase(const_iterator CI) {
    iterator I = const_cast<iterator>(CI);

    assert(this->isReferenceToStorage(CI) &&
           "Iterator to erase is out of bounds.");

    iterator N = I;
    std::move(I + 1, this->end(), I);
    this->pop_back();
    return (N);
  }

  iterator erase(const_iterator CS, const_iterator CE) {
    iterator S = const_cast<iterator>(CS);
    iterator E = const_cast<iterator>(CE);

    assert(this->isRangeInStorage(S, E) && "Range to erase is out of bounds.");

    iterator N = S;
    iterator I = std::move(E, this->end(), S);
    this->destroy_range(I, this->end());
    this->set_size(I - this->begin());
    return (N);
  }

 private:
  template <class ArgType>
  iterator insert_one_impl(iterator I, ArgType&& Elt) {
    static_assert(
        std::is_same<std::remove_const_t<std::remove_reference_t<ArgType>>,
                     T>::value,
        "ArgType must be derived from T!");
    if (I == this->end()) {
      this->push_back(::std::forward<ArgType>(Elt));
      return this->end() - 1;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    size_t Index = I - this->begin();
    std::remove_reference_t<ArgType>* EltPtr =
        this->reserveForParamAndGetAddress(Elt);
    I = this->begin() + Index;

    ::new ((void*)this->end()) T(::std::move(this->back()));

    std::move_backward(I, this->end() - 1, this->end());
    this->set_size(this->size() + 1);

    static_assert(!TakesParamByValue || std::is_same<ArgType, T>::value,
                  "ArgType must be 'T' when taking by value!");
    if (!TakesParamByValue && this->isReferenceToRange(EltPtr, I, this->end()))
      ++EltPtr;

    *I = ::std::forward<ArgType>(*EltPtr);
    return I;
  }

 public:
  iterator insert(iterator I, T&& Elt) {
    return insert_one_impl(I, this->forward_value_param(std::move(Elt)));
  }

  iterator insert(iterator I, const T& Elt) {
    return insert_one_impl(I, this->forward_value_param(Elt));
  }

  iterator insert(iterator I, size_type NumToInsert, ValueParamT Elt) {
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {
      append(NumToInsert, Elt);
      return this->begin() + InsertElt;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    const T* EltPtr = this->reserveForParamAndGetAddress(Elt, NumToInsert);
    I = this->begin() + InsertElt;

    if (size_t(this->end() - I) >= NumToInsert) {
      T* OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);

      if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
        EltPtr += NumToInsert;

      std::fill_n(I, NumToInsert, *EltPtr);
      return I;
    }

    T* OldEnd = this->end();
    this->set_size(this->size() + NumToInsert);
    size_t NumOverwritten = OldEnd - I;
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);

    if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
      EltPtr += NumToInsert;

    std::fill_n(I, NumOverwritten, *EltPtr);
    std::uninitialized_fill_n(OldEnd, NumToInsert - NumOverwritten, *EltPtr);
    return I;
  }

  template <typename ItTy,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>>
  iterator insert(iterator I, ItTy From, ItTy To) {
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {
      append(From, To);
      return this->begin() + InsertElt;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    this->assertSafeToAddRange(From, To);
    size_t NumToInsert = std::distance(From, To);
    reserve(this->size() + NumToInsert);
    I = this->begin() + InsertElt;

    if (size_t(this->end() - I) >= NumToInsert) {
      T* OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);
      std::copy(From, To, I);
      return I;
    }

    T* OldEnd = this->end();
    this->set_size(this->size() + NumToInsert);
    size_t NumOverwritten = OldEnd - I;
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);

    for (T* J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J;
      ++From;
    }

    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  template <typename... ArgTypes>
  reference emplace_back(ArgTypes&&... Args) {
    if (DOMINO_UNLIKELY(this->size() >= this->capacity()))
      return this->growAndEmplaceBack(std::forward<ArgTypes>(Args)...);

    ::new ((void*)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->set_size(this->size() + 1);
    return this->back();
  }

  SmallVectorImpl& operator=(const SmallVectorImpl& RHS);

  SmallVectorImpl& operator=(SmallVectorImpl&& RHS);

  bool operator==(const SmallVectorImpl& RHS) const {
    if (this->size() != RHS.size()) return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }

  bool operator!=(const SmallVectorImpl& RHS) const { return !(*this == RHS); }

  bool operator<(const SmallVectorImpl& RHS) const {
    return std::lexicographical_compare(this->begin(), this->end(), RHS.begin(),
                                        RHS.end());
  }
};

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T>& RHS) {
  if (this == &RHS) return;

  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->Size, RHS.Size);
    std::swap(this->Capacity, RHS.Capacity);
    return;
  }
  this->reserve(RHS.size());
  RHS.reserve(this->size());

  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();

  for (size_type i = 0; i != NumShared; ++i) std::swap((*this)[i], RHS[i]);

  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin() + NumShared, this->end(), RHS.end());
    RHS.set_size(RHS.size() + EltDiff);
    this->destroy_range(this->begin() + NumShared, this->end());
    this->set_size(NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin() + NumShared, RHS.end(), this->end());
    this->set_size(this->size() + EltDiff);
    this->destroy_range(RHS.begin() + NumShared, RHS.end());
    RHS.set_size(NumShared);
  }
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(
    const SmallVectorImpl<T>& RHS) {
  if (this == &RHS) return *this;

  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (RHSSize <= CurSize) {
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
    else
      NewEnd = this->begin();

    this->destroy_range(NewEnd, this->end());
    this->set_size(RHSSize);
    return *this;
  }

  if (this->capacity() < RHSSize) {
    this->clear();
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    std::copy(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  this->uninitialized_copy(RHS.begin() + CurSize, RHS.end(),
                           this->begin() + CurSize);
  this->set_size(RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(SmallVectorImpl<T>&& RHS) {
  if (this == &RHS) return *this;

  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall()) free(this->begin());
    this->BeginX = RHS.BeginX;
    this->Size = RHS.Size;
    this->Capacity = RHS.Capacity;
    RHS.resetToSmall();
    return *this;
  }

  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();

  if (CurSize >= RHSSize) {
    iterator NewEnd = this->begin();
    if (RHSSize) NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    this->destroy_range(NewEnd, this->end());
    this->set_size(RHSSize);
    RHS.clear();
    return *this;
  }

  if (this->capacity() < RHSSize) {
    this->clear();
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    std::move(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  this->uninitialized_move(RHS.begin() + CurSize, RHS.end(),
                           this->begin() + CurSize);
  this->set_size(RHSSize);
  RHS.clear();
  return *this;
}

template <typename T, unsigned N>
struct SmallVectorStorage {
  alignas(T) char InlineElts[N * sizeof(T)];
};

template <typename T>
struct alignas(T) SmallVectorStorage<T, 0> {};

template <typename T, unsigned N>
class SmallVector;

template <typename T>
struct CalculateSmallVectorDefaultInlinedElements {
  static constexpr size_t kPreferredSmallVectorSizeof = 64;
  static_assert(
      sizeof(T) <= 256,
      "You are trying to use a default number of inlined elements for "
      "`SmallVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `SmallVector<T, N>` to make "
      "sure you really want that much inline storage.");
  static constexpr size_t PreferredInlineBytes =
      kPreferredSmallVectorSizeof - sizeof(SmallVector<T, 0>);
  static constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  static constexpr size_t value =
      NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

template <typename T,
          unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value>
class SmallVector : public SmallVectorImpl<T>, SmallVectorStorage<T, N> {
 public:
  SmallVector() : SmallVectorImpl<T>(N) {}

  ~SmallVector() { this->destroy_range(this->begin(), this->end()); }

  explicit SmallVector(size_t Size, const T& Value = T())
      : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  template <typename ItTy,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible<typename std::iterator_traits<
                                  decltype(std::declval<Container>()
                                               .begin())>::iterator_category,
                              std::input_iterator_tag>::value &&
              std::is_convertible<typename std::iterator_traits<
                                      decltype(std::declval<Container>()
                                                   .end())>::iterator_category,
                                  std::input_iterator_tag>::value,
          int> = 0>
  explicit SmallVector(Container&& c) : SmallVectorImpl<T>(N) {
    this->append(c.begin(), c.end());
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  SmallVector(const SmallVector& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty()) SmallVectorImpl<T>::operator=(RHS);
  }

  SmallVector& operator=(const SmallVector& RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector(SmallVector&& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty()) SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible<typename std::iterator_traits<
                                  decltype(std::declval<Container>()
                                               .begin())>::iterator_category,
                              std::input_iterator_tag>::value &&
              std::is_convertible<typename std::iterator_traits<
                                      decltype(std::declval<Container>()
                                                   .end())>::iterator_category,
                                  std::input_iterator_tag>::value,
          int> = 0>
  const SmallVector& operator=(const Container& RHS) {
    this->assign(RHS.begin(), RHS.end());
    return *this;
  }

  SmallVector(SmallVectorImpl<T>&& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty()) SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  SmallVector& operator=(SmallVector&& RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  SmallVector& operator=(SmallVectorImpl<T>&& RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible<typename std::iterator_traits<
                                  decltype(std::declval<Container>()
                                               .begin())>::iterator_category,
                              std::input_iterator_tag>::value &&
              std::is_convertible<typename std::iterator_traits<
                                      decltype(std::declval<Container>()
                                                   .end())>::iterator_category,
                                  std::input_iterator_tag>::value,
          int> = 0>
  const SmallVector& operator=(Container&& C) {
    this->assign(C.begin(), C.end());
    return *this;
  }

  SmallVector& operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template <typename T, unsigned N>
inline size_t capacity_in_bytes(const SmallVector<T, N>& X) {
  return X.capacity_in_bytes();
}

template <typename T, unsigned N>
std::ostream& operator<<(std::ostream& out, const SmallVector<T, N>& list) {
  int i = 0;
  out << "[";
  for (auto e : list) {
    if (i++ > 0) out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

template <typename RangeType>
using ValueTypeFromRangeType =
    typename std::remove_const<typename std::remove_reference<
        decltype(*std::begin(std::declval<RangeType&>()))>::type>::type;

/// Given a range of type R, iterate the entire range and return a
/// SmallVector with elements of the vector.  This is useful, for example,
/// when you want to iterate a range and then sort the results.
template <unsigned Size, typename R>
SmallVector<ValueTypeFromRangeType<R>, Size> to_vector(R&& Range) {
  return {std::begin(Range), std::end(Range)};
}

template <typename R>
SmallVector<ValueTypeFromRangeType<R>,
            CalculateSmallVectorDefaultInlinedElements<
                ValueTypeFromRangeType<R>>::value>
to_vector(R&& Range) {
  return {std::begin(Range), std::end(Range)};
}

}  // namespace domino

namespace std {

/// Implement std::swap in terms of SmallVector swap.
template <typename T>
inline void swap(domino::SmallVectorImpl<T>& LHS,
                 domino::SmallVectorImpl<T>& RHS) {
  LHS.swap(RHS);
}

/// Implement std::swap in terms of SmallVector swap.
template <typename T, unsigned N>
inline void swap(domino::SmallVector<T, N>& LHS,
                 domino::SmallVector<T, N>& RHS) {
  LHS.swap(RHS);
}

}  // namespace std

#endif  // DOMINO_UTIL_SMALL_VECTOR_H_

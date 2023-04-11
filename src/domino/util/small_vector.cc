#include "small_vector.h"

#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace domino {

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
  RHS->reserve(this->size());

  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();

  for (size_type i = 0; i != NumShared; ++i) std::swap((*this)[i], RHS[i]);

  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin() + NumShared, this->end(), RHS.end());
    RHS.set_size(RHS.size() + EltDiff);
    this->destroy_range(this->begin() + NumShared, this->end());
    this->set_size(NumShared);
  } else if (RHS.size() > this->size) {
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

}  // namespace domino
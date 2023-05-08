#ifndef DOMINO_UTIL_INTRUSIVEREFCNTPTR_H_
#define DOMINO_UTIL_INTRUSIVEREFCNTPTR_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <type_traits>

namespace domino {

template <class Derived>
class RefCountedBase {
  mutable unsigned RefCount = 0;

 protected:
  RefCountedBase() = default;
  RefCountedBase(const RefCountedBase &) {}
  RefCountedBase &operator=(const RefCountedBase &) = delete;

#ifndef NDEBUG
  ~RefCountedBase() {
    assert(RefCount == 0 &&
           "Destruction occurred when there are still references to this.");
  }
#else
  ~RefCountedBase() = default;
#endif

 public:
  void Retain() const { ++RefCount; }

  void Release() const {
    assert(RefCount > 0 && "Reference count is already zero.");
    if (--RefCount == 0) {
      delete static_cast<const Derived *>(this);
    }
  }
};

template <class Derived>
class ThreadSafeRefCountedBase {
  mutable std::atomic<int> RefCount{0};

 protected:
  ThreadSafeRefCountedBase() = default;
  ThreadSafeRefCountedBase(const ThreadSafeRefCountedBase &) {}
  ThreadSafeRefCountedBase &operator=(const ThreadSafeRefCountedBase &) =
      delete;

#ifndef NDEBUG
  ~ThreadSafeRefCountedBase() {
    assert(RefCount == 0 &&
           "Destruction occurred when there are still references to this.");
  }
#else
  ~ThreadSafeRefCountedBase() = default;
#endif

 public:
  void Retain() const { RefCount.fetch_add(1, std::memory_order_relaxed); }

  void Release() const {
    int NewRefCount = RefCount.fetch_sub(1, std::memory_order_acq_rel) - 1;
    assert(NewRefCount >= 0 && "Reference count was already zero.");
    if (NewRefCount == 0) delete static_cast<const Derived *>(this);
  }
};

template <typename T>
struct IntrusiveRefCntPtrInfo {
  static void retain(T *obj) { obj->Retain(); }
  static void release(T *obj) { obj->Release(); }
};

template <typename T>
class IntrusiveRefCntPtr {
  T *Obj = nullptr;

 public:
  using element_type = T;

  explicit IntrusiveRefCntPtr() = default;
  IntrusiveRefCntPtr(T *Obj) : Obj(Obj) { retain(); }
  IntrusiveRefCntPtr(const IntrusiveRefCntPtr &Other) : Obj(Other.Obj) {
    retain();
  }
  IntrusiveRefCntPtr(IntrusiveRefCntPtr &&Other) : Obj(Other.Obj) {
    Other.Obj = nullptr;
  }

  template <class X,
            std::enable_if_t<std::is_convertible_v<X *, T *>, bool> = true>
  IntrusiveRefCntPtr(IntrusiveRefCntPtr<X> S) : Obj(S.get()) {
    S.Obj = nullptr;
  }

  template <class X,
            std::enable_if_t<std::is_convertible_v<X *, T *>, bool> = true>
  IntrusiveRefCntPtr(std::unique_ptr<X> S) : Obj(S.release()) {
    retain();
  }

  ~IntrusiveRefCntPtr() { release(); }

  IntrusiveRefCntPtr &operator=(IntrusiveRefCntPtr S) {
    swap(S);
    return *this;
  }

  T &operator*() const { return *Obj; }
  T *operator->() const { return Obj; }
  T *get() const { return Obj; }
  explicit operator bool() const { return !!Obj; }

  void swap(IntrusiveRefCntPtr &Other) { std::swap(Obj, Other.Obj); }

  void reset() {
    release();
    Obj = nullptr;
  }

  void resetWithoutRelease() { Obj = nullptr; }

 private:
  void retain() {
    if (Obj) IntrusiveRefCntPtrInfo<T>::retain(Obj);
  }

  void release() {
    if (Obj) IntrusiveRefCntPtrInfo<T>::release(Obj);
  }

  template <typename X>
  friend class IntrusiveRefCntPtr;
};

template <typename T, typename U>
inline bool operator==(const IntrusiveRefCntPtr<T> &A,
                       const IntrusiveRefCntPtr<U> &B) {
  return A.get() == B.get();
}

template <class T, class U>
inline bool operator!=(const IntrusiveRefCntPtr<T> &A,
                       const IntrusiveRefCntPtr<U> &B) {
  return A.get() != B.get();
}

template <class T, class U>
inline bool operator==(const IntrusiveRefCntPtr<T> &A, U *B) {
  return A.get() == B;
}

template <class T, class U>
inline bool operator!=(const IntrusiveRefCntPtr<T> &A, U *B) {
  return A.get() != B;
}

template <class T, class U>
inline bool operator==(T *A, const IntrusiveRefCntPtr<U> &B) {
  return A == B.get();
}

template <class T, class U>
inline bool operator!=(T *A, const IntrusiveRefCntPtr<U> &B) {
  return A != B.get();
}

template <class T>
bool operator==(std::nullptr_t, const IntrusiveRefCntPtr<T> &B) {
  return !B;
}

template <class T>
bool operator==(const IntrusiveRefCntPtr<T> &A, std::nullptr_t B) {
  return B == A;
}

template <class T>
bool operator!=(std::nullptr_t A, const IntrusiveRefCntPtr<T> &B) {
  return !(A == B);
}

template <class T>
bool operator!=(const IntrusiveRefCntPtr<T> &A, std::nullptr_t B) {
  return !(A == B);
}

template <typename From>
struct simplify_type;

template <class T>
struct simplify_type<IntrusiveRefCntPtr<T>> {
  using SimpleType = T *;

  static SimpleType getSimplifiedValue(IntrusiveRefCntPtr<T> &Val) {
    return Val.get();
  }
};

template <class T>
struct simplify_type<const IntrusiveRefCntPtr<T>> {
  using SimpleType = T *;

  static SimpleType getSimplifiedValue(const IntrusiveRefCntPtr<T> &Val) {
    return Val.get();
  }
};

/// Factory function for creating intrusive ref counted pointers.
template <typename T, typename... Args>
IntrusiveRefCntPtr<T> makeIntrusiveRefCnt(Args &&...A) {
  return IntrusiveRefCntPtr<T>(new T(std::forward<Args>(A)...));
}

}  // namespace domino

#endif  // DOMINO_UTIL_INTRUSIVEREFCNTPTR_H_
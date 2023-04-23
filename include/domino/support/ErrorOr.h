#ifndef DOMINO_SUPPORT_ERROROR_H_
#define DOMINO_SUPPORT_ERROROR_H_

#include <domino/util/AlignOf.h>

#include <cassert>
#include <system_error>
#include <type_traits>

namespace domino {

template <class T>
class ErrorOr {
  template <class OtherT>
  friend class ErrorOr;

  static constexpr bool isRef = std::is_reference<T>::value;

  using wrap = std::reference_wrapper<std::remove_reference_t<T>>;

 public:
  using storage_type = std::conditional_t<isRef, wrap, T>;

 private:
  using reference = std::remove_reference_t<T>&;
  using const_reference = const std::remove_reference_t<T>&;
  using pointer = std::remove_reference_t<T>*;
  using const_pointer = const std::remove_reference_t<T>*;

 public:
  template <class E>
  ErrorOr(E ErrorCode, std::enable_if_t<std::is_error_code_enum_v<E> ||
                                            std::is_error_condition_enum_v<E>,
                                        int> = 0)
      : HasError(true) {
    new (getErrorStorage()) std::error_code(make_error_code(ErrorCode));
  }

  ErrorOr(std::error_code ErrorCode) : HasError(true) {
    new (getErrorStorage()) std::error_code(ErrorCode);
  }

  template <class OtherT>
  ErrorOr(OtherT&& Val,
          std::enable_if_t<std::is_convertible_v<OtherT, T>, int> = 0)
      : HasError(false) {
    new (getStorage()) storage_type(std::forward<OtherT>(Val));
  }

  ErrorOr(const ErrorOr& Other) { copyConstruct(Other); }

  template <class OtherT>
  ErrorOr(const ErrorOr<OtherT>& Other,
          std::enable_if_t<std::is_convertible_v<OtherT, T>, int> = 0) {
    copyConstruct(Other);
  }

  template <class OtherT>
  ErrorOr(const ErrorOr<OtherT>& Other,
          std::enable_if_t<!std::is_convertible_v<OtherT, T>, int> = 0) {
    copyConstruct(Other);
  }

  ErrorOr(ErrorOr&& Other) { moveConstruct(std::move(Other)); }

  template <class OtherT>
  ErrorOr(ErrorOr<OtherT>&& Other,
          std::enable_if_t<std::is_convertible_v<OtherT, T>, int> = 0) {
    moveConstruct(std::move(Other));
  }

  template <class OtherT>
  ErrorOr(ErrorOr<OtherT>&& Other,
          std::enable_if_t<!std::is_convertible_v<OtherT, T>, int> = 0) {
    moveConstruct(std::move(Other));
  }

  ErrorOr& operator=(const ErrorOr& Other) {
    copyAssign(Other);
    return *this;
  }

  ErrorOr& operator=(ErrorOr&& Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  ~ErrorOr() {
    if (!HasError) getStorage()->~storage_type();
  }

  explicit operator bool() const { return !HasError; }

  reference get() { return *getStorage(); }
  const_reference get() const { return *getStorage(); }

  std::error_code getError() const {
    return HasError ? *getErrorStorage() : std::error_code();
  }

  pointer operator->() { return toPointer(getStorage()); }

  const_pointer operator->() const { return toPointer(getStorage()); }

  reference operator*() { return get(); }

  const_reference operator*() const { return get(); }

 private:
  template <class OtherT>
  void copyConstruct(const ErrorOr<OtherT>& Other) {
    if (!Other.HasError) {
      HasError = false;
      new (getStorage()) storage_type(*Other.getStorage());
    } else {
      HasError = true;
      new (getErrorStorage()) std::error_code(Other.getError());
    }
  }

  template <class T1>
  static bool compareThisIfSameType(const T1& a, const T1& b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1& a, const T2& b) {
    return false;
  }

  template <class OtherT>
  void copyAssign(const ErrorOr<OtherT>& Other) {
    if (compareThisIfSameType(*this, Other)) return;

    this->~ErrorOr();
    new (this) ErrorOr(Other);
  }

  template <class OtherT>
  void moveConstruct(ErrorOr<OtherT>&& Other) {
    if (!Other.HasError) {
      HasError = false;
      new (getStorage()) storage_type(std::move(*Other.getStorage()));
    } else {
      HasError = true;
      new (getErrorStorage()) std::error_code(Other.getError());
    }
  }

  template <class OtherT>
  void moveAssign(ErrorOr<OtherT>&& Other) {
    if (compareThisIfSameType(*this, Other)) return;

    this->~ErrorOr();
    new (this) ErrorOr(std::move(Other));
  }

  pointer toPointer(pointer Val) { return Val; }

  const_pointer toPointer(const_pointer Val) { return Val; }

  pointer toPointer(wrap* Val) { return &Val->get(); }

  const_pointer toPointer(const wrap* Val) const { return &Val->get(); }

  storage_type* getStorage() {
    assert(!HasError && "Cannot get storage of ErrorOr with error");
    return reinterpret_cast<storage_type*>(&TStorage);
  }

  const storage_type* getStorage() const {
    assert(!HasError && "Cannot get storage of ErrorOr with error");
    return reinterpret_cast<const storage_type*>(&TStorage);
  }

  std::error_code* getErrorStorage() {
    assert(HasError && "Cannot get error storage of ErrorOr without error");
    return reinterpret_cast<std::error_code*>(&EStorage);
  }

  const std::error_code* getErrorStorage() const {
    assert(HasError && "Cannot get error storage of ErrorOr without error");
    return const_cast<ErrorOr<T>*>(this)->getErrorStorage();
  }

  union {
    AlignedCharArrayUnion<storage_type> TStorage;
    AlignedCharArrayUnion<std::error_code> EStorage;
  };
  bool HasError : 1;
};

template <class T, class E>
std::enable_if_t<std::is_error_code_enum<E>::value ||
                     std::is_error_condition_enum<E>::value,
                 bool>
operator==(const ErrorOr<T>& Err, E Code) {
  return Err.getError() == Code;
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_ERROROR_H_
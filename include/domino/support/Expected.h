#ifndef DOMINO_SUPPORT_EXPECTED_H_
#define DOMINO_SUPPORT_EXPECTED_H_

#include <domino/support/Error.h>
#include <domino/util/AlignOf.h>

namespace domino {

/// Tagged union holding either a T or a Error.
///
/// This class parallels ErrorOr, but replaces error_code with Error. Since
/// Error cannot be copied, this class replaces getError() with
/// takeError(). It also adds an bool errorIsA<ErrT>() method for testing the
/// error class type.
///
/// Example usage of 'Expected<T>' as a function return type:
///
///   @code{.cpp}
///     Expected<int> myDivide(int A, int B) {
///       if (B == 0) {
///         // return an Error
///         return createStringError(inconvertibleErrorCode(),
///                                  "B must not be zero!");
///       }
///       // return an integer
///       return A / B;
///     }
///   @endcode
///
///   Checking the results of to a function returning 'Expected<T>':
///   @code{.cpp}
///     if (auto E = Result.takeError()) {
///       // We must consume the error. Typically one of:
///       // - return the error to our caller
///       // - toString(), when logging
///       // - consumeError(), to silently swallow the error
///       // - handleErrors(), to distinguish error types
///       errs() << "Problem with division " << toString(std::move(E)) << "\n";
///       return;
///     }
///     // use the result
///     outs() << "The answer is " << *Result << "\n";
///   @endcode
///
///  For unit-testing a function returning an 'Expected<T>', see the
///  'EXPECT_THAT_EXPECTED' macros in llvm/Testing/Support/Error.h

template <class T>
class [[nodiscard]] Expected {
  template <class T1>
  friend class ExpectedAsOutParameter;
  template <class OtherT>
  friend class Expected;

  static constexpr bool isRef = std::is_reference<T>::value;

  using wrap = std::reference_wrapper<std::remove_reference_t<T>>;

  using error_type = std::unique_ptr<ErrorInfoBase>;

 public:
  using storage_type = std::conditional_t<isRef, wrap, T>;
  using value_type = T;

 private:
  using reference = std::remove_reference_t<T> &;
  using const_reference = const std::remove_reference_t<T> &;
  using pointer = std::remove_reference_t<T> *;
  using const_pointer = const std::remove_reference_t<T> *;

 public:
  /// Create an Expected<T> error value from the given Error.
  Expected(Error Err)
      : HasError(true)
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
        // Expected is unchecked upon construction in Debug builds.
        ,
        Unchecked(true)
#endif
  {
    assert(Err && "Cannot create Expected<T> from Error success value.");
    new (getErrorStorage()) error_type(Err.takePayload());
  }

  /// Forbid to convert from Error::success() implicitly, this avoids having
  /// Expected<T> foo() { return Error::success(); } which compiles otherwise
  /// but triggers the assertion above.
  Expected(ErrorSuccess) = delete;

  /// Create an Expected<T> success value from the given OtherT value, which
  /// must be convertible to T.
  template <typename OtherT>
  Expected(OtherT &&Val,
           std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : HasError(false)
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
        // Expected is unchecked upon construction in Debug builds.
        ,
        Unchecked(true)
#endif
  {
    new (getStorage()) storage_type(std::forward<OtherT>(Val));
  }

  /// Move construct an Expected<T> value.
  Expected(Expected &&Other) { moveConstruct(std::move(Other)); }

  /// Move construct an Expected<T> value from an Expected<OtherT>, where OtherT
  /// must be convertible to T.
  template <class OtherT>
  Expected(Expected<OtherT> &&Other,
           std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  /// Move construct an Expected<T> value from an Expected<OtherT>, where OtherT
  /// isn't convertible to T.
  template <class OtherT>
  explicit Expected(
      Expected<OtherT> &&Other,
      std::enable_if_t<!std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  /// Move-assign from another Expected<T>.
  Expected &operator=(Expected &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  /// Destroy an Expected<T>.
  ~Expected() {
    if (!HasError)
      getStorage()->~storage_type();
    else
      getErrorStorage()->~error_type();
  }

  /// Return false if there is an error.
  explicit operator bool() {
    return !HasError;
  }

  /// Returns a reference to the stored T value.
  reference get() {
    return *getStorage();
  }

  /// Returns a const reference to the stored T value.
  const_reference get() const {
    return const_cast<Expected<T> *>(this)->get();
  }

  /// Returns \a takeError() after moving the held T (if any) into \p V.
  template <class OtherT>
  Error moveInto(OtherT &Value,
                 std::enable_if_t<std::is_assignable<OtherT &, T &&>::value> * =
                     nullptr) && {
    if (*this) Value = std::move(get());
    return takeError();
  }

  /// Check that this Expected<T> is an error of type ErrT.
  template <typename ErrT>
  bool errorIsA() const {
    return HasError && (*getErrorStorage())->template isA<ErrT>();
  }

  /// Take ownership of the stored error.
  /// After calling this the Expected<T> is in an indeterminate state that can
  /// only be safely destructed. No further calls (beside the destructor) should
  /// be made on the Expected<T> value.
  Error takeError() {
    return HasError ? Error(std::move(*getErrorStorage())) : Error::success();
  }

  /// Returns a pointer to the stored T value.
  pointer operator->() {
    return toPointer(getStorage());
  }

  /// Returns a const pointer to the stored T value.
  const_pointer operator->() const {
    return toPointer(getStorage());
  }

  /// Returns a reference to the stored T value.
  reference operator*() {
    return *getStorage();
  }

  /// Returns a const reference to the stored T value.
  const_reference operator*() const {
    return *getStorage();
  }

 private:
  template <class T1>
  static bool compareThisIfSameType(const T1 &a, const T1 &b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1 &, const T2 &) {
    return false;
  }

  template <class OtherT>
  void moveConstruct(Expected<OtherT> &&Other) {
    HasError = Other.HasError;

    if (!HasError)
      new (getStorage()) storage_type(std::move(*Other.getStorage()));
    else
      new (getErrorStorage()) error_type(std::move(*Other.getErrorStorage()));
  }

  template <class OtherT>
  void moveAssign(Expected<OtherT> &&Other) {
    if (compareThisIfSameType(*this, Other)) return;

    this->~Expected();
    new (this) Expected(std::move(Other));
  }

  pointer toPointer(pointer Val) { return Val; }

  const_pointer toPointer(const_pointer Val) const { return Val; }

  pointer toPointer(wrap *Val) { return &Val->get(); }

  const_pointer toPointer(const wrap *Val) const { return &Val->get(); }

  storage_type *getStorage() {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<storage_type *>(&TStorage);
  }

  const storage_type *getStorage() const {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const storage_type *>(&TStorage);
  }

  error_type *getErrorStorage() {
    assert(HasError && "Cannot get error when a value exists!");
    return reinterpret_cast<error_type *>(&ErrorStorage);
  }

  const error_type *getErrorStorage() const {
    assert(HasError && "Cannot get error when a value exists!");
    return reinterpret_cast<const error_type *>(&ErrorStorage);
  }

  union {
    AlignedCharArrayUnion<storage_type> TStorage;
    AlignedCharArrayUnion<error_type> ErrorStorage;
  };
  bool HasError : 1;
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_EXPECTED_H_
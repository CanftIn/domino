#ifndef DOMINO_SUPPORT_ERROR_H_
#define DOMINO_SUPPORT_ERROR_H_

#include <domino/support/raw_ostream.h>
#include <domino/util/Twine.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace domino {

class ErrorSuccess;

/// Base class for error info classes. Do not extend this directly: Extend
/// the ErrorInfo template subclass instead.
class ErrorInfoBase {
 public:
  virtual ~ErrorInfoBase() = default;

  /// Print an error message to an output stream.
  virtual void log(raw_ostream &OS) const = 0;

  /// Return the error message as a string.
  virtual std::string message() const {
    std::string Msg;
    raw_string_ostream OS(Msg);
    log(OS);
    return OS.str();
  }

  /// Convert this error to a std::error_code.
  ///
  /// This is a temporary crutch to enable interaction with code still
  /// using std::error_code. It will be removed in the future.
  virtual std::error_code convertToErrorCode() const = 0;

  // Returns the class ID for this type.
  static const void *classID() { return &ID; }

  // Returns the class ID for the dynamic type of this ErrorInfoBase instance.
  virtual const void *dynamicClassID() const = 0;

  // Check whether this instance is a subclass of the class identified by
  // ClassID.
  virtual bool isA(const void *const ClassID) const {
    return ClassID == classID();
  }

  // Check whether this instance is a subclass of ErrorInfoT.
  template <typename ErrorInfoT>
  bool isA() const {
    return isA(ErrorInfoT::classID());
  }

 private:
  virtual void anchor();

  static char ID;
};

/// Lightweight error class with error context and mandatory checking.
///
/// Instances of this class wrap a ErrorInfoBase pointer. Failure states
/// are represented by setting the pointer to a ErrorInfoBase subclass
/// instance containing information describing the failure. Success is
/// represented by a null pointer value.
///
/// Instances of Error also contains a 'Checked' flag, which must be set
/// before the destructor is called, otherwise the destructor will trigger a
/// runtime error. This enforces at runtime the requirement that all Error
/// instances be checked or returned to the caller.
///
/// There are two ways to set the checked flag, depending on what state the
/// Error instance is in. For Error instances indicating success, it
/// is sufficient to invoke the boolean conversion operator. E.g.:
///
///   @code{.cpp}
///   Error foo(<...>);
///
///   if (auto E = foo(<...>))
///     return E; // <- Return E if it is in the error state.
///   // We have verified that E was in the success state. It can now be safely
///   // destroyed.
///   @endcode
///
/// A success value *can not* be dropped. For example, just calling 'foo(<...>)'
/// without testing the return value will raise a runtime error, even if foo
/// returns success.
///
/// For Error instances representing failure, you must use either the
/// handleErrors or handleAllErrors function with a typed handler. E.g.:
///
///   @code{.cpp}
///   class MyErrorInfo : public ErrorInfo<MyErrorInfo> {
///     // Custom error info.
///   };
///
///   Error foo(<...>) { return make_error<MyErrorInfo>(...); }
///
///   auto E = foo(<...>); // <- foo returns failure with MyErrorInfo.
///   auto NewE =
///     handleErrors(E,
///       [](const MyErrorInfo &M) {
///         // Deal with the error.
///       },
///       [](std::unique_ptr<OtherError> M) -> Error {
///         if (canHandle(*M)) {
///           // handle error.
///           return Error::success();
///         }
///         // Couldn't handle this error instance. Pass it up the stack.
///         return Error(std::move(M));
///       );
///   // Note - we must check or return NewE in case any of the handlers
///   // returned a new error.
///   @endcode
///
/// The handleAllErrors function is identical to handleErrors, except
/// that it has a void return type, and requires all errors to be handled and
/// no new errors be returned. It prevents errors (assuming they can all be
/// handled) from having to be bubbled all the way to the top-level.
///
/// *All* Error instances must be checked before destruction, even if
/// they're moved-assigned or constructed from Success values that have already
/// been checked. This enforces checking through all levels of the call stack.
class [[nodiscard]] Error {
  // ErrorList needs to be able to yank ErrorInfoBase pointers out of Errors
  // to add to the error list. It can't rely on handleErrors for this, since
  // handleErrors does not support ErrorList handlers.
  friend class ErrorList;

  // handleErrors needs to be able to set the Checked flag.
  template <typename... HandlerTs>
  friend Error handleErrors(Error E, HandlerTs &&...Handlers);

  // Expected<T> needs to be able to steal the payload when constructed from an
  // error.
  template <typename T>
  friend class Expected;

 protected:
  /// Create a success value. Prefer using 'Error::success()' for readability
  Error() {
    setPtr(nullptr);
    setChecked(false);
  }

 public:
  /// Create a success value.
  static ErrorSuccess success();

  // Errors are not copy-constructable.
  Error(const Error &Other) = delete;

  /// Move-construct an error value. The newly constructed error is considered
  /// unchecked, even if the source error had been checked. The original error
  /// becomes a checked Success value, regardless of its original state.
  Error(Error &&Other) {
    setChecked(true);
    *this = std::move(Other);
  }

  /// Create an error value. Prefer using the 'make_error' function, but
  /// this constructor can be useful when "re-throwing" errors from handlers.
  Error(std::unique_ptr<ErrorInfoBase> Payload) {
    setPtr(Payload.release());
    setChecked(false);
  }

  // Errors are not copy-assignable.
  Error &operator=(const Error &Other) = delete;

  /// Move-assign an error value. The current error must represent success, you
  /// you cannot overwrite an unhandled error. The current error is then
  /// considered unchecked. The source error becomes a checked success value,
  /// regardless of its original state.
  Error &operator=(Error &&Other) {
    // Don't allow overwriting of unchecked values.
    assertIsChecked();
    setPtr(Other.getPtr());

    // This Error is unchecked, even if the source error was checked.
    setChecked(false);

    // Null out Other's payload and set its checked bit.
    Other.setPtr(nullptr);
    Other.setChecked(true);

    return *this;
  }

  /// Destroy a Error. Fails with a call to abort() if the error is
  /// unchecked.
  ~Error() {
    assertIsChecked();
    delete getPtr();
  }

  /// Bool conversion. Returns true if this Error is in a failure state,
  /// and false if it is in an accept state. If the error is in a Success state
  /// it will be considered checked.
  explicit operator bool() {
    setChecked(getPtr() == nullptr);
    return getPtr() != nullptr;
  }

  /// Check whether one error is a subclass of another.
  template <typename ErrT>
  bool isA() const {
    return getPtr() && getPtr()->isA(ErrT::classID());
  }

  /// Returns the dynamic class id of this error, or null if this is a success
  /// value.
  const void *dynamicClassID() const {
    if (!getPtr()) return nullptr;
    return getPtr()->dynamicClassID();
  }

 private:
  void assertIsChecked() {}

  ErrorInfoBase *getPtr() const { return Payload; }

  void setPtr(ErrorInfoBase *EI) { Payload = EI; }

  bool getChecked() const { return true; }

  void setChecked(bool V) {}

  std::unique_ptr<ErrorInfoBase> takePayload() {
    std::unique_ptr<ErrorInfoBase> Tmp(getPtr());
    setPtr(nullptr);
    setChecked(true);
    return Tmp;
  }

  friend raw_ostream &operator<<(raw_ostream &OS, const Error &E) {
    if (auto *P = E.getPtr())
      P->log(OS);
    else
      OS << "success";
    return OS;
  }

  ErrorInfoBase *Payload = nullptr;
};

/// Subclass of Error for the sole purpose of identifying the success path in
/// the type system. This allows to catch invalid conversion to Expected<T> at
/// compile time.
class ErrorSuccess final : public Error {};

inline ErrorSuccess Error::success() { return ErrorSuccess(); }

/// Make a Error instance representing failure using the given error info
/// type.
template <typename ErrT, typename... ArgTs>
Error make_error(ArgTs &&...Args) {
  return Error(std::make_unique<ErrT>(std::forward<ArgTs>(Args)...));
}

/// Base class for user error types. Users should declare their error types
/// like:
///
/// class MyError : public ErrorInfo<MyError> {
///   ....
/// };
///
/// This class provides an implementation of the ErrorInfoBase::kind
/// method, which is used by the Error RTTI system.
template <typename ThisErrT, typename ParentErrT = ErrorInfoBase>
class ErrorInfo : public ParentErrT {
 public:
  using ParentErrT::ParentErrT;  // inherit constructors

  static const void *classID() { return &ThisErrT::ID; }

  const void *dynamicClassID() const override { return &ThisErrT::ID; }

  bool isA(const void *const ClassID) const override {
    return ClassID == classID() || ParentErrT::isA(ClassID);
  }
};

/// Special ErrorInfo subclass representing a list of ErrorInfos.
/// Instances of this class are constructed by joinError.
class ErrorList final : public ErrorInfo<ErrorList> {
  // handleErrors needs to be able to iterate the payload list of an
  // ErrorList.
  template <typename... HandlerTs>
  friend Error handleErrors(Error E, HandlerTs &&...Handlers);

  // joinErrors is implemented in terms of join.
  friend Error joinErrors(Error, Error);

 public:
  void log(raw_ostream &OS) const override {
    OS << "Multiple errors:\n";
    for (const auto &ErrPayload : Payloads) {
      ErrPayload->log(OS);
      OS << "\n";
    }
  }

  std::error_code convertToErrorCode() const override;

  // Used by ErrorInfo::classID.
  static char ID;

 private:
  ErrorList(std::unique_ptr<ErrorInfoBase> Payload1,
            std::unique_ptr<ErrorInfoBase> Payload2) {
    assert(!Payload1->isA<ErrorList>() && !Payload2->isA<ErrorList>() &&
           "ErrorList constructor payloads should be singleton errors");
    Payloads.push_back(std::move(Payload1));
    Payloads.push_back(std::move(Payload2));
  }

  static Error join(Error E1, Error E2) {
    if (!E1) return E2;
    if (!E2) return E1;
    if (E1.isA<ErrorList>()) {
      auto &E1List = static_cast<ErrorList &>(*E1.getPtr());
      if (E2.isA<ErrorList>()) {
        auto E2Payload = E2.takePayload();
        auto &E2List = static_cast<ErrorList &>(*E2Payload);
        for (auto &Payload : E2List.Payloads)
          E1List.Payloads.push_back(std::move(Payload));
      } else
        E1List.Payloads.push_back(E2.takePayload());

      return E1;
    }
    if (E2.isA<ErrorList>()) {
      auto &E2List = static_cast<ErrorList &>(*E2.getPtr());
      E2List.Payloads.insert(E2List.Payloads.begin(), E1.takePayload());
      return E2;
    }
    return Error(std::unique_ptr<ErrorList>(
        new ErrorList(E1.takePayload(), E2.takePayload())));
  }

  std::vector<std::unique_ptr<ErrorInfoBase>> Payloads;
};

/// Concatenate errors. The resulting Error is unchecked, and contains the
/// ErrorInfo(s), if any, contained in E1, followed by the
/// ErrorInfo(s), if any, contained in E2.
inline Error joinErrors(Error E1, Error E2) {
  return ErrorList::join(std::move(E1), std::move(E2));
}

class ECError : public ErrorInfo<ECError> {
  friend Error errorCodeToError(std::error_code);

  void anchor() override;

 public:
  void setErrorCode(std::error_code EC) { this->EC = EC; }
  std::error_code convertToErrorCode() const override { return EC; }
  void log(raw_ostream &OS) const override { OS << EC.message(); }

  // Used by ErrorInfo::classID.
  static char ID;

 protected:
  ECError() = default;
  ECError(std::error_code EC) : EC(EC) {}

  std::error_code EC;
};

Error errorCodeToError(std::error_code EC);

class StringError : public ErrorInfo<StringError> {
 public:
  static char ID;

  // Prints EC + S and converts to EC
  StringError(std::error_code EC, const Twine &S = Twine());

  // Prints S and converts to EC
  StringError(const Twine &S, std::error_code EC);

  void log(raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return Msg; }

 private:
  std::string Msg;
  std::error_code EC;
  const bool PrintMsgOnly = false;
};

/// Create formatted StringError object.
template <typename... Ts>
inline Error createStringError(std::error_code EC, char const *Fmt,
                               const Ts &...Vals) {
  std::string Buffer;
  raw_string_ostream Stream(Buffer);
  Stream << format(Fmt, Vals...);
  return make_error<StringError>(Stream.str(), EC);
}

Error createStringError(std::error_code EC, char const *Msg);

inline Error createStringError(std::error_code EC, const Twine &S) {
  return createStringError(EC, S.str().c_str());
}

template <typename... Ts>
inline Error createStringError(std::errc EC, char const *Fmt,
                               const Ts &...Vals) {
  return createStringError(std::make_error_code(EC), Fmt, Vals...);
}

/// This class wraps a filename and another Error.
///
/// In some cases, an error needs to live along a 'source' name, in order to
/// show more detailed information to the user.
class FileError final : public ErrorInfo<FileError> {
  friend Error createFileError(const Twine &, Error);
  friend Error createFileError(const Twine &, size_t, Error);

 public:
  void log(raw_ostream &OS) const override {
    assert(Err && "Trying to log after takeError().");
    OS << "'" << FileName << "': ";
    if (Line) OS << "line " << *Line << ": ";
    Err->log(OS);
  }

  std::string messageWithoutFileInfo() const {
    std::string Msg;
    raw_string_ostream OS(Msg);
    Err->log(OS);
    return OS.str();
  }

  StringRef getFileName() const { return FileName; }

  Error takeError() { return Error(std::move(Err)); }

  std::error_code convertToErrorCode() const override;

  // Used by ErrorInfo::classID.
  static char ID;

 private:
  FileError(const Twine &F, std::optional<size_t> LineNum,
            std::unique_ptr<ErrorInfoBase> E) {
    assert(E && "Cannot create FileError from Error success value.");
    FileName = F.str();
    Err = std::move(E);
    Line = std::move(LineNum);
  }

  static Error build(const Twine &F, std::optional<size_t> Line, Error E) {
    std::unique_ptr<ErrorInfoBase> Payload;
    // handleAllErrors(std::move(E),
    //                 [&](std::unique_ptr<ErrorInfoBase> EIB) -> Error {
    //                   Payload = std::move(EIB);
    //                   return Error::success();
    //                 });
    return Error(
        std::unique_ptr<FileError>(new FileError(F, Line, std::move(Payload))));
  }

  std::string FileName;
  std::optional<size_t> Line;
  std::unique_ptr<ErrorInfoBase> Err;
};

inline void consumeError(Error Err) {
  // TODO
}

inline Error createFileError(const Twine &F, Error E) {
  return FileError::build(F, std::optional<size_t>(), std::move(E));
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_ERROR_H_
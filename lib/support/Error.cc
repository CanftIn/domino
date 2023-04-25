#include <domino/util/Logging.h>
#include <domino/support/Error.h>

namespace {

  enum class ErrorErrorCode : int {
    MultipleErrors = 1,
    FileError,
    InconvertibleError
  };

  // FIXME: This class is only here to support the transition to llvm::Error. It
  // will be removed once this transition is complete. Clients should prefer to
  // deal with the Error value directly, rather than converting to error_code.
  class ErrorErrorCategory : public std::error_category {
  public:
    const char *name() const noexcept override { return "Error"; }

    std::string message(int condition) const override {
      switch (static_cast<ErrorErrorCode>(condition)) {
      case ErrorErrorCode::MultipleErrors:
        return "Multiple errors";
      case ErrorErrorCode::InconvertibleError:
        return "Inconvertible error value. An error has occurred that could "
               "not be converted to a known std::error_code. Please file a "
               "bug.";
      case ErrorErrorCode::FileError:
          return "A file error occurred.";
      }
      DOMINO_ERROR_ABORT("Unhandled error code");
    }
  };

}

ErrorErrorCategory &getErrorErrorCat() {
  static ErrorErrorCategory ErrorErrorCat;
  return ErrorErrorCat;
}


namespace domino {

void ErrorInfoBase::anchor() {}
char ErrorInfoBase::ID = 0;
char ErrorList::ID = 0;
void ECError::anchor() {}
char ECError::ID = 0;
char StringError::ID = 0;
char FileError::ID = 0;

void logAllUnhandledErrors(Error E, raw_ostream &OS, Twine ErrorBanner) {
  if (!E)
    return;
  OS << ErrorBanner;
  //handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
  //  EI.log(OS);
  //  OS << "\n";
  //});
}


std::error_code ErrorList::convertToErrorCode() const {
  return std::error_code(static_cast<int>(ErrorErrorCode::MultipleErrors),
                         getErrorErrorCat());
}

std::error_code inconvertibleErrorCode() {
  return std::error_code(static_cast<int>(ErrorErrorCode::InconvertibleError),
                         getErrorErrorCat());
}

std::error_code FileError::convertToErrorCode() const {
  std::error_code NestedEC = Err->convertToErrorCode();
  if (NestedEC == inconvertibleErrorCode())
    return std::error_code(static_cast<int>(ErrorErrorCode::FileError),
                           getErrorErrorCat());
  return NestedEC;
}

Error errorCodeToError(std::error_code EC) {
  if (!EC)
    return Error::success();
  return Error(std::make_unique<ECError>(ECError(EC)));
}

std::error_code errorToErrorCode(Error Err) {
  std::error_code EC;
  //handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
  //  EC = EI.convertToErrorCode();
  //});
  //if (EC == inconvertibleErrorCode())
  //  report_fatal_error(Twine(EC.message()));
  return EC;
}


StringError::StringError(std::error_code EC, const Twine &S)
    : Msg(S.str()), EC(EC) {}

StringError::StringError(const Twine &S, std::error_code EC)
    : Msg(S.str()), EC(EC), PrintMsgOnly(true) {}

void StringError::log(raw_ostream &OS) const {
  if (PrintMsgOnly) {
    OS << Msg;
  } else {
    OS << EC.message();
    if (!Msg.empty())
      OS << (" " + Msg);
  }
}

std::error_code StringError::convertToErrorCode() const {
  return EC;
}

Error createStringError(std::error_code EC, char const *Msg) {
  return make_error<StringError>(Msg, EC);
}

void report_fatal_error(Error Err, bool GenCrashDiag) {
  assert(Err && "report_fatal_error called with success value");
  std::string ErrMsg;
  {
    raw_string_ostream ErrStream(ErrMsg);
    //logAllUnhandledErrors(std::move(Err), ErrStream);
  }
  //report_fatal_error(Twine(ErrMsg));
}

}  // namespace domino
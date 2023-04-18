#ifndef DOMINO_RPC_UTIL_STRINGSWITCH_H_
#define DOMINO_RPC_UTIL_STIRNGSWITCH_H_

#include <domino/util/StringRef.h>

#include <cassert>
#include <cstring>
#include <optional>
#include <type_traits>

namespace domino {

template <typename T, typename R = T>
class StringSwitch {
  const StringRef Str;

  std::optional<T> Result;

 public:
  explicit StringSwitch(StringRef Str) : Str(Str), Result() {}

  StringSwitch(const StringSwitch&) = delete;

  void operator=(const StringSwitch&) = delete;
  void operator=(const StringSwitch&&) = delete;

  StringSwitch(StringSwitch&& Other)
      : Str(Other.Str), Result(std::move(Other.Result)) {}

  ~StringSwitch() = default;

  StringSwitch& Case(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str == S) Result = std::move(Value);
    return *this;
  }

  StringSwitch& EndsWith(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str.ends_with(S)) Result = std::move(Value);
    return *this;
  }

  StringSwitch& StartsWith(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str.starts_with(S)) Result = std::move(Value);
    return *this;
  }

  [[nodiscard]] std::enable_if_t<std::is_convertible_v<T, R>, R> Default(
      T Value) {
    if (Result) return std::move(*Result);
    return Value;
  }

  [[nodiscard]] operator R() {
    assert(Result && "Fell off the end of a string-switch");
    return std::move(*Result);
  }
};

}  // namespace domino

#endif  // DOMINO_RPC_UTIL_STRINGSWITCH_H_
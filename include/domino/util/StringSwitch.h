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

  StringSwitch(const StringSwitch &) = delete;

  void operator=(const StringSwitch &) = delete;
  void operator=(const StringSwitch &&) = delete;

  StringSwitch(StringSwitch &&Other)
      : Str(Other.Str), Result(std::move(Other.Result)) {}

  ~StringSwitch() = default;

  StringSwitch &Case(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str == S) Result = std::move(Value);
    return *this;
  }

  StringSwitch &EndsWith(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str.ends_with(S)) Result = std::move(Value);
    return *this;
  }

  StringSwitch &StartsWith(StringLiteral S, T Value) {
    if (Result.has_value()) return *this;
    if (Str.starts_with(S)) Result = std::move(Value);
    return *this;
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, T Value) {
    return Case(S0, Value).Case(S1, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      T Value) {
    return Case(S0, Value).Cases(S1, S2, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, StringLiteral S8,
                      T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, S8, Value);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, StringLiteral S8,
                      StringLiteral S9, T Value) {
    return Case(S0, Value).Cases(S1, S2, S3, S4, S5, S6, S7, S8, S9, Value);
  }

  // Case-insensitive case matchers.
  StringSwitch &CaseLower(StringLiteral S, T Value) {
    if (!Result && Str.equals_insensitive(S)) Result = std::move(Value);

    return *this;
  }

  StringSwitch &EndsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.ends_with_insensitive(S)) Result = Value;

    return *this;
  }

  StringSwitch &StartsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.starts_with_insensitive(S)) Result = std::move(Value);

    return *this;
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, T Value) {
    return CaseLower(S0, Value).CaseLower(S1, Value);
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           T Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, Value);
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           StringLiteral S3, T Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, S3, Value);
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           StringLiteral S3, StringLiteral S4, T Value) {
    return CaseLower(S0, Value).CasesLower(S1, S2, S3, S4, Value);
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
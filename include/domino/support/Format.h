#ifndef DOMINO_SUPPORT_FORMAT_H_
#define DOMINO_SUPPORT_FORMAT_H_

#include <domino/util/ArrayRef.h>
#include <domino/util/STLExtras.h>
#include <domino/util/StringRef.h>

#include <cassert>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace domino {

class format_object_base {
 protected:
  format_object_base(const format_object_base&) = default;

  ~format_object_base() = default;

  virtual void home();

  virtual int snprint(char* Buffer, unsigned BufferSize) const = 0;

  const char* Fmt;

 public:
  format_object_base(const char* fmt) : Fmt(fmt) {}

  unsigned print(char* Buffer, unsigned BufferSize) const {
    assert(BufferSize && "Invalid buffer size!");
    int N = snprint(Buffer, BufferSize);
    if (N < 0) return BufferSize * 2;
    if (static_cast<unsigned>(N) >= BufferSize) return N + 1;
    return N;
  }
};

template <typename... Args>
struct validate_format_parameters;
template <typename Arg, typename... Args>
struct validate_format_parameters<Arg, Args...> {
  static_assert(std::is_scalar_v<Arg>,
                "format can't be used with non fundamental / non pointer type");

  validate_format_parameters() { validate_format_parameters<Args...>(); }
};
template <>
struct validate_format_parameters<> {};

template <typename... Ts>
class format_object final : public format_object_base {
  std::tuple<Ts...> Vals;

  template <size_t... Is>
  int snprint_tuple(char* Buffer, unsigned BufferSize,
                    std::index_sequence<Is...>) const {
    return std::snprintf(Buffer, BufferSize, Fmt, std::get<Is>(Vals)...);
  };

 public:
  format_object(const char* fmt, const Ts&... vals)
      : format_object_base(fmt), Vals(std::forward<Ts>(vals)...) {
    validate_format_parameters<Ts...>();
  }

  int snprint(char* Buffer, unsigned BufferSize) const override {
    return snprint_tuple(Buffer, BufferSize,
                         std::make_index_sequence<sizeof...(Ts)>());
  }
};

template <typename... Ts>
inline format_object<Ts...> format(const char* Fmt, const Ts&... Vals) {
  return format_object<Ts...>(Fmt, Vals...);
}

class FormattedString {
 public:
  enum Justification { JustifyNone, JustifyLeft, JustifyRight, JustifyCenter };
  FormattedString(StringRef S, unsigned W, Justification J)
      : Str(S), Width(W), Justify(J) {}

 private:
  StringRef Str;
  unsigned Width;
  Justification Justify;
  friend class raw_ostream;
};

inline FormattedString left_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, FormattedString::JustifyLeft);
}

inline FormattedString right_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, FormattedString::JustifyRight);
}

inline FormattedString center_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, FormattedString::JustifyCenter);
}

class FormattedNumber {
  uint64_t HexValue;
  int64_t DecValue;
  unsigned Width;
  bool Hex;
  bool Upper;
  bool HexPrefix;
  friend class raw_ostream;

 public:
  FormattedNumber(uint64_t HV, int64_t DV, unsigned W, bool H, bool U, bool HP)
      : HexValue(HV), DecValue(DV), Width(W), Hex(H), Upper(U), HexPrefix(HP) {}
};

inline FormattedNumber format_hex(uint64_t HexValue, unsigned Width = 0,
                                  bool Upper = false, bool HexPrefix = true) {
  assert(Width <= 18 && "Hex width must be <= 18");
  return FormattedNumber(HexValue, 0, Width, true, Upper, HexPrefix);
}

inline FormattedNumber format_hex_no_prefix(uint64_t N, unsigned Width,
                                            bool Upper = false) {
  assert(Width <= 16 && "Hex width must be <= 16");
  return FormattedNumber(N, 0, Width, true, Upper, false);
}

inline FormattedNumber format_decimal(int64_t DecValue, unsigned Width) {
  return FormattedNumber(0, DecValue, Width, false, false, false);
}

class FormattedBytes {
  ArrayRef<uint8_t> Bytes;
  std::optional<uint64_t> FirstByteOffset;
  uint32_t IndentLevel;
  uint32_t NumPerLine;
  uint8_t ByteGroupSize;
  bool Upper;
  bool ASCII;
  friend class raw_ostream;

 public:
  FormattedBytes(ArrayRef<uint8_t> B, uint32_t IndentLevel,
                 std::optional<uint64_t> O, uint32_t NumPerLine,
                 uint8_t ByteGroupSize, bool Upper, bool A)
      : Bytes(B),
        FirstByteOffset(O),
        IndentLevel(IndentLevel),
        NumPerLine(NumPerLine),
        ByteGroupSize(ByteGroupSize),
        Upper(Upper),
        ASCII(A) {
    if (ByteGroupSize > NumPerLine) ByteGroupSize = NumPerLine;
  }
};

inline FormattedBytes format_bytes(
    ArrayRef<uint8_t> Bytes,
    std::optional<uint64_t> FirstByteOffset = std::nullopt,
    uint32_t NumPerLine = 16, uint8_t ByteGroupSize = 4,
    uint32_t IndentLevel = 0, bool Upper = false) {
  return FormattedBytes(Bytes, IndentLevel, FirstByteOffset, NumPerLine,
                        ByteGroupSize, Upper, false);
}

inline FormattedBytes format_bytes_with_ascii(
    ArrayRef<uint8_t> Bytes,
    std::optional<uint64_t> FirstByteOffset = std::nullopt,
    uint32_t NumPerLine = 16, uint8_t ByteGroupSize = 4,
    uint32_t IndentLevel = 0, bool Upper = false) {
  return FormattedBytes(Bytes, IndentLevel, FirstByteOffset, NumPerLine,
                        ByteGroupSize, Upper, true);
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_FORMAT_H_
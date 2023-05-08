#include <domino/support/Format.h>
#include <domino/support/NativeFormatting.h>
#include <domino/support/raw_ostream.h>
#include <domino/util/ArrayRef.h>
#include <domino/util/Logging.h>
#include <domino/util/SmallString.h>
#include <domino/util/StringExtras.h>
#include <domino/util/bit.h>

#include <cmath>

using namespace domino;

template <typename T, std::size_t N>
static int format_to_buffer(T Value, char (&Buffer)[N]) {
  char *EndPtr = std::end(Buffer);
  char *CurPtr = EndPtr;

  do {
    *--CurPtr = '0' + char(Value % 10);
    Value /= 10;
  } while (Value);
  return EndPtr - CurPtr;
}

static void writeWithCommas(raw_ostream &S, ArrayRef<char> Buffer) {
  assert(!Buffer.empty());

  ArrayRef<char> ThisGroup;
  int InitialDigits = ((Buffer.size() - 1) % 3) + 1;
  ThisGroup = Buffer.take_front(InitialDigits);
  S.write(ThisGroup.data(), ThisGroup.size());

  Buffer = Buffer.drop_front(InitialDigits);
  assert(Buffer.size() % 3 == 0);
  while (!Buffer.empty()) {
    S << ',';
    ThisGroup = Buffer.take_front(3);
    S.write(ThisGroup.data(), 3);
    Buffer = Buffer.drop_front(3);
  }
}

template <typename T>
static void write_unsigned_impl(raw_ostream &S, T N, size_t MinDigits,
                                IntegerStyle Style, bool IsNegative) {
  static_assert(std::is_unsigned_v<T>, "Value is not unsigned!");

  char NumberBuffer[128];
  std::memset(NumberBuffer, '0', sizeof(NumberBuffer));

  size_t Len = 0;
  Len = format_to_buffer(N, NumberBuffer);

  if (IsNegative) S << '-';

  if (Len < MinDigits && Style != IntegerStyle::Number) {
    for (size_t I = Len; I < MinDigits; ++I) S << '0';
  }

  if (Style == IntegerStyle::Number) {
    writeWithCommas(S, ArrayRef<char>(std::end(NumberBuffer) - Len, Len));
  } else {
    S.write(std::end(NumberBuffer) - Len, Len);
  }
}

template <typename T>
static void write_unsigned(raw_ostream &S, T N, size_t MinDigits,
                           IntegerStyle Style, bool IsNegative = false) {
  // Output using 32-bit div/mod if possible.
  if (N == static_cast<uint32_t>(N))
    write_unsigned_impl(S, static_cast<uint32_t>(N), MinDigits, Style,
                        IsNegative);
  else
    write_unsigned_impl(S, N, MinDigits, Style, IsNegative);
}

template <typename T>
static void write_signed(raw_ostream &S, T N, size_t MinDigits,
                         IntegerStyle Style) {
  static_assert(std::is_signed_v<T>, "Value is not signed!");

  using UnsignedT = std::make_unsigned_t<T>;

  if (N >= 0) {
    write_unsigned(S, static_cast<UnsignedT>(N), MinDigits, Style);
    return;
  }

  UnsignedT UN = -(UnsignedT)N;
  write_unsigned(S, UN, MinDigits, Style, true);
}

void domino::write_integer(raw_ostream &S, unsigned int N, size_t MinDigits,
                           IntegerStyle Style) {
  write_unsigned(S, N, MinDigits, Style);
}

void domino::write_integer(raw_ostream &S, int N, size_t MinDigits,
                           IntegerStyle Style) {
  write_signed(S, N, MinDigits, Style);
}

void domino::write_integer(raw_ostream &S, unsigned long N, size_t MinDigits,
                           IntegerStyle Style) {
  write_unsigned(S, N, MinDigits, Style);
}

void domino::write_integer(raw_ostream &S, long N, size_t MinDigits,
                           IntegerStyle Style) {
  write_signed(S, N, MinDigits, Style);
}

void domino::write_integer(raw_ostream &S, unsigned long long N,
                           size_t MinDigits, IntegerStyle Style) {
  write_unsigned(S, N, MinDigits, Style);
}

void domino::write_integer(raw_ostream &S, long long N, size_t MinDigits,
                           IntegerStyle Style) {
  write_signed(S, N, MinDigits, Style);
}

inline char hexdigit(unsigned X, bool LowerCase = false) {
  assert(X < 16);
  static const char LUT[] = "0123456789ABCDEF";
  const uint8_t Offset = LowerCase ? 32 : 0;
  return LUT[X] | Offset;
}

template <typename T>
[[nodiscard]] int bit_width(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return std::numeric_limits<T>::digits - domino::countl_zero(Value);
}

void domino::write_hex(raw_ostream &S, uint64_t N, HexPrintStyle Style,
                       std::optional<size_t> Width) {
  const size_t kMaxWidth = 128u;

  size_t W = std::min(kMaxWidth, Width.value_or(0u));

  unsigned Nibbles = (domino::bit_width(N) + 3) / 4;
  bool Prefix = (Style == HexPrintStyle::PrefixLower ||
                 Style == HexPrintStyle::PrefixUpper);
  bool Upper =
      (Style == HexPrintStyle::Upper || Style == HexPrintStyle::PrefixUpper);
  unsigned PrefixChars = Prefix ? 2 : 0;
  unsigned NumChars =
      std::max(static_cast<unsigned>(W), std::max(1u, Nibbles) + PrefixChars);

  char NumberBuffer[kMaxWidth];
  ::memset(NumberBuffer, '0', std::size(NumberBuffer));
  if (Prefix) NumberBuffer[1] = 'x';
  char *EndPtr = NumberBuffer + NumChars;
  char *CurPtr = EndPtr;
  while (N) {
    unsigned char x = static_cast<unsigned char>(N) % 16;
    *--CurPtr = hexdigit(x, !Upper);
    N /= 16;
  }

  S.write(NumberBuffer, NumChars);
}

void domino::write_double(raw_ostream &S, double N, FloatStyle Style,
                          std::optional<size_t> Precision) {
  size_t Prec = Precision.value_or(getDefaultPrecision(Style));

  if (std::isnan(N)) {
    S << "nan";
    return;
  } else if (std::isinf(N)) {
    S << (std::signbit(N) ? "-INF" : "INF");
    return;
  }

  char Letter;
  if (Style == FloatStyle::Exponent)
    Letter = 'e';
  else if (Style == FloatStyle::ExponentUpper)
    Letter = 'E';
  else
    Letter = 'f';

  SmallString<8> Spec;
  domino::raw_svector_ostream Out(Spec);
  Out << "%." << Prec << Letter;

  if (Style == FloatStyle::Exponent || Style == FloatStyle::ExponentUpper) {
  }

  if (Style == FloatStyle::Percent) N *= 100.0;

  char Buf[32];
  format(Spec.c_str(), N).snprint(Buf, sizeof(Buf));
  S << Buf;
  if (Style == FloatStyle::Percent) S << '%';
}

bool domino::isPrefixedHexStyle(HexPrintStyle S) {
  return (S == HexPrintStyle::PrefixLower || S == HexPrintStyle::PrefixUpper);
}

size_t domino::getDefaultPrecision(FloatStyle Style) {
  switch (Style) {
    case FloatStyle::Exponent:
    case FloatStyle::ExponentUpper:
      return 6;  // Number of decimal places.
    case FloatStyle::Fixed:
    case FloatStyle::Percent:
      return 2;  // Number of decimal places.
  }
  DOMINO_ERROR_ABORT("Unknown FloatStyle enum");
}

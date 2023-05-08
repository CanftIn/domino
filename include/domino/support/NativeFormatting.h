#ifndef DOMINO_SUPPORT_NATIVEFORMATTING_H_
#define DOMINO_SUPPORT_NATIVEFORMATTING_H_

#include <cstdint>
#include <optional>

namespace domino {

class raw_ostream;
enum class FloatStyle { Exponent, ExponentUpper, Fixed, Percent };
enum class IntegerStyle {
  Integer,
  Number,
};
enum class HexPrintStyle { Upper, Lower, PrefixUpper, PrefixLower };

size_t getDefaultPrecision(FloatStyle Style);

bool isPrefixedHexStyle(HexPrintStyle S);

void write_integer(raw_ostream &S, unsigned int N, size_t MinDigits,
                   IntegerStyle Style);
void write_integer(raw_ostream &S, int N, size_t MinDigits, IntegerStyle Style);
void write_integer(raw_ostream &S, unsigned long N, size_t MinDigits,
                   IntegerStyle Style);
void write_integer(raw_ostream &S, long N, size_t MinDigits,
                   IntegerStyle Style);
void write_integer(raw_ostream &S, unsigned long long N, size_t MinDigits,
                   IntegerStyle Style);
void write_integer(raw_ostream &S, long long N, size_t MinDigits,
                   IntegerStyle Style);

void write_hex(raw_ostream &S, uint64_t N, HexPrintStyle Style,
               std::optional<size_t> Width = std::nullopt);
void write_double(raw_ostream &S, double D, FloatStyle Style,
                  std::optional<size_t> Precision = std::nullopt);

}  // namespace domino

#endif  // DOMINO_SUPPORT_NATIVEFORMATTING_H_

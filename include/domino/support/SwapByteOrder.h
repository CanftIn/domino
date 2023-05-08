#ifndef DOMINO_SUPPORT_SWAPBYTEORDER_H_
#define DOMINO_SUPPORT_SWAPBYTEORDER_H_

#include <domino/util/bit.h>
#include <endian.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace domino {
/// ByteSwap_16 - This function returns a byte-swapped representation of
/// the 16-bit argument.
inline uint16_t ByteSwap_16(uint16_t value) { return domino::byteswap(value); }

/// This function returns a byte-swapped representation of the 32-bit argument.
inline uint32_t ByteSwap_32(uint32_t value) { return domino::byteswap(value); }

/// This function returns a byte-swapped representation of the 64-bit argument.
inline uint64_t ByteSwap_64(uint64_t value) { return domino::byteswap(value); }

namespace sys {

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
constexpr bool IsBigEndianHost = true;
#else
constexpr bool IsBigEndianHost = false;
#endif

static const bool IsLittleEndianHost = !IsBigEndianHost;

inline unsigned char getSwappedBytes(unsigned char C) {
  return domino::byteswap(C);
}
inline signed char getSwappedBytes(signed char C) {
  return domino::byteswap(C);
}
inline char getSwappedBytes(char C) { return domino::byteswap(C); }

inline unsigned short getSwappedBytes(unsigned short C) {
  return domino::byteswap(C);
}
inline signed short getSwappedBytes(signed short C) {
  return domino::byteswap(C);
}

inline unsigned int getSwappedBytes(unsigned int C) {
  return domino::byteswap(C);
}
inline signed int getSwappedBytes(signed int C) { return domino::byteswap(C); }

inline unsigned long getSwappedBytes(unsigned long C) {
  return domino::byteswap(C);
}
inline signed long getSwappedBytes(signed long C) {
  return domino::byteswap(C);
}

inline unsigned long long getSwappedBytes(unsigned long long C) {
  return domino::byteswap(C);
}
inline signed long long getSwappedBytes(signed long long C) {
  return domino::byteswap(C);
}

inline float getSwappedBytes(float C) {
  union {
    uint32_t i;
    float f;
  } in, out;
  in.f = C;
  out.i = domino::byteswap(in.i);
  return out.f;
}

inline double getSwappedBytes(double C) {
  union {
    uint64_t i;
    double d;
  } in, out;
  in.d = C;
  out.i = domino::byteswap(in.i);
  return out.d;
}

template <typename T>
inline std::enable_if_t<std::is_enum<T>::value, T> getSwappedBytes(T C) {
  return static_cast<T>(
      domino::byteswap(static_cast<std::underlying_type_t<T>>(C)));
}

template <typename T>
inline void swapByteOrder(T &Value) {
  Value = getSwappedBytes(Value);
}

}  // namespace sys
}  // namespace domino

#endif  // DOMINO_SUPPORT_SWAPBYTEORDER_H_
#ifndef DOMINO_UTIL_BIT_H_
#define DOMINO_UTIL_BIT_H_

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace domino {

template <
    typename To, typename From,
    typename = std::enable_if_t<sizeof(To) == sizeof(From)>,
    typename = std::enable_if_t<std::is_trivially_constructible<To>::value>,
    typename = std::enable_if_t<std::is_trivially_copyable<To>::value>,
    typename = std::enable_if_t<std::is_trivially_copyable<From>::value>>
[[nodiscard]] inline To bit_cast(const From &from) noexcept {
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] constexpr T byteswap(T V) noexcept {
  if constexpr (sizeof(T) == 1) {
    return V;
  } else if constexpr (sizeof(T) == 2) {
    uint16_t UV = V;
    uint16_t Hi = UV << 8;
    uint16_t Lo = UV >> 8;
    return Hi | Lo;
  } else if constexpr (sizeof(T) == 4) {
    uint32_t UV = V;
    uint32_t Byte0 = UV & 0x000000FF;
    uint32_t Byte1 = UV & 0x0000FF00;
    uint32_t Byte2 = UV & 0x00FF0000;
    uint32_t Byte3 = UV & 0xFF000000;
    return (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
  } else if constexpr (sizeof(T) == 8) {
    uint64_t UV = V;
    uint64_t Hi = domino::byteswap<uint32_t>(UV);
    uint32_t Lo = domino::byteswap<uint32_t>(UV >> 32);
    return (Hi << 32) | Lo;
  } else {
    static_assert(!sizeof(T *), "Don't know how to handle the given type.");
    return 0;
  }
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] constexpr inline bool has_single_bit(T Value) noexcept {
  return (Value != 0) && ((Value & (Value - 1)) == 0);
}

namespace detail {
template <typename T, std::size_t SizeOfT> struct TrailingZerosCounter {
  static unsigned count(T Val) {
    if (!Val)
      return std::numeric_limits<T>::digits;
    if (Val & 0x1)
      return 0;

    // Bisection method.
    unsigned ZeroBits = 0;
    T Shift = std::numeric_limits<T>::digits >> 1;
    T Mask = std::numeric_limits<T>::max() >> Shift;
    while (Shift) {
      if ((Val & Mask) == 0) {
        Val >>= Shift;
        ZeroBits |= Shift;
      }
      Shift >>= 1;
      Mask >>= Shift;
    }
    return ZeroBits;
  }
};

#if defined(__GNUC__) || defined(_MSC_VER)
template <typename T> struct TrailingZerosCounter<T, 4> {
  static unsigned count(T Val) {
    if (Val == 0)
      return 32;

#if __has_builtin(__builtin_ctz) || defined(__GNUC__)
    return __builtin_ctz(Val);
#endif
  }
};

#if !defined(_MSC_VER) || defined(_M_X64)
template <typename T> struct TrailingZerosCounter<T, 8> {
  static unsigned count(T Val) {
    if (Val == 0)
      return 64;

#if __has_builtin(__builtin_ctzll) || defined(__GNUC__)
    return __builtin_ctzll(Val);
#elif defined(_MSC_VER)
    unsigned long Index;
    _BitScanForward64(&Index, Val);
    return Index;
#endif
  }
};
#endif
#endif
} // namespace detail

/// Count number of 0's from the least significant bit to the most
///   stopping at the first 1.
///
/// Only unsigned integral types are allowed.
///
/// Returns std::numeric_limits<T>::digits on an input of 0.
template <typename T> [[nodiscard]] int countr_zero(T Val) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return domino::detail::TrailingZerosCounter<T, sizeof(T)>::count(Val);
}

namespace detail {
template <typename T, std::size_t SizeOfT> struct LeadingZerosCounter {
  static unsigned count(T Val) {
    if (!Val)
      return std::numeric_limits<T>::digits;

    // Bisection method.
    unsigned ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};

} // namespace detail

/// Count number of 0's from the most significant bit to the least
///   stopping at the first 1.
///
/// Only unsigned integral types are allowed.
///
/// Returns std::numeric_limits<T>::digits on an input of 0.
template <typename T> [[nodiscard]] int countl_zero(T Val) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return domino::detail::LeadingZerosCounter<T, sizeof(T)>::count(Val);
}

/// Count the number of ones from the most significant bit to the first
/// zero bit.
///
/// Ex. countl_one(0xFF0FFF00) == 8.
/// Only unsigned integral types are allowed.
///
/// Returns std::numeric_limits<T>::digits on an input of all ones.
template <typename T> [[nodiscard]] int countl_one(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return domino::countl_zero<T>(~Value);
}

/// Count the number of ones from the least significant bit to the first
/// zero bit.
///
/// Ex. countr_one(0x00FF00FF) == 8.
/// Only unsigned integral types are allowed.
///
/// Returns std::numeric_limits<T>::digits on an input of all ones.
template <typename T> [[nodiscard]] int countr_one(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return domino::countr_zero<T>(~Value);
}

/// Returns the number of bits needed to represent Value if Value is nonzero.
/// Returns 0 otherwise.
///
/// Ex. bit_width(5) == 3.
template <typename T> [[nodiscard]] int bit_width(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return std::numeric_limits<T>::digits - domino::countl_zero(Value);
}

/// Returns the largest integral power of two no greater than Value if Value is
/// nonzero.  Returns 0 otherwise.
///
/// Ex. bit_floor(5) == 4.
template <typename T> [[nodiscard]] T bit_floor(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  if (!Value)
    return 0;
  return T(1) << (domino::bit_width(Value) - 1);
}

/// Returns the smallest integral power of two no smaller than Value if Value is
/// nonzero.  Returns 0 otherwise.
///
/// Ex. bit_ceil(5) == 8.
///
/// The return value is undefined if the input is larger than the largest power
/// of two representable in T.
template <typename T> [[nodiscard]] T bit_ceil(T Value) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  if (Value < 2)
    return 1;
  return T(1) << domino::bit_width<T>(Value - 1u);
}

namespace detail {
template <typename T, std::size_t SizeOfT> struct PopulationCounter {
  static int count(T Value) {
    // Generic version, forward to 32 bits.
    static_assert(SizeOfT <= 4, "Not implemented!");
    uint32_t v = Value;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return int(((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
  }
};

template <typename T> struct PopulationCounter<T, 8> {
  static int count(T Value) {
    uint64_t v = Value;
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return int((uint64_t)(v * 0x0101010101010101ULL) >> 56);
  }
};
} // namespace detail

/// Count the number of set bits in a value.
/// Ex. popcount(0xF000F000) = 8
/// Returns 0 if the word is zero.
template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] inline int popcount(T Value) noexcept {
  return detail::PopulationCounter<T, sizeof(T)>::count(Value);
}

}

#endif  // DOMINO_UTIL_BIT_H_
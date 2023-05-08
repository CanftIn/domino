#ifndef DOMINO_SUPPORT_HASH_MD5_H_
#define DOMINO_SUPPORT_HASH_MD5_H_

#include <domino/support/Endian.h>
#include <domino/util/StringRef.h>

#include <array>
#include <cstdint>

namespace domino {

template <unsigned N>
class SmallString;
template <typename T>
class ArrayRef;

class MD5 {
 public:
  struct MD5Result : public std::array<uint8_t, 16> {
    SmallString<32> digest() const;

    uint64_t low() const {
      // Our MD5 implementation returns the result in little endian, so the low
      // word is first.
      using namespace support;
      return endian::read<uint64_t, little, unaligned>(data());
    }

    uint64_t high() const {
      using namespace support;
      return endian::read<uint64_t, little, unaligned>(data() + 8);
    }
    std::pair<uint64_t, uint64_t> words() const {
      using namespace support;
      return std::make_pair(high(), low());
    }
  };

  MD5();

  /// Updates the hash for the byte stream provided.
  void update(ArrayRef<uint8_t> Data);

  /// Updates the hash for the StringRef provided.
  void update(StringRef Str);

  /// Finishes off the hash and puts the result in result.
  void final(MD5Result &Result);

  /// Finishes off the hash, and returns the 16-byte hash data.
  MD5Result final();

  /// Finishes off the hash, and returns the 16-byte hash data.
  /// This is suitable for getting the MD5 at any time without invalidating the
  /// internal state, so that more calls can be made into `update`.
  MD5Result result();

  /// Translates the bytes in \p Res to a hex string that is
  /// deposited into \p Str. The result will be of length 32.
  static void stringifyResult(MD5Result &Result, SmallVectorImpl<char> &Str);

  /// Computes the hash for a given bytes.
  static MD5Result hash(ArrayRef<uint8_t> Data);

 private:
  // Any 32-bit or wider unsigned integer data type will do.
  typedef uint32_t MD5_u32plus;

  // Internal State
  struct {
    MD5_u32plus a = 0x67452301;
    MD5_u32plus b = 0xefcdab89;
    MD5_u32plus c = 0x98badcfe;
    MD5_u32plus d = 0x10325476;
    MD5_u32plus hi = 0;
    MD5_u32plus lo = 0;
    uint8_t buffer[64];
    MD5_u32plus block[16];
  } InternalState;

  const uint8_t *body(ArrayRef<uint8_t> Data);
};

/// Helper to compute and return lower 64 bits of the given string's MD5 hash.
inline uint64_t MD5Hash(StringRef Str) {
  using namespace support;

  MD5 Hash;
  Hash.update(Str);
  MD5::MD5Result Result;
  Hash.final(Result);
  // Return the least significant word.
  return Result.low();
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_HASH_MD5_H_
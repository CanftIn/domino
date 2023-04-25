#ifndef DOMINO_SUPPORT_HASH_SHA1_H
#define DOMINO_SUPPORT_HASH_SHA1_H

#include <array>
#include <cstdint>

namespace domino {

template <typename T>
class ArrayRef;
class StringRef;

/// A class that wrap the SHA1 algorithm.
class SHA1 {
 public:
  SHA1() { init(); }

  /// Reinitialize the internal state
  void init();

  /// Digest more data.
  void update(ArrayRef<uint8_t> Data);

  /// Digest more data.
  void update(StringRef Str);

  /// Return the current raw 160-bits SHA1 for the digested data
  /// since the last call to init(). This call will add data to the internal
  /// state and as such is not suited for getting an intermediate result
  /// (see result()).
  std::array<uint8_t, 20> final();

  /// Return the current raw 160-bits SHA1 for the digested data
  /// since the last call to init(). This is suitable for getting the SHA1 at
  /// any time without invalidating the internal state so that more calls can be
  /// made into update.
  std::array<uint8_t, 20> result();

  /// Returns a raw 160-bit SHA1 hash for the given data.
  static std::array<uint8_t, 20> hash(ArrayRef<uint8_t> Data);

 private:
  /// Define some constants.
  /// "static constexpr" would be cleaner but MSVC does not support it yet.
  enum { BLOCK_LENGTH = 64 };
  enum { HASH_LENGTH = 20 };

  // Internal State
  struct {
    union {
      uint8_t C[BLOCK_LENGTH];
      uint32_t L[BLOCK_LENGTH / 4];
    } Buffer;
    uint32_t State[HASH_LENGTH / 4];
    uint32_t ByteCount;
    uint8_t BufferOffset;
  } InternalState;

  // Helper
  void writebyte(uint8_t data);
  void hashBlock();
  void addUncounted(uint8_t data);
  void pad();

  void final(std::array<uint32_t, HASH_LENGTH / 4> &HashResult);
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_HASH_SHA1_H
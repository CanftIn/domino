#ifndef DOMINO_SUPPORT_FILESYSTEM_UNIQUEID_H_
#define DOMINO_SUPPORT_FILESYSTEM_UNIQUEID_H_

#include <domino/util/DenseMapInfo.h>
#include <domino/util/Hashing.h>

#include <cstdint>
#include <utility>

namespace domino {
namespace sys {
namespace fs {

class UniqueID {
  uint64_t Device;
  uint64_t File;

 public:
  UniqueID() = default;
  UniqueID(uint64_t Device, uint64_t File) : Device(Device), File(File) {}

  bool operator==(const UniqueID &Other) const {
    return Device == Other.Device && File == Other.File;
  }
  bool operator!=(const UniqueID &Other) const { return !(*this == Other); }
  bool operator<(const UniqueID &Other) const {
    /// Don't use std::tie since it bloats the compile time of this header.
    if (Device < Other.Device) return true;
    if (Other.Device < Device) return false;
    return File < Other.File;
  }

  uint64_t getDevice() const { return Device; }
  uint64_t getFile() const { return File; }
};

}  // end namespace fs
}  // end namespace sys

// Support UniqueIDs as DenseMap keys.
template <>
struct DenseMapInfo<domino::sys::fs::UniqueID> {
  static inline domino::sys::fs::UniqueID getEmptyKey() {
    auto EmptyKey = DenseMapInfo<std::pair<uint64_t, uint64_t>>::getEmptyKey();
    return {EmptyKey.first, EmptyKey.second};
  }

  static inline domino::sys::fs::UniqueID getTombstoneKey() {
    auto TombstoneKey =
        DenseMapInfo<std::pair<uint64_t, uint64_t>>::getTombstoneKey();
    return {TombstoneKey.first, TombstoneKey.second};
  }

  static hash_code getHashValue(const domino::sys::fs::UniqueID &Tag) {
    return hash_value(std::make_pair(Tag.getDevice(), Tag.getFile()));
  }

  static bool isEqual(const domino::sys::fs::UniqueID &LHS,
                      const domino::sys::fs::UniqueID &RHS) {
    return LHS == RHS;
  }
};

}  // end namespace domino

#endif  // LLVM_SUPPORT_FILESYSTEM_UNIQUEID_H

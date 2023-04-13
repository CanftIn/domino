#ifndef DOMINO_UTIL_STRINGREF_H_
#define DOMINO_UTIL_STRINGREF_H_

#include <domino/util/iterator_range.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>

namespace domino {

class StringRef {
 public:
  static constexpr size_t npos = ~size_t(0);

  using iterator = const char*;
  using const_iterator = const char*;
  using size_type = size_t;

 private:
  const char* Data = nullptr;

  size_t Length = 0;

  static int compareMemory(const char* Lhs, const char* Rhs, size_t Length) {
    if (Length == 0) return 0;
    return ::memcmp(Lhs, Rhs, Length);
  }

 public:
  StringRef() = default;

  StringRef(std::nullptr_t) = delete;

  constexpr StringRef(const char* Str)
      : Data(Str), Length(Str ? std::char_traits<char>::length(Str) : 0) {}

  constexpr StringRef(const char* data, size_t length)
      : Data(data), Length(length) {}

  StringRef(const std::string& Str) : Data(Str.data()), Length(Str.length()) {}

  constexpr StringRef(std::string_view Str)
      : Data(Str.data()), Length(Str.size()) {}

  iterator begin() const { return Data; }

  iterator end() const { return Data + Length; }

  const unsigned char* bytes_begin() const {
    return reinterpret_cast<const unsigned char*>(begin());
  }

  const unsigned char* bytes_end() const {
    return reinterpret_cast<const unsigned char*>(end());
  }

  iterator_range<const unsigned char*> bytes() const {
    return make_range(bytes_begin(), bytes_end());
  }

  [[nodiscard]] const char* data() const { return Data; }

  [[nodiscard]] constexpr bool empty() const { return Length == 0; }

  [[nodiscard]] constexpr size_t size() const { return Length; }

  [[nodiscard]] char front() const {
    assert(!empty());
    return Data[0];
  }

  [[nodiscard]] char back() const {
    assert(!empty());
    return Data[Length - 1];
  }
};

}  // namespace domino

#endif  // DOMINO_UTIL_STRINGREF_H_